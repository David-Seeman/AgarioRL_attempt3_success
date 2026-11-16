"""Non-periodic creative splitruns by choreography + beam search on the exact simulator.

A run is a sequence of split EVENTS. Each event is one designed move:
  circle   the actor aims at a chosen partner piece (plus an offset)            -- the basic circle shot
  trim     the actor splits off-circle (70-180 deg from the line to the partner) -- shrinks its pieces
  double   a double split aimed at a chosen partner piece
  between  aimed at the gap between two partner pieces
  push     aimed into one of the actor's OWN pieces (its launched half shoves that piece forward)
  both     both players split in the same tick (circle shots)
Parameters per event: wait before the split, aim offset, cursors of both players before and after.
The exact simulator plays each candidate; a BEAM of the best partial runs is kept, so dead ends
(a split whose pieces never take part in an eat, a slow absorption, a drained player, a player pressing
split at the 16-cell cap, more than IDLE_MAX ticks without a split) are pruned and other branches go on.

Ranking (cumulative): + purposeful splits, + young-piece eats, + style bonus per creative move,
- downtime (ticks between splits), - crumbs and excess cells, - mass imbalance, - drift.

    python tools/choreo.py --name mixed --style '{"trim": 3, "double": 3, "between": 2, "push": 3}'
"""
import argparse, json, math, os, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from splitrun.sim import Sim, X, Y, SZ, EX, OWN, BIRTH, CID, G_TICK, BDX, BDY, BD, BOOSTING  # noqa
from splitrun.rlenv import SplitrunEnv  # noqa
from splitrun.replay import Recorder, write_index  # noqa
from tools.freerun import register_split, score_tick, MAXP, MAXID  # noqa
from splitrun import balls as BL  # noqa
BALLS = 0              # number of neutral balls (feeding mode); 0 = none
CHASE = None           # feeding: world position of the target ball; 'chase' candidates lean towards it
CHASE_M = 0.0          # feeding: the target ball's mass
MINSHARE = 0.12
YOUNG = 60             # a cross eat needs a piece younger than this (ticks); huge = allow old-old eats (sweeping)
DIR = np.array([1.0, 0.0])   # dynamic mode: the wanted travel direction (world frame); "ftrim" aims along it

TOTAL = 37000.0
IDLE_MAX = 50
USE_WINDOW = 50        # ticks a split's pieces have to take part in an eat (feeding mode: longer)
MOVES = ("circle", "trim", "double", "between", "push", "cross", "triple", "flythrough", "striple", "wait", "feed", "dash", "strike", "capture", "swallow", "dash2", "ftrim")
KIND = {"double": 2, "triple": 3, "striple": 3}
SCALE = 1.0            # total mass / 37,000 (crumb thresholds scale with it)
AFTER = 4               # ticks simulated after a split before the next event is planned
CK = ("pids", "pn", "pl", "pdl", "pok", "last_split", "eats", "olds", "hold_t", "hold_xy")
HOLD = 22               # after a straight triple the splitter's cursor stays locked on the aim ("n" in imsolo)


class Batch:
    """N worlds + the rule bookkeeping"""
    def __init__(self, N):
        self.N = N
        self.sim = Sim(N)
        self.pids = np.zeros((N, MAXP, MAXID), np.int64); self.pn = np.zeros((N, MAXP), np.int64)
        self.pl = np.zeros((N, MAXP, MAXID), np.int64); self.pdl = np.full((N, MAXP), -1, np.int64)
        self.pok = np.zeros((N, MAXP), np.int64); self.dead = np.zeros(N, bool); self.score = np.zeros(N)
        self.eats = np.zeros(N, np.int64); self.olds = np.zeros(N, np.int64)
        self.last_split = np.zeros(N, np.int64); self.purp = np.zeros(N); self.young = np.zeros(N)
        self.hold_t = np.zeros((N, 2), np.int64); self.hold_xy = np.zeros((N, 2, 2))
        self.gained = np.zeros(N); self.gained_small = np.zeros(N); self.lost = np.zeros(N)
        self.halved = np.zeros(N, bool); self.dasher = np.full(N, -1, np.int64)
        if BALLS:
            # planning batches get their balls copied in by load_from(); only fresh worlds spawn them
            self.bseed = np.zeros(N, np.int64)
            self.bx = np.zeros((N, BALLS)); self.by = np.zeros((N, BALLS)); self.bs = np.zeros((N, BALLS))
            self.bn = np.zeros((N, BALLS), np.int64); self.bt = np.zeros((N, BALLS), np.int64)

    def load_from(self, states, idx):
        """world i <- states[idx[i]] (each state: (sim arrays of 1 world, bookkeeping of 1 world))"""
        for i, j in enumerate(idx):
            ws, cs = states[j]
            for k, v in ws.items():
                getattr(self.sim, k)[i] = v
            for k, v in cs.items():
                getattr(self, k)[i] = v
        self.dead[:] = False
        self.score[:] = 0
        self.purp[:] = 0                          # per-event purpose reward (the node carries the cumulative score)
        self.gained[:] = 0                        # per-event mass eaten from balls (big / small)
        self.gained_small[:] = 0; self.lost[:] = 0; self.halved[:] = False; self.dasher[:] = -1

    def keys(self):
        return CK + (("bx", "by", "bs", "bn", "bt", "bseed") if BALLS else ())

    def state(self, i):
        return ({k: getattr(self.sim, k)[i].copy() for k in SplitrunEnv.SIM_ARRAYS},
                {k: getattr(self, k)[i].copy() for k in self.keys()})

    def step(self, mouse, splits, run=None):
        s = self.sim
        live = ~self.dead if run is None else (~self.dead & run)
        s.active[:] = live.astype(np.int64)
        splits = splits * live[:, None]
        s.step(mouse, splits)
        t = s.g[:, G_TICK]
        for i in np.nonzero(splits.any(1) & live)[0]:
            for p in (0, 1):
                if splits[i, p]:
                    before = self.score[i]
                    register_split(s.ci, s.oc, s.ocn, s.g, i, p, self.pids, self.pn, self.pdl, self.pok,
                                   self.dead, self.score, self.pl, USE_WINDOW)
            self.last_split[i] = t[i]
        before = self.score.copy()
        s.evn[~live] = 0
        score_tick(s.g, s.ev, s.evn, s.active, self.pids, self.pn, self.pdl, self.pok, self.dead, self.score,
                   self.eats, self.olds, self.pl, 0.0, YOUNG)
        if BALLS:
            BL.balls_tick(s.cf, s.ci, s.pc, s.pcn, s.bc, s.bcn, s.oc, s.ocn, s.nf, s.ni, s.nit, s.g, s.P, s.active,
                          self.bx, self.by, self.bs, self.bn, self.bt, self.bseed, self.pids, self.pn, self.pdl, self.pok,
                          self.score, self.gained, self.gained_small, self.lost, 6.0)
        self.purp += np.where(self.dead, 0, self.score - before)      # purpose rewards (PURPOSE per launched use, EAT per eat)
        ex = s.ci[:, :, EX] == 1; own = s.ci[:, :, OWN]; m = s.cf[:, :, SZ] ** 2 * ex
        mA = (m * (own == 0)).sum(1); mB = (m * (own == 1)).sum(1)
        bad = (np.minimum(mA, mB) < MINSHARE * (mA + mB)) | (t - self.last_split > IDLE_MAX)
        bad |= (s.ocn[:, 0] == 0) | (s.ocn[:, 1] == 0)
        self.dead |= bad & live


def projected_com(sim):
    """team centre of mass with every boosting piece counted where its boost will carry it (so a forward trim is
    credited the moment it is thrown, not ~1 s later)"""
    ex = sim.ci[:, :, EX] == 1
    m = sim.cf[:, :, SZ] ** 2 * ex
    bo = (sim.ci[:, :, BOOSTING] == 1) & ex
    px = sim.cf[:, :, X] + np.where(bo, sim.cf[:, :, BDX] * sim.cf[:, :, BD], 0)
    py = sim.cf[:, :, Y] + np.where(bo, sim.cf[:, :, BDY] * sim.cf[:, :, BD], 0)
    M = np.maximum(m.sum(1), 1)
    return np.stack([(px * m).sum(1) / M, (py * m).sum(1) / M], 1)


def pieces(sim, i, p):
    return [(int(sim.ci[i, s_, CID]), sim.cf[i, s_, X], sim.cf[i, s_, Y], sim.cf[i, s_, SZ]) for s_ in sim.oc[i, p, :sim.ocn[i, p]]]


def com(sim, i, p):
    P = pieces(sim, i, p)
    m = np.array([c[3] ** 2 for c in P])
    return np.array([sum(c[1] * c[3] ** 2 for c in P), sum(c[2] * c[3] ** 2 for c in P)]) / max(m.sum(), 1)


def sample(rng, n, style):
    w = np.array([0.0 if (mv in style.get("exclude", ()) or (mv in ("wait", "strike", "feed", "dash", "capture", "swallow", "dash2", "ftrim") and mv not in style))
                  else 1.0 + (0 if mv == "circle" else style.get(mv, 0) * 0.5) for mv in MOVES])
    w /= w.sum()
    mv = rng.choice(len(MOVES), n, p=w)
    wait = rng.integers(0, style.get("gapmax", 12) + 1, n)
    iw = mv == MOVES.index("wait")
    wait[iw] = rng.integers(8, style.get("waitmax", 50) + 1, iw.sum())       # a pause: reorient / let pieces remerge
    ist = mv == MOVES.index("strike")
    wait[ist] = rng.integers(8, 41, ist.sum())                               # strike: drive onto the target ball
    isw = mv == MOVES.index("swallow")
    wait[isw] = rng.integers(6, 31, isw.sum())                               # swallow: the eater's owner drives it in
    return dict(move=mv, actor=rng.integers(0, 2, n), wait=wait,
                j=rng.integers(0, 64, (n, 2)), off=rng.normal(0, 1, (n, 2)) * 350, rho=rng.uniform(400, 2500, n),
                phi=rng.choice([-1, 1], n) * rng.uniform(math.radians(80), math.radians(150), n),   # sideways / slightly back
                c1=rng.uniform(-1, 1, (n, 4)) * 1500, c2=rng.uniform(-1, 1, (n, 4)) * 1500,
                chase=((rng.random(n) < style.get("chase_p", 0.0)) | (mv == MOVES.index("strike")) | (mv == MOVES.index("capture"))).astype(np.int64),
                # the chase point is part of the move, so a committed move replays exactly as planned
                fkind=rng.choice([1, 2, 3], n, p=[0.4, 0.4, 0.2]),
                dt=rng.integers(1, 6, n),
                chase_m=np.full(n, float(CHASE_M)),
                lead=rng.uniform(0, 150, n),
                fang=rng.uniform(-1, 1, n) * math.radians(style.get("ftrim_ang", 50)),
                fpiece=rng.integers(0, 3, n),
                chase_xy=np.tile(np.asarray(CHASE if CHASE is not None else (0.0, 0.0), float), (n, 1)))


def aim(sim, i, a, mv, P):
    """cursor for the split of actor a in world i"""
    q = 1 - a
    A, B = pieces(sim, i, a), pieces(sim, i, q)
    if not A or not B:
        return None
    ca, cb = com(sim, i, a), com(sim, i, q)
    u = cb - ca; u = u / max(np.linalg.norm(u), 1e-6); w = np.array([-u[1], u[0]])
    big = max(A, key=lambda c: c[3])
    Bs = sorted(B, key=lambda c: -c[3])
    name = MOVES[mv]
    if name in ("circle", "double", "cross"):
        t = Bs[P["j"][0] % len(Bs)]
        return np.array(t[1:3]) + P["off"]
    if name == "triple":
        # a straight triple split: the cursor far away, so every launch is parallel (all 8 pieces in a line)
        t = Bs[P["j"][0] % len(Bs)]
        d = np.array(t[1:3]) + P["off"] * 0.3 - np.array(big[1:3])
        return np.array(big[1:3]) + d / max(np.linalg.norm(d), 1e-6) * 30000
    if name in ("feed", "capture") and "chase_xy" in P:
        # feed the partner's piece that is closest to the target ball (the dasher): aim a little ahead of it along
        # its way to the target, so the launched pieces meet it while it travels; its growth lets it swallow
        cxy = np.asarray(P["chase_xy"], float)
        t = min(Bs, key=lambda c: (c[1] - cxy[0]) ** 2 + (c[2] - cxy[1]) ** 2)
        tp = np.array(t[1:3]); dv = cxy - tp; dv = dv / max(np.linalg.norm(dv), 1e-6)
        return tp + dv * P.get("lead", 0.0) + P["off"] * 0.2
    if name == "dash" and "chase_xy" in P:
        # split towards the target ball (the partner catches / feeds on the launched halves as usual)
        cxy = np.asarray(P["chase_xy"], float)
        d = cxy - np.array(big[1:3])
        return np.array(big[1:3]) + d / max(np.linalg.norm(d), 1e-6) * 3000 + P["off"] * 0.5
    if name == "striple":
        # STRAIGHT triple split: only when all of the actor's cells lie on one line pointing at the target
        # (1 cell, or 2 cells lined up); the cursor goes far along that line so all 8 (16) pieces form a line
        if len(A) > 2:
            return None
        t = Bs[P["j"][0] % len(Bs)]
        o = np.array(big[1:3])
        if len(A) == 1:
            d = np.array(t[1:3]) - o
        else:
            other = [c for c in A if c[0] != big[0]][0]
            d = o - np.array(other[1:3])
            if d @ (np.array(t[1:3]) - o) < 0:
                d = -d
            tt = np.array(t[1:3]) - o
            if abs(d[0] * tt[1] - d[1] * tt[0]) / (np.linalg.norm(d) * max(np.linalg.norm(tt), 1)) > math.sin(math.radians(60)):
                return None                                          # target not along the line
        return o + d / max(np.linalg.norm(d), 1e-6) * 30000
    if name == "flythrough":
        # aim at a partner piece that has another partner piece in the way, which the launched half can
        # neither eat nor be eaten by (size ratio < 1.14 both ways): it flies over it to reach the target
        half = big[3] / math.sqrt(2)
        for k in range(len(Bs)):
            t = Bs[(P["j"][0] + k) % len(Bs)]
            if half < 1.14 * t[3]:
                continue
            o = np.array(big[1:3]); d = np.array(t[1:3]) - o; L = np.linalg.norm(d)
            if L < 1 or L > 1400:
                continue
            for b_ in Bs:
                if b_[0] == t[0]:
                    continue
                v = np.array(b_[1:3]) - o; s_ = v @ d / L ** 2
                if 0.1 < s_ < 0.9 and abs(v[0] * d[1] - v[1] * d[0]) / L < b_[3] and half < 1.14 * b_[3] and b_[3] < 1.14 * half:
                    return np.array(t[1:3])
        return None
    if name == "between":
        if len(Bs) < 2:
            return None
        t1 = Bs[P["j"][0] % len(Bs)]; t2 = Bs[P["j"][1] % len(Bs)]
        if t1[0] == t2[0]:
            t2 = Bs[(P["j"][0] + 1) % len(Bs)]
        return (np.array(t1[1:3]) + np.array(t2[1:3])) / 2 + P["off"] * 0.5
    if name == "push":
        own = [c for c in A if c[0] != big[0]]
        if not own:
            return None
        t = own[P["j"][0] % len(own)]
        return np.array(t[1:3]) + P["off"] * 0.5
    if name == "ftrim":
        # FORWARD TRIM (world frame): split along the wanted travel direction DIR (turned by a small angle), from
        # the actor's biggest (fpiece 0), rearmost (1) or frontmost (2) piece; far cursor (rho > 2000) = all
        # cells launch in parallel
        a_ = P.get("fang", 0.0); d = np.array([math.cos(a_) * DIR[0] - math.sin(a_) * DIR[1], math.sin(a_) * DIR[0] + math.cos(a_) * DIR[1]])
        fp = int(P.get("fpiece", 0))
        src = big if fp == 0 else (min(A, key=lambda c: c[1] * DIR[0] + c[2] * DIR[1]) if fp == 1 else max(A, key=lambda c: c[1] * DIR[0] + c[2] * DIR[1]))
        return np.array(src[1:3]) + d * (P["rho"] if P["rho"] < 2000 else 30000)
    # trim: off-circle from the largest piece
    c, s_ = math.cos(P["phi"]), math.sin(P["phi"])
    return np.array(big[1:3]) + P["rho"] * (c * u + s_ * w)


def play(batch, cand, T_after=AFTER, rec=None):
    """play one event per world: wait, split (the move), then T_after ticks. Returns tick of the split."""
    N = batch.N; sim = batch.sim
    fire_t = np.full(N, -1)
    ex0 = sim.ci[:, :, EX] == 1; m0 = sim.cf[:, :, SZ] ** 2 * ex0
    batch.com0 = projected_com(sim)
    frames = []
    ca = np.stack([com(sim, i, 0) for i in range(N)]); cb = np.stack([com(sim, i, 1) for i in range(N)])
    u = cb - ca; u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-6); w = np.stack([-u[:, 1], u[:, 0]], 1)
    fr = lambda org, xy: org + xy[:, :1] * u + xy[:, 1:2] * w
    A1, B1 = fr(ca, cand["c1"][:, 0:2]), fr(cb, cand["c1"][:, 2:4])
    A2, B2 = fr(ca, cand["c2"][:, 0:2]), fr(cb, cand["c2"][:, 2:4])
    if "chase" in cand and cand["chase"].any():
        # lean towards the target ball: each player's cursor sits beyond its own centre in the direction of the
        # target, so the formation keeps its shape and drifts that way while it splitruns (cursors ON the ball
        # would drag every piece, small ones first, into it)
        ch = cand["chase"].astype(bool)
        mid = (ca + cb) / 2
        dv = cand["chase_xy"] - mid
        dv = dv / np.maximum(np.linalg.norm(dv, axis=1, keepdims=True), 1e-6)
        L = 400 + np.abs(cand["c1"][:, :1]) * 0.8
        A1[ch] = (ca + dv * L + cand["c2"][:, 0:2] * 0.3)[ch]; B1[ch] = (cb + dv * L + cand["c2"][:, 2:4] * 0.3)[ch]
        A2[ch] = A1[ch]; B2[ch] = B1[ch]
    sw_ = cand["move"] == MOVES.index("swallow")
    for i in np.nonzero(sw_)[0]:
        # the player owning the piece that can already eat the target (else the biggest safe one) aims exactly at it
        cxy = np.asarray(cand["chase_xy"][i], float); cm = float(cand["chase_m"][i])
        best, bd = -1, 1e18
        for p_ in (0, 1):
            for c_ in pieces(sim, i, p_):
                mm = c_[3] ** 2 / 100
                d_ = math.hypot(c_[1] - cxy[0], c_[2] - cxy[1]) - (1e6 if mm >= 1.32 * cm else 0)
                if mm >= 0.77 * cm and d_ < bd:
                    bd, best = d_, p_
        if best == 0:
            A1[i] = cxy; A2[i] = cxy
        elif best == 1:
            B1[i] = cxy; B2[i] = cxy
    T = int(cand["wait"].max()) + 1 + T_after + (int(cand["dt"].max()) if "dt" in cand else 0)
    kinds = np.array([KIND.get(MOVES[m], 1) for m in cand["move"]])
    if "fkind" in cand:
        # feeds (and captures) can be single, double or triple splits: more, smaller pieces thrown into the dasher
        fm = (cand["move"] == MOVES.index("feed")) | (cand["move"] == MOVES.index("capture"))
        kinds = np.where(fm, cand["fkind"], kinds)
    cap = np.zeros(N, bool)
    for t in range(T):
        mouse = np.zeros((N, 2, 2)); sp = np.zeros((N, 2), np.int64)
        pre = (t <= cand["wait"])[:, None]
        mouse[:, 0] = np.where(pre, A1, A2); mouse[:, 1] = np.where(pre, B1, B2)
        # a player whose straight triple is still settling keeps its cursor locked on the line
        now_t = sim.g[:, G_TICK]
        for p_ in (0, 1):
            h = now_t < batch.hold_t[:, p_]
            if h.any():
                mouse[h, p_] = batch.hold_xy[h, p_]
        nosplit = (cand["move"] == MOVES.index("wait")) | (cand["move"] == MOVES.index("strike")) | (cand["move"] == MOVES.index("swallow"))
        for i in np.nonzero((t == cand["wait"]) & ~batch.dead & ~nosplit)[0]:
            P = {k: v[i] for k, v in cand.items()}
            mv = cand["move"][i]
            actors = (0, 1) if MOVES[mv] in ("cross", "dash2") else (cand["actor"][i],)
            cap[i] = any(sim.ocn[i, a_] >= 14 for a_ in actors)      # selective split at the 16-cell cap
            if P.get("chase_m", 0) > 0 and MOVES[mv] not in ("wait", "strike", "swallow", "capture"):
                # the dasher: a piece that can sit on the target safely (>= 0.77x its mass), nearest to it, if close
                cxy = np.asarray(P["chase_xy"], float); best, bd = -1, 1800.0
                for p_ in (0, 1):
                    for c_ in pieces(sim, i, p_):
                        if c_[3] ** 2 / 100 >= 0.77 * P["chase_m"]:
                            d_ = math.hypot(c_[1] - cxy[0], c_[2] - cxy[1])
                            if d_ < bd:
                                bd, best = d_, p_
                if best >= 0 and best in actors:
                    batch.halved[i] = True          # its owner splits: the dasher would be halved
            for a in actors:
                c = aim(sim, i, a, MOVES.index("dash") if MOVES[mv] in ("capture", "dash2") else mv, P)
                if c is None:
                    continue
                mouse[i, a] = c; sp[i, a] = 1 if MOVES[mv] == "capture" else kinds[i]
                if MOVES[mv] == "striple":
                    batch.hold_t[i, a] = sim.g[i, G_TICK] + 1 + HOLD; batch.hold_xy[i, a] = c
            if sp[i].any():
                fire_t[i] = sim.g[i, G_TICK] + 1
        # CAPTURE, second half: dt ticks after the dash the partner throws mass (single / double / triple split)
        # at the dasher mid-flight, aimed a little ahead of it towards the target
        capm = cand["move"] == MOVES.index("capture")
        for i in np.nonzero(capm & (t == cand["wait"] + cand["dt"]) & ~batch.dead & (batch.dasher >= 0))[0]:
            a = cand["actor"][i]; q = 1 - a
            dz = [c_ for c_ in pieces(sim, i, a) if c_[0] == batch.dasher[i]]
            if not dz:
                continue
            dxy = np.array(dz[0][1:3]); cxy = np.asarray(cand["chase_xy"][i], float)
            dv = cxy - dxy; dv = dv / max(np.linalg.norm(dv), 1e-6)
            mouse[i, q] = dxy + dv * cand["lead"][i] * 2
            sp[i, q] = cand["fkind"][i]
        run_until = cand["wait"] + 1 + T_after + np.where(capm, cand["dt"], 0)
        batch.step(mouse, sp, run=t < run_until)
        for i in np.nonzero(capm & (t == cand["wait"]) & (sp.sum(1) > 0))[0]:
            # the dasher = the actor's newly launched piece closest to the target
            a = cand["actor"][i]; cxy = np.asarray(cand["chase_xy"][i], float)
            born = [(int(sim.ci[i, s_, CID]), sim.cf[i, s_, X], sim.cf[i, s_, Y]) for s_ in sim.oc[i, a, :sim.ocn[i, a]]
                    if sim.ci[i, s_, BIRTH] == sim.g[i, G_TICK]]
            if born:
                batch.dasher[i] = min(born, key=lambda b_: (b_[1] - cxy[0]) ** 2 + (b_[2] - cxy[1]) ** 2)[0]
        if rec is not None:
            rec.after_step(np.trunc(mouse[0]), sp[0])
            if BALLS:      # balls are recorded as owner 2 (ids change when a ball respawns)
                rec.data["frames"][-1].extend([[1_000_000 + k * 10_000 + int(batch.bn[0, k]) % 10_000, 2,
                                                round(float(batch.bx[0, k]), 1), round(float(batch.by[0, k]), 1),
                                                round(float(batch.bs[0, k]), 2), 0] for k in range(BALLS)])
    batch.dead |= (fire_t < 0) & (cand["move"] != MOVES.index("wait")) & (cand["move"] != MOVES.index("strike")) \
        & (cand["move"] != MOVES.index("swallow"))
    batch.cap = cap
    return fire_t


def path_target(tick, center0, style):
    """target point of a dynamic splitrun at server tick `tick`: a loop of radius R around the map centre,
    travelled at `speed` units/s, starting where the pair starts (center0 lies on the loop)"""
    R = style.get("radius", 4500.0); v = style.get("speed", 150.0)
    c0 = np.asarray(center0, float)
    mid = np.asarray(style.get("mid", (0.0, 0.0)), float)       # loop around the map centre
    a0 = math.atan2(c0[1] - mid[1], c0[0] - mid[0])
    ang = a0 + (np.asarray(tick, float) - style.get("t0", 0)) / 25.0 * v / R
    return mid[0] + R * np.cos(ang), mid[1] + R * np.sin(ang)


def score_event(b, cand, style, center0=None):
    """score of one event per world (and prune worlds that break the tidiness limits)"""
    s = b.sim
    ex = s.ci[:, :, EX] == 1; own = s.ci[:, :, OWN]; m = s.cf[:, :, SZ] ** 2 / 100 * ex
    mA = (m * (own == 0)).sum(1); mB = (m * (own == 1)).sum(1)
    share = np.minimum(mA, mB) / np.maximum(mA + mB, 1)
    nA = (ex & (own == 0)).sum(1); nB = (ex & (own == 1)).sum(1)
    sc_ = ((mA + mB) / 37000.0)[:, None]                  # crumb thresholds follow the team's current mass
    crumbs = (ex & (m < 1000 * sc_)).sum(1)
    tiny = (ex & (m < 500 * sc_)).sum(1)
    b.dead |= nA > style.get("hard_cells", 10)
    b.dead |= nB > style.get("hard_cells", 10)
    b.dead |= tiny > style.get("max_tiny", 2)
    gap = cand["wait"] + 1
    names = np.array([MOVES[x] for x in cand["move"]])
    creative = np.array([style.get(x, 0.0) for x in names]) + style.get("capsplit", 0.0) * b.cap
    creative = creative + style.get("food_w", 0.0) * b.gained - style.get("lost_w", 0.0) * b.lost
    if BALLS:
        b.dead |= b.lost > style.get("max_loss", 0.04) * (mA + mB + b.lost)       # never give away real mass
    if BALLS and style.get("danger_w", 0):
        dg = np.zeros(b.N)
        BL.ball_danger(s.cf, s.oc, s.ocn, b.bx, b.by, b.bs, dg)
        creative = creative - style["danger_w"] * dg
    if BALLS and style.get("seek_k", -1) >= 0:
        # hunt the chosen worthy ball without stopping the splitrun: bring a piece that can sit on it safely
        # (>= 0.77x its mass: neither can eat the other) onto it; one quick feed from the partner then takes
        # that piece past 1.3x and it swallows the ball at once
        k_ = style["seek_k"]
        tm = b.bs[:, k_] ** 2 / 100
        d_ = np.hypot(s.cf[:, :, X] - b.bx[:, k_][:, None], s.cf[:, :, Y] - b.by[:, k_][:, None])
        cand_p = ex & (m >= 0.77 * tm[:, None])
        d_c = np.where(cand_p, d_, 1e9).min(1)
        d_big = d_[np.arange(b.N), m.argmax(1)]
        creative = creative - style.get("seek_w", 0.01) * np.where(cand_p.any(1), d_c, d_big + 2000)
        creative = creative - style.get("halve_w", 25.0) * b.halved
        # the partner (the other player) keeps a piece within feeding range of the dasher
        own_ = s.ci[:, :, OWN]
        di = np.where(cand_p, d_, 1e9).argmin(1)
        dx_ = s.cf[np.arange(b.N), di, X]; dy_ = s.cf[np.arange(b.N), di, Y]; downer = own_[np.arange(b.N), di]
        dp = np.hypot(s.cf[:, :, X] - dx_[:, None], s.cf[:, :, Y] - dy_[:, None])
        dp = np.where(ex & (own_ != downer[:, None]), dp, 1e9).min(1)
        near = cand_p.any(1) & (d_c < 2500)
        creative = creative - style.get("feeder_w", 0.01) * np.where(near, np.maximum(dp - 800, 0), 0)
    creative = creative + style.get("few_cells", 0.0) * ((nA <= 2) | (nB <= 2))     # straight triples need 1-2 cells
    ev = b.purp - 0.35 * gap + creative - style.get("crumb_w", 4.0) * crumbs - 4.0 * np.maximum(nA - style.get("max_cells", 6), 0) \
        - 4.0 * np.maximum(nB - style.get("max_cells", 6), 0) - 120 * np.maximum(style.get("share_min", 0.35) - share, 0) \
        - 120 * np.maximum(share - style.get("share_max", 0.5), 0)
    if center0 is not None:
        cx = (s.cf[:, :, X] * m).sum(1) / np.maximum(m.sum(1), 1); cy = (s.cf[:, :, Y] * m).sum(1) / np.maximum(m.sum(1), 1)
        if style.get("path") == "dir":
            # FAST TRANSLATION: reward how far the team's centre advanced along the wanted direction during this
            # event; a little penalty for sideways drift. Near the waypoint: hold there (stationary).
            d = np.asarray(style["dir"], float)
            mv_ = projected_com(s) - b.com0
            prog = mv_ @ d
            lat = np.abs(mv_[:, 0] * d[1] - mv_[:, 1] * d[0])
            ev = ev + style.get("speed_w", 0.05) * prog - style.get("lat_w", 0.02) * lat
        elif style.get("path") == "hold":
            wp = np.asarray(style["waypoint"], float)
            ev = ev - style.get("drift_w", 0.01) * np.maximum(np.hypot(cx - wp[0], cy - wp[1]) - 600, 0)
        elif style.get("path") == "loop":
            # DYNAMIC splitrun: the pair's centre follows a target moving around a big loop
            tx, ty = path_target(s.g[:, G_TICK], center0, style)
            dist = np.hypot(cx - tx, cy - ty)
            ev = ev - style.get("path_w", 0.03) * np.maximum(dist - 600, 0)
        else:
            # stationary: the pair's centre of mass should not wander far from where it started
            drift = np.hypot(cx - center0[0], cy - center0[1])
            ev = ev - style.get("drift_w", 0.01) * np.maximum(drift - 1000, 0)
    return ev


def run(name, start_rows, style, ticks=750, beam=6, M=48, seed=0, log=print):
    rng = np.random.default_rng(seed)
    root = Batch(1)
    root.sim.load(0, start_rows, seed=1)
    root.last_split[:] = root.sim.g[:, G_TICK]
    t_start = int(root.sim.g[0, G_TICK])
    m0 = root.sim.cf[0, :, SZ] ** 2 * (root.sim.ci[0, :, EX] == 1)
    center0 = np.array([(root.sim.cf[0, :, X] * m0).sum(), (root.sim.cf[0, :, Y] * m0).sum()]) / m0.sum()
    # a beam node: (state, cumulative score, action list)
    nodes = [(root.state(0), 0.0, [])]
    best_done = None
    t0 = time.time()
    it = 0
    history = []                      # beams of earlier iterations (for backtracking)
    style0 = dict(style)
    backtracks = 0
    while nodes:
        it += 1
        history.append(nodes)
        N = len(nodes) * M
        b = Batch(N)
        b.load_from([n[0] for n in nodes], np.repeat(np.arange(len(nodes)), M))
        cand = sample(rng, N, style)
        fire_t = play(b, cand)
        ev_score = score_event(b, cand, style, center0)
        parent = np.repeat(np.arange(len(nodes)), M)
        total = np.array([nodes[p][1] for p in parent]) + ev_score
        total[b.dead] = -1e18
        alive = np.nonzero(~b.dead)[0]
        if len(alive) == 0:
            # every branch died: go back ~10 events, fresh random candidates, a bit less style pressure
            if backtracks >= 40 or len(history) < 2:
                break
            backtracks += 1
            back = min(len(history) - 1, 10)
            nodes = history[-back - 1]
            del history[-back - 1:]
            for k_ in ("trim", "double", "between", "push", "cross", "triple", "flythrough", "capsplit", "striple"):
                if k_ in style:
                    style[k_] = style[k_] * 0.8
            log(f"   {name}: all branches died at event {it}; back {back} events (backtrack {backtracks})")
            continue
        # keep the best `beam` children, at most 2 per parent (diversity)
        order = alive[np.argsort(-total[alive])]
        chosen, per = [], {}
        for i in order:
            if per.get(parent[i], 0) >= 2:
                continue
            per[parent[i]] = per.get(parent[i], 0) + 1
            chosen.append(i)
            if len(chosen) >= beam:
                break
        new = []
        for i in chosen:
            act = {k: (v[i].copy() if isinstance(v, np.ndarray) and v.ndim > 1 else v[i].item()) for k, v in cand.items()}
            new.append((b.state(i), float(total[i]), nodes[parent[i]][2] + [act]))
        nodes = new
        el = int(b.sim.g[chosen[0], G_TICK]) - t_start
        if el >= ticks:
            best_done = nodes[0]
            break
        if it % 20 == 0:
            log(f"   {name}: {el / 25:.1f} s, {len(nodes[0][2])} events, best score {nodes[0][1]:.0f} ({time.time() - t0:.0f}s)")
    final = best_done or (nodes[0] if nodes else (history[-1][0] if history else None))
    return final, t_start


def replay(name, start_rows, actions, description=""):
    """re-play a node's action list in a fresh world (deterministic) and record it"""
    b = Batch(1)
    b.sim.load(0, start_rows, seed=1)
    b.last_split[:] = b.sim.g[:, G_TICK]
    rec = Recorder(b.sim, 0, name, description)
    moves = []
    for a in actions:
        cand = {k: np.array([v]) for k, v in a.items()}
        ft = play(b, cand, rec=rec)
        moves.append(dict(tick=int(ft[0]), move=MOVES[a["move"]], actor="AB"[a["actor"]] if MOVES[a["move"]] != "cross" else "AB",
                          cap=bool(b.cap[0])))
    return rec, moves, bool(b.dead[0])


def rows_from_replay(path, frame=0, total=None):
    """a replay frame as a start layout; with `total`, masses are scaled to that total and distances by the
    same factor as the sizes (sqrt of the mass factor), so the layout keeps its shape"""
    r = json.load(open(path))
    cells = r["frames"][frame]
    f = 1.0 if total is None else total / sum(c[4] ** 2 / 100 for c in cells)
    g = math.sqrt(f)
    cx = sum(c[2] for c in cells) / len(cells); cy = sum(c[3] for c in cells) / len(cells)
    return [(c[1], cx + (c[2] - cx) * g, cy + (c[3] - cy) * g, c[4] * g, 200, 0, 0, 0) for c in cells]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="choreo"); ap.add_argument("--style", default="{}")
    ap.add_argument("--start", default="runs/techniques/technique_0.json")
    ap.add_argument("--ticks", type=int, default=750); ap.add_argument("--beam", type=int, default=6)
    ap.add_argument("--m", type=int, default=48); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--total", type=float, default=0, help="scale the start to this total mass")
    a = ap.parse_args()
    style = json.loads(a.style)
    global SCALE
    rows = rows_from_replay(os.path.join(ROOT, a.start), total=a.total or None)
    SCALE = (a.total or 37000) / 37000
    t0 = time.time()
    node, t_start = run(a.name, rows, style, a.ticks, a.beam, a.m, a.seed)
    if node is None:
        print(f"{a.name}: every branch died"); return
    rec, moves, died = replay(a.name, rows, node[2])
    dur = (len(rec.data["inputs"])) / 25
    presses = [t for t, inp in enumerate(rec.data["inputs"]) if inp[0][2] or inp[1][2]]
    gaps = np.diff([0] + presses)
    from collections import Counter
    info = dict(name=a.name, style=style, seconds=dur, died=died, moves=moves, score=node[1],
                mean_gap_ticks=float(gaps.mean()), max_gap_ticks=int(gaps.max()), compute_s=round(time.time() - t0))
    rec.data["description"] = (f"Non-periodic choreography ({a.name}): {dur:.1f} s, {len(presses)} split presses, "
                               f"mean gap {gaps.mean() / 25:.2f} s. Moves: {dict(Counter(m['move'] for m in moves))}.")
    out = os.path.join(ROOT, "runs", "freeruns"); os.makedirs(out, exist_ok=True)
    rec.save(os.path.join(out, a.name + ".json"))
    json.dump(info, open(os.path.join(out, a.name + ".info.json"), "w"), indent=1)
    print(f"{a.name}: {dur:.1f} s, died={died}, {len(presses)} presses, mean gap {gaps.mean():.1f} ticks, max {gaps.max()}, "
          f"moves {dict(Counter(m['move'] for m in moves))} ({info['compute_s']}s)")


if __name__ == "__main__":
    main()
