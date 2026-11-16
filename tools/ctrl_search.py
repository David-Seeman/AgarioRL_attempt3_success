"""Geometry-first search for splitruns with a scripted FEEDBACK controller (no RL, no mass pattern given).

The controller runs a program of K steps, repeated forever. Roles are re-read at every cycle start:
BIG = the player holding more mass right now, SMALL = the other, so one program covers both
same-roles and role-swapping (mirror) splitruns. Step k:

  actor      BIG | SMALL | BOTH            nsplit  1 (single) | 2 (double)
  src        the actor's piece the aim is built on: largest | smallest | nearest to the partner
  tgt        the partner's reference: largest | smallest | nearest to src | partner centre of mass
  frame      u = unit(src -> tgt), w = u rotated +90 deg
  approach   actor cursor = tgt + ax*u + ay*w;  partner cursor = partner centre + px*u + py*w
  trigger    split when |src - tgt| <= D and at least gmin ticks after the previous step
  aim        cursor at the split = tgt + along*u + perp*w  (perp/along let it split sideways/away)

A fixed feedback law applied again and again settles into a repeating orbit when that orbit is stable,
so the mass pattern and the geometry are OUTPUTS, not inputs.

A world fails (and stops) on: a split whose launched pieces are never used within USE ticks (no
purpose), a slow absorption (cross-player eat between two old pieces), a split press that launched
nothing, a trigger that never fires (idle), a player vanishing or dropping below 8% of the mass, drift.
Own merges are allowed. Periodicity: at every cycle start the layout (per role: masses, distances to
both centres of mass) is compared with the previous cycle start.

    python tools/ctrl_search.py --K 3 --share 0.33 --target 0.36 --seed 0 --tag k3
"""
import argparse, json, math, os, sys, time
import numpy as np
from numba import njit
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from splitrun.sim import (Sim, X, Y, SZ, OWN, CID, BIRTH, EX, G_TICK, EV_EATER_OWN, EV_EATEN_OWN, EV_EATER_ID,  # noqa
                          EV_EATEN_ID, EV_EATER_AGE, EV_EATEN_AGE)

TOTAL = 37000.0
NPS = 12
#            actor nsplit src tgt   D     gmin along  perp   ax     ay     px     py
LO = np.array([0,   1,     0,  0,   150,  0,  -1500, -1500, -1500, -1500, -1500, -1500], float)
HI = np.array([3,   3,     3,  4,   1500, 40,  2500,  1500,  1500,  1500,  1500,  1500], float)
DISCRETE = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], bool)
USE = 45           # a split must have a launched piece used within this many ticks
YOUNG = 60
IDLE = 100
MAXT = 48          # tracked launched pieces
NH = 64            # split ring
MAXK = 8
NR = 16            # pieces per role in the descriptor
# int state
S_K, S_TSL, S_CYC, S_BIG, S_DONE, S_NSPLIT, S_CHECK, S_STREAK, S_BEST, S_LASTB, S_FAILT, S_NMATCH, S_NB, S_LAG = range(14)
NSI = 14
LAGS = 3           # an orbit may repeat after 1, 2 or 3 program cycles
# float state
F_CX, F_CY, F_SHARE, F_TPC, F_SHSUM, F_SIM = range(6)
NSF = 6


@njit(cache=True)
def _com(cf, oc, ocn, e, p):
    sx = 0.0; sy = 0.0; sm = 0.0
    for k in range(ocn[e, p]):
        s_ = oc[e, p, k]
        m = cf[e, s_, SZ] * cf[e, s_, SZ]
        sx += cf[e, s_, X] * m; sy += cf[e, s_, Y] * m; sm += m
    if sm == 0:
        return 0.0, 0.0, 0.0
    return sx / sm, sy / sm, sm / 100.0


@njit(cache=True)
def _select(cf, oc, ocn, e, p, mode, rx, ry):
    """slot of player p's piece: 0 largest, 1 smallest, 2 nearest to (rx, ry).
    Pieces within 3% of the extreme size count as tied; ties go to the one nearest (rx, ry)."""
    n = ocn[e, p]
    if n == 0:
        return -1
    ext = cf[e, oc[e, p, 0], SZ]
    for k in range(1, n):
        sz = cf[e, oc[e, p, k], SZ]
        if (mode == 0 and sz > ext) or (mode == 1 and sz < ext):
            ext = sz
    best = -1; bd = 1e18
    for k in range(n):
        s_ = oc[e, p, k]
        sz = cf[e, s_, SZ]
        if mode == 0 and sz < ext / 1.015:
            continue
        if mode == 1 and sz > ext * 1.015:
            continue
        d = math.hypot(cf[e, s_, X] - rx, cf[e, s_, Y] - ry)
        if d < bd:
            bd = d; best = s_
    return best


@njit(cache=True)
def ctrl_tick(cf, ci, oc, ocn, g, prm, K, si, sf, mouse, splits, ang, desc, descn, hcx, active):
    N = cf.shape[0]
    for e in range(N):
        splits[e, 0] = 0; splits[e, 1] = 0
        if si[e, S_DONE] != 0:
            active[e] = 0
            continue
        k = si[e, S_K]
        ax_, ay_, am = _com(cf, oc, ocn, e, 0)
        bx_, by_, bm = _com(cf, oc, ocn, e, 1)
        if k == 0:
            si[e, S_BIG] = 0 if am >= bm else 1          # roles are frozen once step 0 fires
        big = si[e, S_BIG]
        actor_role = int(prm[e, k, 0])          # 0 big, 1 small, 2 both
        nact = 2 if actor_role == 2 else 1
        fire = False
        for r in range(nact):
            p = (big if actor_role != 1 else 1 - big) if r == 0 else 1 - big
            q = 1 - p
            qx, qy, qm = (ax_, ay_, am) if q == 0 else (bx_, by_, bm)
            src = _select(cf, oc, ocn, e, p, int(prm[e, k, 2]), qx, qy)
            if src < 0:
                continue
            sx = cf[e, src, X]; sy = cf[e, src, Y]
            tm = int(prm[e, k, 3])
            if tm == 3:
                tx = qx; ty = qy
            else:
                t_ = _select(cf, oc, ocn, e, q, tm, sx, sy)
                if t_ < 0:
                    continue
                tx = cf[e, t_, X]; ty = cf[e, t_, Y]
            dx = tx - sx; dy = ty - sy
            d = math.hypot(dx, dy)
            if d < 1e-6:
                ux = 1.0; uy = 0.0
            else:
                ux = dx / d; uy = dy / d
            wx = -uy; wy = ux
            if r == 0:
                fire = d <= prm[e, k, 4] and si[e, S_TSL] >= prm[e, k, 5]
            if fire:
                mx = tx + prm[e, k, 6] * ux + prm[e, k, 7] * wx
                my = ty + prm[e, k, 6] * uy + prm[e, k, 7] * wy
                mouse[e, p, 0] = mx; mouse[e, p, 1] = my
                splits[e, p] = int(prm[e, k, 1])
                if r == 0:
                    # split direction vs the direction to the partner (from the actor's centre)
                    pcx, pcy, _ = (ax_, ay_, am) if p == 0 else (bx_, by_, bm)
                    a1 = math.atan2(my - pcy, mx - pcx); a2 = math.atan2(qy - pcy, qx - pcx)
                    da = a1 - a2
                    while da > math.pi:
                        da -= 2 * math.pi
                    while da < -math.pi:
                        da += 2 * math.pi
                    ang[e, k] = da
            else:
                mouse[e, p, 0] = tx + prm[e, k, 8] * ux + prm[e, k, 9] * wx
                mouse[e, p, 1] = ty + prm[e, k, 8] * uy + prm[e, k, 9] * wy
                if nact == 1:
                    mouse[e, q, 0] = qx + prm[e, k, 10] * ux + prm[e, k, 11] * wx
                    mouse[e, q, 1] = qy + prm[e, k, 10] * uy + prm[e, k, 11] * wy
        if fire:
            if k == 0:
                # cycle start: compare the layout with the previous cycle start
                _boundary(cf, oc, ocn, e, si, sf, desc, descn, hcx, ax_, ay_, am, bx_, by_, bm, g[e, G_TICK])
            si[e, S_K] = (k + 1) % K
            if si[e, S_K] == 0:
                si[e, S_CYC] += 1
            si[e, S_TSL] = 0
        else:
            si[e, S_TSL] += 1
            if si[e, S_TSL] > IDLE:
                si[e, S_DONE] = 6; si[e, S_FAILT] = g[e, G_TICK]


@njit(cache=True)
def _boundary(cf, oc, ocn, e, si, sf, desc, descn, hcx, ax_, ay_, am, bx_, by_, bm, tick):
    big = 0 if am >= bm else 1
    cx = (ax_ * am + bx_ * bm) / (am + bm); cy = (ay_ * am + by_ * bm) / (am + bm)
    new = np.zeros((2, NR, 3)); nn = np.zeros(2, np.int64)
    for r in range(2):
        p = big if r == 0 else 1 - big
        n = min(ocn[e, p], NR)
        for k in range(n):
            s_ = oc[e, p, k]
            new[r, k, 0] = cf[e, s_, SZ] * cf[e, s_, SZ] / 100.0
            new[r, k, 1] = math.hypot(cf[e, s_, X] - ax_, cf[e, s_, Y] - ay_) if p == 0 else math.hypot(cf[e, s_, X] - bx_, cf[e, s_, Y] - by_)
            new[r, k, 2] = math.hypot(cf[e, s_, X] - bx_, cf[e, s_, Y] - by_) if p == 0 else math.hypot(cf[e, s_, X] - ax_, cf[e, s_, Y] - ay_)
        # sort by mass descending (insertion sort)
        for i in range(1, n):
            j = i
            while j > 0 and new[r, j, 0] > new[r, j - 1, 0]:
                for f in range(3):
                    t = new[r, j, f]; new[r, j, f] = new[r, j - 1, f]; new[r, j - 1, f] = t
                j -= 1
        nn[r] = n
    match = False
    nb = si[e, S_NB]
    best_sim = 0.0; best_lag = 0
    for lag in range(1, LAGS + 1):
        if nb - lag < 0:
            break
        h = (nb - lag) % LAGS
        if nn[0] != descn[e, h, 0] or nn[1] != descn[e, h, 1]:
            continue
        merr = 0.0; derr = 0.0; cnt = 0
        for r in range(2):
            for k in range(nn[r]):
                merr += abs(new[r, k, 0] - desc[e, h, r, k, 0]) / max(desc[e, h, r, k, 0], 1.0)
                derr += abs(new[r, k, 1] - desc[e, h, r, k, 1]) + abs(new[r, k, 2] - desc[e, h, r, k, 2])
                cnt += 1
        drift = math.hypot(cx - hcx[e, h, 0], cy - hcx[e, h, 1])
        simv = math.exp(-(merr / cnt) / 0.05) * math.exp(-(derr / (2 * cnt)) / 150.0) * math.exp(-drift / 400.0)
        if simv > best_sim:
            best_sim = simv
        if merr / cnt < 0.04 and derr / (2 * cnt) < 70.0 and drift < 150.0:
            match = True
            best_lag = lag
            break
    sf[e, F_SIM] += best_sim
    if match:
        si[e, S_STREAK] += 1
        si[e, S_NMATCH] += 1
        si[e, S_LAG] = best_lag
        sf[e, F_SHSUM] += min(am, bm) / (am + bm)
        sf[e, F_TPC] += (tick - si[e, S_LASTB]) * best_lag
        if si[e, S_STREAK] > si[e, S_BEST]:
            si[e, S_BEST] = si[e, S_STREAK]
    else:
        si[e, S_STREAK] = 0
    h = nb % LAGS
    for r in range(2):
        for k in range(nn[r]):
            for f in range(3):
                desc[e, h, r, k, f] = new[r, k, f]
    descn[e, h, 0] = nn[0]; descn[e, h, 1] = nn[1]
    hcx[e, h, 0] = cx; hcx[e, h, 1] = cy
    si[e, S_NB] = nb + 1
    si[e, S_LASTB] = tick


@njit(cache=True)
def post_tick(cf, ci, oc, ocn, g, ev, evn, si, sf, splits, tid, tsp, tn, sp_tick, sp_used, sp_n, cx0, cy0, strict,
              tuse, tbirth):
    N = cf.shape[0]
    for e in range(N):
        if si[e, S_DONE] != 0:
            continue
        tick = g[e, G_TICK]
        # register launched pieces of this tick's presses
        for p in range(2):
            if splits[e, p] == 0:
                continue
            sid = sp_n[e]
            sp_n[e] += 1
            h = sid % NH
            sp_tick[e, h] = tick; sp_used[e, h] = 0
            born = 0
            for k in range(ocn[e, p]):
                s_ = oc[e, p, k]
                if ci[e, s_, BIRTH] == tick:
                    born += 1
                    if tn[e] < MAXT:
                        tid[e, tn[e]] = ci[e, s_, CID]; tsp[e, tn[e]] = sid; tuse[e, tn[e]] = 0
                        tbirth[e, tn[e]] = tick; tn[e] += 1
            if born == 0:
                si[e, S_DONE] = 2; si[e, S_FAILT] = tick
        if si[e, S_DONE] != 0:
            continue
        for q in range(evn[e]):
            eo = int(ev[e, q, EV_EATER_OWN]); bo = int(ev[e, q, EV_EATEN_OWN])
            if eo == bo:
                continue                         # own merge: allowed
            if ev[e, q, EV_EATER_AGE] > YOUNG and ev[e, q, EV_EATEN_AGE] > YOUNG:
                si[e, S_DONE] = 4; si[e, S_FAILT] = tick
                break
            for j in range(tn[e]):
                if tid[e, j] == int(ev[e, q, EV_EATER_ID]) or tid[e, j] == int(ev[e, q, EV_EATEN_ID]):
                    sp_used[e, tsp[e, j] % NH] = 1
                    tuse[e, j] = 1
        if si[e, S_DONE] != 0:
            continue
        # splits past the window without purpose
        while si[e, S_CHECK] < sp_n[e]:
            h = si[e, S_CHECK] % NH
            if tick - sp_tick[e, h] <= USE:
                break
            if sp_used[e, h] == 0:
                si[e, S_DONE] = 3; si[e, S_FAILT] = tick
                break
            si[e, S_NSPLIT] += 1                 # a clean (purposeful) split
            si[e, S_CHECK] += 1
        if strict:
            for j in range(tn[e]):
                if tuse[e, j] == 0 and tick - tbirth[e, j] > USE:
                    si[e, S_DONE] = 3; si[e, S_FAILT] = tick
                    break
            if si[e, S_DONE] != 0:
                continue
        # forget tracked pieces of checked splits (strict: only once used)
        w = 0
        for j in range(tn[e]):
            if tsp[e, j] >= si[e, S_CHECK] or (strict and tuse[e, j] == 0):
                tid[e, w] = tid[e, j]; tsp[e, w] = tsp[e, j]; tuse[e, w] = tuse[e, j]; tbirth[e, w] = tbirth[e, j]; w += 1
        tn[e] = w
        ax_, ay_, am = _com(cf, oc, ocn, e, 0)
        bx_, by_, bm = _com(cf, oc, ocn, e, 1)
        if ocn[e, 0] == 0 or ocn[e, 1] == 0 or min(am, bm) < 0.08 * (am + bm):
            si[e, S_DONE] = 7; si[e, S_FAILT] = tick
            continue
        cx = (ax_ * am + bx_ * bm) / (am + bm); cy = (ay_ * am + by_ * bm) / (am + bm)
        if math.hypot(cx - cx0[e], cy - cy0[e]) > 2500:
            si[e, S_DONE] = 8; si[e, S_FAILT] = tick


def start_rows(share, layout, rng=None):
    """BIG and SMALL pieces (powers-of-two splits of each player's mass) placed on a line."""
    mb, ms = TOTAL * (1 - share), TOTAL * share
    nb, ns = layout
    rows = []
    sz = lambda m: math.sqrt(m * 100)
    for i in range(nb):
        rows.append((0, -900 - i * 1.05 * (sz(mb / nb) * 2) / 2, (i - (nb - 1) / 2) * sz(mb / nb) * 1.9, sz(mb / nb), 200, 0, 0, 0))
    for i in range(ns):
        rows.append((1, 700 + i * 0.5 * sz(ms / ns), (i - (ns - 1) / 2) * sz(ms / ns) * 1.9, sz(ms / ns), 200, 0, 0, 0))
    return rows


class Runner:
    def __init__(self, N, K):
        self.N, self.K = N, K
        self.sim = Sim(N)

    def run(self, V, starts, T=1000, record=None, trace=None, strict=False):
        """V (N, K*NPS) raw params; starts: list of row lists (world i uses starts[i % len])"""
        N, K, sim = len(V), self.K, self.sim
        prm = np.zeros((self.N, K, NPS)); prm[:N] = V.reshape(N, K, NPS)
        prm[N:] = prm[0]
        prm[:, :, DISCRETE] = np.floor(prm[:, :, DISCRETE])
        for i in range(self.N):
            sim.load(i, starts[i % len(starts)], seed=i + 1)
        sim.active[:] = 1
        si = np.zeros((self.N, NSI), np.int64); sf = np.zeros((self.N, NSF))
        ang = np.zeros((self.N, MAXK)); desc = np.zeros((self.N, LAGS, 2, NR, 3)); descn = np.zeros((self.N, LAGS, 2), np.int64)
        hcx = np.zeros((self.N, LAGS, 2))
        tid = np.zeros((self.N, MAXT), np.int64); tsp = np.zeros((self.N, MAXT), np.int64); tn = np.zeros(self.N, np.int64)
        sp_tick = np.zeros((self.N, NH), np.int64); sp_used = np.zeros((self.N, NH), np.int64); sp_n = np.zeros(self.N, np.int64)
        cx0 = np.zeros(self.N); cy0 = np.zeros(self.N)
        tuse = np.zeros((self.N, MAXT), np.int64); tbirth = np.zeros((self.N, MAXT), np.int64)
        for i in range(self.N):
            c = sim.cells(i); M = sum(x[4] ** 2 for x in c)
            cx0[i] = sum(x[2] * x[4] ** 2 for x in c) / M; cy0[i] = sum(x[3] * x[4] ** 2 for x in c) / M
        mouse = np.zeros((self.N, 2, 2)); splits = np.zeros((self.N, 2), np.int64)
        rec = None
        if record is not None:
            from splitrun.replay import Recorder
            rec = Recorder(sim, record, "controller")
        angs = []
        for t in range(T):
            lastb = si[trace, S_LASTB] if trace is not None else 0
            ctrl_tick(sim.cf, sim.ci, sim.oc, sim.ocn, sim.g, prm, K, si, sf, mouse, splits, ang, desc, descn, hcx, sim.active)
            if trace is not None and si[trace, S_LASTB] != lastb:
                print(f"  t={t} cycle start | streak {si[trace, S_STREAK]} lag {si[trace, S_LAG]}")
            if not sim.active.any():
                break
            sim.step(mouse, splits)
            if rec is not None:
                if si[record, S_DONE] == 0:
                    rec.after_step(np.trunc(mouse[record]), splits[record])
            post_tick(sim.cf, sim.ci, sim.oc, sim.ocn, sim.g, sim.ev, sim.evn, si, sf, splits, tid, tsp, tn,
                      sp_tick, sp_used, sp_n, cx0, cy0, strict, tuse, tbirth)
        surv = np.where(si[:, S_DONE] == 0, T, si[:, S_FAILT])
        return dict(si=si[:N], sf=sf[:N], ang=ang[:N, :K], surv=surv[:N], rec=rec, angs=angs)


def fitness(res, K, target, T):
    si, sf = res["si"], res["sf"]
    best = si[:, S_BEST].astype(float)
    nm = np.maximum(si[:, S_NMATCH], 1)
    share = np.where(si[:, S_NMATCH] > 0, sf[:, F_SHSUM] / nm, 0.5)
    tpc = np.where(si[:, S_NMATCH] > 0, sf[:, F_TPC] / nm, 999)
    w_imb = np.where(share <= target, 1.0, np.exp(-(share - target) / 0.04))
    a = np.degrees(res["ang"])
    spread = a.max(1) - a.min(1)
    nov = 0.5 * np.minimum(1.0, spread / 60.0)
    speed = np.clip(25.0 * K / np.maximum(tpc, 1) / 2.0, 0, 1)        # splits per second / 2
    per = np.minimum(best, 12) * K
    clean = np.minimum(si[:, S_NSPLIT], 60) * 0.1
    simb = np.minimum(sf[:, F_SIM], 15) * 0.5
    F = res["surv"] / T + clean + simb + per * w_imb * (1 + nov + 0.3 * speed)
    return F, dict(best=best, share=share, tpc=tpc, spread=spread, clean=si[:, S_NSPLIT], done=si[:, S_DONE], lag=si[:, S_LAG])


def signature(V_row, K, res_row_share, masses=None):
    p = V_row.reshape(K, NPS)
    return tuple((int(p[k, 0]), int(p[k, 1])) for k in range(K)), round(float(res_row_share), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--share", type=float, default=0.33, help="start: smaller player's share of the mass")
    ap.add_argument("--layout", default="2,1", help="start: pieces of BIG,SMALL")
    ap.add_argument("--target", type=float, default=0.38, help="orbits count fully only if small share <= this")
    ap.add_argument("--double", action="store_true", help="at least one step is a double split")
    ap.add_argument("--pop", type=int, default=2048); ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--T", type=int, default=1000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--strict", action="store_true", help="every launched piece must be used (not just one per split)")
    ap.add_argument("--ban", default="", help="jsonl of earlier finds: penalise the same actor/split program")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    K = a.K
    lo = np.tile(LO, K); hi = np.tile(HI, K); disc = np.tile(DISCRETE, K)
    lay = tuple(int(x) for x in a.layout.split(","))
    starts = [start_rows(a.share, lay)]
    banned = set()
    if a.ban and os.path.exists(a.ban):
        for line in open(a.ban):
            r = json.loads(line)
            banned.add(tuple(map(tuple, r["program"])))
    run = Runner(a.pop, K)

    def fix(V):
        V = np.clip(V, lo, hi - 1e-6)
        if a.double:
            P = V.reshape(len(V), K, NPS)
            nod = (np.floor(P[:, :, 1]) < 2).all(1)
            P[nod, rng.integers(0, K, nod.sum()), 1] = 2.5
        return V

    def score(V):
        res = run.run(V, starts, a.T, strict=a.strict)
        F, info = fitness(res, K, a.target, a.T)
        if banned:
            for i in range(len(V)):
                prog = tuple((int(V[i].reshape(K, NPS)[k, 0]), int(V[i].reshape(K, NPS)[k, 1])) for k in range(K))
                if info["best"][i] >= 3 and prog in banned and info["share"][i] <= a.target:
                    F[i] = res["surv"][i] / a.T + 1.0      # a known program: only a little credit
        return F, info

    V = fix(rng.uniform(lo, hi, (a.pop, len(lo))))
    F, info = score(V)
    ne = 64
    t0 = time.time()
    out_dir = os.path.join(ROOT, "runs", "controller"); os.makedirs(out_dir, exist_ok=True)
    for gen in range(a.gens):
        o = np.argsort(-F)
        el, ef = V[o[:ne]], F[o[:ne]]
        i_best = o[0]
        print(f"gen {gen:3d} best F {F[i_best]:.2f} | streak {info['best'][i_best]:.0f} cycles, share {info['share'][i_best]:.2f}, "
              f"{info['tpc'][i_best]:.0f} ticks/cycle, angle spread {info['spread'][i_best]:.0f} | "
              f"lag {info['lag'][i_best]} | clean splits {info['clean'][i_best]} | worlds with a 3+ cycle orbit: {(info['best'] >= 3).sum()} | "
              f"fails {np.bincount(info['done'], minlength=9).tolist()} ({time.time() - t0:.0f}s)", flush=True)
        scale = 0.10 * 0.96 ** gen
        pa = el[rng.integers(0, ne, a.pop)]; pb = el[rng.integers(0, ne, a.pop)]
        mask = rng.random(pa.shape) < 0.5
        cross = rng.random(a.pop) < 0.3
        kids = np.where(cross[:, None] & mask, pb, pa)
        kids = kids + rng.normal(0, 1, kids.shape) * (hi - lo) * scale * ~disc
        flip = (rng.random(kids.shape) < 0.04) & disc
        kids = np.where(flip, rng.uniform(lo, hi, kids.shape), kids)
        kids[:ne] = el
        V = fix(kids)
        F, info = score(V)
        if gen % 10 == 9:
            np.save(os.path.join(out_dir, f"pop_{a.tag}_s{a.seed}.npy"), V[np.argsort(-F)[:ne]])
    o = np.argsort(-F)
    np.save(os.path.join(out_dir, f"pop_{a.tag}_s{a.seed}.npy"), V[o[:ne]])
    # report distinct good programs
    seen = set(); outp = os.path.join(out_dir, "finds.jsonl")
    for i in o[:ne]:
        if info["best"][i] < 3:
            continue
        P = V[i].reshape(K, NPS)
        prog = [[int(P[k, 0]), int(P[k, 1]), int(P[k, 2]), int(P[k, 3])] for k in range(K)]
        key = (tuple(map(tuple, prog)), round(float(info["share"][i]), 2))
        if key in seen:
            continue
        seen.add(key)
        r = dict(tag=a.tag, seed=a.seed, K=K, lag=int(info["lag"][i]), strict=a.strict, T=a.T, program=[p[:2] for p in prog], full_program=prog, params=V[i].tolist(),
                 streak=float(info["best"][i]), share=float(info["share"][i]), ticks_per_cycle=float(info["tpc"][i]),
                 spread=float(info["spread"][i]), start=dict(share=a.share, layout=list(lay)), F=float(F[i]))
        with open(outp, "a") as f:
            f.write(json.dumps(r) + "\n")
        print("FIND", {k: v for k, v in r.items() if k != "params"}, flush=True)


if __name__ == "__main__":
    main()
