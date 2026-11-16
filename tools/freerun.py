"""Non-periodic splitruns: a model-predictive controller on the exact simulator (no RL, no fixed technique).

Every REPLAN ticks the real world is cloned into NP planner worlds. Each tries one plan:
    who splits (A, B or both), single or double, after a short wait w (the gap since the last split is
    capped at GAPMAX ticks: little downtime), aimed at angle phi (all 360 deg, measured from the line to
    the partner) and distance rho from the splitter's centre, with cursors for both players before and
    after the split (in the pair frame).
It is simulated for w + HORIZON ticks and scored:
    + EAT per cross-player eat that involves a young piece (a splitrun move)
    - dead end: a split whose pieces (parents or launched) take part in no cross eat within USE ticks
      (no purpose), a slow absorption (eat between two old pieces), a split press that launches nothing
      (16-cell cap), a player below MINSHARE of the mass
    - waiting, drift, too many cells / crumbs at the horizon; + optional style bonuses
The best plan is executed for REPLAN ticks, then everything is re-planned from the real state.
The real run stops on the same dead ends (or after --ticks). Result: a replay + stats.

    python tools/freerun.py --name balanced_fast --ticks 750
"""
import argparse, json, math, os, sys, time
import numpy as np
from numba import njit
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from splitrun.sim import (Sim, X, Y, SZ, OWN, CID, BIRTH, EX, G_TICK, EV_EATER_OWN, EV_EATEN_OWN, EV_EATER_ID,  # noqa
                          EV_EATEN_ID, EV_EATER_AGE, EV_EATEN_AGE, EV_MASS)
from splitrun.rlenv import SplitrunEnv  # noqa
from splitrun.replay import Recorder, write_index  # noqa

TOTAL = 37000.0
USE = 50            # a split must be followed by a cross eat involving one of its pieces within this many ticks
YOUNG = 60          # cross eats need at least one piece younger than this (no slow absorption)
GAPMAX = 20         # at most this many ticks between split presses (0.8 s)
MINSHARE = 0.12
HORIZON = 36
MAXP = 12           # pending split purposes tracked
MAXID = 32
EAT = 0.5
PURPOSE = 6.0      # a split whose pieces did take part in a cross eat
DEAD = -40.0
# plan vector: actor(0 A,1 B,2 both), kind, wait, phi, rho, A1x A1y B1x B1y (before), A2x A2y B2x B2y (after)
NPAR = 13
LO = np.array([0, 1, 0, -math.pi, 200, -1500, -1500, -1500, -1500, -1500, -1500, -1500, -1500], float)
HI = np.array([3, 3, GAPMAX, math.pi, 2500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500], float)


@njit(cache=True)
def register_split(ci, oc, ocn, g, e, p, pids, pn, pdl, pok, dead, score, plaunch, use=USE):
    """remember every cell of player p right after its split (parents + launched): the split's pieces"""
    tick = g[e, G_TICK]
    born = 0
    for k in range(ocn[e, p]):
        if ci[e, oc[e, p, k], BIRTH] == tick:
            born += 1
    if born == 0:
        dead[e] = True; score[e] += DEAD                 # split press that launched nothing
        return
    slot = -1
    for s in range(MAXP):
        if pdl[e, s] < 0:
            slot = s
            break
    if slot < 0:
        return
    n = 0
    for k in range(ocn[e, p]):
        if n < MAXID:
            pids[e, slot, n] = ci[e, oc[e, p, k], CID]
            plaunch[e, slot, n] = 1 if ci[e, oc[e, p, k], BIRTH] == tick else 0
            n += 1
    pn[e, slot] = n
    pdl[e, slot] = tick + use
    pok[e, slot] = 0


@njit(cache=True)
def score_tick(g, ev, evn, active, pids, pn, pdl, pok, dead, score, eats, old_eats, plaunch, trimw, young=YOUNG):
    N = ev.shape[0]
    for e in range(N):
        if active[e] == 0 or dead[e]:
            continue
        tick = g[e, G_TICK]
        for q in range(evn[e]):
            eo = int(ev[e, q, EV_EATER_OWN]); bo = int(ev[e, q, EV_EATEN_OWN])
            if eo == bo:
                continue                                  # own merge: allowed
            if ev[e, q, EV_EATER_AGE] > young and ev[e, q, EV_EATEN_AGE] > young:
                dead[e] = True; score[e] += DEAD; old_eats[e] += 1
                break
            score[e] += EAT; eats[e] += 1
            a = int(ev[e, q, EV_EATER_ID]); b = int(ev[e, q, EV_EATEN_ID])
            for s in range(MAXP):
                if pdl[e, s] < 0 or pok[e, s] == 1:
                    continue
                for k in range(pn[e, s]):
                    if pids[e, s, k] == a or pids[e, s, k] == b:
                        pok[e, s] = 1
                        # a launched piece used: a real splitrun move; only a parent used: a trim
                        score[e] += PURPOSE if plaunch[e, s, k] == 1 else trimw
                        break
        if dead[e]:
            continue
        for s in range(MAXP):
            if pdl[e, s] >= 0 and pok[e, s] == 0 and tick > pdl[e, s]:
                dead[e] = True; score[e] += DEAD
                break
            if pdl[e, s] >= 0 and pok[e, s] == 1:
                pdl[e, s] = -1                           # satisfied: free the slot


@njit(cache=True)
def danger(cf, ci, oc, ocn, g, dead, out):
    """how close two OLD cross-player pieces (one able to eat the other) are to a slow absorption"""
    N = cf.shape[0]
    for e in range(N):
        out[e] = 0.0
        if dead[e]:
            continue
        tick = g[e, G_TICK]
        for i in range(ocn[e, 0]):
            a = oc[e, 0, i]
            for j in range(ocn[e, 1]):
                b = oc[e, 1, j]
                sa = cf[e, a, SZ]; sb = cf[e, b, SZ]
                if sa >= 1.14 * sb:
                    big, small, sbig, ssmall = a, b, sa, sb
                elif sb >= 1.14 * sa:
                    big, small, sbig, ssmall = b, a, sb, sa
                else:
                    continue
                # only pieces that are (or soon will be) old count
                if tick - ci[e, big, BIRTH] < YOUNG - 20 or tick - ci[e, small, BIRTH] < YOUNG - 20:
                    continue
                gap = math.hypot(cf[e, a, X] - cf[e, b, X], cf[e, a, Y] - cf[e, b, Y]) - (sbig - ssmall / 3)
                if gap < 400:
                    out[e] += (400 - gap) / 400


def coms(sim, e):
    out = []
    for p in range(2):
        s = sim.oc[e, p, :sim.ocn[e, p]]
        if len(s) == 0:
            out.append((np.zeros(2), 0.0)); continue
        m = sim.cf[e, s, SZ] ** 2
        out.append((np.array([(sim.cf[e, s, X] * m).sum(), (sim.cf[e, s, Y] * m).sum()]) / m.sum(), m.sum() / 100))
    return out


class Planner:
    def __init__(self, NP=384, iters=2, seed=0, style=None):
        self.NP, self.iters = NP, iters
        self.sim = Sim(NP)
        self.rng = np.random.default_rng(seed)
        self.prev = None
        self.style = style or {}

    def plan(self, world, pend, wmax, center0):
        (ca, ma), (cb, mb) = coms(world, 0)
        u = cb - ca; u = u / max(np.linalg.norm(u), 1e-6); w = np.array([-u[1], u[0]])
        lo, hi = LO.copy(), HI.copy(); hi[2] = wmax + 0.999
        mu, sd = (lo + hi) / 2, (hi - lo) / 2
        best_v, best_s = None, -1e18
        for it in range(self.iters):
            V = self.rng.normal(mu, sd, (self.NP, NPAR))
            if it == 0:
                V[: self.NP // 2] = self.rng.uniform(lo, hi, (self.NP // 2, NPAR))
                if self.prev is not None:
                    V[0] = self.prev
            V = np.clip(V, lo, hi - 1e-6)
            s = self.simulate(world, V, pend, ca, cb, u, w, center0)
            o = np.argsort(-s)
            if s[o[0]] > best_s:
                best_s, best_v = s[o[0]], V[o[0]].copy()
            el = V[o[: max(8, self.NP // 10)]]
            mu = el.mean(0); sd = el.std(0) + (hi - lo) * 0.03
            # discrete dims: keep sampling a little outside the elite
            sd[0] = max(sd[0], 0.6); sd[1] = max(sd[1], 0.5)
        self.prev = best_v.copy()
        return best_v, (ca, cb, u, w), best_s

    def simulate(self, world, V, pend, ca, cb, u, w, center0):
        P, NP = self.sim, self.NP
        for kk in SplitrunEnv.SIM_ARRAYS:
            getattr(P, kk)[:] = getattr(world, kk)[0][None]
        P.active[:] = 1
        actor = np.floor(V[:, 0]).astype(int); kind = np.floor(V[:, 1]).astype(int); wait = np.floor(V[:, 2]).astype(int)
        fr = lambda org, xy: org[None] + xy[:, :1] * u[None] + xy[:, 1:2] * w[None]
        A1, B1, A2, B2 = fr(ca, V[:, 5:7]), fr(cb, V[:, 7:9]), fr(ca, V[:, 9:11]), fr(cb, V[:, 11:13])
        # split aim: angle phi from the splitter's line to its partner (mirrored for B), distance rho
        c, s_ = np.cos(V[:, 3]), np.sin(V[:, 3])
        dA = c[:, None] * u[None] + s_[:, None] * w[None]
        dB = -c[:, None] * u[None] - s_[:, None] * w[None]
        aimA = ca[None] + dA * V[:, 4:5]; aimB = cb[None] + dB * V[:, 4:5]
        pids = np.zeros((NP, MAXP, MAXID), np.int64); pn = np.zeros((NP, MAXP), np.int64)
        plaunch = np.zeros((NP, MAXP, MAXID), np.int64)
        pdl = np.full((NP, MAXP), -1, np.int64); pok = np.zeros((NP, MAXP), np.int64)
        for j, (ids, lf, dl) in enumerate(pend[:MAXP - 2]):
            pids[:, j, :len(ids)] = ids[:MAXID]; plaunch[:, j, :len(ids)] = lf[:MAXID]
            pn[:, j] = min(len(ids), MAXID); pdl[:, j] = dl
        dead = np.zeros(NP, bool); score = np.zeros(NP); eats = np.zeros(NP, np.int64); olds = np.zeros(NP, np.int64)
        mouse = np.zeros((NP, 2, 2)); splits = np.zeros((NP, 2), np.int64)
        T = int(wait.max()) + HORIZON
        for t in range(T):
            before = (t < wait)[:, None]
            mouse[:, 0] = np.where(before, A1, A2); mouse[:, 1] = np.where(before, B1, B2)
            splits[:] = 0
            at = t == wait
            if at.any():
                sa = at & (actor != 1); sb = at & (actor != 0)
                mouse[sa, 0] = aimA[sa]; mouse[sb, 1] = aimB[sb]
                splits[sa, 0] = kind[sa]; splits[sb, 1] = kind[sb]
            P.active[:] = (~dead).astype(np.int64)
            P.step(mouse, splits)
            if at.any():
                for i in np.nonzero(at & ~dead)[0]:
                    for p in ((0,) if actor[i] == 0 else (1,) if actor[i] == 1 else (0, 1)):
                        register_split(P.ci, P.oc, P.ocn, P.g, i, p, pids, pn, pdl, pok, dead, score, plaunch)
            score_tick(P.g, P.ev, P.evn, P.active, pids, pn, pdl, pok, dead, score, eats, olds, plaunch,
                       float(self.style.get("trim", 0.0)))
            if dead.all():
                break
        # horizon state: viability
        dg = np.zeros(NP)
        danger(P.cf, P.ci, P.oc, P.ocn, P.g, dead, dg)
        score -= 8 * dg
        ex = P.ci[:, :, EX] == 1
        own = P.ci[:, :, OWN]
        m = P.cf[:, :, SZ] ** 2 / 100 * ex
        mA = (m * (own == 0)).sum(1); mB = (m * (own == 1)).sum(1)
        share = np.minimum(mA, mB) / np.maximum(mA + mB, 1)
        nA = (ex & (own == 0)).sum(1); nB = (ex & (own == 1)).sum(1)
        crumbs = (ex & (m < 900)).sum(1)
        lo_sh = self.style.get("share_min", 0.3); hi_sh = self.style.get("share_max", 0.5)
        score += np.where(share < MINSHARE, DEAD, 0)
        score -= 60 * np.maximum(lo_sh - share, 0) + 60 * np.maximum(share - hi_sh, 0)
        mc = self.style.get("max_cells", 5)
        score -= 6 * np.maximum(nA - mc, 0) + 6 * np.maximum(nB - mc, 0) + self.style.get("crumb_w", 6) * crumbs
        score -= 0.25 * wait
        # stationary: the pair's centre stays near where it started
        cx = ((P.cf[:, :, X] * m).sum(1) / np.maximum(m.sum(1), 1), (P.cf[:, :, Y] * m).sum(1) / np.maximum(m.sum(1), 1))
        drift = np.hypot(cx[0] - center0[0], cx[1] - center0[1])
        score -= 0.004 * np.maximum(drift - 600, 0)
        # style
        st = self.style
        if st:
            off = np.abs(V[:, 3]) > math.radians(60)
            score += st.get("off", 0) * off + st.get("double", 0) * (kind == 2) + st.get("both", 0) * (actor == 2)
        return score


def start_rows(a_mass, b_mass, gap=1800.0):
    rows = []
    for p, M in enumerate((a_mass, b_mass)):
        n = int(math.ceil(M / 22000))
        for k in range(n):
            s = math.sqrt(M / n * 100)
            x = (-1 if p == 0 else 1) * gap / 2
            y = (k - (n - 1) / 2) * 2.05 * s
            rows.append((p, x + (-1 if p == 0 else 1) * (s if n > 1 else 0) * 0.3, y, s, 200, 0, 0, 0))
    return rows


def rows_from_replay(path, frame=0):
    r = json.load(open(path))
    return [(c[1], c[2], c[3], c[4], 200, 0, 0, 0) for c in r["frames"][frame]]


def run(name, a_mass=18500, b_mass=18500, ticks=750, NP=384, seed=0, style=None, replan=2, log=print, rows=None):
    world = Sim(1)
    world.load(0, rows if rows is not None else start_rows(a_mass, b_mass), seed=1)
    if rows is not None:
        (_, a_mass), (_, b_mass) = coms(world, 0)
    rec = Recorder(world, 0, name)
    pl = Planner(NP=NP, seed=seed, style=style)
    (ca, _), (cb, _) = coms(world, 0)
    center0 = (ca * a_mass + cb * b_mass) / (a_mass + b_mass)
    pend = []                        # real pending split purposes: (ids, deadline)
    pids = np.zeros((1, MAXP, MAXID), np.int64); pn = np.zeros((1, MAXP), np.int64)
    plaunch = np.zeros((1, MAXP, MAXID), np.int64)
    pdl = np.full((1, MAXP), -1, np.int64); pok = np.zeros((1, MAXP), np.int64)
    dead = np.zeros(1, bool); score = np.zeros(1); eats = np.zeros(1, np.int64); olds = np.zeros(1, np.int64)
    last_split, t_plan, plan, frame = 0, -99, None, None
    splits_log, reason = [], "time limit"
    tick = 0
    t0 = time.time()
    while tick < ticks:
        if plan is None or tick - t_plan >= replan:
            live = [(pids[0, s, :pn[0, s]].copy(), plaunch[0, s, :pn[0, s]].copy(), int(pdl[0, s]))
                    for s in range(MAXP) if pdl[0, s] >= 0 and pok[0, s] == 0]
            wmax = max(0, GAPMAX - (tick - last_split))
            plan, frame, sc = pl.plan(world, live, wmax, center0)
            t_plan = tick
        ca, cb, u, w = frame
        el = tick - t_plan
        wait = int(plan[2]) - el
        fr = lambda org, xy: org + xy[0] * u + xy[1] * w
        mouse = np.zeros((1, 2, 2)); sp = np.zeros((1, 2), np.int64)
        actor, kind = int(plan[0]), int(plan[1])
        if wait > 0:
            mouse[0, 0] = fr(ca, plan[5:7]); mouse[0, 1] = fr(cb, plan[7:9])
        else:
            mouse[0, 0] = fr(ca, plan[9:11]); mouse[0, 1] = fr(cb, plan[11:13])
        fired = wait == 0
        if fired:
            c, s_ = math.cos(plan[3]), math.sin(plan[3])
            if actor != 1:
                mouse[0, 0] = ca + (c * u + s_ * w) * plan[4]; sp[0, 0] = kind
            if actor != 0:
                mouse[0, 1] = cb + (-c * u - s_ * w) * plan[4]; sp[0, 1] = kind
        world.active[:] = 1
        world.step(mouse, sp)
        rec.after_step(np.trunc(mouse[0]), sp[0])
        tick += 1
        if fired:
            for p in ((0,) if actor == 0 else (1,) if actor == 1 else (0, 1)):
                register_split(world.ci, world.oc, world.ocn, world.g, 0, p, pids, pn, pdl, pok, dead, score, plaunch)
                splits_log.append(dict(tick=tick, player=p, kind=kind, phi=round(math.degrees(plan[3]), 1)))
            last_split = tick
            plan = None
        score_tick(world.g, world.ev, world.evn, world.active, pids, pn, pdl, pok, dead, score, eats, olds, plaunch, 0.0)
        (_, ma), (_, mb) = coms(world, 0)
        if dead[0]:
            reason = "old-piece absorption" if olds[0] else ("a split without purpose" if any(pdl[0] >= 0) else "split launched nothing")
            break
        if min(ma, mb) < MINSHARE * (ma + mb) or world.ocn[0, 0] == 0 or world.ocn[0, 1] == 0:
            reason = "one player drained"; break
        if tick - last_split > GAPMAX + 5:
            reason = "downtime limit"; break
        if tick % 125 == 0:
            log(f"   {name}: {tick / 25:.0f}s, {len(splits_log)} splits, {int(eats[0])} eats, A {ma:,.0f} / B {mb:,.0f}, "
                f"cells {world.ocn[0, 0]}/{world.ocn[0, 1]} ({time.time() - t0:.0f}s)")
    return rec, dict(name=name, ticks=tick, seconds=tick / 25, reason=reason, splits=splits_log, eats=int(eats[0]),
                     start=[a_mass, b_mass], style=style or {}, compute_s=round(time.time() - t0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="freerun")
    ap.add_argument("--a", type=float, default=18500); ap.add_argument("--b", type=float, default=18500)
    ap.add_argument("--ticks", type=int, default=750); ap.add_argument("--np", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", default="", help="replay to take the starting layout from (frame 0)")
    ap.add_argument("--style", default="{}", help='JSON, e.g. {"off": 3, "double": 2, "both": 1, "balance": 0.3}')
    a = ap.parse_args()
    rows = rows_from_replay(a.start) if a.start else None
    rec, info = run(a.name, a.a, a.b, a.ticks, a.np, a.seed, json.loads(a.style), rows=rows)
    out = os.path.join(ROOT, "runs", "freeruns"); os.makedirs(out, exist_ok=True)
    rec.data["description"] = f"{info['seconds']:.1f} s, {len(info['splits'])} splits, stopped: {info['reason']}"
    rec.save(os.path.join(out, a.name + ".json"))
    json.dump(info, open(os.path.join(out, a.name + ".info.json"), "w"), indent=1)
    gaps = np.diff([0] + [s["tick"] for s in info["splits"]])
    print(f"{a.name}: {info['seconds']:.1f} s, {len(info['splits'])} split presses, {info['eats']} eats, "
          f"mean gap {gaps.mean() if len(gaps) else 0:.1f} ticks, stopped: {info['reason']} ({info['compute_s']}s)")


if __name__ == "__main__":
    main()
