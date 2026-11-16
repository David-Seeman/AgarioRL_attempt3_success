"""Scripted splitrun controller: model-predictive control with the exact simulator (no RL).

The technique (split order + intended eats per split, from the mass bookkeeping in
tools/random_techniques.py) says WHAT must happen. This controller finds HOW, live:

  every REPLAN ticks: clone the real world into NP planner worlds; each tries a plan
      wait w ticks (actor cursor A1, partner cursor P1)  ->  split, aimed in direction phi
      (any of 360 deg) at distance rho from the splitting piece  ->  cursors A2 / P2 while pieces fly
  score each plan over its horizon:
      +10 per intended eat that happens (pending eats of earlier splits + this split's eats)
      -12 per unintended cross-player eat or own merge, near-miss credit for missing eats,
      -0.05 per tick of waiting, -drift
  cross-entropy method: refine the sampling distribution a few rounds, execute the best plan's first
  REPLAN ticks, replan from the real state.

The real world fails when an unintended eat / merge happens or an intended eat does not happen within
PEND_T ticks of its split. Output: a replay + a summary line in runs/controller/mpc_results.jsonl.

    python tools/mpc_splitrun.py runs/techniques/random_e2_imb2_good.json 12 --cycles 20
"""
import argparse, json, math, os, sys, time
import numpy as np
from numba import njit
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from splitrun.sim import (Sim, X, Y, SZ, EV_EATER_OWN, EV_EATEN_OWN, EV_MASS, EV_EATER_MASS)  # noqa
from splitrun.rlenv import SplitrunEnv  # noqa
from splitrun.replay import Recorder, write_index  # noqa

TOTAL = 37000.0
TOL = 0.07          # mass tolerance when matching an eat to an intended one
PEND_T = 70         # an intended eat must happen within this many ticks of its split
NEW_T = 40          # the planner wants a split's own eats within this many ticks
NPAR = 11           # w, A1(2), P1(2), phi, rho, A2(2), P2(2)
LO = np.array([0, -1600, -1600, -1600, -1600, -math.pi, 150, -1600, -1600, -1600, -1600], float)
HI = np.array([40, 1600, 1600, 1600, 1600, math.pi, 2200, 1600, 1600, 1600, 1600], float)
MAXX = 24           # expected eats tracked per plan


@njit(cache=True)
def near_miss(cf, oc, ocn, xo, xm, xem, xn, best):
    """for each expected eat (eater owner xo, eaten mass xm, eater mass xem): smallest
    (distance - eat reach) over matching piece pairs; keeps the running minimum in best"""
    N = cf.shape[0]
    for e in range(N):
        for j in range(xn[e]):
            a = xo[e, j]; b = 1 - a
            for i in range(ocn[e, a]):
                s1 = oc[e, a, i]
                m1 = cf[e, s1, SZ] * cf[e, s1, SZ] / 100
                if abs(m1 - xem[e, j]) > 0.25 * xem[e, j] and m1 < xem[e, j]:
                    continue
                for k in range(ocn[e, b]):
                    s2 = oc[e, b, k]
                    m2 = cf[e, s2, SZ] * cf[e, s2, SZ] / 100
                    if abs(m2 - xm[e, j]) > 0.25 * xm[e, j]:
                        continue
                    d = math.hypot(cf[e, s1, X] - cf[e, s2, X], cf[e, s1, Y] - cf[e, s2, Y])
                    gap = d - (cf[e, s1, SZ] - cf[e, s2, SZ] / 3)
                    if gap < best[e, j]:
                        best[e, j] = gap


def com(sim, e, p):
    c = [(sim.cf[e, s, X], sim.cf[e, s, Y], sim.cf[e, s, SZ] ** 2) for s in sim.oc[e, p, :sim.ocn[e, p]]]
    m = sum(w for *_, w in c)
    return np.array([sum(x * w for x, _, w in c) / m, sum(y * w for _, y, w in c) / m]), m / 100


def find_piece(sim, e, p, mass):
    best, bd = -1, 1e18
    for s in sim.oc[e, p, :sim.ocn[e, p]]:
        d = abs(sim.cf[e, s, SZ] ** 2 / 100 - mass)
        if d < bd:
            best, bd = s, d
    return best


class Controller:
    def __init__(self, t, NP=384, iters=3, horizon=45, replan=3, seed=0):
        self.t = t
        self.E = len(t["seq"])
        self.mirror = t["mirror"]
        self.NP, self.iters, self.H, self.replan = NP, iters, horizon, replan
        self.rng = np.random.default_rng(seed)
        self.plan = Sim(NP)
        self.mu = (LO + HI) / 2; self.sd = (HI - LO) / 2
        self.prev_best = None

    def event_eats(self, k, cyc):
        """intended eats of event k in world terms: (eater owner, eaten mass, eater mass)"""
        sw = int(self.mirror and cyc % 2 == 1)
        return [(a ^ sw, m * TOTAL, em * TOTAL) for (a, b, m, em) in self.t["eats"][k]]

    def actor(self, k, cyc):
        sw = int(self.mirror and cyc % 2 == 1)
        return self.t["seq"][k][0] ^ sw, self.t["seq"][k][1]

    def splitter_mass(self, k, cyc):
        """mass of the actor's piece whose launched half takes part in this split's first eat"""
        a, kind = self.actor(k, cyc)
        for (eo, m, em) in self.event_eats(k, cyc):
            return em * 2 ** kind if eo == a else m * 2 ** kind
        return None

    def decide(self, world, k, cyc, pending, tick, commit=None):
        """returns (plan vector, frame) for event k given the pending eats"""
        NP, P = self.NP, self.plan
        a, kind = self.actor(k, cyc)
        q = 1 - a
        ca, _ = com(world, 0, a); cq, _ = com(world, 0, q)
        u = cq - ca; u /= max(np.linalg.norm(u), 1e-6); w = np.array([-u[1], u[0]])
        sm = self.splitter_mass(k, cyc)
        new = self.event_eats(k, cyc)
        pend = [(o, m, em, max(1, PEND_T - 15 - (tick - ts))) for (o, m, em, ts) in pending]
        mu, sd = self.mu.copy(), self.sd.copy()
        if self.prev_best is not None:
            mu = self.prev_best.copy(); mu[0] = max(0, mu[0] - self.replan)
            sd = (HI - LO) * 0.25
        best_v, best_s = None, -1e18
        for it in range(self.iters):
            V = self.rng.normal(mu, sd, (NP, NPAR))
            if it == 0:
                V[: NP // 4] = self.rng.uniform(LO, HI, (NP // 4, NPAR))
                if self.prev_best is not None:
                    V[NP // 4] = mu
            if commit is not None:
                # keep following the cursors already in use (the plan that justified the last split relied on them)
                ca_ = commit[a] - ca; cq_ = commit[q] - cq
                f = [ca_ @ u, ca_ @ w, cq_ @ u, cq_ @ w]
                nc = NP // 8
                V[-nc:, 1:5] = f
                V[-nc:, 7:11] = f
                V[-nc // 2:, 0] = self.rng.integers(0, HI[0], nc // 2)
            V = np.clip(V, LO, HI)
            V[:, 0] = np.round(V[:, 0])
            s = self.simulate(world, V, a, q, kind, sm, ca, cq, u, w, pend, new)
            if commit is not None:
                ca_ = commit[a] - ca; cq_ = commit[q] - cq
                dev = np.hypot(V[:, 1] - ca_ @ u, V[:, 2] - ca_ @ w) + np.hypot(V[:, 3] - cq_ @ u, V[:, 4] - cq_ @ w)
                s = s - 0.001 * dev
            o = np.argsort(-s)
            if s[o[0]] > best_s:
                best_s, best_v = s[o[0]], V[o[0]].copy()
                self.best_complete = bool(self.complete[o[0]])
            el = V[o[:max(8, NP // 12)]]
            mu = el.mean(0); sd = el.std(0) + (HI - LO) * 0.02
        self.prev_best = best_v
        # never commit to a split whose plan does not deliver every intended eat: keep manoeuvring
        if not self.best_complete and best_v[0] < 1:
            best_v = best_v.copy(); best_v[0] = self.replan
        return best_v, (a, q, kind, sm, ca, cq, u, w), best_s

    def simulate(self, world, V, a, q, kind, sm, ca, cq, u, w, pend, new):
        P, NP = self.plan, self.NP
        for kk in SplitrunEnv.SIM_ARRAYS:
            getattr(P, kk)[:] = getattr(world, kk)[0][None]
        P.active[:] = 1
        wait = V[:, 0].astype(int)
        to = lambda org, xy: org[None] + xy[:, :1] * u[None] + xy[:, 1:2] * w[None]
        A1 = to(ca, V[:, 1:3]); P1 = to(cq, V[:, 3:5]); A2 = to(ca, V[:, 7:9]); P2 = to(cq, V[:, 9:11])
        dirs = np.stack([np.cos(V[:, 5]) * u[0] - np.sin(V[:, 5]) * w[0], np.cos(V[:, 5]) * u[1] - np.sin(V[:, 5]) * w[1]], 1)
        exp_all = pend + [(o, m, em, -1) for (o, m, em) in new]
        nx = min(len(exp_all), MAXX)
        xo = np.zeros((NP, MAXX), np.int64); xm = np.zeros((NP, MAXX)); xem = np.zeros((NP, MAXX))
        dl = np.zeros(MAXX, int)
        for j, (o, m, em, d_) in enumerate(exp_all[:nx]):
            xo[:, j] = o; xm[:, j] = m; xem[:, j] = em; dl[j] = d_
        xn = np.full(NP, nx, np.int64)
        best = np.full((NP, MAXX), 3000.0)
        got = np.zeros((NP, MAXX), bool)
        eat_t = np.full((NP, MAXX), -1)
        ontime = np.zeros((NP, MAXX), bool)
        self.eat_t = eat_t
        score = np.zeros(NP)
        dead = np.zeros(NP, bool)
        T = int(HI[0]) + self.H
        mouse = np.zeros((NP, 2, 2)); splits = np.zeros((NP, 2), np.int64)
        npd = len(pend)
        fired = np.zeros(NP, bool)
        t_fire = np.full(NP, 10 ** 6)
        for t in range(T):
            # a split may only fire once its wait is over AND the earlier splits' eats have all happened
            ready = (t >= wait) & got[:, :min(npd, nx)].all(1) & ~fired
            before = ~fired & ~ready
            at = ready
            mouse[:, a] = np.where(before[:, None], A1, A2)
            mouse[:, q] = np.where(before[:, None], P1, P2)
            splits[:] = 0
            if at.any():
                ii = np.nonzero(at)[0]
                for i in ii:
                    s_ = find_piece(P, i, a, sm) if sm else -1
                    org = P.cf[i, s_, :2] if s_ >= 0 else ca
                    mouse[i, a] = org + dirs[i] * V[i, 6]
                splits[ii, a] = kind
                fired[ii] = True; t_fire[ii] = t
            P.active[:] = (~dead).astype(np.int64)
            P.step(mouse, splits)
            near_miss(P.cf, P.oc, P.ocn, xo, xm, xem, xn, best)
            for i in np.nonzero((P.evn > 0) & ~dead)[0]:
                for ev in P.ev[i, :P.evn[i]]:
                    eo, bo = int(ev[EV_EATER_OWN]), int(ev[EV_EATEN_OWN])
                    if eo == bo:
                        score[i] -= 12; dead[i] = True; break
                    m, em = ev[EV_MASS], ev[EV_EATER_MASS]
                    # new eats only count after this plan's split
                    lim = nx if t > t_fire[i] else min(npd, nx)
                    j = next((j for j in range(lim) if not got[i, j] and xo[i, j] == eo and abs(xm[i, j] - m) < TOL * xm[i, j]
                              and abs(xem[i, j] - em) < TOL * xem[i, j] + 0.15 * xm[i, j]), None)
                    if j is None:
                        score[i] -= 12; dead[i] = True; break
                    late = t > (dl[j] if dl[j] >= 0 else t_fire[i] + NEW_T)
                    got[i, j] = True
                    eat_t[i, j] = t
                    score[i] -= 0.15 * t                 # sooner is better (and survives replanning)
                    if not late:
                        ontime[i, j] = True
                    if late:
                        score[i] -= 10               # too late: the real run would already have failed
            if dead.all():
                break
        score -= 0.05 * np.minimum(t_fire, T)
        score -= 5 * ~fired
        score += 10 * got[:, :nx].sum(1)
        self.complete = ontime[:, :nx].all(1) & ~dead & fired
        score += 10 * self.complete
        miss = ~got[:, :nx]
        score += (miss * (3.0 * np.exp(-np.maximum(best[:, :nx], 0) / 250.0))).sum(1)
        # stay stationary
        cx = np.array([np.linalg.norm(com_pair(P, i) - (ca + cq) / 2) for i in range(NP)])
        score -= 0.002 * np.maximum(cx - 400, 0)
        return score


def com_pair(sim, e):
    c = [(sim.cf[e, s, X], sim.cf[e, s, Y], sim.cf[e, s, SZ] ** 2) for p in range(2) for s in sim.oc[e, p, :sim.ocn[e, p]]]
    m = sum(w for *_, w in c)
    return np.array([sum(x * w for x, _, w in c) / m, sum(y * w for _, y, w in c) / m])


def start_rows(t, gap=900.0):
    """each player's pieces in a compact row; players side by side"""
    rows = []
    for p, ms in enumerate((t["masses_a"], t["masses_b"])):
        sizes = [math.sqrt(m * TOTAL * 100) for m in ms]
        y = -sum(sizes) + sizes[0]
        for s in sizes:
            x = (-1 if p == 0 else 1) * (gap / 2 + s * 0.6)
            rows.append((p, x, y, s, 100, 0, 0, 0))
            y += 2 * s + 20
    return rows


def run(t, name, cycles=20, NP=384, seed=0, log=print, max_ticks=3000, rows=None):
    ctl = Controller(t, NP=NP, seed=seed)
    world = Sim(1)
    world.load(0, rows if rows is not None else start_rows(t), seed=1)
    rec = Recorder(world, 0, name)
    k, cyc, tick = 0, 0, 0
    pending = []                                   # (eater owner, eaten mass, eater mass, tick of split)
    plan, frame, t_plan = None, None, 0
    commit = None
    done_cycles, reason = 0, "max ticks"
    split_ticks = []
    while tick < max_ticks and cyc < cycles:
        if plan is None or tick - t_plan >= ctl.replan:
            plan, frame, sc = ctl.decide(world, k, cyc, pending, tick, commit)
            plan_ok = ctl.best_complete
            t_plan = tick
        a, q, kind, sm, ca, cq, u, w = frame
        el = tick - t_plan
        wait = int(plan[0]) - el
        to = lambda org, xy: org + xy[0] * u + xy[1] * w
        mouse = np.zeros((1, 2, 2)); splits = np.zeros((1, 2), np.int64)
        fired = False
        if wait > 0 or (wait == 0 and (pending or not plan_ok)):
            mouse[0, a] = to(ca, plan[1:3]); mouse[0, q] = to(cq, plan[3:5])
            if wait == 0:
                plan[0] += 1                          # earlier eats still pending: keep waiting
        elif wait == 0:
            s_ = find_piece(world, 0, a, sm) if sm else -1
            org = world.cf[0, s_, :2] if s_ >= 0 else ca
            d = np.cos(plan[5]) * u - np.sin(plan[5]) * w
            mouse[0, a] = org + d * plan[6]; mouse[0, q] = to(cq, plan[9:11])
            splits[0, a] = kind
            fired = True
        else:
            mouse[0, a] = to(ca, plan[7:9]); mouse[0, q] = to(cq, plan[9:11])
        world.step(mouse, splits)
        rec.after_step(np.trunc(mouse[0]), splits[0])
        commit = mouse[0].copy()
        if fired:
            commit[a] = to(ca, plan[7:9]); commit[q] = to(cq, plan[9:11])
        tick += 1
        if fired:
            split_ticks.append((tick, k, cyc))
            pending += [(o, m, em, tick) for (o, m, em) in ctl.event_eats(k, cyc)]
            k += 1
            if k == ctl.E:
                k = 0; cyc += 1; done_cycles = cyc
            plan = None; ctl.prev_best = None
        bad = None
        for ev in world.events(0):
            eo, bo = int(ev[EV_EATER_OWN]), int(ev[EV_EATEN_OWN])
            if eo == bo:
                bad = "own merge"; break
            m, em = ev[EV_MASS], ev[EV_EATER_MASS]
            j = next((j for j, (o, pm, pem, _) in enumerate(pending) if o == eo and abs(pm - m) < TOL * pm
                      and abs(pem - em) < TOL * pem + 0.15 * pm), None)
            if j is None:
                bad = f"unintended eat ({'AB'[eo]} ate {m:.0f})"; break
            pending.pop(j)
        if bad is None:
            late = [p for p in pending if tick - p[3] > PEND_T]
            if late:
                bad = f"intended eat never happened ({'AB'[late[0][0]]} eats {late[0][1]:.0f})"
        if bad:
            reason = bad
            break
        if tick % 100 == 0:
            log(f"   tick {tick}: cycle {cyc}, event {k}, pending {len(pending)}")
    # the cycle in progress is incomplete
    return rec, done_cycles, reason, split_ticks, tick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("catalogue"); ap.add_argument("index", type=int)
    ap.add_argument("--cycles", type=int, default=20); ap.add_argument("--np", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", default="", help="solver population (.npy): start from its best layout")
    a = ap.parse_args()
    t = json.load(open(a.catalogue))[a.index]
    rows = None
    if a.start:
        import tools.solve_technique as st
        tech = st.Technique(t)
        V = np.load(a.start)
        c, _, _ = st.evaluate(tech, V[:64], cycles=1, upto=1, close=False)
        rows = tech.rows(V[int(np.argmin(c))])
    base = os.path.basename(a.catalogue)[:-5]
    name = f"mpc_{base}_{a.index}"
    t0 = time.time()
    rec, n, reason, st, ticks = run(t, name, a.cycles, a.np, a.seed, rows=rows)
    out = os.path.join(ROOT, "runs", "controller"); os.makedirs(out, exist_ok=True)
    share = min(sum(t["masses_a"]), sum(t["masses_b"]))
    rec.data["description"] = (f"MPC controller on {base} #{a.index}: {n} complete cycles, stopped: {reason}. "
                               f"Smaller player's share {share:.2f}; split order {t['seq']}{' (mirror)' if t['mirror'] else ''}.")
    rec.save(os.path.join(out, name + ".json"))
    per = np.diff([s[0] for s in st if s[1] == 0]).mean() if n >= 2 else None
    r = dict(catalogue=base, index=a.index, cycles=n, reason=reason, ticks=ticks, share=share,
             ticks_per_cycle=per, seconds=round(time.time() - t0))
    with open(os.path.join(out, "mpc_results.jsonl"), "a") as f:
        f.write(json.dumps(r) + "\n")
    print(r, flush=True)


if __name__ == "__main__":
    main()
