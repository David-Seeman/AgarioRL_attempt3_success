"""Non-periodic creative splitruns: base circle policy + planned creative maneuvers.

Real-run rules (tools/freerun.py): every split's pieces must take part in a cross eat within USE ticks,
no slow absorption between two old pieces, no split press that launches nothing, neither player below
MINSHARE, at most GAPMAX ticks without a split press.
"""
import argparse, json, math, os, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from splitrun.sim import Sim, X, Y, SZ, EX, OWN, BIRTH, CID, G_TICK  # noqa
from splitrun.rlenv import SplitrunEnv, ACTION_REPEAT  # noqa
from splitrun.replay import Recorder, write_index  # noqa
from tools.freerun import register_split, score_tick, coms, rows_from_replay, MAXP, MAXID, MINSHARE, GAPMAX  # noqa
from tools.base_policy import BasePolicy  # noqa

IDLE_MAX = 50           # hard limit on ticks without any split press (2 s); downtime is minimised and reported


class Checker:
    """the real-run rules for a batch of N worlds"""
    def __init__(self, N):
        self.pids = np.zeros((N, MAXP, MAXID), np.int64); self.pn = np.zeros((N, MAXP), np.int64)
        self.pl = np.zeros((N, MAXP, MAXID), np.int64); self.pdl = np.full((N, MAXP), -1, np.int64)
        self.pok = np.zeros((N, MAXP), np.int64); self.dead = np.zeros(N, bool); self.score = np.zeros(N)
        self.eats = np.zeros(N, np.int64); self.olds = np.zeros(N, np.int64)
        self.last_split = np.zeros(N, np.int64); self.reason = [""] * N
        self.why = np.zeros(N, np.int64); self.when = np.zeros(N, np.int64)   # 1 absorption 2 drained 3 idle 4 no purpose / no-op

    def copy_from(self, other, i=0):
        for k in ("pids", "pn", "pl", "pdl", "pok", "last_split"):
            getattr(self, k)[:] = getattr(other, k)[i][None]

    def after_step(self, sim, splits):
        t = sim.g[:, G_TICK]
        for i in np.nonzero(splits.any(1) & ~self.dead)[0]:
            for p in (0, 1):
                if splits[i, p]:
                    register_split(sim.ci, sim.oc, sim.ocn, sim.g, i, p, self.pids, self.pn, self.pdl, self.pok,
                                   self.dead, self.score, self.pl)
            self.last_split[i] = t[i]
        score_tick(sim.g, sim.ev, sim.evn, sim.active, self.pids, self.pn, self.pdl, self.pok, self.dead, self.score,
                   self.eats, self.olds, self.pl, 0.0)
        ex = sim.ci[:, :, EX] == 1; own = sim.ci[:, :, OWN]; m = sim.cf[:, :, SZ] ** 2 * ex
        mA = (m * (own == 0)).sum(1); mB = (m * (own == 1)).sum(1)
        drained = np.minimum(mA, mB) < MINSHARE * (mA + mB)
        idle = t - self.last_split > IDLE_MAX
        was = self.dead.copy()
        self.dead |= drained | idle
        new = self.dead & ~self.why.astype(bool)
        for i in np.nonzero(new)[0]:
            self.why[i] = 1 if self.olds[i] else 2 if drained[i] else 3 if idle[i] else 4
            self.when[i] = t[i]
        return self.dead


def base_rollout(sim, pol, chk, ticks, rec=None):
    """run the base policy in every live world of sim for `ticks` ticks"""
    for _ in range(0, ticks, ACTION_REPEAT):
        since = sim.g[:, G_TICK] - chk.last_split
        mouse, splits = pol.act(sim, since)
        for r in range(ACTION_REPEAT):
            sim.active[:] = (~chk.dead).astype(np.int64)
            sp = splits if r == 0 else np.zeros_like(splits)
            sim.step(mouse, sp)
            if rec is not None:
                rec.after_step(np.trunc(mouse[0]), sp[0])
            chk.after_step(sim, sp)
        if chk.dead.all():
            break


def orbit_starts(pol, n_want=6, ticks=500, settle=50, log=print):
    """start states ON the base policy's own orbit: run it from the technique pools, keep worlds that
    survive `ticks` under the real-run rules, snapshot them `settle` ticks in"""
    from splitrun.rlenv import TechSpec, StartPool
    spec = TechSpec(json.load(open(os.path.join(ROOT, "runs/techniques/catalogue_s3_e2_p8.json")))[4])
    states = []
    for pth in ("pool_t7.pkl", "pool_t4.pkl", "pool_t0.pkl", "pool_t6.pkl"):
        pool = StartPool(spec).load(os.path.join(ROOT, "runs", "rl", pth))
        idx = np.linspace(0, len(pool.states) - 1, 96).astype(int)
        N = len(idx)
        sim = Sim(N)
        for j, i in enumerate(idx):
            for k, v in pool.states[i][0].items():
                getattr(sim, k)[j] = v
        chk = Checker(N); chk.last_split[:] = sim.g[:, G_TICK]
        base_rollout(sim, pol, chk, settle)
        snap = {k: getattr(sim, k).copy() for k in SplitrunEnv.SIM_ARRAYS}
        ck = {k: getattr(chk, k).copy() for k in ("pids", "pn", "pl", "pdl", "pok", "last_split")}
        alive0 = ~chk.dead
        base_rollout(sim, pol, chk, ticks - settle)
        good = np.nonzero(alive0 & ~chk.dead)[0]
        log(f"   {pth}: {len(good)}/{N} worlds survive {ticks / 25:.0f} s on the base policy")
        for j in good:
            states.append(({k: v[j].copy() for k, v in snap.items()}, {k: v[j].copy() for k, v in ck.items()}))
        if len(states) >= n_want:
            break
    return states


TYPES = ("trim", "double", "between", "push")
H_CHECK = 125          # a maneuver is accepted only if the base policy survives this long after it
WMAX = 10


def piece_table(sim, i, p):
    return [(int(sim.ci[i, s_, CID]), sim.cf[i, s_, X], sim.cf[i, s_, Y], sim.cf[i, s_, SZ]) for s_ in sim.oc[i, p, :sim.ocn[i, p]]]


def aim_of(sim, i, m, base_aim=None):
    """cursor for maneuver m (dict of arrays, row i) in world i at the moment it fires"""
    a = m["actor"][i]; q = 1 - a
    A = piece_table(sim, i, a); B = piece_table(sim, i, q)
    if not A or not B:
        return None
    (ca, _), (cq, _) = coms(sim, i)[a], coms(sim, i)[q]
    big = max(A, key=lambda c: c[3])
    u = cq - ca; u = u / max(np.linalg.norm(u), 1e-6); w = np.array([-u[1], u[0]])
    typ = m["type"][i]
    if typ == "between" and len(B) >= 2:
        j1 = m["pick"][i, 0] % len(B); j2 = m["pick"][i, 1] % len(B)
        if j1 == j2:
            j2 = (j1 + 1) % len(B)
        p1, p2 = np.array(B[j1][1:3]), np.array(B[j2][1:3])
        d_ = p2 - p1; d_ = d_ / max(np.linalg.norm(d_), 1e-6)
        return (p1 + p2) / 2 + m["perp"][i] * np.array([-d_[1], d_[0]])
    if typ == "push" and len(A) >= 2:
        others = [c for c in A if c[0] != big[0]]
        tgt = others[m["pick"][i, 0] % len(others)]
        return np.array(tgt[1:3]) + m["perp"][i] * w
    if base_aim is not None and typ in ("trim", "double"):
        # rotate the base policy's own aim around the splitter's largest piece
        v = base_aim - np.array(big[1:3])
        c, s_ = math.cos(m["phi"][i]), math.sin(m["phi"][i])
        return np.array(big[1:3]) + np.array([c * v[0] - s_ * v[1], s_ * v[0] + c * v[1]])
    return np.array(big[1:3]) + m["rho"][i] * (math.cos(m["phi"][i]) * u + math.sin(m["phi"][i]) * w)


def run_batch(sim, chk, pol, m, ticks, rec=None, stats=None):
    """step every world `ticks` ticks: base policy, except that world i fires maneuver m (if given) after
    m["wait"][i] ticks. Decisions every ACTION_REPEAT ticks from the current phase (ticks must be even)."""
    N = sim.cf.shape[0]
    fired = np.zeros(N, bool) if m is not None else np.ones(N, bool)
    t = 0
    while t < ticks:
        since = sim.g[:, G_TICK] - chk.last_split
        mouse, splits = pol.act(sim, since)
        for r in range(ACTION_REPEAT):
            sp = splits.copy() if r == 0 else np.zeros_like(splits)
            now = np.zeros(N, bool)
            if m is not None:
                # the maneuver takes over the base policy's next split press (timing unchanged)
                now = (sp.sum(1) > 0) & ~fired & ~chk.dead
                for i in np.nonzero(now)[0]:
                    a = 0 if sp[i, 0] > 0 else 1
                    m["actor"][i] = a
                    aim = aim_of(sim, i, m, base_aim=mouse[i, a].copy())
                    if aim is None:
                        now[i] = False
                        continue
                    mouse[i, a] = aim; sp[i, a] = m["kind"][i]
                    if stats is not None:
                        stats.on_fire(sim, i, a)
                fired |= now
            sim.active[:] = (~chk.dead).astype(np.int64)
            sim.step(mouse, sp)
            if rec is not None:
                rec.after_step(np.trunc(mouse[0]), sp[0])
            chk.after_step(sim, sp)
            if stats is not None:
                stats.after(sim, now)
            t += 1
        if chk.dead.all():
            break
    return fired


class ManeuverStats:
    """did the maneuver do what its type promises?"""
    def __init__(self, N, m):
        self.m = m
        self.fire_t = np.full(N, -1); self.launched = [set() for _ in range(N)]; self.older = [dict() for _ in range(N)]
        self.used_l = np.zeros(N, bool); self.pushed_ids = [set() for _ in range(N)]; self.push_ok = np.zeros(N, bool)
        self.gap = np.zeros(N, int)

    def on_fire(self, sim, i, a):
        self.older[i] = {c[0]: c[3] for c in piece_table(sim, i, a)}

    def after(self, sim, now):
        for i in np.nonzero(now)[0]:
            a = self.m["actor"][i]
            self.fire_t[i] = sim.g[i, G_TICK]
            self.launched[i] = {int(sim.ci[i, s_, CID]) for s_ in sim.oc[i, a, :sim.ocn[i, a]] if sim.ci[i, s_, BIRTH] == sim.g[i, G_TICK]}
        act = self.fire_t >= 0
        for i in np.nonzero(act)[0]:
            age = sim.g[i, G_TICK] - self.fire_t[i]
            if age > 45:
                continue
            a = self.m["actor"][i]
            if self.m["type"][i] == "push" and age >= 13:
                # own launched piece in contact with an older own piece (rigid push)
                A = piece_table(sim, i, a)
                L = [c for c in A if c[0] in self.launched[i]]; O = [c for c in A if c[0] in self.older[i]]
                for l_ in L:
                    for o in O:
                        if math.hypot(l_[1] - o[1], l_[2] - o[2]) < l_[3] + o[3] - 5:
                            self.pushed_ids[i].add(o[0])
            if sim.evn[i]:
                for ev in sim.ev[i, :sim.evn[i]]:
                    eo, bo, ea, eb = int(ev[0]), int(ev[1]), int(ev[3]), int(ev[4])
                    if eo == bo:
                        continue
                    if ea in self.launched[i] or eb in self.launched[i]:
                        self.used_l[i] = True
                    if bo == a and eb in self.pushed_ids[i]:
                        self.push_ok[i] = True


def sample_maneuvers(rng, n, types):
    typ = rng.choice(types, n)
    phi = rng.uniform(-math.radians(35), math.radians(35), n)          # doubles: near the base aim
    tr = typ == "trim"
    phi[tr] = rng.choice([-1, 1], tr.sum()) * rng.uniform(math.radians(70), math.radians(150), tr.sum())
    return dict(type=typ, actor=rng.integers(0, 2, n), wait=rng.integers(0, WMAX + 1, n),
                kind=np.where(typ == "double", 2, 1), phi=phi, rho=rng.uniform(300, 2600, n),
                pick=rng.integers(0, 1000, (n, 2)), perp=rng.uniform(-250, 250, n))


def plan_maneuver(world, chk0, pol, types, rng, NP=160, exclude=()):
    sim = Sim(NP)
    for k in SplitrunEnv.SIM_ARRAYS:
        getattr(sim, k)[:] = getattr(world, k)[0][None]
    chk = Checker(NP); chk.copy_from(chk0)
    m = sample_maneuvers(rng, NP, types)
    st = ManeuverStats(NP, m)
    fired = run_batch(sim, chk, pol, m, 30 + H_CHECK, stats=st)
    typ = m["type"]
    ok = fired & ~chk.dead & (st.fire_t >= 0)
    ok &= np.where(typ == "trim", ~st.used_l, True)
    ok &= np.where((typ == "double") | (typ == "between"), st.used_l, True)
    ok &= np.where(typ == "push", st.push_ok, True)
    if not ok.any():
        return None
    score = -(st.fire_t - world.g[0, G_TICK]).astype(float) + rng.random(NP) * 0.01
    score[~ok] = -1e9
    i = int(np.argmax(score))
    return {k: (v[i:i + 1].copy() if isinstance(v, np.ndarray) else v) for k, v in m.items()}


def snapshot(world, chk, rec):
    return ({k: getattr(world, k).copy() for k in SplitrunEnv.SIM_ARRAYS},
            {k: getattr(chk, k).copy() for k in ("pids", "pn", "pl", "pdl", "pok", "last_split", "dead", "eats", "olds", "score")},
            len(rec.data["frames"]), len(rec.data["inputs"]), len(rec.data["events"]))


def restore(world, chk, rec, snap):
    ws, cs, nf, ni, ne = snap
    for k, v in ws.items():
        getattr(world, k)[:] = v
    for k, v in cs.items():
        getattr(chk, k)[:] = v
    del rec.data["frames"][nf:]; del rec.data["inputs"][ni:]; del rec.data["events"][ne:]


def creative_run(name, start, pol, types, ticks=750, every=40, seed=0, log=print):
    rng = np.random.default_rng(seed)
    world = Sim(1)
    for k, v in start[0].items():
        getattr(world, k)[0] = v
    chk = Checker(1)
    for k, v in start[1].items():
        getattr(chk, k)[0] = v
    rec = Recorder(world, 0, name)
    t0 = world.g[0, G_TICK]
    moves, stack, retries = [], [], 0
    next_try = 0
    while world.g[0, G_TICK] - t0 < ticks:
        el = world.g[0, G_TICK] - t0
        if el >= next_try:
            m = plan_maneuver(world, chk, pol, types, rng)
            if m is not None:
                stack.append((snapshot(world, chk, rec), len(moves)))
                st = ManeuverStats(1, m)
                for _ in range(30):
                    fired = run_batch(world, chk, pol, m, 2, rec=rec, stats=st)
                    if fired[0] or chk.dead[0]:
                        break
                    m_ = m                                   # not fired yet: keep waiting for the base split
                moves.append(dict(tick=int(st.fire_t[0] - t0), type=str(m["type"][0]), actor="AB"[int(m["actor"][0])],
                                  kind=int(m["kind"][0])))
                next_try = world.g[0, G_TICK] - t0 + every
                continue
            next_try = el + 10
        run_batch(world, chk, pol, None, 2, rec=rec)
        if chk.dead[0]:
            if stack and retries < 30:
                snap, nm = stack.pop()
                restore(world, chk, rec, snap); del moves[nm:]
                retries += 1
                next_try = world.g[0, G_TICK] - t0 + 4          # re-plan a little later (new random candidates)
                continue
            break
    dur = (world.g[0, G_TICK] - t0) / 25
    return rec, dict(name=name, seconds=float(dur), moves=moves, died=bool(chk.dead[0]), retries=retries, types=list(types))


if __name__ == "__main__":
    import pickle
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="creative"); ap.add_argument("--types", default="trim,double,between,push")
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=40); ap.add_argument("--ticks", type=int, default=750)
    a = ap.parse_args()
    pol = BasePolicy("discover_s0")
    starts = pickle.load(open(os.path.join(ROOT, "runs", "freeruns", "orbit_starts.pkl"), "rb"))
    t0 = time.time()
    rec, info = creative_run(a.name, starts[a.start], pol, a.types.split(","), a.ticks, a.every, a.seed)
    out = os.path.join(ROOT, "runs", "freeruns"); os.makedirs(out, exist_ok=True)
    rec.save(os.path.join(out, a.name + ".json"))
    json.dump(info, open(os.path.join(out, a.name + ".info.json"), "w"), indent=1)
    from collections import Counter
    print(f"{a.name}: {info['seconds']:.1f} s, died={info['died']}, maneuvers {dict(Counter(x['type'] for x in info['moves']))}, "
          f"retries {info['retries']} ({time.time() - t0:.0f}s)")
