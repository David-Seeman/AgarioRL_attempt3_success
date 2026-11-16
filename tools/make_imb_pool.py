"""Start pool of IMBALANCED worlds for discovery RL (one player holds ~15-35% of the mass).

States are taken from the imbalanced mass patterns' partly-solved geometry (tools/solve_technique.py
elites): the static start, and a snapshot at every split press of one played cycle (pieces in flight).
Plus the short chains found by tools/screen_many.py.

    python tools/make_imb_pool.py runs/rl/pool_imb.pkl
"""
import glob, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import tools.solve_technique as st  # noqa
from splitrun.sim import Sim  # noqa
from splitrun.rlenv import TechSpec, StartPool, SplitrunEnv  # noqa


def main(out):
    ref = json.load(open(os.path.join(ROOT, "runs/techniques/random_e2_imb2_good.json")))[12]
    pool = StartPool(TechSpec(ref))
    for f in sorted(glob.glob(os.path.join(ROOT, "runs/techniques/random_e[23]_imb2_good_t*_elite.npy"))) + \
            sorted(glob.glob(os.path.join(ROOT, "runs/techniques/random_e[23]_imb2_good_t*_p1.npy"))):
        cat = f.split("_t")[0] + ".json"
        k = int(f.split("_t")[-1].split("_")[0])
        tech = st.Technique(json.load(open(cat))[k])
        tech.HI[[tech.npos + 7 * e + 2 for e in range(tech.E)]] = 40
        V = np.load(f)[:32]
        c, _, _ = st.evaluate(tech, V, cycles=1, upto=tech.E, close=False)
        V = V[c < 5.5]                               # at most one intended eat missed in the cycle
        if not len(V):
            print(f"{os.path.basename(f)}: no usable candidates"); continue
        n0 = len(pool.states)
        sim0 = Sim(1)
        for v in V:
            sim0.load(0, tech.rows(v), seed=1)
            pool.add_from_sim(sim0, 0, 0, 0, [])
        snaps = []

        def hook(sim, tc, cyc, tev, alive, splits):
            for i in np.nonzero(alive & (splits.sum(1) > 0))[0]:
                if cyc[i] == 0:
                    snaps.append({kk: getattr(sim, kk)[i].copy() for kk in SplitrunEnv.SIM_ARRAYS})
        st.evaluate(tech, V, cycles=1, upto=tech.E, close=False, hook=hook)
        for s_ in snaps:
            pool.states.append((s_, 0, 0, []))
        print(f"{os.path.basename(f)}: {len(V)} usable candidates -> {len(pool.states) - n0} states", flush=True)
    for f in glob.glob(os.path.join(ROOT, "runs/rl/chainpool_random_e[23]_imb2_good_*.pkl")):
        extra = StartPool(pool.spec).load(f).states
        pool.states += [(s_, 0, 0, []) for s_, *_ in extra]
        print(f"{os.path.basename(f)}: {len(extra)} states")
    pool.save(out)
    print(f"pool: {len(pool.states)} states -> {out}")




def filter_pool(path, out, ticks=40):
    """drop states that break on their own (slow absorption / one-sided) within `ticks` of doing nothing"""
    import json
    from splitrun.rlenv_discover import DiscoverEnv
    from splitrun.rlenv import T_DONE, T_TICKS
    ref = json.load(open(os.path.join(ROOT, "runs/techniques/random_e2_imb2_good.json")))[12]
    spec = TechSpec(ref)
    pool = StartPool(spec).load(path)
    n = len(pool.states)
    env = DiscoverEnv(spec, pool, n, seed=0)

    class Pick:                      # restore world i into env i
        def __init__(self): self.i = 0
        def integers(self, n_): self.i += 1; return self.i - 1
    rng = Pick()
    for e in range(n):
        pool.restore(env, e, rng)
    bad = np.zeros(n, bool)

    def no_reset(idx):                # a finished world stays finished (and is judged by its reason)
        bad[idx] |= np.isin(env.tr[idx, T_DONE], [3, 4, 7, 9])
        env.tr[idx, T_DONE] = np.maximum(env.tr[idx, T_DONE], 1)
    env.reset = no_reset
    for _ in range(ticks // 2):
        env.advance(_centre_mouse(env), np.zeros((n, 2), np.int64), 2)
    keep = [s_ for s_, b in zip(pool.states, bad) if not b]
    pool.states = keep
    pool.save(out)
    print(f"kept {len(keep)} / {n} states -> {out}")


def _centre_mouse(env):
    from splitrun.rlenv import _com
    sim = env.sim
    m = np.zeros((sim.cf.shape[0], 2, 2))
    for e in range(sim.cf.shape[0]):
        for p in range(2):
            x, y, _ = _com(sim.cf, sim.oc, sim.ocn, e, p)
            m[e, p] = (x, y)
    return m


if __name__ == "__main__":
    if sys.argv[1] == "--filter":
        filter_pool(sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1])
