"""Overnight screening of many-split technique candidates, discarding early.

For each structurally distinct family (not plain A/B alternation), take its best mass patterns and:
  1. curriculum (can every intended eat of one cycle be made?)        -> discard if not
  2. greedy chain from a quiet cycle boundary (does it keep going?)    -> discard if < 2 cycles
Survivors are ranked by chain length; their chain states are saved as RL start pools.

    python tools/screen_many.py --worker 0 --workers 2
Results append to runs/techniques/screen.jsonl (one line per candidate).
"""
import argparse, json, os, sys, time
from collections import defaultdict
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import tools.solve_technique as st  # noqa
import tools.build_chain as bc  # noqa
from tools.technique_families import family, is_circle  # noqa
from splitrun.rlenv import TechSpec, StartPool  # noqa

FILES = ["random_e4_ranked.json", "random_e5_ranked.json", "random_e6_ranked.json",
         "random_e4_AABB_ranked.json", "random_e3_AAB_ranked.json", "random_e3_AAA_ranked.json"]


def candidates(per_family=2, files=FILES, all_families=False):
    fams = defaultdict(list)
    for f in files:
        p = os.path.join(ROOT, "runs", "techniques", f)
        if not os.path.exists(p):
            continue
        for i, t in enumerate(json.load(open(p))):
            k = family(t)
            if all_families or not is_circle(k):
                fams[k].append((t["margin"], f, i))
    out = []
    for k, m in sorted(fams.items(), key=lambda kv: -max(x[0] for x in kv[1])):
        for margin, f, i in sorted(m, reverse=True)[:per_family]:
            out.append((f, i, "/".join(map(str, k))))
    return out


def curriculum(tech, rng, pop=2048, gens=25):
    V = rng.uniform(tech.LO, tech.HI, (pop, tech.dim))
    for j in range(1, tech.E + 1):
        fn = st.robust(tech, lambda X, j=j: st.evaluate(tech, X, cycles=1, upto=j, close=False)[0], K=2)
        c = fn(V)
        el = V[np.argsort(c)[:128]]; ec = np.sort(c)[:128]
        for g in range(1, gens + 1):
            kids = np.clip(el[rng.integers(0, 128, pop)] + rng.normal(0, 1, (pop, tech.dim)) * (tech.HI - tech.LO) * 0.08 * 0.95 ** g,
                           tech.LO, tech.HI)
            ck = fn(kids)
            allV = np.concatenate([el, kids]); allc = np.concatenate([ec, ck])
            o = np.argsort(allc)[:128]; el, ec = allV[o], allc[o]
            if ec[0] < 0.5:
                break
        if ec[0] > 3:
            return None, j, ec[0]                    # this split's eats cannot be made: discard
        V = el[rng.integers(0, 128, pop)]; V[:128] = el
    return el, tech.E, ec[0]


def chain(tech, t, v, rng, quiet=80, cycles=10):
    st.ONORBIT = True
    gi = tech.npos + 7 * (tech.E - 1) + 2
    tech.HI[gi] = max(tech.HI[gi], quiet)
    v = v.copy(); v[gi] = quiet
    tracker, pool = bc.Tracker(tech), StartPool(TechSpec(t))
    state, cyc, n_ok = None, 0, 0
    for n in range(cycles):
        if n > 0:
            v2, cost = bc.retune(tech, v, state, cyc, rng, 1024)
            if cost >= 9.5:
                v2, cost = bc.resolve(tech, v, state, cyc, rng)
            v = v2
        before = len(pool.states)
        cost, end, _ = bc.run_one(tech, v, state, cyc, tracker, pool)
        if end is None or cost >= 9.5:
            del pool.states[before:]
            break
        n_ok += 1
        state, cyc = end, cyc + 1
    return n_ok, pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0); ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--files", nargs="*", default=FILES)
    ap.add_argument("--all-families", action="store_true", help="also screen plain A/B alternation (circle) families")
    ap.add_argument("--per-family", type=int, default=2)
    ap.add_argument("--out", default="screen.jsonl")
    a = ap.parse_args()
    cands = candidates(a.per_family, a.files, a.all_families)
    mine = cands[a.worker::a.workers]
    out = os.path.join(ROOT, "runs", "techniques", a.out)
    done = set()
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line); done.add((r["file"], r["index"]))
    print(f"worker {a.worker}: {len(mine)} candidates", flush=True)
    for f, i, fam in mine:
        if (f, i) in done:
            continue
        t0 = time.time()
        t = json.load(open(os.path.join(ROOT, "runs", "techniques", f)))[i]
        tech = st.Technique(t)
        rng = np.random.default_rng(i)
        st.ONORBIT = False
        el, stage, cost = curriculum(tech, rng)
        res = dict(file=f, index=i, family=fam, splits=tech.E, curriculum_ok=el is not None,
                   failed_split=None if el is not None else stage, chain=0)
        if el is not None:
            n_ok, pool = chain(tech, t, el[0], rng)
            res["chain"] = n_ok
            if n_ok >= 2:
                p = os.path.join(ROOT, "runs", "rl", f"chainpool_{f[:-5]}_{i}.pkl")
                pool.save(p)
                res["pool"] = p
                res["pool_states"] = len(pool.states)
        res["seconds"] = round(time.time() - t0)
        with open(out, "a") as fh:
            fh.write(json.dumps(res) + "\n")
        print(res, flush=True)


if __name__ == "__main__":
    main()
