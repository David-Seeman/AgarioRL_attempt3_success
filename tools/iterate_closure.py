"""Close a technique's cycle by FIXED-POINT ITERATION on the starting layout.

Direct search for a periodic start fails when the end layout has a different SHAPE from the start
(e.g. A ends up between B's two pieces): the search cannot jump between shapes without breaking eats.
Instead:
    start_{n+1} = end layout of the cycle run from start_n   (positions only; masses/counts repeat)
and after each move re-tune only the controls (split aims, movement, timings) so every intended eat
happens again from the new start. If the start -> end map settles, the start is on the orbit.

    python tools/iterate_closure.py <catalogue> <index> <population.npy> [--iters 20]
"""
import argparse, json, os, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import tools.solve_technique as st  # noqa
from splitrun.metrics import _descriptor, _layout_error  # noqa


def end_positions(tech, end_cells):
    """end layout -> the parameter layout (pieces in catalogue order: role A by mass desc, then role B;
    positions relative to piece 0)"""
    swap = 1 if tech.mirror else 0
    cs = sorted(end_cells, key=lambda c: ((c[1] ^ swap), -c[4]))
    P = np.array([[c[2], c[3]] for c in cs])
    P = P - P[0]
    return P[1:].ravel()


def retune(tech, v, rng, pop=1024, gens=20, log=print):
    """keep the start positions, re-optimise the controls until all eats of one cycle happen"""
    ctrl = np.arange(tech.npos, tech.dim)
    V = np.repeat(v[None], pop, 0)
    V[1:, ctrl] += rng.normal(0, 1, (pop - 1, len(ctrl))) * (tech.HI - tech.LO)[ctrl] * 0.03
    V = np.clip(V, tech.LO, tech.HI)
    fn = st.robust(tech, lambda X: st.evaluate(tech, X, cycles=1, upto=tech.E, close=False)[0])
    c = fn(V)
    o = np.argsort(c)[:64]; el, ec = V[o], c[o]
    scale = 0.03
    for g in range(gens):
        if ec[0] < 0.7:
            break
        kids = el[rng.integers(0, len(el), pop)].copy()
        kids[:, ctrl] += rng.normal(0, 1, (pop, len(ctrl))) * (tech.HI - tech.LO)[ctrl] * scale
        kids = np.clip(kids, tech.LO, tech.HI)
        ck = fn(kids)
        allV = np.concatenate([el, kids]); allc = np.concatenate([ec, ck])
        o = np.argsort(allc)[:64]; el, ec = allV[o], allc[o]
        scale *= 0.93
    return el[0], ec[0]


def iterate(tech, v, iters=20, seed=0, log=print):
    rng = np.random.default_rng(seed)
    st.ONORBIT = True
    hist = []
    for it in range(iters):
        ends = {}
        c, cyc, rec = st.evaluate(tech, v[None], cycles=1, ends=ends)
        if 0 not in ends or len(ends[0]) != tech.n:
            log(f"   iter {it}: cycle not clean (cost {c[0]:.1f}) -> re-tuning controls")
            v, cc = retune(tech, v, rng)
            log(f"      re-tuned: eats cost {cc:.2f}")
            if cc > 5:
                return v, False, hist
            continue
        start = [[i, r[0], r[1], r[2], r[3]] for i, r in enumerate(tech.rows(v))]
        a, b = _descriptor(start, 0), _descriptor(ends[0], 1 if tech.mirror else 0)
        err = _layout_error(a, b) if a[0] == b[0] else np.inf
        hist.append(err)
        log(f"   iter {it}: start->end layout change {err:.0f} units")
        if err < 25:
            return v, True, hist
        # move the start to where the cycle ended (a fixed point of this map is on the orbit)
        v = v.copy()
        v[:tech.npos] = np.clip(end_positions(tech, ends[0]), tech.LO[:tech.npos], tech.HI[:tech.npos])
    return v, False, hist


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("catalogue"); ap.add_argument("index", type=int); ap.add_argument("population")
    ap.add_argument("--iters", type=int, default=20); ap.add_argument("--tries", type=int, default=4)
    ap.add_argument("--last-gap", type=int, default=70)
    a = ap.parse_args()
    tech = st.Technique(json.load(open(a.catalogue))[a.index])
    tech.HI[tech.npos + 7 * (tech.E - 1) + 2] = a.last_gap
    st.ONORBIT = True
    popu = np.load(a.population)
    c, _, _ = st.evaluate(tech, popu, cycles=1, upto=tech.E, close=False)
    order = np.argsort(c)
    base = os.path.basename(a.catalogue)[:-5]
    for j in order[:a.tries]:
        t0 = time.time()
        print(f"candidate {j} (eats cost {c[j]:.2f})", flush=True)
        v, ok, hist = iterate(tech, popu[j], a.iters)
        if ok:
            n, cc, _ = st.chain_test(tech, v, 100)
            print(f"   CLOSED; chain test {n}/100 ({time.time() - t0:.0f}s)", flush=True)
            np.save(os.path.join(ROOT, "runs", "techniques", f"{base}_t{a.index}_closed.npy"), np.repeat(v[None], 16, 0))
            if n >= 100:
                break
