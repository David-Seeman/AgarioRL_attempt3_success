"""Imbalanced splitruns built from ONE geometric primitive: the convergent volley.

The cursor is placed ON a partner piece T. Every cell of the splitter launches along the line from
itself to T; a launched piece interacts with T only if its parent is within ~820 + eat reach of T
(cells further away "fall short" and simply keep their launched half). The controller chooses the
formation, so it chooses WHICH cells are in range; arrival order = nearest first.
Interaction at T, in arrival order (T grows as it eats):
    T >= 1.3 x piece  -> T eats it          piece >= 1.3 x T -> the piece eats T (T is gone;
    otherwise         -> rejected (no clean outcome)            later arrivals find nothing)
A split is (actor, single|double, target rank j, ordered subset of the actor's cells in range).
Masses: iterate the bookkeeping to its attracting fixed point; legality checked on the settled orbit.
Also reported: the oldest piece age in cycles (old same-owner pieces eventually merge).

    python tools/volley_techniques.py --events 2
"""
import argparse, itertools, json, os, time
from multiprocessing import Pool
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RATIO, MARGIN = 1.3, 0.04
MINFRAC, MAXFRAC = 0.02, 22500 / 37000
MAXC = 4           # cells per player


def volley(pieces, actor, kind, j, rng_cells, check, age):
    """pieces: list of [owner, mass, born_cycle]; returns (new pieces, eats) or None"""
    mine = sorted([p for p in pieces if p[0] == actor], key=lambda p: -p[1])
    other = sorted([p for p in pieces if p[0] != actor], key=lambda p: -p[1])
    if j >= len(other) or any(i >= len(mine) for i in rng_cells):
        return None
    f = 0.5 ** kind
    new_mine = []
    arrivals = []
    for i, p in enumerate(mine):
        new_mine.append([actor, p[1] * f, p[2]])                         # stayer keeps its age
        launched = [[actor, p[1] * f, age] for _ in range(2 ** kind - 1)]
        if i in rng_cells:
            arrivals.append((rng_cells.index(i), launched))
        else:
            new_mine += launched                                         # fell short
    T = list(other[j]); alive = True
    eats = []
    for _, group in sorted(arrivals, key=lambda x: x[0]):
        for L in group:
            if not alive:
                new_mine.append(L); continue
            if T[1] >= RATIO * (1 + MARGIN) * L[1] or not check and T[1] >= L[1]:
                eats.append((1 - actor, actor, L[1], T[1])); T[1] += L[1]
            elif L[1] >= RATIO * (1 + MARGIN) * T[1] or not check and L[1] > T[1]:
                eats.append((actor, 1 - actor, T[1], L[1])); L[1] += T[1]; alive = False
                new_mine.append(L)
            else:
                return None
    if not eats:
        return None
    rest = [p for k, p in enumerate(other) if k != j] + ([T] if alive else [])
    return new_mine + rest, eats


def run(prog, x0, mirror, cycles=150):
    na, nb, events = prog
    pieces = [[0, m, 0] for m in x0[:na]] + [[1, m, 0] for m in x0[na:]]
    hist, last, maxage = [], None, 0
    for c in range(cycles):
        sw = mirror and c % 2 == 1
        late = c >= cycles - 6
        eats_c = []
        for e, (actor, kind, j, rc) in enumerate(events):
            r = volley(pieces, actor ^ sw, kind, j, list(rc), late, c + 1)
            if r is None:
                return None
            pieces, ev = r
            if len([p for p in pieces if p[0] == 0]) > MAXC or len([p for p in pieces if p[0] == 1]) > MAXC:
                return None
            if late and any(p[1] < MINFRAC or p[1] > MAXFRAC for p in pieces):
                return None
            eats_c.append([(a ^ sw, b ^ sw, m, em) for a, b, m, em in ev])
        ro = lambda o: o ^ (1 if (mirror and c % 2 == 0) else 0)
        a_ = sorted([p[1] for p in pieces if p[0] == ro(0)], reverse=True)
        b_ = sorted([p[1] for p in pieces if p[0] == ro(1)], reverse=True)
        if len(a_) != na or len(b_) != nb:
            return None
        hist.append(np.array(a_ + b_)); last = eats_c
        if late:
            maxage = max(maxage, max(c + 1 - p[2] for p in pieces))
    if np.abs(hist[-1] - hist[-2]).max() > 1e-5:
        return None
    return hist[-1], last, maxage


def programs(E):
    out = []
    for na in range(1, MAXC + 1):
        for nb in range(1, MAXC + 1):
            def rec(k, cnt, evs):
                if k == E:
                    out.append((na, nb, tuple(evs)))
                    return
                for actor in ((0,) if k == 0 else (0, 1)):
                    for kind in (1, 2):
                        n = cnt[actor]
                        if kind == 2 and n > 2:
                            continue
                        for j in range(cnt[1 - actor]):
                            for r in range(1, n + 1):
                                for rc in itertools.permutations(range(n), r):
                                    evs2 = evs + [(actor, kind, j, rc)]
                                    # counts are mass dependent (who eats whom), so only bound them loosely here
                                    rec(k + 1, cnt, evs2)
            rec(0, [na, nb], [])
    return out


def worker(args):
    progs, seed = args
    rng = np.random.default_rng(seed)
    found = []
    for prog in progs:
        na, nb, evs = prog
        for mirror in (False, True):
            for _ in range(4):
                x0 = rng.dirichlet(np.ones(na + nb) * 2)
                r = run(prog, x0, mirror)
                if r is None:
                    continue
                x, eats, maxage = r
                found.append(dict(masses_a=x[:na].tolist(), masses_b=x[na:].tolist(), mirror=mirror,
                                  seq=[[a, k] for a, k, _, _ in evs], volleys=[[a, k, j, list(rc)] for a, k, j, rc in evs],
                                  eats=[[list(e) for e in ev] for ev in eats], max_age_cycles=int(maxage)))
                break
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=2)
    ap.add_argument("--max-share", type=float, default=0.42)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time()
    P = programs(a.events)
    print(f"{len(P):,} volley programs with {a.events} splits", flush=True)
    chunks = [(P[i::a.procs * 8], i) for i in range(a.procs * 8)]
    with Pool(a.procs) as pool:
        res = [f for part in pool.map(worker, chunks) for f in part]
    uniq = {}
    for t in res:
        t["share"] = min(sum(t["masses_a"]), sum(t["masses_b"]))
        key = (tuple(np.round(t["masses_a"], 3)), tuple(np.round(t["masses_b"], 3)), t["mirror"],
               str([[e[0], e[1]] for ev in t["eats"] for e in ev]))
        uniq.setdefault(key, t)
    allt = sorted(uniq.values(), key=lambda t: t["share"])
    for t in allt:
        t["rho"] = 0.0; t["margin"] = 0.0
        t["log"] = [f"{'AB'[v[0]]} {'single' if v[1] == 1 else 'double'} at {'AB'[1 - v[0]]}'s #{v[2]} (cells in range {v[3]}): " + "; ".join(
            f"{'AB'[x]} {em * 37000:,.0f} eats {'AB'[y]} {m * 37000:,.0f}" for x, y, m, em in ev) for v, ev in zip(t["volleys"], t["eats"])]
    path = os.path.join(ROOT, "runs", "techniques", f"volley_e{a.events}.json")
    json.dump(allt, open(path, "w"), indent=1)
    imb = [t for t in allt if t["share"] <= a.max_share]
    print(f"{len(allt)} distinct techniques, {len(imb)} with small share <= {a.max_share} -> {path} ({time.time() - t0:.0f}s)")
    for t in imb[:40]:
        print(f"  share {t['share']:.2f} {'MIRROR' if t['mirror'] else 'same  '} max age {t['max_age_cycles']} cyc | A " +
              " + ".join(f"{m * 37000:,.0f}" for m in t["masses_a"]) + " | B " + " + ".join(f"{m * 37000:,.0f}" for m in t["masses_b"]))
        for l in t["log"]:
            print("        " + l)


if __name__ == "__main__":
    main()
