"""Enumerate IMBALANCED splitrun mass programs whose splits can be aimed in closed form.

"Aimable" split (so a controller can compute the cursor instead of searching for it):
  * the splitting player has 1 or 2 cells (single split) or 1 cell (double split);
  * every eat of the split involves one of the pieces launched by that split (a young piece);
  * each piece takes part in at most one eat per split;
  * a 2-cell single split may use both launched pieces: the aim point is where the lines
    cell1 -> target1 and cell2 -> target2 cross (one cursor, two shots).
Pieces are referred to by rank (owner, index in that player's list sorted by mass, launched listed
after stayers). Masses come from iterating the bookkeeping to its attracting fixed point (like
tools/random_techniques.py); every eat needs >= 1.3x (+4% margin) at the fixed point.

    python tools/aimable_techniques.py --events 2 --max-share 0.4
Output: runs/techniques/aimable_e{E}.json (catalogue format + "shots": per split, which launched
piece does what to which partner piece).
"""
import argparse, itertools, json, os, sys, time
from multiprocessing import Pool
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RATIO, MARGIN = 1.3, 0.04
MINFRAC, MAXFRAC = 0.02, 22500 / 37000


def step(pieces, actor, kind, shots, check=True):
    """pieces: list of (owner, mass) sorted per owner by mass desc (A first).
    shots: tuple over the actor's launched pieces (in order of the actor's cells, rank order):
        None | ('eats', j) | ('eaten', j)   with j = partner piece rank
    returns new pieces (sorted) and the eats [(eater owner, eaten owner, eaten mass, eater mass)], or None"""
    mine = [m for o, m in pieces if o == actor]
    other = [m for o, m in pieces if o != actor]
    f = 0.5 ** kind
    stay = [m * f for m in mine]
    launched = [m * f for m in mine for _ in range(2 ** kind - 1)]
    other = list(other)
    gone_l, gone_o, eats = set(), set(), []
    gain_l = [0.0] * len(launched)
    gain_o = [0.0] * len(other)
    for i, s in enumerate(shots):
        if s is None:
            continue
        how, j = s
        if j >= len(other) or j in gone_o:
            return None
        if how == 'eats':
            ml, mo = launched[i] + gain_l[i], other[j] + gain_o[j]
            if check and ml < RATIO * (1 + MARGIN) * mo:
                return None
            gain_l[i] += mo; gone_o.add(j)
            eats.append((actor, 1 - actor, mo, ml))
        else:
            ml, mo = launched[i] + gain_l[i], other[j] + gain_o[j]
            if check and mo < RATIO * (1 + MARGIN) * ml:
                return None
            gain_o[j] += ml; gone_l.add(i)
            eats.append((1 - actor, actor, ml, mo))
    new_mine = stay + [launched[i] + gain_l[i] for i in range(len(launched)) if i not in gone_l]
    new_other = [other[j] + gain_o[j] for j in range(len(other)) if j not in gone_o]
    out = [(actor, m) for m in sorted(new_mine, reverse=True)] + [(1 - actor, m) for m in sorted(new_other, reverse=True)]
    return sorted(out, key=lambda p: (p[0], -p[1])), eats


def run(prog, x0, mirror, cycles=160):
    na, nb, events = prog
    pieces = [(0, m) for m in sorted(x0[:na], reverse=True)] + [(1, m) for m in sorted(x0[na:], reverse=True)]
    hist, last = [], None
    for c in range(cycles):
        sw = mirror and c % 2 == 1
        eats_c = []
        for actor, kind, shots in events:
            late = c >= cycles - 4                     # legality only on the settled orbit
            r = step(pieces, actor ^ sw, kind, shots, check=late)
            if r is None:
                return None
            pieces, ev = r
            if late and any(m < MINFRAC or m > MAXFRAC for _, m in pieces):
                return None
            eats_c.append([(a ^ sw, b ^ sw, m, em) for a, b, m, em in ev])   # role terms
        own = lambda o: sorted([m for p, m in pieces if p == (o ^ (1 if (mirror and c % 2 == 0) else 0))], reverse=True)
        a_, b_ = own(0), own(1)
        if len(a_) != na or len(b_) != nb:
            return None
        hist.append(np.array(a_ + b_)); last = eats_c
    if np.abs(hist[-1] - hist[-2]).max() > 1e-5 or (mirror is False and np.abs(hist[-1] - hist[-3]).max() > 1e-5):
        return None
    return hist[-1], last


MAXCELLS_SPLIT = 3   # a single split may come from up to 3 cells ...
MAXSHOTS = 2         # ... but at most 2 of its launched pieces are aimed at something (one cursor, two lines)
MAXP = 5             # pieces per player at any time


def programs(E, max_pieces=4):
    """all structural programs with E events (counts checked; masses later)"""
    out = []
    for na in range(1, max_pieces + 1):
        for nb in range(1, max_pieces + 1):
            # enumerate event sequences with piece counts tracked
            def rec(k, cnt, evs):
                if k == E:
                    out.append((na, nb, tuple(evs)))
                    return
                for actor in (0, 1):
                    for kind in (1, 2):
                        n_act = cnt[actor]
                        if (kind == 1 and n_act > MAXCELLS_SPLIT) or (kind == 2 and n_act != 1):
                            continue
                        nl = n_act * (2 ** kind - 1)
                        n_oth = cnt[1 - actor]
                        opts = [None] + [(h, j) for h in ('eats', 'eaten') for j in range(n_oth)]
                        for shots in itertools.product(opts, repeat=nl):
                            used = [s[1] for s in shots if s is not None]
                            if len(used) != len(set(used)) or not used or len(used) > MAXSHOTS:
                                continue                  # one eat per partner piece; every split does something
                            eaten_l = sum(1 for s in shots if s and s[0] == 'eaten')
                            eaten_o = sum(1 for s in shots if s and s[0] == 'eats')
                            c2 = list(cnt)
                            c2[actor] = n_act + nl - eaten_l
                            c2[1 - actor] = n_oth - eaten_o
                            if min(c2) < 1 or max(c2) > MAXP:
                                continue
                            rec(k + 1, c2, evs + [(actor, kind, shots)])
            rec(0, [na, nb], [])
    return out


def worker(args):
    progs, seed = args
    rng = np.random.default_rng(seed)
    found = []
    for prog in progs:
        na, nb, evs = prog
        if evs[0][0] != 0:
            continue                                   # A acts first (A/B relabelling)
        for mirror in (False, True):
            for _ in range(3):
                x0 = rng.dirichlet(np.ones(na + nb) * 2)
                r = run(prog, x0, mirror)
                if r is None:
                    continue
                x, eats = r
                found.append(dict(masses_a=x[:na].tolist(), masses_b=x[na:].tolist(), mirror=mirror,
                                  seq=[[a, k] for a, k, _ in evs], shots=[list(s) for _, _, s in evs],
                                  eats=[[list(e) for e in ev] for ev in eats]))
                break
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=2)
    ap.add_argument("--max-share", type=float, default=0.4)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--samples", type=int, default=0, help="sample this many random programs instead of enumerating")
    a = ap.parse_args()
    t0 = time.time()
    if a.samples:
        jobs = [(a.events, a.samples // (a.procs * 8), i) for i in range(a.procs * 8)]
        with Pool(a.procs) as pool:
            res = [f for part in pool.map(sample_worker, jobs) for f in part]
        print(f"{a.samples:,} sampled programs with {a.events} splits", flush=True)
    else:
        P = programs(a.events)
        print(f"{len(P):,} structural programs with {a.events} splits ({time.time() - t0:.0f}s)", flush=True)
        chunks = [(P[i::a.procs * 8], i) for i in range(a.procs * 8)]
        with Pool(a.procs) as pool:
            res = [f for part in pool.map(worker, chunks) for f in part]
    # distinct by rounded masses + sequence
    uniq = {}
    for t in res:
        share = min(sum(t["masses_a"]), sum(t["masses_b"]))
        if share > a.max_share:
            continue
        t["share"] = share
        key = (tuple(np.round(t["masses_a"], 3)), tuple(np.round(t["masses_b"], 3)), str(t["seq"]), t["mirror"])
        uniq.setdefault(key, t)
    out = sorted(uniq.values(), key=lambda t: t["share"])
    for t in out:
        t["rho"] = 0.0; t["margin"] = 0.0
        t["log"] = [f"{'AB'[a]} {'single' if k == 1 else 'double'}: " + "; ".join(
            f"{'AB'[x]} {em * 37000:,.0f} eats {'AB'[y]} {m * 37000:,.0f}" for x, y, m, em in ev) for (a, k), ev in zip(t["seq"], t["eats"])]
    path = os.path.join(ROOT, "runs", "techniques", f"aimable_e{a.events}{'_sampled' if a.samples else ''}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"{len(res)} fixed points, {len(out)} distinct with small share <= {a.max_share} -> {path} ({time.time() - t0:.0f}s)")
    for t in out[:40]:
        print(f"  share {t['share']:.2f} {'MIRROR' if t['mirror'] else 'same  '} A " + " + ".join(f"{m * 37000:,.0f}" for m in t["masses_a"]) +
              " | B " + " + ".join(f"{m * 37000:,.0f}" for m in t["masses_b"]) + " || " + " || ".join(t["log"]))


def random_programs(rng, E, n):
    """n random structural programs (same rules as programs(), sampled instead of enumerated)"""
    out = []
    while len(out) < n:
        na, nb = int(rng.integers(1, 5)), int(rng.integers(1, 4))
        cnt = [na, nb]; evs = []; ok = True
        for k in range(E):
            actor = 0 if k == 0 else int(rng.integers(0, 2))
            kind = 2 if (cnt[actor] == 1 and rng.random() < 0.3) else 1
            n_act = cnt[actor]
            if kind == 1 and n_act > MAXCELLS_SPLIT:
                ok = False; break
            nl = n_act * (2 ** kind - 1)
            n_oth = cnt[1 - actor]
            nshots = int(rng.integers(1, min(MAXSHOTS, nl, n_oth) + 1))
            which_l = rng.choice(nl, nshots, replace=False)
            which_o = rng.choice(n_oth, nshots, replace=False)
            shots = [None] * nl
            for i, j in zip(which_l, which_o):
                shots[i] = ('eats' if rng.random() < 0.5 else 'eaten', int(j))
            eaten_l = sum(1 for s in shots if s and s[0] == 'eaten')
            eaten_o = sum(1 for s in shots if s and s[0] == 'eats')
            cnt = list(cnt)
            cnt[actor] = n_act + nl - eaten_l; cnt[1 - actor] = n_oth - eaten_o
            if min(cnt) < 1 or max(cnt) > MAXP:
                ok = False; break
            evs.append((actor, kind, tuple(shots)))
        if ok and cnt == [na, nb] or (ok and cnt == [nb, na]):
            out.append((na, nb, tuple(evs)))
    return out


def sample_worker(args):
    E, n, seed = args
    rng = np.random.default_rng(seed)
    return worker((random_programs(rng, E, n), seed + 1))


if __name__ == "__main__":
    main()

