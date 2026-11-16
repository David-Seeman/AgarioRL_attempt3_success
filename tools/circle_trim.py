"""Imbalanced CIRCLE splitruns with TRIM splits (the structure human imbalanced splitruns use).

Circle split (A or B): a parallel volley. Every cell launches its half in the same direction; each
launched piece lands on at most one partner piece (or on nothing) and the 1.3x rule decides who eats.
Trim split (t = the big player A, single; T = double): a volley aimed off-circle at empty space, no
eats; it only makes A's pieces (and later its backpieces) smaller.
Program = pattern string (e.g. "tAB", "AtAB", "tABAB") + for every circle split the landing matching.
Masses: attracting fixed point of the bookkeeping (legality checked on the settled orbit only).

    python tools/circle_trim.py tAB AtB tABAB AtBAB ABtAB
"""
import itertools, json, os, sys, time
from multiprocessing import Pool
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import aimable_techniques as at  # noqa  (step(): volley bookkeeping with 'eats'/'eaten' shots)

RATIO, MARGIN = 1.3, 0.04
MINFRAC, MAXFRAC = 0.02, 22500 / 37000


def step(pieces, actor, kind, shots, check=True):
    """shots per launched piece: None | ('eats', (j,)) | ('eats', (j1, j2)) (flies through two, in order) |
    ('eaten', j) (partner piece j swallows it; j may swallow several)"""
    mine = [m for o, m in pieces if o == actor]
    other = [m for o, m in pieces if o != actor]
    f = 0.5 ** kind
    stay = [m * f for m in mine]
    launched = [m * f for m in mine for _ in range(2 ** kind - 1)]
    gone_o, eats, keep_l = set(), [], []
    gain_o = [0.0] * len(other)
    for i, sh in enumerate(shots):
        L = launched[i]
        if sh is None:
            keep_l.append(L); continue
        if sh[0] == 'eats':
            for j in sh[1]:
                if j >= len(other) or j in gone_o:
                    return None
                mo = other[j] + gain_o[j]
                if check and L < RATIO * (1 + MARGIN) * mo:
                    return None
                eats.append((actor, 1 - actor, mo, L)); L += mo; gone_o.add(j)
            keep_l.append(L)
        else:
            j = sh[1]
            if j >= len(other) or j in gone_o:
                return None
            mo = other[j] + gain_o[j]
            if check and mo < RATIO * (1 + MARGIN) * L:
                return None
            eats.append((1 - actor, actor, L, mo)); gain_o[j] += L
    new_mine = stay + keep_l
    new_other = [other[j] + gain_o[j] for j in range(len(other)) if j not in gone_o]
    out = [(actor, m) for m in sorted(new_mine, reverse=True)] + [(1 - actor, m) for m in sorted(new_other, reverse=True)]
    return sorted(out, key=lambda p: (p[0], -p[1])), eats


def run(prog, x0, cycles=160):
    na, nb, events = prog
    pieces = [(0, m) for m in sorted(x0[:na], reverse=True)] + [(1, m) for m in sorted(x0[na:], reverse=True)]
    hist = []
    for c in range(cycles):
        late = c >= cycles - 4
        eats_c = []
        for actor, kind, shots in events:
            r = step(pieces, actor, kind, shots, check=late)
            if r is None:
                return None
            pieces, ev = r
            if late and any(m < MINFRAC or m > MAXFRAC for _, m in pieces):
                return None
            eats_c.append(ev)
        a_ = sorted([m for p, m in pieces if p == 0], reverse=True); b_ = sorted([m for p, m in pieces if p == 1], reverse=True)
        if len(a_) != na or len(b_) != nb:
            return None
        hist.append(np.array(a_ + b_))
    if np.abs(hist[-1] - hist[-2]).max() > 1e-5:
        return None
    return hist[-1], eats_c


MAXC = 3
MIDC = 8            # pieces a player may hold mid-cycle


def programs(pattern, na, nb):
    out = []

    def rec(k, cnt, evs):
        if k == len(pattern):
            if cnt == [na, nb]:
                out.append((na, nb, tuple(evs)))
            return
        ch = pattern[k]
        actor = 0 if ch in "AtT" else 1
        kind = 2 if ch == "T" else 1
        n = cnt[actor]
        nl = n * (2 ** kind - 1)
        if n + nl > MIDC:
            return
        if ch in "tT":
            c2 = list(cnt); c2[actor] = n + nl
            if c2[actor] <= MIDC:
                rec(k + 1, c2, evs + [(actor, kind, (None,) * nl)])
            return
        no = cnt[1 - actor]
        opts = [None] + [('eats', (j,)) for j in range(no)] + [('eaten', j) for j in range(no)] + \
               [('eats', (j1, j2)) for j1 in range(no) for j2 in range(no) if j1 != j2]
        for shots in itertools.product(opts, repeat=nl):
            if all(s_ is None for s_ in shots):
                continue
            eaten_o = [j for s_ in shots if s_ and s_[0] == 'eats' for j in s_[1]]
            swallow = {s_[1] for s_ in shots if s_ and s_[0] == 'eaten'}
            if len(eaten_o) != len(set(eaten_o)) or swallow & set(eaten_o):
                continue                     # a partner piece is eaten at most once, and not while swallowing
            if sum(1 for s_ in shots if s_ and s_[0] == 'eats' and len(s_[1]) == 2) > 1:
                continue                     # at most one fly-through double eat per split
            el = sum(1 for s_ in shots if s_ and s_[0] == 'eaten'); eo = len(eaten_o)
            c2 = list(cnt); c2[actor] = n + nl - el; c2[1 - actor] = cnt[1 - actor] - eo
            if min(c2) < 1 or max(c2) > MIDC:
                continue
            rec(k + 1, c2, evs + [(actor, kind, shots)])
    rec(0, [na, nb], [])
    return out


def solve(args):
    pattern, na, nb, seed = args
    rng = np.random.default_rng(seed)
    found = []
    for prog in programs(pattern, na, nb):
        for _ in range(3):
            x0 = rng.dirichlet(np.ones(na + nb) * 2)
            r = run(prog, x0)
            if r is None:
                continue
            x, eats = r
            sa, sb = sum(x[:na]), sum(x[na:])
            if sb < sa:                                  # A must be the big player
                found.append(dict(pattern=pattern, masses_a=x[:na].tolist(), masses_b=x[na:].tolist(), mirror=False,
                                  seq=[[a, k] for a, k, _ in prog[2]], shots=[[list(x) if isinstance(x, tuple) else x for x in s] for _, _, s in prog[2]],
                                  eats=[[list(e) for e in ev] for ev in eats], share=float(sb)))
            break
    return found


def main(patterns):
    t0 = time.time()
    jobs = [(p, na, nb, i) for i, (p, na, nb) in enumerate((p, na, nb) for p in patterns for na in range(1, MAXC + 1) for nb in range(1, MAXC + 1))]
    print(f"{sum(len(programs(p, na, nb)) for p, na, nb, _ in jobs):,} programs", flush=True)
    with Pool(8) as pool:
        res = [f for part in pool.map(solve, jobs) for f in part]
    uniq = {}
    for t in res:
        key = (t["pattern"], tuple(np.round(t["masses_a"], 3)), tuple(np.round(t["masses_b"], 3)))
        uniq.setdefault(key, t)
    out = sorted(uniq.values(), key=lambda t: (sum(len(e) for e in t["eats"]), t["share"]))
    for t in out:
        t["rho"] = 0.0; t["margin"] = 0.0
        t["log"] = [f"{'AB'[a]} {'trim ' if not ev else ''}{'single' if k == 1 else 'double'}: " + "; ".join(
            f"{'AB'[x]} {em * 37000:,.0f} eats {'AB'[y]} {m * 37000:,.0f}" for x, y, m, em in ev) for (a, k), ev in zip(t["seq"], t["eats"])]
    path = os.path.join(ROOT, "runs", "techniques", "circle_trim.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"{len(out)} distinct imbalanced circle+trim techniques -> {path} ({time.time() - t0:.0f}s)")
    for i, t in enumerate(out[:30]):
        print(f"#{i} {t['pattern']} small share {t['share']:.2f} (A is {(1 - t['share']) / t['share']:.1f}x B) | A " +
              " + ".join(f"{m * 37000:,.0f}" for m in t["masses_a"]) + " | B " + " + ".join(f"{m * 37000:,.0f}" for m in t["masses_b"]))
        for l in t["log"]:
            print("      " + l)


if __name__ == "__main__":
    main(sys.argv[1:])
