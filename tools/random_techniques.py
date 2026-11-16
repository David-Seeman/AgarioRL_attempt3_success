"""Find clean periodic techniques with MANY splits per cycle by sampling random split/eat programs.

Exhaustive enumeration explodes beyond 2 splits per cycle, so instead:
  1. draw a random program: a cycle of E split events (actor, single/double) and, after each,
     which pieces eat which -- referring to pieces by RANK (player, mass order, stayer/launched);
     every eat involves a flying piece (launched by this or the previous event);
  2. run its mass bookkeeping from random starting masses, cycle after cycle (with A<->B swapped
     each cycle for mirror techniques);
  3. a real technique is an attracting fixed point (stability rho < 1), so its masses settle onto
     the periodic pattern by themselves; accept if they do, with every eat >= 1.3x (+margin) and no
     piece above the 22,500 autosplit limit.
Output is in the catalogue format used by tools/solve_technique.py.

    python tools/random_techniques.py --events 4 --samples 2000000
"""
import argparse, json, os, sys, time
from multiprocessing import Pool
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RATIO = 1.3
MARGIN = 0.04          # eats need >= 1.3 * (1 + MARGIN)
MINFRAC = 0.02         # every piece >= 2% of the total (740 mass)
MAXFRAC = 22500 / 37000


def split_list(pieces, actor, k, ev):
    """pieces: list of [owner, mass, gen]; canonical order: A by mass desc, then B by mass desc.
    After a split the actor's pieces are listed stayer, launched(s) for each old piece in rank order."""
    out = []
    for p in pieces:
        if p[0] != actor:
            out.append(p)
            continue
        f = 0.5 ** k
        out.append([actor, p[1] * f, p[2]])
        for _ in range(2 ** k - 1):
            out.append([actor, p[1] * f, ev])
    return out


def canon(pieces):
    a = sorted([p for p in pieces if p[0] == 0], key=lambda p: -p[1])
    b = sorted([p for p in pieces if p[0] == 1], key=lambda p: -p[1])
    return a + b


def run_program(prog, x0, mirror, cycles=40, record=False):
    """returns (ok, masses at cycle start, eats log of the last cycle, rho estimate)"""
    na, nb = prog["na"], prog["nb"]
    pieces = [[0, m, -9] for m in x0[:na]] + [[1, m, -9] for m in x0[na:]]
    pieces = canon(pieces)
    hist = []
    last_eats = None
    for c in range(cycles):
        cyc_eats = []
        ev_states = []
        swap = mirror and c % 2 == 1
        for e, (actor, k, eats) in enumerate(prog["events"]):
            actor ^= swap
            gen = c * 100 + e
            sp = split_list(pieces, actor, k, gen)
            n = len(sp)
            eaten, gain, ev_eats = set(), {}, []
            for a_ref, b_ref in eats:
                # refs are (role, index within that player's post-split list)
                ia = _ref(sp, a_ref, swap)
                ib = _ref(sp, b_ref, swap)
                if ia is None or ib is None or ia in eaten or ib in eaten or ib in gain:
                    return (False,)
                pa, pb = sp[ia], sp[ib]
                if pa[0] == pb[0] or not (pa[2] >= gen - 1 or pb[2] >= gen - 1):
                    return (False,)
                ma = pa[1] + gain.get(ia, 0)
                if ma < RATIO * (1 + MARGIN) * pb[1]:
                    return (False,)
                ev_eats.append((pa[0], pb[0], pb[1], ma))
                gain[ia] = gain.get(ia, 0) + pb[1]
                eaten.add(ib)
            pieces = canon([[p[0], p[1] + gain.get(i, 0), p[2]] for i, p in enumerate(sp) if i not in eaten])
            if any(p[1] < MINFRAC or p[1] > MAXFRAC for p in pieces):
                return (False,)
            cyc_eats.append(ev_eats)
            ev_states.append((sorted([p[1] for p in pieces if p[0] == 0], reverse=True),
                              sorted([p[1] for p in pieces if p[0] == 1], reverse=True)))
        # masses at the start of the next cycle, in role order (swap back for mirror)
        sw = mirror and c % 2 == 0
        ma_ = sorted([p[1] for p in pieces if p[0] == (1 if sw else 0)], reverse=True)
        mb_ = sorted([p[1] for p in pieces if p[0] == (0 if sw else 1)], reverse=True)
        if len(ma_) != na or len(mb_) != nb:
            return (False,)
        hist.append(np.array(ma_ + mb_))
        last_eats = cyc_eats
        last_states = ev_states
        start_state = cyc_start if c > 0 else None
        cyc_start = (sorted([p[1] for p in pieces if p[0] == 0], reverse=True),
                     sorted([p[1] for p in pieces if p[0] == 1], reverse=True))
    d = [np.abs(hist[i + 1] - hist[i]).max() for i in range(len(hist) - 1)]
    if d[-1] > 1e-5:
        return (False,)
    # primitive period: the state must not come back (as itself or mirrored) before the cycle ends
    same = lambda s1, s2: len(s1[0]) == len(s2[0]) and len(s1[1]) == len(s2[1]) and \
        np.allclose(s1[0], s2[0], atol=1e-6) and np.allclose(s1[1], s2[1], atol=1e-6)
    s0 = start_state
    for st in last_states[:-1]:
        if same(st, s0) or same((st[1], st[0]), s0):
            return (False,)
    # rho: how fast differences shrank
    early = [x for x in d[:10] if x > 1e-12]
    rho = (d[-1] / early[0]) ** (1 / (len(d) - 1)) if early and d[-1] > 0 else 0.0
    return (True, hist[-1], last_eats, rho)


def _ref(sp, ref, swap):
    role, idx = ref
    owner = role ^ swap
    lst = [i for i, p in enumerate(sp) if p[0] == owner]
    return lst[idx] if idx < len(lst) else None


FORCED = None      # e.g. "AAA": force the actor order (A = player 0), all single splits
P_DOUBLE = 0.12    # probability that a split is a double split
IDENTITY_ONLY = False
MAX_SHARE = 1.0    # keep only techniques where some player holds <= this share at the cycle start


def random_program(rng, E, x0=None):
    """Build a program while tracking masses from x0, so every chosen eat is legal (>= 1.3x) at the
    time it is chosen. Pieces are referred to by (role, index in that player's post-split list)."""
    na, nb = int(rng.integers(1, 5)), int(rng.integers(1, 5))
    if x0 is None:
        x0 = rng.dirichlet(np.ones(na + nb) * 2)
    pieces = canon([[0, m, -9] for m in x0[:na]] + [[1, m, -9] for m in x0[na:]])
    events = []
    actor = int(rng.integers(0, 2))
    for e in range(E):
        trim = False
        if FORCED:
            ch = FORCED[e]
            trim = ch in "tT"                          # trim: the big player (A) splits off-circle, no eats
            actor = 0 if ch in "AatT" else 1
            k = 2 if ch in "abT" else 1
        else:
            if e > 0 and rng.random() < 0.75:
                actor = 1 - actor                      # mostly alternate: A, B, A, B ...
            k = 2 if rng.random() < P_DOUBLE else 1
        sp = split_list(pieces, actor, k, e)
        if sum(1 for p in sp if p[0] == actor) > 16:
            return None
        idx = {o: [i for i, p in enumerate(sp) if p[0] == o] for o in (0, 1)}
        role_of = lambda i: (sp[i][0], idx[sp[i][0]].index(i))
        flying = [i for i, p in enumerate(sp) if p[2] >= e - 1]
        order = list(flying)
        rng.shuffle(order)
        eaten, gain, eats = set(), {}, []
        for f in ([] if trim else order):
            if f in eaten or rng.random() < 0.1:
                continue
            # legal partners: f eats q, or q eats f
            opts = []
            for q in range(len(sp)):
                if sp[q][0] == sp[f][0] or q in eaten:
                    continue
                if q not in gain and sp[f][1] + gain.get(f, 0) >= RATIO * (1 + MARGIN) * sp[q][1]:
                    opts.append((f, q))
                if f not in gain and sp[q][1] + gain.get(q, 0) >= RATIO * (1 + MARGIN) * sp[f][1]:
                    opts.append((q, f))
            if not opts:
                continue
            a_, b_ = opts[int(rng.integers(0, len(opts)))]
            eats.append((role_of(a_), role_of(b_)))
            gain[a_] = gain.get(a_, 0) + sp[b_][1]
            eaten.add(b_)
        if not eats and not trim:
            return None
        pieces = canon([[p[0], p[1] + gain.get(i, 0), p[2]] for i, p in enumerate(sp) if i not in eaten])
        events.append((actor, k, eats))
    counts = [sum(1 for p in pieces if p[0] == 0), sum(1 for p in pieces if p[0] == 1)]
    return dict(na=na, nb=nb, events=events, end_counts=counts, x0=x0)


def worker(args):
    seed, n, E, forced, p_double, identity_only, max_share = args
    global FORCED, P_DOUBLE, IDENTITY_ONLY, MAX_SHARE
    FORCED, P_DOUBLE, IDENTITY_ONLY, MAX_SHARE = forced, p_double, identity_only, max_share
    rng = np.random.default_rng(seed)
    found = []
    for _ in range(n):
        prog = random_program(rng, E)
        if prog is None:
            continue
        ca, cb = prog["end_counts"]
        for mirror in ((False,) if IDENTITY_ONLY else (False, True)):
            want = (prog["nb"], prog["na"]) if mirror else (prog["na"], prog["nb"])
            if (ca, cb) != want:
                continue
            if mirror and prog["events"][0][0] != 0:
                continue
            for t_ in range(3):
                x0 = prog["x0"] if t_ == 0 else rng.dirichlet(np.ones(prog["na"] + prog["nb"]) * 2)
                r = run_program(prog, x0, mirror)
                if r[0] and min(sum(r[1][:prog["na"]]), sum(r[1][prog["na"]:])) > MAX_SHARE:
                    r = (False,)
                if r[0]:
                    found.append((prog, mirror, r[1].tolist(), r[2], r[3]))
                    break
    return found


def to_catalogue(prog, mirror, x, eats, rho):
    na = prog["na"]
    if mirror:   # the last recorded cycle had A and B swapped; relabel eats by role
        eats = [[(1 - eo, 1 - bo, m, em) for eo, bo, m, em in ev] for ev in eats]
    seq = [[a, k] for a, k, _ in prog["events"]]
    used = sum(len(e) for _, _, e in prog["events"])
    launched = sum((2 ** k - 1) * 0 for _, k, _ in prog["events"])
    log = []
    for (a, k, _), ev in zip(prog["events"], eats):
        desc = "; ".join(f"{'AB'[eo]} {em * 37000:,.0f} eats {'AB'[bo]} {m * 37000:,.0f}" for eo, bo, m, em in ev)
        log.append(f"{'AB'[a]} {'single' if k == 1 else 'double'}: {desc}")
    return dict(masses_a=x[:na], masses_b=x[na:], seq=seq, mirror=mirror, rho=float(rho),
                margin=float(min(em - RATIO * m for ev in eats for _, _, m, em in ev)),
                log=log, eats=[[(eo, bo, m, em) for eo, bo, m, em in ev] for ev in eats])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=4)
    ap.add_argument("--samples", type=int, default=1_000_000)
    ap.add_argument("--procs", type=int, default=7)
    ap.add_argument("--actors", default=None, help='force the split order, e.g. "AAA" (lowercase = double split)')
    ap.add_argument("--p-double", type=float, default=0.12)
    ap.add_argument("--identity-only", action="store_true")
    ap.add_argument("--max-share", type=float, default=1.0, help="small player's share of the mass at the cycle start <= this")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    t0 = time.time()
    chunk = a.samples // (a.procs * 8)
    with Pool(a.procs) as pool:
        res = [f for part in pool.map(worker, [(s, chunk, a.events, a.actors, a.p_double, a.identity_only, a.max_share) for s in range(a.procs * 8)]) for f in part]
    uniq = {}
    for prog, mirror, x, eats, rho in res:
        key = (tuple((ev[0], ev[1]) for ev in prog["events"]), mirror,
               tuple(sorted(np.round(np.array(x[:prog["na"]]) * 200).astype(int))),
               tuple(sorted(np.round(np.array(x[prog["na"]:]) * 200).astype(int))))
        if key not in uniq:
            uniq[key] = to_catalogue(prog, mirror, x, eats, rho)
    techs = sorted(uniq.values(), key=lambda t: (len(t["masses_a"]) + len(t["masses_b"]), t["rho"]))
    out = os.path.join(ROOT, "runs", "techniques", f"random_e{a.events}{'_' + a.actors if a.actors else ''}{a.tag}.json")
    json.dump(techs, open(out, "w"), indent=1)
    print(f"{a.samples:,} programs in {time.time() - t0:.0f}s -> {len(res)} hits, {len(uniq)} distinct techniques -> {out}")
    for i, t in enumerate(techs[:15]):
        print(f"\n#{i} {'MIRROR' if t['mirror'] else 'IDENTITY'} rho={t['rho']:.2f} margin={t['margin']:.3f} "
              f"start A " + " + ".join(f"{m * 37000:,.0f}" for m in t["masses_a"]) + " | B " + " + ".join(f"{m * 37000:,.0f}" for m in t["masses_b"]))
        for l in t["log"]:
            print("    " + l)
