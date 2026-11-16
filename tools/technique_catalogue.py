"""Catalogue of clean periodic splitrun techniques (mass bookkeeping only; geometry comes later).

A technique is a cycle of split EVENTS. At each event one player presses split (single or
double), which halves (or quarters) EVERY one of its pieces, and then some eats happen. Rules:

  * every eat involves a flying piece: one launched by this event or by the previous one
    (the partner can split right back into a backpiece while the first pieces are still flying)
  * eater mass >= 1.3 x eaten mass; a launched piece may eat up to 2 pieces
  * every event has at least one eat (no split without a purpose)
  * no merges (re-merging takes 20-30 s)

The cycle is periodic when every piece's mass is back to its starting value (up to relabelling),
either for the same player (IDENTITY) or with A and B swapped (MIRROR).  Masses are linear in the
starting masses, so each candidate is a linear program: maximise the eat margin delta in
eater >= 1.3 eaten + delta (total mass 1).

Mass STABILITY: if the starting masses are slightly off (as they will be in any learned/RL run),
the same cycle maps the error e -> A e.  rho = spectral radius of A on mass-preserving errors.
rho < 1: errors shrink every cycle, the technique pulls itself back to the exact pattern.
rho = 1: errors persist (neutral).  rho > 1: errors grow, the technique drifts apart.
"""
import itertools
import sys
import time

import numpy as np
from scipy.optimize import linprog

RATIO = 1.3
MINMASS = 0.015


class Pc:
    __slots__ = ("owner", "coef", "gen", "name")

    def __init__(self, owner, coef, gen, name):
        self.owner, self.coef, self.gen, self.name = owner, coef, gen, name


def split(pieces, actor, k, ev):
    out = []
    for p in pieces:
        if p.owner != actor:
            out.append(p)
            continue
        f = 0.5 ** k
        out.append(Pc(actor, p.coef * f, p.gen, p.name))                    # stays: keeps identity
        for j in range(2 ** k - 1):
            out.append(Pc(actor, p.coef * f, ev, f"{'AB'[actor]}{ev}.{len(out)}"))
    return out


def eat_sets(pieces, ev, max_eats=3):
    """all sets of eats for this event: list of (eater, [eaten...])"""
    flying = [i for i, p in enumerate(pieces) if p.gen >= ev - 1 and p.gen >= 0]
    n = len(pieces)
    # candidate (eater, eaten) pairs
    pairs = [(a, b) for a in range(n) for b in range(n)
             if pieces[a].owner != pieces[b].owner and (a in flying or b in flying)]
    results = []

    def rec(i, chosen, eaten, eaters, count):
        if i == len(pairs):
            if chosen:
                results.append(list(chosen))
            return
        rec(i + 1, chosen, eaten, eaters, count)
        a, b = pairs[i]
        if count >= max_eats or b in eaten or a in eaten or b in eaters:
            return
        if eaters.get(a, 0) >= (2 if pieces[a].gen == ev else 1):
            return
        chosen.append((a, b))
        eaten.add(b)
        eaters[a] = eaters.get(a, 0) + 1
        rec(i + 1, chosen, eaten, eaters, count + 1)
        chosen.pop()
        eaten.discard(b)
        eaters[a] -= 1
        if eaters[a] == 0:
            del eaters[a]

    rec(0, [], set(), {}, 0)
    return results


def apply(pieces, eats):
    gain, eaten = {}, set()
    rows = []
    evs = []
    # an eater that eats twice: the second eat needs the mass after the first
    for a, b in eats:
        cur = pieces[a].coef + gain.get(a, 0)
        rows.append((cur, pieces[b].coef))
        evs.append((pieces[a].owner, pieces[b].owner, pieces[b].coef, cur))
        gain[a] = gain.get(a, 0) + pieces[b].coef
        eaten.add(b)
    out = [Pc(p.owner, p.coef + gain.get(i, 0), p.gen, p.name) for i, p in enumerate(pieces) if i not in eaten]
    return out, rows, evs


def solve(n, owners, rows, final, mirror):
    tgt = [(1 - o) if mirror else o for o in owners]
    fa = [p for p in final if p.owner == 0]
    fb = [p for p in final if p.owner == 1]
    ta = [i for i in range(n) if tgt[i] == 0]
    tb = [i for i in range(n) if tgt[i] == 1]
    if len(fa) != len(ta) or len(fb) != len(tb):
        return None
    F = [p.coef for p in fa + fb]
    best = None
    for pa in itertools.permutations(ta):
        for pb in itertools.permutations(tb):
            perm = list(pa) + list(pb)
            A = np.array(F) - np.eye(n)[perm]          # final_j - x_perm(j) = 0
            # quick rejection: need a non-trivial null space
            sv = np.linalg.svd(A, compute_uv=False)
            if sv[-1] > 1e-9:
                continue
            A_eq = np.hstack([A, np.zeros((n, 1))])
            A_eq = np.vstack([A_eq, np.append(np.ones(n), 0)])
            b_eq = np.append(np.zeros(n), 1)
            A_ub = [np.append(-(ea - RATIO * eb), 1.0) for ea, eb in rows]
            res = linprog(np.append(np.zeros(n), -1.0), A_ub=A_ub, b_ub=np.zeros(len(A_ub)), A_eq=A_eq, b_eq=b_eq,
                          bounds=[(MINMASS, 1)] * n + [(None, 1)], method="highs")
            if res.status == 0 and res.x[-1] > 1e-4 and (best is None or res.x[-1] > best[0]):
                x = res.x[:n]
                # stability: error map e -> P^-1 F e
                Pm = np.eye(n)[perm]
                Amap = np.linalg.solve(Pm, np.array(F))
                ev = np.linalg.eigvals(Amap)
                ev = sorted(ev, key=lambda z: -abs(z))
                # drop the eigenvalue 1 that belongs to the periodic solution itself
                k = int(np.argmin([abs(z - 1) for z in ev]))
                rest = [z for i, z in enumerate(ev) if i != k]
                rho = max([abs(z) for z in rest], default=0.0)
                best = (res.x[-1], x, rho)
    return best


def catalogue(max_start=3, max_events=3, max_pieces=8):
    found = []
    for na in range(1, max_start + 1):
        for nb in range(1, max_start + 1):
            n = na + nb
            owners = [0] * na + [1] * nb
            start = [Pc(o, np.eye(n)[i], -5, f"{'AB'[o]}{i}") for i, o in enumerate(owners)]

            def rec(pieces, ev, rows, seq, log, evlist):
                if ev > 0:
                    for mirror in (True, False):
                        if not mirror and ev < 2:
                            continue
                        if mirror and seq[0][0] != 0:
                            continue
                        sol = solve(n, owners, rows, pieces, mirror)
                        if sol:
                            found.append(dict(margin=sol[0], x=sol[1], rho=sol[2], na=na, nb=nb, mirror=mirror,
                                              seq=list(seq), log=list(log),
                                              eats=[[(a, b, float(c @ sol[1]), float(ec @ sol[1])) for a, b, c, ec in e] for e in evlist]))
                if ev == max_events:
                    return
                for actor in (0, 1):
                    if ev == 0 and actor == 1:
                        continue
                    for k in (1, 2):
                        sp = split(pieces, actor, k, ev)
                        if len(sp) > max_pieces:
                            continue
                        for eats in eat_sets(sp, ev):
                            after, r, evs = apply(sp, eats)
                            desc = ", ".join(f"{sp[a].name}->eats {sp[b].name}" for a, b in eats)
                            rec(after, ev + 1, rows + r, seq + [(actor, k)],
                                log + [f"{'AB'[actor]} {'single' if k == 1 else 'double'}: {desc}"], evlist + [evs])
            rec(start, 0, [], [], [], [])
    return found


def signature(t):
    xa = tuple(sorted(np.round(t["x"][:t["na"]] * 96).astype(int)))
    xb = tuple(sorted(np.round(t["x"][t["na"]:] * 96).astype(int)))
    return (xa, xb, tuple(t["seq"]), t["mirror"])


if __name__ == "__main__":
    import json, os
    ms = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    me = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    mp = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    t0 = time.time()
    res = catalogue(ms, me, mp)
    uniq = {}
    for t in res:
        k = signature(t)
        if k not in uniq or t["margin"] > uniq[k]["margin"]:
            uniq[k] = t
    print(f"{len(res)} raw solutions, {len(uniq)} distinct techniques ({time.time() - t0:.0f}s)")
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "techniques")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"catalogue_s{ms}_e{me}_p{mp}.json"), "w") as fh:
        json.dump([dict(masses_a=[float(v) for v in t["x"][:t["na"]]], masses_b=[float(v) for v in t["x"][t["na"]:]],
                        seq=[list(e) for e in t["seq"]], mirror=t["mirror"], rho=float(t["rho"]), margin=float(t["margin"]),
                        log=t["log"], eats=t["eats"]) for t in sorted(uniq.values(), key=lambda t: (len(t["seq"]), t["rho"]))], fh, indent=1)
    for t in sorted(uniq.values(), key=lambda t: (len(t["seq"]), t["rho"], -t["margin"])):
        m = lambda v: f"{v * 37000:,.0f}"
        print(f"\n[{'MIRROR' if t['mirror'] else 'IDENTITY'}] events={len(t['seq'])} stability rho={t['rho']:.2f} margin={t['margin']:.3f}")
        print("   start A: " + " + ".join(m(v) for v in t["x"][:t["na"]]) + "   |  B: " + " + ".join(m(v) for v in t["x"][t["na"]:]))
        for l in t["log"]:
            print("     " + l)
