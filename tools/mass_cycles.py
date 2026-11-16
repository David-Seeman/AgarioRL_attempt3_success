"""Which clean, periodic splitrun patterns are possible at all? (mass bookkeeping only)

Ignores geometry. A pattern is a short program of turns. In each turn the acting player
presses split (single or double), which halves EVERY one of its pieces; then some eats happen.
Every eat involves a freshly launched piece:

    backpiece eat   a launched piece of the actor eats a piece of the partner
    fed             a partner piece eats a launched piece of the actor ("launch into getting eaten")

An eat needs eater mass >= 1.3 x eaten mass (imsolo: size >= 1.14 x). Merges are not allowed
(pieces here need 20-30 s to re-merge, far longer than a turn).

The pattern is periodic when, after the program, every piece's mass is back where it started,
up to relabelling the pieces, optionally with A and B swapped (the "mirror" turn: A acts, and
the result is B's starting position, so B can then act the same way).

All masses are linear in the starting masses, so for each program "is there a periodic version
and how much eat-margin does it have?" is a linear program.  We maximise the smallest margin
delta in  m_eater >= 1.3 m_eaten + delta  (total mass = 1).
"""
import itertools
import sys

import numpy as np
from scipy.optimize import linprog

RATIO = 1.3
MINMASS = 0.02      # every piece at least 2% of the total


class Piece:
    def __init__(self, owner, coef, fresh=False, name=""):
        self.owner, self.coef, self.fresh, self.name = owner, coef, fresh, name


def split_pieces(pieces, actor, k):
    out = []
    for p in pieces:
        if p.owner != actor:
            out.append(Piece(p.owner, p.coef, False, p.name))
            continue
        f = 0.5 ** k
        out.append(Piece(actor, p.coef * f, False, p.name + "b"))          # stays behind
        for j in range(2 ** k - 1):
            out.append(Piece(actor, p.coef * f, True, p.name + f"L{j}"))    # launched
    return out


def eat_options(pieces, actor):
    """yield (list of eats [(eater_idx, eaten_idx)], ineq rows) -- each piece in at most one eat,
    every launched piece used (every split serves a purpose)"""
    fresh = [i for i, p in enumerate(pieces) if p.fresh]
    other = [i for i, p in enumerate(pieces) if p.owner != actor]
    # for each launched piece: idle is NOT allowed; it either eats an `other` piece or is eaten by one
    choices = []
    for f in fresh:
        c = [("eat", f, o) for o in other] + [("fed", o, f) for o in other]
        choices.append(c)
    for combo in itertools.product(*choices):
        used = set()
        ok = True
        eats = []
        for kind, a, b in combo:
            if a in used or b in used:
                ok = False
                break
            used.add(a)
            used.add(b)
            eats.append((a, b))
        if ok:
            yield eats


def apply_eats(pieces, eats):
    eaten = {b for _, b in eats}
    gain = {}
    for a, b in eats:
        gain[a] = gain.get(a, 0) + pieces[b].coef
    out = []
    for i, p in enumerate(pieces):
        if i in eaten:
            continue
        out.append(Piece(p.owner, p.coef + gain.get(i, 0), False, p.name))
    return out


def solve(n, ineqs, final, init_owner_counts, swap):
    """find x (starting masses) with the final pieces equal to the (swapped) start, max margin"""
    fa = [p for p in final if p.owner == 0]
    fb = [p for p in final if p.owner == 1]
    ta = [i for i in range(n) if init_owner_counts[i] == (1 if swap else 0)]
    tb = [i for i in range(n) if init_owner_counts[i] == (0 if swap else 1)]
    if len(fa) != len(ta) or len(fb) != len(tb):
        return None
    best = None
    for pa in itertools.permutations(ta):
        for pb in itertools.permutations(tb):
            A_eq, b_eq = [], []
            for p, t in list(zip(fa, pa)) + list(zip(fb, pb)):
                row = p.coef.copy()
                row[t] -= 1
                A_eq.append(np.append(row, 0.0))
                b_eq.append(0.0)
            A_eq.append(np.append(np.ones(n), 0.0))
            b_eq.append(1.0)
            # maximise delta:  -(eater - 1.3 eaten) + delta <= 0 ;  x_i >= MINMASS
            A_ub = [np.append(-(ea - RATIO * eb), 1.0) for ea, eb in ineqs]
            b_ub = [0.0] * len(A_ub)
            res = linprog(np.append(np.zeros(n), -1.0), A_ub=A_ub or None, b_ub=b_ub or None,
                          A_eq=A_eq, b_eq=b_eq, bounds=[(MINMASS, 1)] * n + [(None, 1)], method="highs")
            if res.status == 0 and res.x[-1] > 1e-6 and (best is None or res.x[-1] > best[0]):
                best = (res.x[-1], res.x[:n], (pa, pb))
    return best


def programs(n_a, n_b, max_turns):
    """turn sequences: tuples of (actor, split_k)"""
    for L in range(1, max_turns + 1):
        for seq in itertools.product([(0, 1), (0, 2), (1, 1), (1, 2)], repeat=L):
            if seq[0][0] != 0:
                continue          # A acts first w.l.o.g.
            yield seq


def search(max_pieces=3, max_turns=2, max_total_pieces=8):
    found = []
    for n_a in range(1, max_pieces + 1):
        for n_b in range(1, max_pieces + 1):
            n = n_a + n_b
            owners = [0] * n_a + [1] * n_b
            for seq in programs(n_a, n_b, max_turns):
                start = [Piece(o, np.eye(n)[i], False, ("a" if o == 0 else "b") + str(i)) for i, o in enumerate(owners)]

                def rec(pieces, t, ineqs, log):
                    if t == len(seq):
                        for swap in (True, False):
                            if not swap and len(seq) % 2 == 1 and False:
                                continue
                            sol = solve(n, ineqs, pieces, owners, swap)
                            if sol:
                                found.append((sol[0], n_a, n_b, seq, swap, list(log), sol[1]))
                        return
                    actor, k = seq[t]
                    sp = split_pieces(pieces, actor, k)
                    if len(sp) > max_total_pieces:
                        return
                    for eats in eat_options(sp, actor):
                        rows = [(sp[a].coef, sp[b].coef) for a, b in eats]
                        desc = [f"{sp[a].name}({'AB'[sp[a].owner]}) eats {sp[b].name}({'AB'[sp[b].owner]})" for a, b in eats]
                        rec(apply_eats(sp, eats), t + 1, ineqs + rows,
                            log + [f"{'AB'[actor]} {'single' if k == 1 else 'double'} split: " + "; ".join(desc)])
                rec(start, 0, [], [])
    found.sort(key=lambda z: -z[0])
    return found


if __name__ == "__main__":
    mp = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    mt = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    res = search(mp, mt)
    print(f"{len(res)} periodic patterns found (pieces per player <= {mp}, turns per period <= {mt})")
    seen = set()
    for margin, na, nb, seq, swap, log, x in res[:40]:
        key = (na, nb, tuple(log))
        if key in seen:
            continue
        seen.add(key)
        print(f"\nmargin {margin:.3f} | start A={na} pieces, B={nb} pieces | returns {'MIRRORED (A<->B)' if swap else 'to itself'}")
        print("  start masses (fraction of total): A " + ", ".join(f"{v:.3f}" for v in x[:na]) + " | B " + ", ".join(f"{v:.3f}" for v in x[na:]))
        for l in log:
            print("   ", l)
