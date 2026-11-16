"""Splitrun detector.

The user's definition: two players move in circles around each other by repeatedly eating each
other's backpieces, and for this project the run must be CLEAN and PERIODIC -- every split has a
purpose and the relative positions of the balls repeat (nearly) every cycle, for many cycles,
without the pair travelling.  `splitrun_verdict(replay)` is the test.
"""
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------- periodicity
# The user's definition of a proper splitrun: CLEAN and PERIODIC -- the relative positions of the
# balls repeat (nearly) every cycle, every split has a purpose.  The layout is compared through a
# descriptor that ignores where the pair is and which way it faces: every piece's mass plus all
# pairwise distances, pieces ordered by (player, mass).  "Half-period with swap" means the layout
# repeats with A and B exchanging roles (as in a mirror turn).

def _descriptor(frame, swap):
    """pieces ordered by (player, mass); returns (counts, masses, positions, tie groups)"""
    import itertools
    cells = sorted(frame, key=lambda c: ((c[1] ^ swap), -c[4]))
    counts = (sum(1 for c in cells if (c[1] ^ swap) == 0), sum(1 for c in cells if (c[1] ^ swap) == 1))
    P = np.array([[c[2], c[3]] for c in cells])
    m = np.array([c[4] ** 2 / 100 for c in cells])
    # equal-mass pieces of the same player (e.g. the two halves of a split) have no natural order
    groups, i = [], 0
    while i < len(cells):
        j = i + 1
        while j < len(cells) and (cells[j][1] ^ swap) == (cells[i][1] ^ swap) and abs(m[j] - m[i]) < 0.01 * m[i]:
            j += 1
        groups.append(list(range(i, j)))
        i = j
    # try every order within small tie groups (a split's two halves); big groups keep one order
    perms = [list(itertools.permutations(g)) if len(g) <= 3 else [tuple(g)] for g in groups]
    orders = [list(itertools.chain(*combo)) for combo in itertools.islice(itertools.product(*perms), 24)]
    return counts, m, P, orders


def _layout_error(a, b):
    """mean |pairwise-distance difference| between two layouts, best over tie orderings of b"""
    iu = np.triu_indices(len(a[1]), 1)
    Da = np.linalg.norm(a[2][:, None] - a[2][None], axis=2)[iu]
    best = np.inf
    for o in b[3]:
        Pb = b[2][o]
        Db = np.linalg.norm(Pb[:, None] - Pb[None], axis=2)[iu]
        best = min(best, np.abs(Da - Db).mean())
    return best


@dataclass
class PeriodReport:
    period: int = 0            # ticks
    swap: bool = False         # layout repeats with A and B exchanging roles
    median_error: float = np.inf   # units, mean |distance difference| between layouts one period apart
    periodic_ticks: int = 0    # longest stretch of ticks whose layout repeats within tolerance
    periodic_cycles: float = 0.0


def periodicity(frames, events=(), min_period=20, max_period=300, tol=60.0, mass_tol=0.05, stride=5):
    """frames: replay frames (lists of [id, owner, x, y, size, flags]); events: replay events
    ([tick, eater_owner, eaten_owner, mass, kind]).  A period only counts if at least one
    cross-player eat happens inside it -- a layout that merely sits still is not a cycle."""
    T = len(frames)
    best = PeriodReport()
    eat_ticks = np.zeros(T + 1, np.int64)
    for ev in events:
        if ev[1] != ev[2] and ev[0] <= T:
            eat_ticks[ev[0]] += 1
    cum = np.concatenate([[0], np.cumsum(eat_ticks)])
    descs = {s: [_descriptor(f, s) for f in frames] for s in (0, 1)}
    for swap in (0, 1):
        for L in range(min_period, min(max_period, T - 1) + 1):
            ts = np.arange(0, T - L, stride)
            ok = np.zeros(len(ts), bool)
            errs = np.full(len(ts), np.inf)
            for k, t in enumerate(ts):
                if cum[min(t + L + 1, T)] - cum[t + 1] == 0:
                    continue
                a = descs[0][t]
                # "nearly the same": an eat may land a few ticks earlier or later than last cycle
                for dt in (0, -1, 1, -2, 2, -3, 3):
                    u = t + L + dt
                    if u <= t or u >= T:
                        continue
                    b = descs[swap][u]
                    if a[0] != b[0] or len(a[1]) < 2 or np.abs(a[1] - b[1]).mean() > mass_tol * a[1].mean():
                        continue
                    errs[k] = min(errs[k], _layout_error(a, b))
                    if errs[k] < tol:
                        break
                ok[k] = errs[k] < tol
                if k >= 12 and not ok[k - 12:k + 1].any():
                    break          # nothing periodic for a whole minute-scale window: next L
            if not ok.any():
                continue
            run = longest = 0
            for v in ok:
                run = run + 1 if v else 0
                longest = max(longest, run)
            med = float(np.median(errs[ok]))
            longest *= stride
            if (longest, -med) > (best.periodic_ticks, -best.median_error):
                best = PeriodReport(L, bool(swap), med, longest, (longest + L) / L)
    return best


def splitrun_verdict(replay, min_cycles=10, max_drift=3000.0):
    """The Stage-2 test of the user's definition, on a replay dict.
    Proper splitrun  <=>  the layout repeats (clean + periodic) for >= min_cycles cycles, mass
    crosses in BOTH directions, the pair stays put (centre drift <= max_drift) and both survive.
    Also reports how far the pair rotates (circling)."""
    frames, events = replay["frames"], replay.get("events", [])
    per = periodicity(frames, events)
    com = []
    axis = []
    for f in frames:
        P = np.array([[c[2], c[3]] for c in f]); M = np.array([c[4] ** 2 for c in f]); O = np.array([c[1] for c in f])
        com.append((P * M[:, None]).sum(0) / M.sum())
        if (O == 0).any() and (O == 1).any():
            ca = (P[O == 0] * M[O == 0, None]).sum(0) / M[O == 0].sum()
            cb = (P[O == 1] * M[O == 1, None]).sum(0) / M[O == 1].sum()
            axis.append(cb - ca)
    com = np.array(com)
    drift = float(np.linalg.norm(com - com[0], axis=1).max())
    ax = np.array(axis)
    ang = np.unwrap(np.arctan2(ax[:, 1], ax[:, 0])) if len(ax) > 1 else np.zeros(1)
    rotations = float(abs(ang[-1] - ang[0]) / (2 * np.pi))
    both_ways = any(e[1] == 0 and e[2] == 1 for e in events) and any(e[1] == 1 and e[2] == 0 for e in events)
    alive = all(any(c[1] == p for c in frames[-1]) for p in (0, 1))
    ok = per.periodic_cycles >= min_cycles and both_ways and drift <= max_drift and alive
    return {"splitrun": ok, "period_ticks": per.period, "roles_swap": per.swap,
            "periodic_cycles": round(per.periodic_cycles, 1), "layout_error": round(per.median_error, 1),
            "rotations": round(rotations, 2), "centre_drift": round(drift), "mass_both_ways": both_ways, "both_alive": alive}


def phase_periodicity(replay, n_splits, mirror, tol=60.0):
    """Periodicity measured at matching phases instead of a fixed number of ticks: cycle starts
    are taken from the split presses themselves (every `n_splits` presses; for mirror techniques
    every half-cycle with the players' roles swapped), and consecutive cycle-start layouts are
    compared. A splitrun whose cycles take 25 ticks one time and 27 the next still counts, as long
    as the layout repeats.
    Returns dict(cycles, repeating_cycles, longest_repeating_run, median_error, median_cycle_ticks)."""
    frames, inputs = replay["frames"], replay.get("inputs", [])
    presses = []
    for t, inp in enumerate(inputs):
        for p in range(2):
            n = inp[p][2] if len(inp[p]) > 2 else 0
            if n:
                presses.append(t)
    starts = presses[::n_splits]
    errs, lens = [], []
    for k in range(1, len(starts)):
        a = _descriptor(frames[starts[k - 1]], 0)
        b = _descriptor(frames[starts[k]], 1 if mirror else 0)
        lens.append(starts[k] - starts[k - 1])
        if a[0] != b[0] or np.abs(a[1] - b[1]).mean() > 0.05 * a[1].mean():
            errs.append(np.inf)
        else:
            errs.append(_layout_error(a, b))
    ok = [e < tol for e in errs]
    run = best = 0
    for v in ok:
        run = run + 1 if v else 0
        best = max(best, run)
    fin = [e for e in errs if np.isfinite(e)]
    return dict(cycles=len(errs), repeating_cycles=int(sum(ok)), longest_repeating_run=best,
                median_error=float(np.median(fin)) if fin else float("inf"),
                median_cycle_ticks=float(np.median(lens)) if lens else 0.0)
