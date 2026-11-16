"""Solve ONE turn of the 2+3 splitrun for a mirror-periodic geometry.

Variables (turn frame of the actor X: X-big at origin, Y-small on the +x axis):
  layout   xs_x xs_y  yb_x yb_y  ym_x ym_y  ys_x          (ys_y = 0)
  split    px py                  X's cursor on the split tick
  moves    ax ay  bx by           cursors of X and Y for the rest of the turn
  delay    ticks until the next split (Y's turn)
Cost = 10 per missing intended eat + 10 per unintended eat (incl. own merges)
       + layout mismatch at the next split (the new actor Y's frame, roles mirrored) / 100.
"""
import argparse, math, os, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from splitrun.sim import Sim, X, Y, SZ, EV_EATER_OWN, EV_EATEN_OWN, EV_MASS  # noqa
from splitrun.pattern23 import start_rows, TOTAL  # noqa

NAMES = ["xs_x", "xs_y", "yb_x", "yb_y", "ym_x", "ym_y", "ys_x", "px", "py", "ax", "ay", "bx", "by", "delay"]
LO = np.array([-2600, -2600, -2600, -2600, -3000, -3000, 900, -3000, -3000, -3000, -3000, -3000, -3000, 15])
HI = np.array([2600, 2600, 2600, 2600, 3000, 3000, 1500, 3000, 3000, 3000, 3000, 3000, 3000, 70])


def layout_of(v):
    return np.array([[0, 0], [v[0], v[1]], [v[2], v[3]], [v[4], v[5]], [v[6], 0]], float)


def frame_coords(pts, origin, target, flip):
    u = target - origin
    u = u / max(np.linalg.norm(u), 1e-9)
    w = np.array([-u[1], u[0]]) * (-1 if flip else 1)
    r = pts - origin
    return np.stack([r @ u, r @ w], 1)


def simulate(V, record=None):
    E = len(V)
    sim = Sim(E)
    for e in range(E):
        L = layout_of(V[e])
        sim.load(e, start_rows([L[1, 0], L[1, 1], L[2, 0], L[2, 1], L[3, 0], L[3, 1], L[4, 0], L[4, 1]]), seed=e + 1)
    delay = np.round(V[:, 13]).astype(int)
    good = np.zeros(E); bad = np.zeros(E)
    end_state = [None] * E
    rec = None
    if record is not None:
        from splitrun.replay import Recorder
        rec = Recorder(sim, record, "turn23")
    for t in range(int(HI[13]) + 1):
        mouse = np.zeros((E, 2, 2)); splits = np.zeros((E, 2), np.int64)
        if t == 0:
            mouse[:, 0] = V[:, 7:9]
            splits[:, 0] = 1
        else:
            mouse[:, 0] = V[:, 9:11]
        mouse[:, 1] = V[:, 11:13]
        sim.step(mouse, splits)
        if rec is not None:
            rec.after_step(np.trunc(mouse[record]), splits[record])
        for e in np.nonzero(sim.evn)[0]:
            for ev in sim.ev[e, :sim.evn[e]]:
                if ev[EV_EATER_OWN] != ev[EV_EATEN_OWN] and abs(ev[EV_MASS] / TOTAL - 1 / 12) < 0.01:
                    good[e] += 1
                else:
                    bad[e] += 1
        for e in np.nonzero(delay == t + 1)[0]:
            end_state[e] = sim.cells(e)
            sim.active[e] = 0
    cost = np.zeros(E)
    init = np.array([layout_of(v) for v in V])
    for e in range(E):
        c = end_state[e]
        a = sorted([x for x in c if x[1] == 1], key=lambda x: -x[4])   # new actor Y
        q = sorted([x for x in c if x[1] == 0], key=lambda x: -x[4])   # new partner X
        miss = 10 * abs(2 - good[e]) + 10 * bad[e]
        if len(a) != 2 or len(q) != 3:
            cost[e] = miss + 50
            continue
        pts = np.array([[a[0][2], a[0][3]], [a[1][2], a[1][3]], [q[0][2], q[0][3]], [q[1][2], q[1][3]], [q[2][2], q[2][3]]])
        best = min(np.abs(frame_coords(pts, pts[0], pts[4], f) - init[e]).mean() for f in (False, True))
        cost[e] = miss + best / 100
    return cost, good, bad, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=8192)
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--max-delay", type=int, default=70)
    ap.add_argument("--out", default="turn_elite.npy")
    a = ap.parse_args()
    HI[13] = a.max_delay
    rng = np.random.default_rng(2)
    t0 = time.time()
    V = rng.uniform(LO, HI, (a.pop, len(LO)))
    cost, good, bad, _ = simulate(V)
    print(f"gen 0: best cost {cost.min():.2f}; candidates with both intended eats and nothing else: {((good == 2) & (bad == 0)).sum()}")
    elite = V[np.argsort(cost)[:128]]; ecost = np.sort(cost)[:128]
    for g in range(1, a.gens + 1):
        scale = 0.08 * (0.93 ** g)
        kids = elite[rng.integers(0, 128, a.pop)] + rng.normal(0, 1, (a.pop, len(LO))) * (HI - LO) * scale
        kids = np.clip(kids, LO, HI)
        c, gd, bd, _ = simulate(kids)
        allV = np.concatenate([elite, kids]); allc = np.concatenate([ecost, c])
        o = np.argsort(allc)[:128]
        elite, ecost = allV[o], allc[o]
        if g % 5 == 0 or g == 1:
            print(f"gen {g}: best cost {ecost[0]:.3f} (layout mismatch {100 * (ecost[0] % 10):.0f} units), "
                  f"clean candidates this gen {((gd == 2) & (bd == 0)).sum()}  ({time.time() - t0:.0f}s)")
    out = os.path.join(ROOT, "runs", "stage2_pattern23")
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, a.out), elite)
    print("best:", dict(zip(NAMES, np.round(elite[0], 0))))


if __name__ == "__main__":
    main()
