"""Chain the solved 2+3 turn: at every split, re-read the real piece positions, rebuild the turn
frame (actor big -> partner smallest), and replay the solved turn's cursors in that frame.

    python tools/run_pattern23.py [--turns 40] [--record name]
"""
import argparse, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from splitrun.sim import Sim, EV_EATER_OWN, EV_EATEN_OWN, EV_MASS  # noqa
from splitrun.pattern23 import start_rows, TOTAL  # noqa
from splitrun.replay import Recorder, write_index  # noqa
from tools.solve_turn23 import layout_of, frame_coords  # noqa


def pieces(sim, p):
    c = sorted([x for x in sim.cells(0) if x[1] == p], key=lambda x: -x[4])
    return np.array([[x[2], x[3]] for x in c])


def run(v, turns, record=None, desc=""):
    sim = Sim(1)
    L = layout_of(v)
    sim.load(0, start_rows([L[1, 0], L[1, 1], L[2, 0], L[2, 1], L[3, 0], L[3, 1], L[4, 0], L[4, 1]]), seed=1)
    rec = Recorder(sim, 0, record or "pattern23", desc) if record else None
    delay = int(round(v[13]))
    init = L
    actor, flip = 0, False
    clean_turns, errs, log = 0, [], []
    for k in range(turns):
        a, q = pieces(sim, actor), pieces(sim, 1 - actor)
        if len(a) != 2 or len(q) != 3:
            log.append(f"turn {k}: pieces {len(a)}+{len(q)} -> pattern broken")
            break
        pts = np.concatenate([a, q])
        # choose the handedness that matches the solved layout best, then measure the layout error
        e0 = [np.abs(frame_coords(pts, pts[0], pts[4], f) - init).mean() for f in (False, True)]
        flip = bool(np.argmin(e0))
        errs.append(min(e0))
        o = pts[0]
        u = (pts[4] - o) / np.linalg.norm(pts[4] - o)
        w = np.array([-u[1], u[0]]) * (-1 if flip else 1)
        to_world = lambda xy: o + xy[0] * u + xy[1] * w
        good = bad = 0
        for t in range(delay):
            mouse = np.zeros((1, 2, 2)); splits = np.zeros((1, 2), np.int64)
            if t == 0:
                mouse[0, actor] = to_world(v[7:9]); splits[0, actor] = 1
            else:
                mouse[0, actor] = to_world(v[9:11])
            mouse[0, 1 - actor] = to_world(v[11:13])
            sim.step(mouse, splits)
            if rec:
                rec.after_step(np.trunc(mouse[0]), splits[0])
            for ev in sim.events(0):
                if ev[EV_EATER_OWN] != ev[EV_EATEN_OWN] and abs(ev[EV_MASS] / TOTAL - 1 / 12) < 0.01:
                    good += 1
                else:
                    bad += 1
        ok = good == 2 and bad == 0
        clean_turns += ok
        log.append(f"turn {k}: {'clean' if ok else 'NOT clean'} (intended eats {good}, other {bad}), "
                   f"layout error at split {errs[-1]:.1f}")
        if not ok:
            break
        actor = 1 - actor
    return clean_turns, errs, log, rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=40)
    ap.add_argument("--record", default=None)
    ap.add_argument("--which", type=int, default=0)
    ap.add_argument("--file", default="turn_elite.npy")
    a = ap.parse_args()
    v = np.load(os.path.join(ROOT, "runs", "stage2_pattern23", a.file))[a.which]
    n, errs, log, rec = run(v, a.turns, a.record)
    print("\n".join(log))
    print(f"clean turns: {n}")
    if rec:
        rec.save(os.path.join(ROOT, "runs", "stage2_pattern23", a.record + ".json"))
        write_index(os.path.join(ROOT, "runs"))
