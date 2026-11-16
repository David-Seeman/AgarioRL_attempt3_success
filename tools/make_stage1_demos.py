"""Stage-1 visual checks: scripted scenarios rendered for the viewer.

    python tools/make_stage1_demos.py      -> runs/stage1/*.json + runs/index.json
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from splitrun.sim import Sim, CID, OWN, X, Y, SZ, BIRTH  # noqa: E402
from splitrun.replay import Recorder, server_trace_to_replay, write_index  # noqa: E402
import tests.test_fidelity as tf  # noqa: E402

OUT = os.path.join(ROOT, "runs", "stage1")
M0 = 18500.0
R0 = float(np.sqrt(M0 * 100))   # "size" (= radius) of an 18.5k cell


def cells_of(sim, e, p):
    """[(id, x, y, size, birth)] of player p, oldest first (server ownedCells order)"""
    return [(int(sim.ci[e, s, CID]), sim.cf[e, s, X], sim.cf[e, s, Y], sim.cf[e, s, SZ], int(sim.ci[e, s, BIRTH]))
            for s in sim.oc[e, p, :sim.ocn[e, p]]]


def centroid(sim, e, p):
    c = cells_of(sim, e, p)
    if not c:
        return np.zeros(2)
    m = np.array([k[3] ** 2 for k in c])
    return np.array([(m * [k[1] for k in c]).sum(), (m * [k[2] for k in c]).sum()]) / m.sum()


def run(name, desc, cells, controller, ticks, seed=1):
    sim = Sim(1)
    sim.load(0, cells, seed=seed)
    rec = Recorder(sim, 0, name, desc)
    for t in range(ticks):
        mouse, splits = controller(sim, t)
        mouse = np.asarray(mouse, float).reshape(1, 2, 2)
        splits = np.asarray(splits, np.int64).reshape(1, 2)
        sim.step(mouse, splits)
        rec.after_step(np.trunc(mouse[0]), splits[0])
        if sim.ocn[0, 0] == 0 or sim.ocn[0, 1] == 0:
            break
    rec.save(os.path.join(OUT, name + ".json"))
    ev = rec.data["events"]
    print(f"{name:22s} ticks={len(rec.data['frames']) - 1:5d}  eats={sum(1 for x in ev if x[4] == 'eat')}"
          f"  merges={sum(1 for x in ev if x[4] == 'merge')}  final masses A={sum(c[3]**2 for c in cells_of(sim,0,0))/100:,.0f}"
          f" B={sum(c[3]**2 for c in cells_of(sim,0,1))/100:,.0f}")
    return rec.data


def main():
    # 1. single split: A splits once to the right, then holds the cursor still. Pieces
    #    of 9,250 mass should re-merge after round(0.75 * 962) = 721 ticks (~28.8 s).
    A0, B0 = (-3000.0, 0.0), (3000.0, 0.0)
    hold = (A0[0] + 600, 0.0)
    run("01_single_split",
        "Player A (18.5k) splits once to the right and then holds its cursor still. The two 9,250-mass "
        "halves should fly ~820 units, push against each other, and merge back after ~29 s (721 ticks). "
        "B is idle.",
        [[0, *A0, R0, 3000, 0, 0, 0], [1, *B0, R0, 3000, 0, 0, 0]],
        lambda sim, t: ([[A0[0] + 3000, 0] if t == 0 else hold, B0], [1 if t == 0 else 0, 0]), 1000)

    # 2. double split: 4 x 4,625 pieces, merge after round(0.75 * 680) = 510 ticks (~20.4 s)
    run("02a_double_split_same_tick",
        "Player A (18.5k) double-splits to the right with BOTH splits processed in the same server tick, then "
        "holds still. Four 4,625-mass pieces in a line; they merge back after ~20 s (510 ticks). B is idle. "
        "Compare with 02b: which one looks like an imsolo double split?",
        [[0, *A0, R0, 3000, 0, 0, 0], [1, *B0, R0, 3000, 0, 0, 0]],
        lambda sim, t: ([[A0[0] + 3000, 0] if t == 0 else hold, B0], [2 if t == 0 else 0, 0]), 800)
    run("02b_double_split_1tick_gap",
        "Same as 02a, but the second split is processed one server tick (40 ms) after the first, like the "
        "client's fallback double-split macro. The first piece has already flown ~87 units, so the pieces spread "
        "differently.",
        [[0, *A0, R0, 3000, 0, 0, 0], [1, *B0, R0, 3000, 0, 0, 0]],
        lambda sim, t: ([[A0[0] + 3000, 0] if t <= 1 else hold, B0], [1 if t <= 1 else 0, 0]), 800)

    # 3. the elementary splitrun step: A double-splits away (upwards), leaving a quarter-mass
    #    backpiece; B single-splits into that backpiece and eats it.
    A1, B1 = (-1500.0, 0.0), (0.0, 0.0)
    state = {}

    def ctl3(sim, t):
        a_cells = cells_of(sim, 0, 0)
        if t == 5:
            state["back"] = a_cells[0][0]          # A's oldest cell stays behind = backpiece
        back = next((c for c in a_cells if c[0] == state.get("back")), None)
        ma = [A1[0], -4000]
        mb = [back[1], back[2]] if back is not None else list(centroid(sim, 0, 1))
        return [ma, mb], [2 if t == 5 else 0, 1 if t == 8 else 0]
    run("03_backpiece_eat",
        "The basic splitrun ingredient. A (18.5k) double-splits upwards (same tick), leaving 4 quarter pieces "
        "(4,625) stacked in a line; B (18.5k) single-splits at A's backpiece 3 ticks later. When the 13-tick "
        "no-collide window ends, A's stacked pieces spring apart, and B's 9,250 launched half (>1.3x bigger) "
        "eats the quarter pieces it reaches.",
        [[0, *A1, R0, 3000, 0, 0, 0], [1, *B1, R0, 3000, 0, 0, 0]], ctl3, 150)

    # 4. autosplit: A (20k) eats B's launched 5k piece -> 25k > 22,500 -> autosplits in a random direction
    run("04_autosplit",
        "Autosplit check. A is a single 20k cell; B (10k) splits at A, A eats the 5k piece and reaches 25k, "
        "which is over the 22,500 single-cell limit, so A autosplits into two 12.5k cells (random angle).",
        [[0, 0, 0, float(np.sqrt(20000 * 100)), 3000, 0, 0, 0], [1, 1900, 0, float(np.sqrt(10000 * 100)), 3000, 0, 0, 0]],
        lambda sim, t: ([[0, 0], [0, 0]], [0, 1 if t == 2 else 0]), 100, seed=7)

    # 5. fidelity, visually: the same random input tape through the real server and the simulator
    tape = tf.make_tape(0, 400)
    trace = tf.run_server(tape)
    rep = server_trace_to_replay(trace, tape, "05a_random_splits_REAL_SERVER",
                                 "Random aggressive splitting by both players, run through the REAL imsolo server "
                                 "code. Compare with 05b (same inputs, simulator) using 'overlay'.", json.load(open(os.path.join(ROOT, "env_settings.json"))))
    with open(os.path.join(OUT, "05a_random_splits_REAL_SERVER.json"), "w") as f:
        json.dump(rep, f, separators=(",", ":"))
    it = iter(tape["inputs"])
    run("05b_random_splits_SIM",
        "Same input tape as 05a, run through the simulator. Should be indistinguishable from 05a.",
        [[c["owner"], c["x"], c["y"], c["size"], c["age"], 0, 0, 0] for c in tape["cells"]],
        lambda sim, t: (lambda row: ([row[0][:2], row[1][:2]], [row[0][2], row[1][2]]))(next(it)), len(trace), seed=tape["seed"])

    items = write_index(os.path.join(ROOT, "runs"))
    print(f"{len(items)} replays indexed")


if __name__ == "__main__":
    main()
