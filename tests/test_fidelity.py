"""Differential test: splitrun/sim.py vs the real imsolo server code, cell by cell, every tick.

    python tests/test_fidelity.py [n_scenarios] [ticks]

Scenarios are random but aggressive: two players start close together with lots of
mass and press single/double splits at random while aiming around each other, so
every rule gets exercised (cross-player eats, own-cell merges, rigid pile-ups, the
16-cell cap, autosplit above 22,500 mass, and wall bounces in the 'wall' family).
"""
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from splitrun.sim import Sim  # noqa: E402


def make_tape(seed, ticks):
    rng = np.random.default_rng(seed)
    family = ["close", "close", "autosplit", "wall", "far"][seed % 5]
    mapw = 9071
    cx, cy = rng.uniform(-3000, 3000, 2)
    if family == "wall":
        cx, cy = mapw - 1500, rng.uniform(-3000, 3000)
    cells = []
    if family == "autosplit":
        # one player well above the 22,500-mass autosplit threshold
        masses = [rng.uniform(23000, 60000), rng.uniform(5000, 18500)]
    else:
        masses = [rng.uniform(8000, 18500), rng.uniform(8000, 18500)]
    gap = rng.uniform(1500, 3500) if family != "far" else rng.uniform(4000, 6000)
    ang = rng.uniform(0, 2 * np.pi)
    for p in range(2):
        sgn = -1 if p == 0 else 1
        x = cx + sgn * np.cos(ang) * gap / 2
        y = cy + sgn * np.sin(ang) * gap / 2
        cells.append({"owner": p, "x": float(x), "y": float(y), "size": float(np.sqrt(masses[p] * 100)),
                      "age": int(rng.integers(0, 3000))})
    inputs = []
    targets = [None, None]
    split_p = rng.uniform(0.02, 0.12)
    for t in range(ticks):
        row = []
        for p in range(2):
            if targets[p] is None or rng.random() < 0.08:
                targets[p] = (cx + rng.normal(0, 2500), cy + rng.normal(0, 2500))
            s = 0
            if rng.random() < split_p:
                s = 1 if rng.random() < 0.6 else 2
            row.append([float(targets[p][0]), float(targets[p][1]), s])
        inputs.append(row)
    return {"seed": int(seed) + 1, "cells": cells, "inputs": inputs, "family": family}


def run_server(tape):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(tape, f)
        name = f.name
    out = subprocess.run(["node", os.path.join(ROOT, "tools", "trace_server.js"), name],
                         capture_output=True, text=True, check=True).stdout
    os.unlink(name)
    return json.loads(out)["ticks"]


def run_sim(tapes):
    sim = Sim(len(tapes))
    for e, tp in enumerate(tapes):
        rows = [[c["owner"], c["x"], c["y"], c["size"], c["age"], 0, 0, 0] for c in tp["cells"]]
        sim.load(e, rows, seed=tp["seed"])
    T = max(len(tp["inputs"]) for tp in tapes)
    out = [[] for _ in tapes]
    for t in range(T):
        mouse = np.zeros((len(tapes), 2, 2))
        splits = np.zeros((len(tapes), 2), np.int64)
        for e, tp in enumerate(tapes):
            if t < len(tp["inputs"]):
                for p in range(2):
                    mouse[e, p] = tp["inputs"][t][p][:2]
                    splits[e, p] = tp["inputs"][t][p][2]
            else:
                sim.active[e] = 0
        sim.step(mouse, splits)
        for e in range(len(tapes)):
            if sim.active[e]:
                out[e].append(sim.cells(e))
                if sim.ocn[e, 0] == 0 or sim.ocn[e, 1] == 0:
                    sim.active[e] = 0
    return out, sim


def compare(server, sim):
    """returns (first tick with a topology mismatch or None, max abs position/size error)"""
    maxerr = 0.0
    n = min(len(server), len(sim))
    if len(server) != len(sim):
        return n, maxerr
    for t in range(n):
        a, b = server[t], sim[t]
        if len(a) != len(b) or any(x[0] != y[0] or x[1] != y[1] for x, y in zip(a, b)):
            return t, maxerr
        for x, y in zip(a, b):
            maxerr = max(maxerr, abs(x[2] - y[2]), abs(x[3] - y[3]), abs(x[4] - y[4]), abs(x[5] - y[5]))
    return None, maxerr


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    ticks = int(sys.argv[2]) if len(sys.argv) > 2 else 750
    tapes = [make_tape(s, ticks) for s in range(n)]
    with ThreadPoolExecutor(8) as ex:
        server = list(ex.map(run_server, tapes))
    sim_out, sim = run_sim(tapes)
    bad = 0
    total_states = 0
    fam_stats = {}
    for i, tp in enumerate(tapes):
        t_bad, err = compare(server[i], sim_out[i])
        total_states += sum(len(x) for x in server[i])
        cells_max = max(len(x) for x in server[i])
        fam_stats.setdefault(tp["family"], []).append(cells_max)
        status = "OK " if t_bad is None and err < 1e-6 else "BAD"
        if status == "BAD":
            bad += 1
        print(f"{status} scenario {i:3d} [{tp['family']:9s}] ticks={len(server[i]):4d} max_cells_in_world={cells_max:2d} "
              f"max_err={err:.2e}" + ("" if t_bad is None else f"  first mismatch at tick {t_bad}"))
    print(f"\n{n - bad}/{n} scenarios identical; {total_states} cell-states compared; "
          f"quadtree overflow flags: {int(sim.g[:, 3].max())}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
