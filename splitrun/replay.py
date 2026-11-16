"""Record episodes from a Sim env (or a real-server trace) into the viewer's replay format.

Replay JSON:
  { name, description, source: "sim" | "server", tick_rate, map: [x, y, w, h],
    frames:  [ [ [id, owner, x, y, size, flags], ... ], ... ]   flags: 1 boosting, 2 can merge
    inputs:  [ [[mx, my, splits], [mx, my, splits]], ... ]      what each player pressed that tick
    events:  [ [tick, eater_owner, eaten_owner, mass, kind], ... ] kind: "eat" (cross-player) | "merge" }
"""
import json
import os

import numpy as np

from .sim import (BOOSTING, CANMERGE, CID, EV_EATEN_OWN, EV_EATER_OWN, EV_MASS, OWN, SZ, X, Y, BD)


class Recorder:
    def __init__(self, sim, e, name, description=""):
        self.sim, self.e = sim, e
        s = sim.settings
        self.data = {"name": name, "description": description, "source": "sim",
                     "tick_rate": s["serverFrequency"],
                     "map": [s["worldMapX"], s["worldMapY"], s["worldMapW"], s["worldMapH"]],
                     "frames": [], "inputs": [], "events": []}
        self.frame()

    def frame(self, inputs=None):
        sim, e = self.sim, self.e
        cells = []
        for i in range(sim.pcn[e]):
            s_ = sim.pc[e, i]
            flags = (1 if sim.ci[e, s_, BOOSTING] else 0) | (2 if sim.ci[e, s_, CANMERGE] else 0)
            cells.append([int(sim.ci[e, s_, CID]), int(sim.ci[e, s_, OWN]), round(float(sim.cf[e, s_, X]), 1),
                          round(float(sim.cf[e, s_, Y]), 1), round(float(sim.cf[e, s_, SZ]), 2), flags])
        self.data["frames"].append(cells)
        if inputs is not None:
            self.data["inputs"].append(inputs)
            t = len(self.data["frames"]) - 1
            for ev in sim.events(e):
                kind = "merge" if ev[EV_EATER_OWN] == ev[EV_EATEN_OWN] else "eat"
                self.data["events"].append([t, int(ev[EV_EATER_OWN]), int(ev[EV_EATEN_OWN]),
                                            round(float(ev[EV_MASS]), 1), kind])

    def after_step(self, mouse, splits):
        """call after sim.step(); mouse (2,2), splits (2,) for this env"""
        self.frame([[float(mouse[p][0]), float(mouse[p][1]), int(splits[p])] for p in range(2)])

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.data, f, separators=(",", ":"))


def server_trace_to_replay(trace_ticks, tape, name, description, settings):
    """real-server trace (tools/trace_server.js output) -> replay (no flags/events: display only)"""
    frames = [[[i + 1, c["owner"], c["x"], c["y"], c["size"], 0] for i, c in enumerate(tape["cells"])]]
    for t in trace_ticks:
        frames.append([[c[0], c[1], round(c[2], 1), round(c[3], 1), round(c[4], 2), 1 if c[5] > 0 else 0] for c in t])
    return {"name": name, "description": description, "source": "server", "tick_rate": settings["serverFrequency"],
            "map": [settings["worldMapX"], settings["worldMapY"], settings["worldMapW"], settings["worldMapH"]],
            "frames": frames, "inputs": [[i[0], i[1]] for i in tape["inputs"][:len(trace_ticks)]], "events": []}


def write_index(runs_dir):
    """runs/index.json lists the replays under runs/ for the viewer. Automatic RL training evaluations
    (runs/rl/<run>/eval_*.json) would bury everything else, so only the 2 latest per run are listed."""
    items = []
    for root, _, files in os.walk(runs_dir):
        if os.sep + "old" in root:
            continue
        evals = sorted(f for f in files if f.startswith("eval_") and f.endswith(".json"))
        skip = set(evals[:-2])
        for fn in sorted(files):
            if fn in skip:
                continue
            if fn.endswith(".json") and fn not in ("index.json", "live.json"):
                p = os.path.join(root, fn)
                try:
                    with open(p) as f:
                        d = json.load(f)
                except (json.JSONDecodeError, OSError):      # another process is mid-write
                    continue
                if not isinstance(d, dict) or "frames" not in d:
                    continue
                items.append({"path": os.path.relpath(p, runs_dir), "name": d.get("name", fn),
                              "description": d.get("description", ""), "source": d.get("source", "sim"),
                              "ticks": len(d["frames"]) - 1})
    with open(os.path.join(runs_dir, "index.json"), "w") as f:
        json.dump(items, f, indent=1)
    return items
