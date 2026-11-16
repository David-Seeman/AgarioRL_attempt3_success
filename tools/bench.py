"""Throughput of the batched simulator under aggressive random splitting."""
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from splitrun.sim import Sim

def run(E, T=300):
    rng = np.random.default_rng(0)
    sim = Sim(E)
    for e in range(E):
        sim.load(e, [[0, -1200, 0, 1360, 2000, 0, 0, 0], [1, 1200, 0, 1360, 2000, 0, 0, 0]], seed=e + 1)
    mouse = rng.normal(0, 2500, (E, 2, 2))
    sim.step(mouse, np.zeros((E, 2), np.int64))  # compile / warm
    t0 = time.time(); ncells = 0
    for t in range(T):
        if t % 10 == 0:
            mouse = rng.normal(0, 2500, (E, 2, 2))
        splits = (rng.random((E, 2)) < 0.06) * rng.integers(1, 3, (E, 2))
        sim.step(mouse, splits)
        ncells += sim.pcn.sum()
    dt = time.time() - t0
    return E * T / dt, ncells / (E * T)

for E in (1, 64, 512, 2048):
    sps, avg = run(E)
    print(f"E={E:5d}: {sps:10,.0f} env-ticks/s   (avg {avg:.1f} cells per world)")
