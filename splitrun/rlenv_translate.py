"""Translation environment: the discovery splitrun (clean, purposeful splits; same termination rules) plus a
reward for moving the whole team in a requested direction as fast as possible.

* direction: a random unit vector per episode, turned by a random angle every TURN ticks; it is given to the
  policy in the pair frame (A's centre -> B's centre), as 2 extra observation features
* progress reward: PROG_W x (advance of the team's PROJECTED centre of mass along the direction), where pieces
  in flight count where their boost will carry them -- a forward trim is credited when it is thrown
* small penalty for sideways drift
"""
import math
import numpy as np
from .rlenv_discover import DiscoverEnv
from .rlenv import T_TICKS
from .sim import X, Y, SZ, EX, OWN, BDX, BDY, BD, BOOSTING

PROG_W = 0.25
LAT_W = 0.01
TURN = 750
EDGE = 7000.0              # beyond this (map is +-9,071) the requested direction turns back to the middle
DEATH_PEN = 150.0          # ending a run early costs about a minute of progress: sustained speed, not sprints


def projected_com(sim, idx=None):
    cf, ci = sim.cf, sim.ci
    if idx is not None:
        cf, ci = cf[idx], ci[idx]
    ex = ci[:, :, EX] == 1
    m = cf[:, :, SZ] ** 2 * ex
    bo = (ci[:, :, BOOSTING] == 1) & ex
    px = cf[:, :, X] + np.where(bo, cf[:, :, BDX] * cf[:, :, BD], 0)
    py = cf[:, :, Y] + np.where(bo, cf[:, :, BDY] * cf[:, :, BD], 0)
    M = np.maximum(m.sum(1), 1)
    return np.stack([(px * m).sum(1) / M, (py * m).sum(1) / M], 1)


class TranslateEnv(DiscoverEnv):
    def __init__(self, spec, pool, n_envs, max_ticks=1500, seed=0, **kw):
        N = n_envs
        self.dir = np.zeros((N, 2)); self.dir[:, 0] = 1
        self.pcom = np.zeros((N, 2))
        self.ep_prog = np.zeros(N)
        self.done_speeds = []
        self._rng = np.random.default_rng(seed + 12345)
        super().__init__(spec, pool, n_envs, max_ticks, seed, **kw)
        self.max_drift = 1e9                    # travelling is the point: no "stay near the start" cutoff

    def reset(self, idx):
        super().reset(idx)
        a = self._rng.uniform(0, 2 * math.pi, len(np.atleast_1d(idx)))
        self.dir[idx] = np.stack([np.cos(a), np.sin(a)], 1)
        self.pcom[idx] = projected_com(self.sim, idx)

    def advance(self, mouse, splits, repeat):
        before_ticks = self.tr[:, T_TICKS].copy()
        prog_before = self.ep_prog.copy()
        obs, rew, term, trunc, info = super().advance(mouse, splits, repeat)
        done = term | trunc
        pc = projected_com(self.sim)
        mv = pc - self.pcom
        prog = (mv * self.dir).sum(1)
        lat = np.abs(mv[:, 0] * self.dir[:, 1] - mv[:, 1] * self.dir[:, 0])
        live = ~done
        r = np.where(live, PROG_W * prog - LAT_W * lat, 0.0) - DEATH_PEN * term
        rew = rew + r
        self.ep_ret += r
        self.ep_prog += np.where(live, prog, 0.0)
        if info is not None:
            idx = np.nonzero(done)[0]
            for j, e in enumerate(idx):
                t = max(int(info["ticks"][j]), 1)
                self.done_speeds.append(float(self.ep_prog[e]) / t * 25)
            self.ep_prog[idx] = 0
            info["speed"] = np.array(self.done_speeds[-len(idx):]) if len(idx) else None
        self.pcom = pc
        # near the map edge: point the requested direction back towards the middle
        edge = live & (np.abs(pc).max(1) > EDGE)
        if edge.any():
            back = -pc[edge] / np.maximum(np.linalg.norm(pc[edge], axis=1, keepdims=True), 1e-6)
            self.dir[edge] = np.where(((self.dir[edge] * back).sum(1) < 0.3)[:, None], back, self.dir[edge])
        # new direction now and then (episodes teach steering, not just one heading)
        turn = live & (self.tr[:, T_TICKS] // TURN > before_ticks // TURN)
        if turn.any():
            a = np.arctan2(self.dir[turn, 1], self.dir[turn, 0]) + self._rng.uniform(-math.pi, math.pi, turn.sum())
            self.dir[turn] = np.stack([np.cos(a), np.sin(a)], 1)
        return self.observe(), rew, term, trunc, info

    def observe(self):
        base = super().observe()
        sim = self.sim
        ex = sim.ci[:, :, EX] == 1; own = sim.ci[:, :, OWN]; m = sim.cf[:, :, SZ] ** 2 * ex
        ca = np.stack([(sim.cf[:, :, X] * m * (own == 0)).sum(1), (sim.cf[:, :, Y] * m * (own == 0)).sum(1)], 1) / np.maximum((m * (own == 0)).sum(1), 1)[:, None]
        cb = np.stack([(sim.cf[:, :, X] * m * (own == 1)).sum(1), (sim.cf[:, :, Y] * m * (own == 1)).sum(1)], 1) / np.maximum((m * (own == 1)).sum(1), 1)[:, None]
        u = cb - ca; u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-6); w = np.stack([-u[:, 1], u[:, 0]], 1)
        if not hasattr(self, "obs2"):
            self.obs2 = np.zeros((base.shape[0], base.shape[1] + 2), np.float32)
        self.obs2[:, :base.shape[1]] = base
        self.obs2[:, -2] = (self.dir * u).sum(1)
        self.obs2[:, -1] = (self.dir * w).sum(1)
        return self.obs2
