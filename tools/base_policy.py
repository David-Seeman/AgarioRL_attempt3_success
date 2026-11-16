"""The trained discovery policy as a BASE controller for any batch of simulator worlds.

It keeps a circle-style splitrun going. Creative maneuvers are planned on top of it
(tools/creative_runs.py): each maneuver is checked by simulating the base policy afterwards.

    pol = BasePolicy("discover_s0")
    mouse, splits = pol.act(sim, since_split)      # one decision (applied for ACTION_REPEAT ticks)
"""
import os, sys
import numpy as np
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import __main__
import rl.ppo as pp
__main__.RunningNorm = pp.RunningNorm            # checkpoints pickled the class from the training script
from splitrun.rlenv import build_obs, env_actions, OBS_DIM, NTI, NTF, T_SINCE_SPLIT, T_SINCE_PROG, T_BOUND, ACTION_REPEAT  # noqa

torch.set_num_threads(2)


class BasePolicy:
    def __init__(self, run="discover_s0"):
        d = torch.load(os.path.join(ROOT, "runs", "rl", run, "ckpt.pt"), weights_only=False)
        self.pol = pp.Policy(OBS_DIM); self.pol.load_state_dict(d["pol"]); self.pol.eval()
        self.norm = d["norm"]

    def act(self, sim, since_split, since_prog=None):
        N = sim.cf.shape[0]
        tr = np.zeros((N, NTI), np.int64); trf = np.zeros((N, NTF))
        tr[:, T_SINCE_SPLIT] = since_split
        tr[:, T_SINCE_PROG] = since_split if since_prog is None else since_prog
        tr[:, T_BOUND] = 10                          # "in an orbit"
        obs = np.zeros((N, OBS_DIM))
        build_obs(sim.cf, sim.ci, sim.oc, sim.ocn, sim.g, tr, trf, 0, 1, np.zeros(N, np.int64), obs)
        with torch.no_grad():
            normal, cat = self.pol.dist(torch.as_tensor(self.norm(obs)))
            a = pp.to_env_action(normal.mean.numpy(), cat.probs.argmax(-1).numpy())
        mouse = np.zeros((N, 2, 2)); splits = np.zeros((N, 2), np.int64)
        env_actions(sim.cf, sim.oc, sim.ocn, tr, 0, a, mouse, splits)
        return mouse, splits
