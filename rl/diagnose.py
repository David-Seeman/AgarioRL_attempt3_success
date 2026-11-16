"""Why do episodes end? Roll out a run's current policy (and an idle baseline) and count reasons."""
import sys, json, collections, os
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import __main__
import rl.ppo as pp
__main__.RunningNorm = pp.RunningNorm           # checkpoints pickled the class from the training script
from splitrun.rlenv import SplitrunEnv, TechSpec, StartPool, OBS_DIM
NAMES = {0: 'max length', 2: 'wrong split', 3: 'own merge', 4: 'unintended eat', 5: 'intended eat timeout',
         6: 'no progress', 7: 'player vanished', 8: 'drift'}


def diagnose(run, n=256, steps=300):
    args = json.load(open(os.path.join(ROOT, 'runs', 'rl', run, 'args.json')))
    spec = TechSpec(json.load(open(args['cat']))[args['tech']])
    pool = StartPool(spec).load(args['pool'])
    d = torch.load(os.path.join(ROOT, 'runs', 'rl', run, 'ckpt.pt'), weights_only=False)
    pol = pp.Policy(OBS_DIM); pol.load_state_dict(d['pol']); norm = d['norm']
    out = {}
    for mode in ('policy', 'idle'):
        env = SplitrunEnv(spec, pool, n, max_ticks=600, seed=1); obs = env.observe()
        reasons = collections.Counter(); m = 0; cyc = []
        for s in range(steps):
            if mode == 'policy':
                with torch.no_grad():
                    nd, c = pol.dist(torch.as_tensor(norm(obs)))
                    a = pp.to_env_action(nd.sample().numpy(), c.sample().numpy())
            else:
                a = np.zeros((n, 2, 3))
            obs, r, term, trunc, info = env.step(a)
            if info is not None:
                for rr, cc in zip(info['reason'], info['cycles']):
                    reasons[NAMES.get(int(rr), rr)] += 1; m += 1; cyc.append(cc)
        out[mode] = (m, {k: f"{v / m:.0%}" for k, v in reasons.most_common()}, float(np.mean(cyc)) if cyc else 0)
    return out


if __name__ == "__main__":
    for run in sys.argv[1:]:
        for mode, (m, rs, c) in diagnose(run).items():
            print(f"{run} [{mode}] {m} episodes, mean cycles {c:.2f}: {rs}")
