"""Benchmark a translation policy: fixed heading (+x), deterministic actions, n worlds from the orbit starts.
Reports per-world speed along the heading (units/s) and survival (s); saves the best world as a replay.

    python rl/translate_bench.py translate_b --n 8 --secs 60
"""
import argparse, json, os, sys
import numpy as np
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import __main__
import rl.ppo as pp
__main__.RunningNorm = pp.RunningNorm
import splitrun.rlenv_translate as RT  # noqa
from splitrun.rlenv import TechSpec, StartPool, T_TICKS  # noqa
from splitrun.replay import Recorder, write_index  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("--n", type=int, default=8); ap.add_argument("--secs", type=float, default=60)
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    RT.TURN = 10 ** 9                                     # keep the heading fixed
    spec = TechSpec(json.load(open(os.path.join(ROOT, "runs/techniques/catalogue_s3_e2_p8.json")))[4])
    pool = StartPool(spec).load(os.path.join(ROOT, "runs/rl/pool_orbit.pkl"))
    d = torch.load(os.path.join(ROOT, "runs", "rl", a.run, "ckpt.pt"), weights_only=False)
    ticks = int(a.secs * 25)
    env = RT.TranslateEnv(spec, pool, a.n, max_ticks=ticks, seed=7)
    env.dir[:] = (1.0, 0.0)
    obs = env.observe()
    pol = pp.Policy(obs.shape[1]); pol.load_state_dict(d["pol"]); norm = d["norm"]
    recs = {e: Recorder(env.sim, e, "translate") for e in range(a.n)}
    start = RT.projected_com(env.sim).copy()
    alive = np.ones(a.n, bool); surv = np.zeros(a.n); dist = np.zeros(a.n)
    for step in range(ticks // 2):
        with torch.no_grad():
            nd, c = pol.dist(torch.as_tensor(norm(obs)))
        obs, r, term, trunc, info = env.step(pp.to_env_action(nd.mean.numpy(), c.probs.argmax(-1).numpy()))
        done = term | trunc
        pc = RT.projected_com(env.sim)
        for e in range(a.n):
            if not alive[e]:
                continue
            if done[e]:
                alive[e] = False
                continue
            recs[e].after_step(np.trunc(env.mouse[e]), env.splits[e])
            surv[e] = env.tr[e, T_TICKS] / 25
            dist[e] = pc[e, 0] - start[e, 0]
        env.dir[:] = (1.0, 0.0)
        if not alive.any():
            break
    speed = dist / np.maximum(surv, 1e-6)
    for e in range(a.n):
        print(f"  world {e}: survived {surv[e]:5.1f} s, speed {speed[e]:6.1f} units/s")
    print(f"{a.run}: mean speed {speed.mean():.0f} units/s, mean survival {surv.mean():.1f} s "
          f"(best {speed[np.argmax(surv * speed)]:.0f} u/s over {surv[np.argmax(surv * speed)]:.1f} s)")
    if a.save:
        e = int(np.argmax(surv * speed))
        rec = recs[e]
        rec.data["name"] = a.save
        rec.data["description"] = (f"RL translation policy ({a.run}), fixed heading +x: {surv[e]:.1f} s at {speed[e]:.0f} units/s "
                                   "along the heading.")
        os.makedirs(os.path.join(ROOT, "runs", "dynamic"), exist_ok=True)
        rec.save(os.path.join(ROOT, "runs", "dynamic", a.save + ".json"))
        write_index(os.path.join(ROOT, "runs"))


if __name__ == "__main__":
    main()
