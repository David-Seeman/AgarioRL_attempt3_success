"""Roll out a discovery run's policy deterministically; measure how far launched pieces fly before
their eat, the repeat period, and save a replay under runs/rl/discoveries/."""
import sys, json, os
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import __main__
import rl.ppo as pp
__main__.RunningNorm = pp.RunningNorm
from splitrun.rlenv import TechSpec, StartPool, OBS_DIM, T_PN, T_BOUND, T_DONE, T_TICKS
from splitrun.rlenv_discover import DiscoverEnv
from splitrun.sim import EV_EATER_OWN, EV_EATEN_OWN, EV_EATER_AGE, EV_EATEN_AGE, EV_MASS
from splitrun.replay import Recorder, write_index
from splitrun.metrics import phase_periodicity, splitrun_verdict


def inspect(run, n=16, ticks=1500, min_flight=0):
    args = json.load(open(os.path.join(ROOT, "runs", "rl", run, "args.json")))
    spec = TechSpec(json.load(open(args["cat"]))[args["tech"]])
    pool = StartPool(spec).load(args["pool"])
    for p in args.get("extra_pools", []):
        pool.states += StartPool(spec).load(p).states
    d = torch.load(os.path.join(ROOT, "runs", "rl", run, "ckpt.pt"), weights_only=False)
    pol = pp.Policy(OBS_DIM); pol.load_state_dict(d["pol"]); norm = d["norm"]
    env = DiscoverEnv(spec, pool, n, max_ticks=ticks, seed=7, min_flight=args.get("min_flight", 0))
    recs = {e: Recorder(env.sim, e, run) for e in range(n)}
    alive = np.ones(n, bool); ages = [[] for _ in range(n)]
    obs = env.observe()
    for s in range(ticks // 2):
        with torch.no_grad():
            nd, c = pol.dist(torch.as_tensor(norm(obs)))
            a = pp.to_env_action(nd.mean.numpy(), c.probs.argmax(-1).numpy())
        obs, r, term, trunc, info = env.step(a)
        for e in range(n):
            if alive[e]:
                for q in range(env.sim.evn[e]):
                    ev = env.sim.ev[e, q]
                    if ev[EV_EATER_OWN] != ev[EV_EATEN_OWN]:
                        ages[e].append(min(ev[EV_EATER_AGE], ev[EV_EATEN_AGE]))
                if term[e] or trunc[e]:
                    alive[e] = False
                else:
                    recs[e].after_step(np.trunc(env.mouse[e]), env.splits[e])
    lens = [len(recs[e].data["frames"]) for e in range(n)]
    best = int(np.argmax(lens))
    rec = recs[best].data
    per = None
    for k in range(1, 9):
        for mir in (False, True):
            r = phase_periodicity(rec, k, mir)
            if r["cycles"] and r["repeating_cycles"] / r["cycles"] >= 0.8:
                per = (k, mir, r); break
        if per: break
    ag = np.array(ages[best])
    v = splitrun_verdict(rec)
    return rec, per, ag, v, lens


if __name__ == "__main__":
    out = os.path.join(ROOT, "runs", "rl", "discoveries"); os.makedirs(out, exist_ok=True)
    for run in sys.argv[1:]:
        rec, per, ag, v, lens = inspect(run)
        k, mir, r = per if per else (None, None, {})
        desc = (f"Discovered by RL ({run}, no target technique). Repeats every {k} splits"
                f"{' with A/B swapped' if mir else ''} (phase-aligned: {r.get('repeating_cycles')}/{r.get('cycles')} cycles "
                f"repeat, median layout error {r.get('median_error', 0):.1f} units, {r.get('median_cycle_ticks')} ticks per cycle). "
                f"Rotations {v['rotations']}, centre drift {v['centre_drift']} units. Launched pieces eat/are eaten after a median "
                f"{np.median(ag):.0f} ticks of flight ({(ag <= 3).mean():.0%} within 3 ticks = point-blank).")
        rec["name"] = run; rec["description"] = desc
        json.dump(rec, open(os.path.join(out, run + ".json"), "w"), separators=(",", ":"))
        print(run, "| episode lengths", sorted(lens)[-4:], "|", desc)
    write_index(os.path.join(ROOT, "runs"))
