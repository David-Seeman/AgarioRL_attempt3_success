"""One-screen status of all RL runs: trend of the last iterations + process health."""
import csv, glob, os, subprocess, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
alive = subprocess.run(["pgrep", "-fl", "rl/ppo.py"], capture_output=True, text=True).stdout
for d in sorted(glob.glob(os.path.join(ROOT, "runs", "rl", "*", "log.csv"))):
    run = d.split("/")[-2]
    rows = list(csv.DictReader(open(d)))
    if not rows:
        print(f"{run}: no rows yet"); continue
    last = rows[-1]
    def avg(key, n):
        v = [float(r[key]) for r in rows[-n:] if r[key] not in ("", "nan")]
        return sum(v) / len(v) if v else float("nan")
    evs = [(r["iter"], r["eval_cycles_mean"], r["eval_cycles_max"]) for r in rows if r["eval_cycles_mean"] not in ("",)]
    age = time.time() - os.path.getmtime(d)
    running = f"--run {run}" in alive
    print(f"{run}: {'RUNNING' if running else 'STOPPED'} (log updated {age:.0f}s ago) iter {last['iter']} "
          f"steps {int(float(last['env_steps'])):,} | last 20 iters: ep cycles {avg('ep_cycles', 20):.2f} "
          f"(max {max(float(r['max_cycles']) for r in rows[-20:] if r['max_cycles'] != 'nan'):.1f}), "
          f"ep ticks {avg('ep_ticks', 20):.0f}, survived {avg('survived_frac', 20):.2f} | evals {evs[-3:]}"
          + (f" | pool {last.get('pool_size')}" if last.get('pool_size') else ""))
