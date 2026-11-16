"""Benchmark the fast-translating splitrun: start at x=-7,000 heading for x=+8,000, play up to --secs seconds.
Reports speed along the direction (units/s), sideways drift, downtime, survival.

    python tools/speed_bench.py --style '{"speed_w": 0.08}' --seeds 0 1
"""
import argparse, json, os, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import tools.live_controller as LC  # noqa
import tools.choreo as C0  # noqa


def bench(style, seed, secs, lookahead=10, beam=24, M=64, save=None, total=None):
    L = LC.Live(beam=beam, M=M, lookahead=lookahead, seed=seed, pace=False, mode="dynamic",
                out=os.path.join(ROOT, "runs", "speed_bench_tmp"))
    L.style.update(style)
    if total:
        L.total = total
        C0.SCALE = total / 37000
    L.start_at = (-7000.0, 0.0)
    L.new_run()
    L.waypoint = np.array([8000.0, 0.0])
    t0 = time.time()
    track = []
    import tools.choreo as C
    from collections import defaultdict
    mstat = defaultdict(lambda: [0, 0.0])
    while len(L.rec.data["frames"]) < secs * 25 and time.time() - t0 < secs * 8:
        L.steer()
        if not L.expand():
            if not L.backtrack():
                break
            continue
        if len(L.nodes[0][2]) >= L.L:
            mv = C.MOVES[L.nodes[0][2][0]["move"]]
            _, c0 = L.team()
            if not L.commit() or not L.nodes:
                break
            _, c = L.team(); track.append((len(L.rec.data["frames"]) / 25, c[0], c[1]))
            mstat[mv][0] += 1; mstat[mv][1] += float(c[0] - c0[0])
    I = L.rec.data["inputs"]
    presses = [t for t, inp in enumerate(I) if inp[0][2] or inp[1][2]]
    gaps = np.diff([0] + presses) / 25 if presses else np.array([99.0])
    tr = np.array(track) if track else np.zeros((1, 3))
    dur = len(L.rec.data["frames"]) / 25
    res = dict(seed=seed, seconds=round(dur, 1), speed=round(float((tr[-1, 1] - (-7000)) / max(dur, 1e-6)), 1),
               max_lateral=round(float(np.abs(tr[:, 2]).max()), 0), mean_gap=round(float(gaps.mean()), 2),
               max_gap=round(float(gaps.max()), 2), splits=len(presses), wall=round(time.time() - t0),
               moves={k: (v[0], round(v[1] / max(v[0], 1))) for k, v in sorted(mstat.items(), key=lambda kv: -kv[1][0])})
    if save:
        L.rec.data["name"] = save
        L.rec.data["description"] = (f"Fast-translating splitrun benchmark: {res['seconds']} s, {res['speed']} units/s along the "
                                     f"direction, gap {res['mean_gap']} s average (max {res['max_gap']} s), style {style}")
        L.rec.save(os.path.join(ROOT, "runs", "dynamic", save + ".json"))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="{}"); ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--secs", type=float, default=30); ap.add_argument("--save", default="")
    ap.add_argument("--lookahead", type=int, default=10); ap.add_argument("--total", type=float, default=0)
    ap.add_argument("--young", type=int, default=60)
    a = ap.parse_args()
    C0.YOUNG = a.young
    for sd in a.seeds:
        print(bench(json.loads(a.style), sd, a.secs, lookahead=a.lookahead, save=(a.save + f"_s{sd}") if a.save else None, total=a.total or None), flush=True)
