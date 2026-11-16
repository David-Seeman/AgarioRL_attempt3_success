"""Split-direction profile of a replay: is it "just a circle", or does it split in unexpected directions?

For every split press, angle = direction of the split (from the splitting player's mass centre to its cursor)
measured against the direction to the other player's mass centre (0 = straight at the partner, 90 = sideways,
180 = directly away). Signed so that + is the side the pair rotates to.
In a circle technique every split is aimed about the same way (a narrow band of small angles).

    python tools/split_directions.py runs/techniques/technique_0.json runs/techniques/technique_4.json
"""
import json, math, sys
import numpy as np


def centre(frame, owner):
    c = [(x, y, s * s) for _, o, x, y, s, _ in frame if o == owner]
    if not c:
        return None
    m = sum(w for *_, w in c)
    return np.array([sum(x * w for x, _, w in c) / m, sum(y * w for _, y, w in c) / m])


def profile(replay):
    F, I = replay["frames"], replay["inputs"]
    out = []
    for t, inp in enumerate(I):
        for p in range(2):
            if inp[p][2] <= 0:
                continue
            fr = F[t]                      # the state the press acted on
            me, other = centre(fr, p), centre(fr, 1 - p)
            if me is None or other is None:
                continue
            d = np.array(inp[p][:2]) - me
            a = np.array(other) - me
            ang = math.degrees(math.atan2(a[0] * d[1] - a[1] * d[0], a @ d))
            out.append((t, p, int(inp[p][2]), ang))
    if not out:
        return out
    # sign convention: + in the direction the pair mostly turns
    s = np.sign(np.median([o[3] for o in out])) or 1
    return [(t, p, k, s * a) for t, p, k, a in out]


def summary(prof):
    a = np.array([x[3] for x in prof])
    return dict(splits=len(a), mean=float(np.mean(a)), spread=float(np.std(a)),
                sideways=float(np.mean(np.abs(a) > 60)), against=float(np.mean(a < -20)))


if __name__ == "__main__":
    for f in sys.argv[1:]:
        r = json.load(open(f)); pr = profile(r)
        s = summary(pr)
        print(f"{f.split('/')[-1]:45s} splits {s['splits']:3d} | mean angle {s['mean']:6.1f} spread {s['spread']:5.1f} "
              f"| >60deg {s['sideways']:.2f} | turning against {s['against']:.2f}")
        print("     first angles: " + " ".join(f"{'AB'[p]}{'x2' if k > 1 else ''}:{a:.0f}" for t, p, k, a in pr[:12]))
