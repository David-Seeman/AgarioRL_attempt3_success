"""What actually happened in a replay (independent of what the planner intended).

Per split press (player p at tick t, cursor c):
  angle     between the split direction (p's centre of mass -> cursor) and the direction to the partner
  off-circle|angle| > 60 deg
  double    two presses in one tick
  trim      none of the pieces it launched was used (eaten, or grew by eating) within 45 ticks
  between   the cursor sits in the gap between two partner pieces (projection 0.2-0.8 along the segment,
            within 0.35 x its length of it, outside both pieces)
  push      after the no-collision window (13 ticks) one of p's launched pieces touches an OLDER piece of p,
            and that older piece is eaten by the partner within 40 ticks of the split
Downtime: gaps between consecutive split presses (either player).

    python tools/analyze_freerun.py runs/freeruns/*.json
"""
import json, math, sys
import numpy as np


def com(frame, p):
    c = [x for x in frame if x[1] == p]
    m = sum(x[4] ** 2 for x in c)
    return np.array([sum(x[2] * x[4] ** 2 for x in c) / m, sum(x[3] * x[4] ** 2 for x in c) / m]) if m else None


def analyze(r):
    F, I = r["frames"], r["inputs"]
    byid = [{c[0]: c for c in fr} for fr in F]
    presses = []
    for t, inp in enumerate(I):
        for p in (0, 1):
            if inp[p][2]:
                presses.append((t, p, inp[p][2], np.array(inp[p][:2])))
    res = dict(seconds=len(I) / 25, presses=len(presses), doubles=0, triples=0, off_circle=0, trims=0, between=0, pushes=0,
               both=0, crossing=0, flythrough=0, capsplit=0, straight_triples=0)
    ticks_pressed = sorted({t for t, *_ in presses})
    gaps = np.diff([0] + ticks_pressed) if ticks_pressed else np.array([len(I)])
    res["mean_gap_s"] = float(gaps.mean() / 25); res["max_gap_s"] = float(gaps.max() / 25)
    res["both"] = sum(1 for t in ticks_pressed if I[t][0][2] and I[t][1][2])
    angles = []
    for t, p, k, cur in presses:
        fr = F[t]                                   # state the press acted on
        me, other = com(fr, p), com(fr, 1 - p)
        if me is None or other is None:
            continue
        d, a = cur - me, other - me
        ang = math.degrees(math.atan2(a[0] * d[1] - a[1] * d[0], a @ d))
        angles.append(ang)
        res["doubles"] += k == 2
        res["triples"] += k >= 3
        if k >= 3 and t + 16 < len(F):
            # straight? all of the splitter's pieces lie on one line (PCA: sideways spread small compared with
            # their sizes) 12 ticks later, or just before its next split if that comes sooner
            nxt = next((u for u in range(t + 1, len(I)) if I[u][p][2]), len(I))
            P_ = [c for c in F[min(t + 12, nxt)] if c[1] == p]
            if len(P_) >= 6:
                xy = np.array([c[2:4] for c in P_]); xy = xy - xy.mean(0)
                _, sv, vt = np.linalg.svd(xy, full_matrices=False)
                side = np.abs(xy @ vt[1])
                res["straight_triples"] += bool(side.max() < 0.6 * np.mean([c[4] for c in P_]))
        res["capsplit"] += sum(1 for c in fr if c[1] == p) >= 14
        res["off_circle"] += abs(ang) > 60
        # launched pieces = new ids of player p right after the press
        after = F[t + 1] if t + 1 < len(F) else fr
        old_ids = {c[0] for c in fr if c[1] == p}
        new = [c for c in after if c[1] == p and c[0] not in old_ids]
        used = False
        for c in new:
            m0 = c[4] ** 2
            for u in range(t + 1, min(len(F), t + 46)):
                x = byid[u].get(c[0])
                if x is None or x[4] ** 2 > m0 * 1.05:
                    used = True
                    break
            if used:
                break
        res["trims"] += (not used) and len(new) > 0
        # between: some launched piece's flight line (parent -> 820 units towards the cursor) crosses the
        # open gap between two separated partner pieces
        B = [c for c in fr if c[1] != p]
        parents = [c for c in fr if c[1] == p]
        hit = False
        for par in parents:
            o = np.array(par[2:4]); dv = cur - o; n_ = np.linalg.norm(dv)
            if n_ < 1:
                continue
            e_ = o + dv / n_ * 820
            for i in range(len(B)):
                for j in range(i + 1, len(B)):
                    p1, p2 = np.array(B[i][2:4]), np.array(B[j][2:4])
                    L = np.linalg.norm(p2 - p1)
                    if L - B[i][4] - B[j][4] <= 0:
                        continue                                   # touching/overlapping: no gap
                    d1, d2 = e_ - o, p2 - p1
                    den = d1[0] * d2[1] - d1[1] * d2[0]
                    if abs(den) < 1e-9:
                        continue
                    w_ = p1 - o
                    s1 = (w_[0] * d2[1] - w_[1] * d2[0]) / den
                    s2 = (w_[0] * d1[1] - w_[1] * d1[0]) / den
                    if 0 < s1 <= 1 and B[i][4] / L < s2 < 1 - B[j][4] / L:
                        hit = True
        res["between"] += hit
        # push: an older own piece touched by a launched piece moves faster than its cursor could move it,
        # and is eaten by the partner within 40 ticks
        new_ids = {c[0] for c in new}
        pos0 = {c[0]: (c[2], c[3], c[4]) for c in fr if c[1] == p}
        for u in range(t + 13, min(len(F), t + 41)):
            mine = [c for c in F[u] if c[1] == p]
            L_ = [c for c in mine if c[0] in new_ids]
            done = False
            for o in [c for c in mine if c[0] in old_ids]:
                if not any(math.hypot(l_[2] - o[2], l_[3] - o[3]) < l_[4] + o[4] - 5 for l_ in L_):
                    continue
                x0, y0, s0 = pos0[o[0]]
                speed = 88 * s0 ** -0.4396754 * 1.4
                disp = math.hypot(o[2] - x0, o[3] - y0)
                if disp > 1.5 * speed * (u - t) + 60:
                    # eaten soon after?
                    for v in range(u, min(len(F), t + 41)):
                        if o[0] not in byid[v]:
                            res["pushes"] += 1; done = True
                            break
                if done:
                    break
            if done:
                break
    # crossing: both players split in the same tick and their launch lines (big piece -> cursor) cross
    for t in ticks_pressed:
        if not (I[t][0][2] and I[t][1][2]):
            continue
        segs = []
        for p in (0, 1):
            mine = [c for c in F[t] if c[1] == p]
            if not mine:
                break
            big = max(mine, key=lambda c: c[4]); o = np.array(big[2:4]); d = np.array(I[t][p][:2]) - o
            segs.append((o, o + d / max(np.linalg.norm(d), 1e-6) * 820))
        if len(segs) == 2:
            (a0, a1), (b0, b1) = segs
            d1, d2 = a1 - a0, b1 - b0; den = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(den) > 1e-9:
                w_ = b0 - a0
                s1 = (w_[0] * d2[1] - w_[1] * d2[0]) / den; s2 = (w_[0] * d1[1] - w_[1] * d1[0]) / den
                res["crossing"] += 0 <= s1 <= 1 and 0 <= s2 <= 1
    # fly-through: a launched piece's path covers a partner piece it can neither eat nor be eaten by
    # (size ratio < 1.14), and a smaller partner piece further along that path is eaten within 30 ticks
    for t, p, k, cur in presses:
        fr = F[t]; after = F[t + 1] if t + 1 < len(F) else fr
        old_ids = {c[0] for c in fr if c[1] == p}
        for c in [c for c in after if c[1] == p and c[0] not in old_ids]:
            par = min((x for x in fr if x[1] == p), key=lambda x: math.hypot(x[2] - c[2], x[3] - c[3]))
            o = np.array(par[2:4]); d = cur - o; n_ = np.linalg.norm(d)
            if n_ < 1:
                continue
            d = d / n_
            B = [x for x in fr if x[1] != p]
            blockers = [x for x in B if 0 < (np.array(x[2:4]) - o) @ d < 820 and abs(d[0] * (x[3] - o[1]) - d[1] * (x[2] - o[0])) < x[4]
                        and c[4] < 1.14 * x[4] and x[4] < 1.14 * c[4]]
            if not blockers:
                continue
            bd = min((np.array(x[2:4]) - o) @ d for x in blockers)
            targets = {x[0] for x in B if (np.array(x[2:4]) - o) @ d > bd and c[4] >= 1.14 * x[4]}
            if any(i_ not in byid[u] for i_ in targets for u in range(t + 1, min(len(F), t + 31))):
                res["flythrough"] += 1
                break
    res["angles_spread"] = float(np.std(angles)) if angles else 0.0
    return res


if __name__ == "__main__":
    for f in sys.argv[1:]:
        if f.endswith(".info.json"):
            continue
        r = json.load(open(f))
        a = analyze(r)
        print(f"{f.split('/')[-1]:28s} {a['seconds']:5.1f}s presses {a['presses']:3d} | gap mean {a['mean_gap_s']:.2f}s max {a['max_gap_s']:.2f}s | "
              f"off-circle {a['off_circle']:2d} trims {a['trims']:2d} doubles {a['doubles']:2d} between {a['between']:2d} "
              f"pushes {a['pushes']:2d} both {a['both']:2d} crossing {a['crossing']:2d} triples {a['triples']:2d} "
              f"flythrough {a['flythrough']:2d} capsplit {a['capsplit']:2d} straight-triples {a['straight_triples']:2d} | angle spread {a['angles_spread']:.0f}")
