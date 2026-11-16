"""Group techniques into structurally different FAMILIES and pick one representative each.

Family key: the order in which players split (canonical under A/B relabelling and cyclic re-start;
mirror techniques use their full cycle), which splits are doubles, and piece counts. Two players
simply alternating (ABAB...) are the 'splits in a circle' family; others (AABB, AAAB, AABA, with
doubles, ...) are structurally different.

    python tools/technique_families.py runs/techniques/random_e4_ranked.json [...more files]
"""
import json, sys
from collections import defaultdict


def canon_seq(seq, mirror):
    s = [(a, k) for a, k in seq]
    if mirror:                                       # full cycle = sequence, then sequence with A<->B swapped
        s = s + [(1 - a, k) for a, k in seq]
    words = []
    n = len(s)
    for r in range(n):
        rot = s[r:] + s[:r]
        for swap in (0, 1):
            words.append("".join(("AB"[a ^ swap]).lower() if k == 2 else "AB"[a ^ swap] for a, k in rot))
    return min(words)


def family(t):
    seq = canon_seq(t["seq"], t["mirror"])
    return (seq, "mirror" if t["mirror"] else "identity", tuple(sorted([len(t["masses_a"]), len(t["masses_b"])])))


def is_circle(key):
    s = key[0].upper()
    return all(s[i] != s[i + 1] for i in range(len(s) - 1)) and s[0] != s[-1]


if __name__ == "__main__":
    fams = defaultdict(list)
    for f in sys.argv[1:]:
        for i, t in enumerate(json.load(open(f))):
            fams[family(t)].append((f, i, t))
    rows = sorted(fams.items(), key=lambda kv: (is_circle(kv[0]), -len(kv[1])))
    print(f"{len(rows)} families ({sum(1 for k, _ in rows if not is_circle(k))} not simple alternation)")
    for key, members in rows[:40]:
        f, i, t = max(members, key=lambda m: m[2]["margin"])
        print(f"{'circle ' if is_circle(key) else 'DISTINCT'} {key[0]:14s} {key[1]:8s} pieces {key[2]}  "
              f"members {len(members):3d}  best: {f.split('/')[-1]}#{i} margin {t['margin']:.3f}")
