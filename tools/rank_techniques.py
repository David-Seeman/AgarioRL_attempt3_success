"""Rank sampled techniques for geometry solving: purposeful, simple, alternating, robust.

    python tools/rank_techniques.py runs/techniques/random_e4.json [top]
Writes <file>_ranked.json (deduplicated: A/B relabelling and cyclic re-starts collapse)."""
import json, re, sys
import numpy as np


def launched_used(t):
    """fraction of launched pieces that eat or are eaten in the event that launched them or the next"""
    # the log records eats; the catalogue format lost piece identities, so use counts:
    # launched pieces per event = actor pieces * (2^k - 1); pieces involved in eats <= 2 per eat
    return None


def key(t):
    ma = tuple(sorted(np.round(np.array(t["masses_a"]) * 400).astype(int)))
    mb = tuple(sorted(np.round(np.array(t["masses_b"]) * 400).astype(int)))
    pieces = tuple(sorted([ma, mb]))
    ev = tuple(sorted(tuple(sorted(np.round(np.array([m for _, _, m, _ in e]) * 400).astype(int))) for e in t["eats"]))
    return (pieces, ev, len(t["seq"]))


def plausible(t, max_eats=3, max_chain=2):
    """geometric plausibility: at most `max_eats` eats per split, and no piece eats more than
    `max_chain` pieces in one split (an eater's mass grows by what it ate: em2 = em1 + m1)"""
    for ev in t["eats"]:
        if len(ev) > max_eats:
            return False
        chains = []
        for (eo, bo, m, em) in ev:
            for c in chains:
                if c["owner"] == eo and abs(c["mass"] - em) < 1e-4:
                    c["n"] += 1
                    c["mass"] = em + m
                    break
            else:
                chains.append({"owner": eo, "mass": em + m, "n": 1})
        if max(c["n"] for c in chains) > max_chain:
            return False
    return True


def score(t):
    n = len(t["masses_a"]) + len(t["masses_b"])
    alt = sum(1 for i in range(1, len(t["seq"])) if t["seq"][i][0] != t["seq"][i - 1][0]) / max(1, len(t["seq"]) - 1)
    eats = sum(len(e) for e in t["eats"])
    doubles = sum(1 for a, k in t["seq"] if k == 2)
    # every split should have a purpose: eats per split, ideally >= 2 (its launched piece + a response)
    return (-n, alt, min(eats / len(t["seq"]), 2.0), t["margin"], -doubles)


if __name__ == "__main__":
    techs = json.load(open(sys.argv[1]))
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    uniq = {}
    for t in techs:
        k = key(t)
        if k not in uniq or t["margin"] > uniq[k]["margin"]:
            uniq[k] = t
    ranked = sorted([t for t in uniq.values() if plausible(t)], key=score, reverse=True)
    print(f"{len(ranked)} plausible (<= 3 eats per split, no piece eating more than 2 at once)")
    json.dump(ranked, open(sys.argv[1][:-5] + "_ranked.json", "w"), indent=1)
    print(f"{len(techs)} -> {len(uniq)} after merging relabellings/re-starts")
    for i, t in enumerate(ranked[:top]):
        print(f"\n[{i}] {'MIRROR' if t['mirror'] else 'IDENTITY'} splits={len(t['seq'])} pieces={len(t['masses_a'])}+{len(t['masses_b'])} "
              f"margin={t['margin']:.3f}  start A " + " + ".join(f"{m * 37000:,.0f}" for m in t["masses_a"])
              + " | B " + " + ".join(f"{m * 37000:,.0f}" for m in t["masses_b"]))
        for l in t["log"]:
            print("    " + l)
