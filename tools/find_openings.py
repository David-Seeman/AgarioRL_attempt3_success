"""Openings: from two single 18.5k cells, which short split/eat sequences reach a technique's
starting masses?  (mass bookkeeping only, same rules as tools/technique_catalogue.py; here every
mass is known, so each candidate sequence is simply checked.)

    python tools/find_openings.py runs/techniques/catalogue_s3_e2_p8.json [max_events]
"""
import json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.technique_catalogue import Pc, split, eat_sets  # noqa

RATIO = 1.3
TOTAL = 37000.0
TOL = 0.15          # every piece within 15% of the technique's mass (the technique then self-corrects)


def run(max_events, targets):
    """targets: list of (index, sorted masses A, sorted masses B). Returns {index: [openings]}"""
    found = {i: [] for i, _, _ in targets}
    start = [Pc(0, np.array([0.5]), -5, "A"), Pc(1, np.array([0.5]), -5, "B")]

    def matches(pieces):
        ma = sorted([p.coef[0] for p in pieces if p.owner == 0], reverse=True)
        mb = sorted([p.coef[0] for p in pieces if p.owner == 1], reverse=True)
        hits = []
        for i, ta, tb in targets:
            for swap in (False, True):
                a, b = (mb, ma) if swap else (ma, mb)
                if len(a) == len(ta) and len(b) == len(tb):
                    err = max(np.max(np.abs(np.array(a) - ta) / ta), np.max(np.abs(np.array(b) - tb) / tb))
                    if err <= TOL:
                        hits.append((i, swap, err))
        return hits

    def rec(pieces, ev, log):
        for i, swap, err in matches(pieces):
            found[i].append((list(log), swap, err))
        if ev == max_events:
            return
        for actor in (0, 1):
            for k in (1, 2):
                sp = split(pieces, actor, k, ev)
                if len(sp) > 9:
                    continue
                opts = eat_sets(sp, ev) + [[]]          # an opening split may be pure setup
                for eats in opts:
                    ok, gain, eaten = True, {}, set()
                    for a, b in eats:
                        ma = sp[a].coef[0] + gain.get(a, 0)
                        if ma < RATIO * sp[b].coef[0]:
                            ok = False
                            break
                        gain[a] = gain.get(a, 0) + sp[b].coef[0]
                        eaten.add(b)
                    if not ok:
                        continue
                    after = [Pc(p.owner, p.coef + gain.get(j, 0), p.gen, p.name) for j, p in enumerate(sp) if j not in eaten]
                    desc = ", ".join(f"{'AB'[sp[a].owner]} {sp[a].coef[0] * TOTAL:,.0f} eats {'AB'[sp[b].owner]} "
                                     f"{sp[b].coef[0] * TOTAL:,.0f}" for a, b in eats) or "no eat (setup)"
                    rec(after, ev + 1, log + [f"{'AB'[actor]} {'single' if k == 1 else 'double'} split: {desc}"])

    rec(start, 0, [])
    return found


if __name__ == "__main__":
    techs = json.load(open(sys.argv[1]))
    me = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    targets = [(i, sorted(t["masses_a"], reverse=True), sorted(t["masses_b"], reverse=True)) for i, t in enumerate(techs)]
    found = run(me, targets)
    for i, t in enumerate(techs):
        ops = sorted(found[i], key=lambda o: (len(o[0]), o[2]))
        print(f"\ntechnique {i}: start A " + " + ".join(f"{m * TOTAL:,.0f}" for m in t["masses_a"]) + " | B "
              + " + ".join(f"{m * TOTAL:,.0f}" for m in t["masses_b"]) + f"  -> {len(ops)} openings with <= {me} splits")
        for log, swap, err in ops[:3]:
            print(f"   [worst piece off by {err * 100:.0f}%] " + ("(roles swapped) " if swap else "") + " ; ".join(log))
