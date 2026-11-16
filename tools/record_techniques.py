"""Record every solved technique as a replay with a plain-language description.

    python tools/record_techniques.py runs/techniques/catalogue_s3_e2_p8.json
Uses runs/techniques/<catalogue>_t<k>_elite.npy written by tools/solve_technique.py; picks the
elite that chains longest; writes runs/techniques/<name>.json and a summary table.
"""
import json, os, re, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.solve_technique import Technique, chain_test, TOTAL  # noqa
from splitrun.metrics import splitrun_verdict  # noqa
from splitrun.replay import write_index  # noqa


def describe(t):
    """plain words for one cycle, from the catalogue's log + eat masses"""
    lines = []
    fmt = lambda m: f"{m * TOTAL:,.0f}"
    for k, ((actor, kind), log, eats) in enumerate(zip(t["seq"], t["log"], t["eats"])):
        who = "AB"[actor]
        parts = []
        if "->" not in log:                    # sampled techniques: masses only
            parts = [f"{'AB'[a]}'s {em * TOTAL:,.0f} eats {'AB'[b]}'s {m * TOTAL:,.0f}" for a, b, m, em in eats]
            lines.append(f"{who} {'single' if kind == 1 else 'double'}-splits; " + "; ".join(parts) + ".")
            continue
        pairs = re.findall(r"(\S+)->eats (\S+)", log.split(": ", 1)[1])
        for (p, q), (a, b, m, *_) in zip(pairs, eats):
            launched_p, launched_q = "." in p, "." in q
            eater = f"{'AB'[a]}'s {'just-launched ' if launched_p and p.startswith(who) else 'launched ' if launched_p else ''}piece"
            eaten = f"{'AB'[b]}'s {'launched piece' if launched_q else 'backpiece'} ({fmt(m)})"
            parts.append(f"{eater} eats {eaten}")
        lines.append(f"{who} {'single' if kind == 1 else 'double'}-splits; " + "; ".join(parts) + ".")
    return " ".join(lines)


def main(cat_path, only=None):
    techs = json.load(open(cat_path))
    base = os.path.basename(cat_path)[:-5]
    out = os.path.join(ROOT, "runs", "techniques")
    rows = []
    is_base = base == "catalogue_s3_e2_p8"
    for k, t in enumerate(techs):
        if only is not None and k not in only:
            continue
        f = os.path.join(out, f"{base}_t{k}_elite.npy")
        if not os.path.exists(f):
            continue
        tech = Technique(t)
        elite = np.load(f)
        best = None
        for j in range(min(16, len(elite))):
            n, c, _ = chain_test(tech, elite[j], 60)
            if best is None or n > best[0] or (n == best[0] and c < best[1]):
                best = (n, c, j)
        if best[0] < 30 or (is_base and k == 5):   # unsolved geometry, or base #5 (= the 2+3 as two turns)
            print(f"technique {k}: skipped (clean cycles {best[0]})")
            continue
        v = elite[best[2]]
        period = sum(int(round(v[tech.npos + 7 * e + 2])) for e in range(tech.E))
        n, _, rec = chain_test(tech, v, 60, record=0)
        n_splits = len(t["seq"])
        name = f"technique_{k}" if is_base else f"{n_splits}split_{base.replace('random_', '').replace('_ranked', '')}_{k}"
        start = ("A: " + " + ".join(f"{m * TOTAL:,.0f}" for m in t["masses_a"]) + " | B: "
                 + " + ".join(f"{m * TOTAL:,.0f}" for m in t["masses_b"]))
        verdict = splitrun_verdict(rec.data)
        stab = f"{t['rho']:.2f}" if is_base else "self-correcting"
        rec.data["name"] = name
        rec.data["description"] = (f"{name} ({'mirror' if t['mirror'] else 'repeats itself'} every "
                                   f"{len(t['seq'])} split{'s' if len(t['seq']) > 1 else ''}; cycle {period} ticks = "
                                   f"{period / 25:.2f} s; mass stability {stab}). Start {start}. {describe(t)} "
                                   f"Clean cycles in this run: {n}/60.")
        rec.save(os.path.join(out, name + ".json"))
        rows.append(dict(k=k, clean=n, period=period, splits=len(t["seq"]), rho=t["rho"], mirror=t["mirror"],
                         start=start, rotations=verdict["rotations"], drift=verdict["centre_drift"],
                         splitrun=verdict["splitrun"], text=describe(t)))
        print(f"technique {k}: clean {n}/60, cycle {period} ticks, rho {t['rho']:.2f}, verdict {verdict}", flush=True)
    json.dump(rows, open(os.path.join(out, f"{base}_summary.json"), "w"), indent=1)
    write_index(os.path.join(ROOT, "runs"))


if __name__ == "__main__":
    main(sys.argv[1], [int(x) for x in sys.argv[2:]] or None)
