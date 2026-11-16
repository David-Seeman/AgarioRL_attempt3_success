"""Rebuild runs/freeruns/catalogue.json (the viewer's "Non-periodic catalogue" tab) from the replays."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.analyze_freerun import analyze  # noqa
from splitrun.replay import write_index  # noqa

META = {
 "small_doubles":    ("Small 3.5k", "Double splits (3.5k)", "Total mass 3,500. Double splits most of the time, with a few sideways trims."),
 "small_triples":    ("Small 3.5k", "Triple splits (3.5k)", "Total mass 3,500. Ordinary triple splits (NOT straight: fired from several cells side by side, and the cursor moved on right after), plus doubles."),
 "small_cross":      ("Small 3.5k", "Crossing + fly-through (3.5k)", "Total mass 3,500. Both players often split at each other in the same tick; launched pieces fly over a same-size partner piece to eat a smaller one behind it."),
 "medium_doubles":   ("Medium 10k", "Double splits + trims (10k)", "Total mass 10,000. Double splits, and trims aimed sideways / slightly back whose pieces the partner eats soon after."),
 "medium_triples":   ("Medium 10k", "Triples + trims (10k)", "Total mass 10,000. Ordinary (not straight) triple splits, doubles, sideways trims and crossing splits."),
 "medium_flythrough":("Medium 10k", "Fly-through (10k)", "Total mass 10,000. Fly-through shots and crossing splits; the shortest downtime of the catalogue."),
 "straight_triples": ("Standard 37k", "Straight triple splits (37k)", "Straight triple splits: fired from 1 cell (or 2 lined-up cells) with the cursor held far along that line for ~1 s (like 'n' in imsolo), so all 8 pieces lie on one line. Opening triple by A; a second one by B at 12 s. Plus many double splits."),
 "tempo":      ("Standard 37k", "Tempo circle", "Least downtime among the 37k runs; circle shots with some between-gap and off-circle splits."),
 "trims":      ("Standard 37k", "Trim-heavy", "Many off-circle trims (70-180 deg from the partner) to shrink the splitter's pieces."),
 "doubles":    ("Standard 37k", "Double splits", "About every second split is a double split, aimed at a chosen partner piece."),
 "between":    ("Standard 37k", "Between the pieces", "Splits aimed through the open gap between two partner pieces."),
 "push":       ("Standard 37k", "Aimed into own pieces", "Splits aimed into the splitter's own pieces (made before the push-split clarification; not a true push-split study)."),
 "both":       ("Standard 37k", "Simultaneous", "Both players often split in the same tick."),
 "mixed":      ("Standard 37k", "Mixed bag", "Trims, double splits, between-gap and simultaneous splits."),
 "imbalanced": ("Standard 37k", "Imbalanced (~2.7x)", "The smaller player holds about 27% of the mass (the other ~2.7x)."),
 "large_45k":  ("Large 45k+", "Large 45k", "Total mass 45,000; pieces peak just under the 22,500 autosplit limit. Double splits and sideways trims."),
 "large_60k":  ("Large 45k+", "Large 60k with autosplits", "Total mass 60,000; pieces grow past 22,500 and autosplit. Circle shots, between-gap splits, trims."),
 "large_cap":  ("Large 45k+", "Large 60k, many cells", "Total mass 60,000, about 9.5 cells per player and many double splits; aimed for selective splits at the 16-cell cap but managed only one."),
}


def autosplits(r):
    return sum(1 for f in r["frames"][:-1] for c in f if c[4] ** 2 / 100 > 22500)


def main():
    cat = []
    for k, (grp, title, blurb) in META.items():
        p = os.path.join(ROOT, "runs", "freeruns", k + ".json")
        if not os.path.exists(p):
            continue
        r = json.load(open(p)); a = {kk: (v.item() if hasattr(v, "item") else v) for kk, v in analyze(r).items()}
        info = json.load(open(p.replace(".json", ".info.json")))
        sh = [min(sum(c[4] ** 2 for c in f if c[1] == q) for q in (0, 1)) / sum(c[4] ** 2 for c in f) for f in r["frames"]]
        if k == "straight_triples":
            title = f"Straight triple splits (37k)"
        a.update(path=f"freeruns/{k}.json", title=title, blurb=blurb, style=info.get("style", {}), group=grp,
                 small_share=sum(sh) / len(sh), autosplits=autosplits(r), total_mass=round(sum(c[4] ** 2 / 100 for c in r["frames"][0])))
        extra = f" Autosplits: {a['autosplits']}." if a["autosplits"] else ""
        r["name"] = k
        r["description"] = (f"{title}: {a['seconds']:.1f} s, {a['presses']} split presses, downtime {a['mean_gap_s']:.2f} s average "
                            f"(max {a['max_gap_s']:.2f} s). Double {a['doubles']}, triple {a['triples']}, off-circle {a['off_circle']}, "
                            f"between-gap {a['between']}, crossing {a['crossing']}, fly-through {a['flythrough']}, simultaneous {a['both']}, "
                            f"verified straight triples {a['straight_triples']}.{extra} " + blurb)
        json.dump(r, open(p, "w"), separators=(",", ":"))
        cat.append(a)
    json.dump(cat, open(os.path.join(ROOT, "runs", "freeruns", "catalogue.json"), "w"), indent=1)
    write_index(os.path.join(ROOT, "runs"))
    print(len(cat), "catalogue entries")


if __name__ == "__main__":
    main()
