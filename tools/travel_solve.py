"""Solve a catalogue technique as a TRANSLATING periodic splitrun from scratch: cycle frames with a fixed world
orientation (tools/solve_technique.WORLD_FRAME), closing = the same layout shifted (mirror cycles may reflect
sideways), reward TRANS_W x net travel per tick. Periodic, so sustainable by construction.

    python tools/travel_solve.py runs/techniques/catalogue_s3_e2_p8.json 0 --trans 0.5
"""
import argparse, json, os, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import tools.solve_technique as st  # noqa
from tools.translate_technique import speed_of  # noqa
from splitrun.replay import write_index  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("catalogue"); ap.add_argument("which", type=int)
    ap.add_argument("--trans", type=float, default=0.5); ap.add_argument("--pop", type=int, default=4096)
    ap.add_argument("--gens", type=int, default=30); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-gap", type=int, default=16)
    ap.add_argument("--strict", type=float, default=0); ap.add_argument("--local", action="store_true"); ap.add_argument("--robust", type=int, default=1)
    a = ap.parse_args()
    st.WORLD_FRAME = True
    st.TRANS_W = a.trans
    st.STRICT_LAY = a.strict
    st.LOCAL_FRAMES = a.local; st.ROBUST_K = a.robust
    techs = json.load(open(a.catalogue)); tech = st.Technique(techs[a.which])
    gi = [tech.npos + 7 * e + 2 for e in range(tech.E)]
    tech.HI[gi] = a.max_gap
    name = f"wtravel_t{a.which}_w{a.trans}_s{a.seed}{'_local' if a.local else ''}{'_r%d' % a.robust if a.robust > 1 else ''}{'_strict' if a.strict else ''}"
    t0 = time.time()
    print(f"{name}: {'MIRROR' if tech.mirror else 'IDENTITY'} {tech.seq}", flush=True)
    elite, ec = st.solve(tech, a.pop, a.gens, seed=a.seed)
    out = os.path.join(ROOT, "runs", "travel"); os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, name + ".npy"), elite)
    res = []
    for j in range(min(24, len(elite))):
        n, sp, _ = speed_of(tech, elite[j], cycles=60)
        per = sum(int(round(elite[j][g])) for g in gi)
        res.append((n, sp, per, j))
    res.sort(key=lambda r: (-(r[0] >= 60), -r[1]))
    for n, sp, per, j in res[:6]:
        print(f"   elite #{j}: {n}/60 clean cycles, {sp:.0f} units/s, cycle {per} ticks", flush=True)
    n, sp, per, j = res[0]
    n, sp, rec = speed_of(tech, elite[j], cycles=100)
    rec.data["name"] = name
    rec.data["description"] = (f"Translating periodic splitrun (technique {a.which}, world-fixed frame): {n}/100 clean cycles, "
                               f"{sp:.0f} units/s, cycle {per} ticks.")
    os.makedirs(os.path.join(ROOT, "runs", "dynamic"), exist_ok=True)
    rec.save(os.path.join(ROOT, "runs", "dynamic", name + ".json"))
    write_index(os.path.join(ROOT, "runs"))
    print(f"FINAL {name}: {n}/100 cycles, {sp:.0f} units/s, cycle {per} ticks ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
