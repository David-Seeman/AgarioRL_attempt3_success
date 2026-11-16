"""Turn a solved stationary splitrun technique into a TRANSLATING one by continuation.

Start from the technique's solved geometry, replace the stay-stationary drift penalty by a reward for net
displacement per tick (tools/solve_technique.TRANS_W), raise it step by step and re-optimise; keep only
solutions that still chain 100 clean cycles. Reports speed (units/s) and saves a replay of the best.

    python tools/translate_technique.py runs/techniques/catalogue_s3_e2_p8.json 0
"""
import json, os, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import tools.solve_technique as st  # noqa
from splitrun.replay import write_index  # noqa


def speed_of(tech, v, cycles=40):
    n, c, rec = st.chain_test(tech, v, cycles=cycles, record=0)
    F = rec.data["frames"]
    com = lambda f: np.array([sum(c_[2] * c_[4] ** 2 for c_ in f), sum(c_[3] * c_[4] ** 2 for c_ in f)]) / sum(c_[4] ** 2 for c_ in f)
    d = np.linalg.norm(com(F[-1]) - com(F[0]))
    return n, d / (len(F) / 25), rec


def main(cat, k, seed=0):
    techs = json.load(open(cat)); tech = st.Technique(techs[k])
    base = os.path.basename(cat)[:-5]
    el = np.load(os.path.join(ROOT, "runs", "techniques", f"{base}_t{k}_elite.npy"))
    tech.HI[[tech.npos + 7 * e + 2 for e in range(tech.E)]] = np.maximum(tech.HI[[tech.npos + 7 * e + 2 for e in range(tech.E)]], el[:, [tech.npos + 7 * e + 2 for e in range(tech.E)]].max(0) + 2)
    rng = np.random.default_rng(seed)
    n0, v0, _ = speed_of(tech, el[0])
    print(f"start: chains {n0}/40 cycles, speed {v0:.0f} units/s", flush=True)
    best = (v0, el[0].copy())
    V = el.copy()
    for w in (0.5, 1.0, 2.0, 4.0, 8.0):
        st.TRANS_W = w
        fn = lambda X: st.evaluate(tech, X, cycles=10)[0]
        el2, ec = st.evolve(tech, np.concatenate([V] * 8)[:1024], fn, rng, 25, -1e9, scale0=0.02, label=f"w={w}")
        ok = []
        for v in el2[:12]:
            n, sp, _ = speed_of(tech, v)
            ok.append((n, sp, v))
        good = [(sp, v) for n, sp, v in ok if n >= 40]
        if good:
            sp, v = max(good, key=lambda x: x[0])
            print(f"TRANS_W {w}: best chaining solution {sp:.0f} units/s ({len(good)}/12 chain 40 cycles)", flush=True)
            if sp > best[0]:
                best = (sp, v.copy())
            V = el2
        else:
            print(f"TRANS_W {w}: nothing chains 40 cycles any more -- stop", flush=True)
            break
    sp, v = best
    n, sp2, rec = speed_of(tech, v, cycles=100)
    rec.data["name"] = f"translating_{base}_t{k}"
    rec.data["description"] = (f"Technique {k} turned into a translating splitrun by continuation: {n}/100 clean cycles, "
                               f"{sp2:.0f} units/s net travel (periodic, so sustainable by construction).")
    os.makedirs(os.path.join(ROOT, "runs", "dynamic"), exist_ok=True)
    rec.save(os.path.join(ROOT, "runs", "dynamic", f"translating_t{k}.json"))
    np.save(os.path.join(ROOT, "runs", "dynamic", f"translating_t{k}.npy"), v)
    write_index(os.path.join(ROOT, "runs"))
    print(f"FINAL technique {k}: {n}/100 cycles, {sp2:.0f} units/s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 0)
