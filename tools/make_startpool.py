"""Build an RL start pool for a technique from scripted solutions (tools/solve_technique.py elites).

Runs the scripted solution and snapshots the FULL world state (positions, momentum, ages) at many
moments, together with which split comes next, the role parity, and the intended eats still pending.
Only snapshots from clean stretches are kept (after a warm-up cycle when the solution chains).

    python tools/make_startpool.py <catalogue.json> <index> <elite.npy> <out.pkl> [--cycles 40]
"""
import argparse, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import tools.solve_technique as st  # noqa
from splitrun.rlenv import TechSpec, StartPool  # noqa
from splitrun.sim import EV_EATER_OWN, EV_EATEN_OWN, EV_MASS, EV_EATER_MASS  # noqa


def build(cat_path, k, elite_path, cycles=40, n_elite=8, every=3):
    t = json.load(open(cat_path))[k]
    tech = st.Technique(t)
    spec = TechSpec(t)
    pool = StartPool(spec)
    el = np.load(elite_path)[:n_elite]
    N = len(el)
    E = tech.E
    pend = [[] for _ in range(N)]
    snaps = [[] for _ in range(N)]
    state = {"t": 0}

    def hook(sim, tc, cyc, tev, alive, splits):
        for i in range(N):
            if not alive[i]:
                continue
            sw = int(tech.mirror and cyc[i] % 2 == 1)
            # splits pressed on this tick -> their intended eats become pending (player terms)
            for kk in range(E):
                if tev[i, kk] == tc[i]:
                    for (eo, bo, m, em) in tech.eats[kk]:
                        pend[i].append([eo ^ sw, m * st.TOTAL, em * st.TOTAL, 0])
            for q in range(sim.evn[i]):
                ev = sim.ev[i, q]
                eo = int(ev[EV_EATER_OWN])
                if eo == int(ev[EV_EATEN_OWN]):
                    continue
                for j, pp in enumerate(pend[i]):
                    if pp[0] == eo and abs(pp[1] - ev[EV_MASS]) < 0.03 * pp[1] + 40 and abs(pp[2] - ev[EV_EATER_MASS]) < 0.03 * pp[2] + 40:
                        pend[i].pop(j)
                        break
            for pp in pend[i]:
                pp[3] += 1
            fired = int(np.sum(tev[i, :E] <= tc[i]))
            if fired >= E:
                nxt, par = 0, (cyc[i] + 1) % 2
            else:
                nxt, par = fired, cyc[i] % 2
            if (cyc[i] >= 1 or cycles == 1) and state["t"] % every == 0:
                snaps[i].append(({kk: getattr(sim, kk)[i].copy() for kk in pool_arrays}, nxt, par,
                                 [tuple(pp) for pp in pend[i]]))
        state["t"] += 1

    pool_arrays = ("cf", "ci", "pc", "pcn", "bc", "bcn", "oc", "ocn", "nf", "ni", "nit", "g")
    cost, done, _ = st.evaluate(tech, el, cycles=cycles, hook=hook)
    kept = 0
    for i in range(N):
        good = done[i] >= (cycles if cycles > 1 else 1)
        if not good and cycles > 1 and done[i] >= 3:
            # keep the clean part: drop the last cycle before the failure
            good = True
            snaps[i] = snaps[i][: int(len(snaps[i]) * (done[i] - 1) / max(done[i], 1))]
        if good:
            for s_ in snaps[i]:
                pool.states.append(s_)
                kept += 1
    return pool, kept, done


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("catalogue"); ap.add_argument("index", type=int); ap.add_argument("elite"); ap.add_argument("out")
    ap.add_argument("--cycles", type=int, default=40)
    a = ap.parse_args()
    pool, kept, done = build(a.catalogue, a.index, a.elite, a.cycles)
    pool.save(a.out)
    print(f"{kept} start states from elites with clean cycles {list(done)} -> {a.out}")


def build_from_replay(cat_path, k, replay_path, every=3, warmup_cycles=1):
    """Re-simulate a recorded replay (deterministic) and snapshot states + tracker info, as long as
    every split press and every eat matches the technique."""
    from splitrun.sim import Sim
    t = json.load(open(cat_path))[k]
    tech = st.Technique(t)
    spec = TechSpec(t)
    pool = StartPool(spec)
    rep = json.load(open(replay_path))
    sim = Sim(1)
    rows = [[c[1], c[2], c[3], c[4], 100, 0, 0, 0] for c in sorted(rep["frames"][0], key=lambda c: c[0])]
    sim.load(0, rows, seed=1)
    E = tech.E
    nxt, cyc, pend, t_ = 0, 0, [], 0
    arrays = ("cf", "ci", "pc", "pcn", "bc", "bcn", "oc", "ocn", "nf", "ni", "nit", "g")
    ok_cycles = 0
    for inp in rep["inputs"]:
        mouse = np.array([[inp[0][0], inp[0][1]], [inp[1][0], inp[1][1]]], float)[None]
        spl = np.array([[inp[0][2], inp[1][2]]], np.int64)
        sw = int(tech.mirror and cyc % 2 == 1)
        # presses, in technique order
        pressed = [int(spl[0, 0]), int(spl[0, 1])]
        for _ in range(2):
            if pressed == [0, 0]:
                break
            p_exp = tech.seq[nxt][0] ^ sw
            if pressed[p_exp] != tech.seq[nxt][1]:
                break
            pressed[p_exp] = 0
            for (eo, bo, m, em) in tech.eats[nxt]:
                pend.append([eo ^ sw, m * st.TOTAL, em * st.TOTAL, 0])
            nxt += 1
            if nxt == E:
                nxt, cyc = 0, cyc + 1
                ok_cycles += 1
                sw = int(tech.mirror and cyc % 2 == 1)
        if pressed != [0, 0]:
            print("   replay: a split press does not match the technique; stopping")
            break
        sim.step(mouse, spl)
        bad = False
        for ev in sim.events(0):
            eo = int(ev[EV_EATER_OWN])
            if eo == int(ev[EV_EATEN_OWN]):
                bad = True
                break
            j = next((j for j, pp in enumerate(pend) if pp[0] == eo and abs(pp[1] - ev[EV_MASS]) < 0.03 * pp[1] + 40
                      and abs(pp[2] - ev[EV_EATER_MASS]) < 0.03 * pp[2] + 40), None)
            if j is None:
                bad = True
                break
            pend.pop(j)
        if bad or any(pp[3] > 60 for pp in pend):
            print(f"   replay diverged from the technique after {ok_cycles} cycles; stopping")
            break
        for pp in pend:
            pp[3] += 1
        if ok_cycles >= warmup_cycles and t_ % every == 0:
            pool.states.append(({kk: getattr(sim, kk)[0].copy() for kk in arrays}, nxt, cyc % 2,
                                [tuple(pp) for pp in pend]))
        t_ += 1
    return pool, ok_cycles
