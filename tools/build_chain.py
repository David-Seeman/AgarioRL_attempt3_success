"""Greedy chain builder: many consecutive CLEAN cycles of a technique, starting each cycle from the
exact full world state the previous one ended in (ages, pieces in flight -- everything).

    cycle 0: the curriculum solution from its own starting layout
    cycle n: re-tune only the controls (split aims/timings/movement) so that, starting from the real
             end state of cycle n-1, every intended eat happens and nothing else; a soft penalty on how
             much the layout changes over the cycle pushes the chain towards a repeating orbit
The chain's states are saved as an RL start pool (states with the next split + pending eats), and
the per-cycle layout change shows whether it is converging to a periodic splitrun.

    python tools/build_chain.py <catalogue> <index> <curriculum_population.npy> <out_pool.pkl> [--cycles 20]
"""
import argparse, json, os, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import tools.solve_technique as st  # noqa
from splitrun.rlenv import TechSpec, StartPool  # noqa
from splitrun.replay import Recorder, write_index  # noqa
from splitrun.sim import EV_EATER_OWN, EV_EATEN_OWN, EV_MASS, EV_EATER_MASS  # noqa
from splitrun.metrics import _descriptor, _layout_error  # noqa

ARR = ("cf", "ci", "pc", "pcn", "bc", "bcn", "oc", "ocn", "nf", "ni", "nit", "g")


class Tracker:
    """pending intended eats + next split index across the whole chain (for pool snapshots)"""

    def __init__(self, tech):
        self.tech, self.pend, self.nxt, self.par = tech, [], 0, 0

    def on_tick(self, sim, tc, cyc, tev, splits_now):
        tech = self.tech
        sw = int(tech.mirror and cyc % 2 == 1)
        for kk in range(tech.E):
            if tev[kk] == tc:
                for (eo, bo, m, em) in tech.eats[kk]:
                    self.pend.append([eo ^ sw, m * st.TOTAL, em * st.TOTAL, 0])
        ok = True
        for q in range(sim.evn[0]):
            ev = sim.ev[0, q]
            eo = int(ev[EV_EATER_OWN])
            if eo == int(ev[EV_EATEN_OWN]):
                ok = False
                continue
            j = next((j for j, pp in enumerate(self.pend) if pp[0] == eo and abs(pp[1] - ev[EV_MASS]) < 0.03 * pp[1] + 40
                      and abs(pp[2] - ev[EV_EATER_MASS]) < 0.03 * pp[2] + 40), None)
            if j is None:
                ok = False
            else:
                self.pend.pop(j)
        for pp in self.pend:
            pp[3] += 1
        fired = int(np.sum(tev[:tech.E] <= tc))
        self.nxt, self.par = (0, (cyc + 1) % 2) if fired >= tech.E else (fired, cyc % 2)
        return ok


def run_one(tech, v, init_state, cyc0, tracker=None, pool=None, rec_sim=None, every=2):
    """run one cycle of candidate v from init_state; returns (cost, end_state, start_cells, end_cells)"""
    box = {}

    def hook(sim, tc, cyc, tev, alive, splits):
        if not alive[0]:
            return
        if tracker is not None:
            tracker.on_tick(sim, tc[0], cyc[0], tev[0], splits[0])
            if pool is not None and tc[0] % every == 0:
                pool.states.append(({k: getattr(sim, k)[0].copy() for k in ARR}, tracker.nxt, tracker.par,
                                    [tuple(pp) for pp in tracker.pend]))
        if tc[0] + 1 == tev[0, tech.E]:
            box["end"] = {k: getattr(sim, k)[0].copy() for k in ARR}
            box["end_cells"] = sim.cells(0)
        if "start_cells" not in box:
            box["start_cells"] = None
    ends = {}
    c, cyc, rec = st.evaluate(tech, v[None], cycles=1, init_state=init_state, cyc0=cyc0, hook=hook, ends=ends,
                              record=0 if rec_sim else None)
    return c[0], box.get("end"), rec


PRE_LO = np.array([-3000, -3000, -3000, -3000, 0]); PRE_HI = np.array([3000, 3000, 3000, 3000, 100])


def resolve_pre(tech, v, init_state, cyc0, rng, pop=2048, gens=25, log=print):
    """like resolve(), but also searches a repositioning phase (both players' cursors + duration)
    before the cycle's first split, from the fixed start state"""
    D = tech.dim + 5
    LO = np.concatenate([tech.LO, PRE_LO]); HI = np.concatenate([tech.HI, PRE_HI])
    V = rng.uniform(LO, HI, (pop, D))
    V[0, :tech.dim] = v
    ev = lambda X, **kw: st.evaluate(tech, X[:, :tech.dim], cycles=1, init_state=init_state, cyc0=cyc0,
                                     preroll=X[:, tech.dim:], **kw)[0]
    for j in range(1, tech.E + 1):
        fn = lambda X, j=j: ev(X, upto=j, close=False)
        c = fn(V)
        el = V[np.argsort(c)[:128]]; ec = np.sort(c)[:128]
        for g in range(1, gens + 1):
            kids = np.clip(el[rng.integers(0, 128, pop)] + rng.normal(0, 1, (pop, D)) * (HI - LO) * 0.08 * 0.93 ** g, LO, HI)
            ck = fn(kids)
            allV = np.concatenate([el, kids]); allc = np.concatenate([ec, ck])
            o = np.argsort(allc)[:128]; el, ec = allV[o], allc[o]
            if ec[0] < 0.5:
                break
        log(f"      repositioning + split {j}: eats cost {ec[0]:.2f}")
        if ec[0] > 3:
            break
        V = el[rng.integers(0, 128, pop)]; V[:128] = el
    c = ev(el)
    j = int(np.argmin(c))
    return el[j], c[j]


def resolve(tech, v, init_state, cyc0, rng, pop=2048, gens=25, log=print):
    """full per-split curriculum for the controls from a FIXED start state (positions are ignored)"""
    V = rng.uniform(tech.LO, tech.HI, (pop, tech.dim))
    V[0] = v
    for j in range(1, tech.E + 1):
        fn = lambda X, j=j: st.evaluate(tech, X, cycles=1, upto=j, close=False, init_state=init_state, cyc0=cyc0)[0]
        c = fn(V)
        el = V[np.argsort(c)[:128]]; ec = np.sort(c)[:128]
        for g in range(1, gens + 1):
            kids = np.clip(el[rng.integers(0, 128, pop)] + rng.normal(0, 1, (pop, tech.dim)) * (tech.HI - tech.LO) * 0.08 * 0.93 ** g,
                           tech.LO, tech.HI)
            ck = fn(kids)
            allV = np.concatenate([el, kids]); allc = np.concatenate([ec, ck])
            o = np.argsort(allc)[:128]; el, ec = allV[o], allc[o]
            if ec[0] < 0.5:
                break
        V = el[rng.integers(0, 128, pop)]; V[:128] = el
    # among the eat-correct controls, prefer a clean closed cycle with the smallest layout change
    c = st.evaluate(tech, el, cycles=1, init_state=init_state, cyc0=cyc0)[0]
    j = int(np.argmin(c))
    return el[j], c[j]


def retune(tech, v, init_state, cyc0, rng, pop=1024, gens=30, log=print):
    ctrl = np.arange(tech.npos, tech.dim)
    V = np.repeat(v[None], pop, 0)
    V[1:, ctrl] += rng.normal(0, 1, (pop - 1, len(ctrl))) * (tech.HI - tech.LO)[ctrl] * 0.02
    V = np.clip(V, tech.LO, tech.HI)
    fn = lambda X: st.evaluate(tech, X, cycles=1, init_state=init_state, cyc0=cyc0)[0]
    c = fn(V)
    o = np.argsort(c)[:64]; el, ec = V[o], c[o]
    scale, best, stuck = 0.03, ec[0], 0
    for g in range(gens):
        if ec[0] < 4.5:
            break
        kids = el[rng.integers(0, len(el), pop)].copy()
        kids[:, ctrl] += rng.normal(0, 1, (pop, len(ctrl))) * (tech.HI - tech.LO)[ctrl] * scale
        kids = np.clip(kids, tech.LO, tech.HI)
        ck = fn(kids)
        allV = np.concatenate([el, kids]); allc = np.concatenate([ec, ck])
        o = np.argsort(allc)[:64]; el, ec = allV[o], allc[o]
        if ec[0] < best - 1e-3:
            best, stuck = ec[0], 0
        else:
            stuck += 1
            scale = max(scale * 0.8, 0.002)
            if stuck % 6 == 0:
                scale = 0.03
    return el[0], ec[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("catalogue"); ap.add_argument("index", type=int); ap.add_argument("population"); ap.add_argument("out")
    ap.add_argument("--cycles", type=int, default=20); ap.add_argument("--pop", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", type=int, default=0, help="force the last gap to this many ticks (quiet boundary)")
    a = ap.parse_args()
    t = json.load(open(a.catalogue))[a.index]
    tech = st.Technique(t)
    st.ONORBIT = True               # score the layout change of every cycle (push towards a repeat)
    rng = np.random.default_rng(a.seed)
    popu = np.load(a.population)
    c0, _, _ = st.evaluate(tech, popu, cycles=1, upto=tech.E, close=False)
    v = popu[int(np.argmin(c0))].copy()
    if a.quiet:
        gi = tech.npos + 7 * (tech.E - 1) + 2
        tech.HI[gi] = max(tech.HI[gi], a.quiet)
        v[gi] = a.quiet
    tracker, pool = Tracker(tech), StartPool(TechSpec(t))
    state, cyc = None, 0
    ctrl_prev = v[tech.npos:].copy()
    log = []
    t0 = time.time()
    for n in range(a.cycles):
        if n > 0:
            v2, cost = retune(tech, v, state, cyc, rng, a.pop)
            if cost >= 9.5:                    # local re-tune failed: full curriculum from this state
                v2, cost = resolve(tech, v, state, cyc, rng)
            v = v2
        # run the chosen controls once more for real, recording pool states
        snap_before = len(pool.states)
        tr_backup = (list(map(list, tracker.pend)), tracker.nxt, tracker.par)
        cost, end, _ = run_one(tech, v, state, cyc, tracker, pool)
        if end is None or cost >= 9.5:          # failures always add >= 10
            del pool.states[snap_before:]
            print(f"cycle {n}: could not find clean controls (cost {cost:.2f}); chain ends", flush=True)
            break
        dctrl = np.abs(v[tech.npos:] - ctrl_prev).mean() / np.abs(tech.HI - tech.LO)[tech.npos:].mean()
        ctrl_prev = v[tech.npos:].copy()
        print(f"cycle {n}: clean (cost {cost:.2f}); controls changed {dctrl * 100:.1f}% ; pool {len(pool.states)} "
              f"({time.time() - t0:.0f}s)", flush=True)
        log.append(float(cost))
        state, cyc = end, cyc + 1
        pool.save(a.out)
    print(f"chain of {len(log)} clean cycles; pool {len(pool.states)} states -> {a.out}")


if __name__ == "__main__":
    main()
