"""Closed-loop geometry for any technique (mass pattern + split sequence).

Instead of replaying fixed aim points at fixed times, every tick the controller looks up the ACTUAL
pieces it needs (by mass) and reacts to where they are:

  for the next event k (actor X, partner Y), its PRIMARY eat names two pieces:
      splitter  = X's piece that is about to split (mass = 2 x its launched half)
      target    = the partner piece the launched half must eat, or (fed) must be eaten by
  approach:   X steers towards a standoff point next to the target: distance `stand`, at angle
              `theta` from the target's outward direction (target minus Y's centre of mass)
  trigger:    X splits as soon as |splitter - target| <= `trig` (and at least `gmin` ticks after the
              previous split) -- the split happens when the geometry is right, not at a fixed time
  aim:        cursor = target + `ext` * (unit vector splitter->target) + `perp` * (sideways)
  partner:    Y's cursor = Y's centre of mass + (`px`, `py`) in the same target frame

Per event 7 parameters (theta, stand, trig, ext, perp, px, py) + a global min gap `gmin`.
Starting positions are also searched (the controller then corrects them within the first cycle).

Scoring over CYCLES cycles; a world stops as soon as it goes wrong (unintended eat, own merge,
a split that never triggers, a missing piece), which prunes bad candidates early:
  10 per missing/unintended eat, layout change between consecutive cycle starts (capped, so a
  closed cycle always beats a broken one), centre drift, and TIME_W * mean cycle length.
"""
import argparse, json, math, os, sys, time
import numpy as np
from numba import njit
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from splitrun.sim import Sim, X, Y, SZ, EV_EATER_OWN, EV_EATEN_OWN, EV_MASS, EV_EATER_MASS  # noqa
from splitrun.metrics import _descriptor, _layout_error  # noqa

TOTAL = 37000.0
TIME_W = 0.02
NPE = 7                     # params per event
TIMEOUT = 90                # ticks an event may wait for its trigger


class Tech:
    def __init__(self, t):
        self.t = t
        self.ma, self.mb = t["masses_a"], t["masses_b"]
        self.n = len(self.ma) + len(self.mb)
        self.seq = t["seq"]
        self.E = len(self.seq)
        self.mirror = t["mirror"]
        self.eats = t["eats"]
        # primary eat of each event: the first eat involving the actor's freshly launched half.
        # launched-half mass = eater mass (actor eats) or eaten mass (actor is fed); splitter = 2^k x that
        self.prim = np.zeros((self.E, 4))          # [kind(0 eats,1 fed), splitter mass, target mass, k]
        for k, ((actor, kk), ev) in enumerate(zip(self.seq, self.eats)):
            pick = None
            for (eo, bo, m, em) in ev:
                if eo == actor:
                    pick = (0, em * 2 ** kk, m)
                    break
                if bo == actor:
                    pick = (1, m * 2 ** kk, em)
                    break
            if pick is None:
                eo, bo, m, em = ev[0]
                pick = (0, em, m)
            self.prim[k] = [pick[0], pick[1] * TOTAL, pick[2] * TOTAL, kk]
        self.actors = np.array([a for a, _ in self.seq], np.int64)
        self.npos = 2 * (self.n - 1)
        self.dim = self.npos + NPE * self.E + 1
        lo = [-2800] * self.npos + [-math.pi, 300, 400, -600, -900, -3000, -3000] * self.E + [0]
        hi = [2800] * self.npos + [math.pi, 2500, 1900, 1500, 900, 3000, 3000] * self.E + [25]
        self.LO, self.HI = np.array(lo, float), np.array(hi, float)

    def rows(self, v):
        owners = [0] * len(self.ma) + [1] * len(self.mb)
        masses = list(self.ma) + list(self.mb)
        pos = [(0.0, 0.0)] + [(v[2 * i], v[2 * i + 1]) for i in range(self.n - 1)]
        return [[o, x, y, math.sqrt(m * TOTAL * 100), 100, 0, 0, 0] for o, (x, y), m in zip(owners, pos, masses)]


@njit(cache=True)
def _find(cf, oc, ocn, e, p, mass):
    best = -1
    bd = 1e18
    for k in range(ocn[e, p]):
        s_ = oc[e, p, k]
        m = cf[e, s_, SZ] * cf[e, s_, SZ] / 100
        d = abs(m - mass)
        if d < bd:
            bd = d
            best = s_
    if best >= 0 and bd > 0.15 * mass:
        return -1
    return best


@njit(cache=True)
def control(cf, oc, ocn, V, npos, prim, actors, mirror, state, mouse, splits, fired):
    """state (N,4): [event, ticks since last split, cycle, broken]; fired (N,) set to 1 when a cycle
    starts (event 0 fires) so Python can take the layout snapshot."""
    N = cf.shape[0]
    E = prim.shape[0]
    for e in range(N):
        splits[e, 0] = 0
        splits[e, 1] = 0
        fired[e] = 0
        if state[e, 3] != 0:
            continue
        k = state[e, 0]
        sw = (state[e, 2] % 2) if mirror else 0
        a = actors[k] ^ sw
        q = 1 - a
        s_ = _find(cf, oc, ocn, e, a, prim[k, 1])
        t_ = _find(cf, oc, ocn, e, q, prim[k, 2])
        if s_ < 0 or t_ < 0:
            # the piece this split needs does not exist yet (e.g. still being formed by an eat): wait
            state[e, 1] += 1
            if state[e, 1] > 90:
                state[e, 3] = 2
            continue
        b = npos + 7 * k
        theta = V[e, b]
        stand = V[e, b + 1]
        trig = V[e, b + 2]
        ext = V[e, b + 3]
        perp = V[e, b + 4]
        px = V[e, b + 5]
        py = V[e, b + 6]
        # partner centre of mass
        mx = 0.0
        my = 0.0
        mm = 0.0
        for j in range(ocn[e, q]):
            s2 = oc[e, q, j]
            w = cf[e, s2, SZ] * cf[e, s2, SZ]
            mx += cf[e, s2, X] * w
            my += cf[e, s2, Y] * w
            mm += w
        mx /= mm
        my /= mm
        tx = cf[e, t_, X]
        ty = cf[e, t_, Y]
        ox = tx - mx
        oy = ty - my
        on = math.hypot(ox, oy)
        if on < 1:
            ox = 1.0
            oy = 0.0
            on = 1.0
        ox /= on
        oy /= on
        wx = -oy
        wy = ox
        sx = cf[e, s_, X]
        sy = cf[e, s_, Y]
        dx = tx - sx
        dy = ty - sy
        d = math.hypot(dx, dy)
        state[e, 1] += 1
        # partner keeps its formation in the target frame
        mouse[e, q, 0] = mx + px * ox + py * wx
        mouse[e, q, 1] = my + px * oy + py * wy
        if d <= trig and state[e, 1] >= V[e, V.shape[1] - 1]:
            ux = dx / max(d, 1.0)
            uy = dy / max(d, 1.0)
            mouse[e, a, 0] = tx + ext * ux - perp * uy
            mouse[e, a, 1] = ty + ext * uy + perp * ux
            splits[e, a] = int(prim[k, 3])
            if k == 0:
                fired[e] = 1
            state[e, 0] = k + 1
            state[e, 1] = 0
            if state[e, 0] == E:
                state[e, 0] = 0
                state[e, 2] += 1
        elif state[e, 1] > 90:
            state[e, 3] = 3                  # the trigger never came
        else:
            # approach: standoff point beside the target, angle theta from its outward direction
            c = math.cos(theta)
            s = math.sin(theta)
            dirx = c * ox - s * oy
            diry = s * ox + c * oy
            gx = tx + stand * dirx
            gy = ty + stand * diry
            # steer the splitter (not the whole blob) onto the standoff point
            mouse[e, a, 0] = gx + (gx - sx) * 0.5
            mouse[e, a, 1] = gy + (gy - sy) * 0.5


def evaluate(tech, V, cycles=6, record=None):
    N = len(V)
    sim = Sim(N)
    for i in range(N):
        sim.load(i, tech.rows(V[i]), seed=i + 1)
    state = np.zeros((N, 4), np.int64)
    mouse = np.zeros((N, 2, 2)); splits = np.zeros((N, 2), np.int64); fired = np.zeros(N, np.int64)
    cost = np.zeros(N)
    alive = np.ones(N, bool)
    snaps = [[] for _ in range(N)]          # layout at each cycle start
    starts = [[] for _ in range(N)]         # tick of each cycle start
    cyc_eats = [[] for _ in range(N)]
    exp = [(a, m, em) for ev in tech.eats for (a, _, m, em) in ev]
    rec = None
    if record is not None:
        from splitrun.replay import Recorder
        rec = Recorder(sim, record, "closedloop")
    T = cycles * (TIMEOUT + 30) * tech.E + 50
    com_of = lambda cells: np.array([sum(c[2] * c[4] ** 2 for c in cells), sum(c[3] * c[4] ** 2 for c in cells)]) / sum(c[4] ** 2 for c in cells)

    def score_cycle(i, c):
        sw = int(tech.mirror and c % 2 == 1)
        got = list(cyc_eats[i])
        miss = 0
        for (a, m, em) in exp:
            a ^= sw
            j = next((j for j, g in enumerate(got) if g[0] == a and abs(g[1] - m) < 0.03 * m + 0.001
                      and abs(g[2] - em) < 0.03 * em + 0.001), None)
            if j is None:
                miss += 1
            else:
                got.pop(j)
        return miss + len(got)

    for t in range(T):
        control(sim.cf, sim.oc, sim.ocn, V, tech.npos, tech.prim, tech.actors, tech.mirror, state, mouse, splits, fired)
        # prune: anything broken stops now
        broken = alive & (state[:, 3] != 0)
        if broken.any():
            cost[broken] += 10 * (cycles - state[broken, 2]) + 5
            alive &= ~broken
        # a new cycle starts when event 0 fires: score the finished cycle, snapshot the layout
        for i in np.nonzero(alive & (fired == 1))[0]:
            c = state[i, 2]                      # cycles completed so far (event 0 just fired)
            cells = sim.cells(i)
            if snaps[i]:
                cost[i] += 10 * score_cycle(i, c - 1)
                cyc_eats[i] = []
                prev = snaps[i][-1]
                a_ = _descriptor(prev, 0)
                b_ = _descriptor(cells, 1 if tech.mirror else 0)
                if a_[0] != b_[0] or np.abs(a_[1] - b_[1]).mean() > 0.05 * a_[1].mean():
                    cost[i] += 10 * (cycles - c) + 5
                    alive[i] = False
                    continue
                if len(snaps[i]) >= 2:           # the first cycle is a warm-up for the layout
                    lay = _layout_error(a_, b_)
                    drift = np.linalg.norm(com_of(cells) - com_of(prev))
                    cost[i] += 4 * (1 - np.exp(-lay / 300)) + min(1.0, max(0.0, drift - 30) / 300)
            snaps[i].append(cells)
            starts[i].append(t)
            if c >= cycles:
                alive[i] = False
        sim.active[:] = alive
        sim.step(mouse, splits)
        if rec is not None:
            rec.after_step(np.trunc(mouse[record]), splits[record])
        for i in np.nonzero(sim.evn & alive)[0]:
            for ev in sim.ev[i, :sim.evn[i]]:
                if ev[EV_EATER_OWN] == ev[EV_EATEN_OWN]:
                    cost[i] += 10 * (cycles - state[i, 2]) + 5     # own merge: never part of a technique
                    alive[i] = False
                    break
                cyc_eats[i].append((int(ev[EV_EATER_OWN]), ev[EV_MASS] / TOTAL, ev[EV_EATER_MASS] / TOTAL))
        if not alive.any():
            break
    done = np.array([len(s) - 1 for s in snaps])
    per = np.array([np.mean(np.diff(s)) if len(s) > 1 else 999 for s in starts])
    cost += TIME_W * np.minimum(per, 999)
    cost += 10 * np.maximum(0, cycles - np.maximum(done, 0)) * (done < cycles)
    return cost, done, per, rec


def evolve(tech, V, fn, rng, gens, n_elite=96, scale0=0.06, log=None, label=""):
    pop = len(V)
    c = fn(V)
    o = np.argsort(c)[:n_elite]
    el, ec = V[o], c[o]
    scale, best, stuck = scale0, ec[0], 0
    for g in range(1, gens + 1):
        kids = np.clip(el[rng.integers(0, len(el), pop)] + rng.normal(0, 1, (pop, tech.dim)) * (tech.HI - tech.LO) * scale,
                       tech.LO, tech.HI)
        c = fn(kids)
        allV = np.concatenate([el, kids]); allc = np.concatenate([ec, c])
        o = np.argsort(allc)[:n_elite]
        el, ec = allV[o], allc[o]
        if ec[0] < best - 1e-3:
            best, stuck = ec[0], 0
        else:
            stuck += 1
            scale = max(scale * 0.85, 0.004)
            if stuck % 8 == 0:
                scale = scale0
        if log and g % 10 == 0:
            log(f"   {label} gen {g}: best cost {ec[0]:.2f}")
    return el, ec


def solve(tech, pop=4096, gens=40, seed=0, log=print):
    rng = np.random.default_rng(seed)
    V = rng.uniform(tech.LO, tech.HI, (pop, tech.dim))
    for cyc, g in ((3, gens), (6, gens // 2)):
        el, ec = evolve(tech, V, lambda X_: evaluate(tech, X_, cycles=cyc)[0], rng, g, log=log, label=f"{cyc} cycles")
        V = el[rng.integers(0, len(el), pop)]
        V[:len(el)] = el
    return el, ec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("catalogue")
    ap.add_argument("which", type=int, nargs="+")
    ap.add_argument("--pop", type=int, default=4096)
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    techs = json.load(open(a.catalogue))
    base = os.path.basename(a.catalogue)[:-5]
    for k in a.which:
        tech = Tech(techs[k])
        t0 = time.time()
        print(f"technique {k}: {'MIRROR' if tech.mirror else 'IDENTITY'} {tech.seq}", flush=True)
        el, ec = solve(tech, a.pop, a.gens, a.seed)
        np.save(os.path.join(ROOT, "runs", "techniques", f"{base}_t{k}_cl.npy"), el)
        c, done, per, _ = evaluate(tech, el[:16], cycles=60)
        j = int(np.lexsort((c, -done))[0])
        print(f"   best cost {ec[0]:.2f}; 60-cycle test: {done[j]}/60 clean cycles, mean cycle {per[j]:.0f} ticks "
              f"({time.time() - t0:.0f}s)", flush=True)
