"""Find fast, repeatable geometry for a catalogue technique (tools/technique_catalogue.py).

Searched per technique (all positions/cursors in the CYCLE FRAME: origin = the first actor's
biggest piece, x-axis towards the partner's centre of mass, y-axis +/-90 deg):
  layout      starting positions of every piece
  per event   split cursor, gap until the next event (ticks), both players' cursors in that gap
  flip        whether the handedness alternates between cycles

Scored over CYCLES consecutive cycles, each re-framed on the real pieces (so a solution must
chain, not just close one cycle):
  +10 per missing or unintended eat (checked per event: eater's player + eaten mass), own merges
  + layout error at each cycle end vs its start (roles swapped for MIRROR techniques) / 100
  + TIME_W * cycle length (ticks)  -- less downtime is better

    python tools/solve_technique.py runs/techniques/catalogue_s3_e2_p8.json 4
"""
import argparse, json, math, os, sys, time
import numpy as np
from numba import njit
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from splitrun.sim import Sim, X, Y, SZ, EV_EATER_OWN, EV_EATEN_OWN, EV_MASS, EV_EATER_MASS  # noqa
from splitrun.metrics import _descriptor, _layout_error  # noqa

TOTAL = 37000.0
TIME_W = 0.004
STRICT_LAY = 0.0       # > 0: a cycle whose layout is off by more than this (units) fails outright
TRANS_W = 0.0          # > 0: reward net displacement per tick (a TRANSLATING periodic splitrun) instead of penalising drift
CYCLES = 3
WORLD_FRAME = False    # True: cycle frames keep a FIXED world orientation (x = travel direction); only the origin follows
                       # the pieces, so a closed cycle is a pure translation (no turning) and mirror cycles zig-zag along x
MAXGAP = 40


@njit(cache=True)
def frames_of(cf, oc, ocn, actor, origin, u):
    for e in range(cf.shape[0]):
        a = actor[e]
        q = 1 - a
        big = -1
        for k in range(ocn[e, a]):
            s_ = oc[e, a, k]
            if big < 0 or cf[e, s_, SZ] > cf[e, big, SZ]:
                big = s_
        mx = 0.0
        my = 0.0
        m = 0.0
        for k in range(ocn[e, q]):
            s_ = oc[e, q, k]
            w = cf[e, s_, SZ] * cf[e, s_, SZ]
            mx += cf[e, s_, X] * w
            my += cf[e, s_, Y] * w
            m += w
        if big < 0 or m == 0:
            origin[e, 0] = 0.0
            origin[e, 1] = 0.0
            u[e, 0] = 1.0
            u[e, 1] = 0.0
            continue
        origin[e, 0] = cf[e, big, X]
        origin[e, 1] = cf[e, big, Y]
        dx = mx / m - origin[e, 0]
        dy = my / m - origin[e, 1]
        d = math.hypot(dx, dy)
        if d < 1:
            d = 1.0
            dx = 1.0
            dy = 0.0
        u[e, 0] = dx / d
        u[e, 1] = dy / d


def world_fix(u):
    if WORLD_FRAME:
        u[:, 0] = 1.0; u[:, 1] = 0.0


def orient_error(a, b):
    """world-frame layouts: mean position error about the centre (no rotation allowed; reflection in x allowed)"""
    Pa = a[2] - a[2].mean(0)
    best = np.inf
    for o in b[3]:
        Pb = b[2][o] - b[2][o].mean(0)
        best = min(best, np.linalg.norm(Pa - Pb, axis=1).mean(), np.linalg.norm(Pa - Pb * np.array([1, -1]), axis=1).mean())
    return best


def frame_layout(cells, swap, origin, u, h):
    """{(role, mass key): [frame coords...]} of a cell list"""
    w = np.array([-u[1], u[0]]) * h
    out = {}
    for c in cells:
        key = (c[1] ^ swap, int(round(c[4] * c[4] / 100 / TOTAL * 400)))
        r = np.array([c[2], c[3]]) - origin
        out.setdefault(key, []).append(np.array([r @ u, r @ w]))
    return out


def layout_mismatch(a, ref):
    err, n = 0.0, 0
    for k, pts in a.items():
        if k not in ref:
            return np.inf
        for p in pts:
            err += min(np.linalg.norm(p - q) for q in ref[k])
            n += 1
    return err / max(n, 1)


@njit(cache=True)
def eat_gaps(cf, oc, ocn, tgt_owner, tgt_mass, active, gaps):
    """gaps[e, k] = min(gaps, how far the closest eligible pair is from being able to eat) for each
    intended eat k (eater player tgt_owner[e,k], eaten mass tgt_mass[e,k] in total-mass fractions)"""
    for e in range(cf.shape[0]):
        if active[e] == 0:
            continue
        for k in range(tgt_owner.shape[1]):
            a = tgt_owner[e, k]
            if a < 0:
                continue
            m = tgt_mass[e, k] * 37000.0
            best = 1e9
            for x in range(ocn[e, a]):
                i = oc[e, a, x]
                si = cf[e, i, SZ]
                for y in range(ocn[e, 1 - a]):
                    j = oc[e, 1 - a, y]
                    sj = cf[e, j, SZ]
                    mj = sj * sj / 100
                    if abs(mj - m) > 0.12 * m or si < 1.12 * sj:
                        continue
                    d = math.hypot(cf[e, i, X] - cf[e, j, X], cf[e, i, Y] - cf[e, j, Y]) - (si - sj / 3)
                    if d < best:
                        best = d
            if best < gaps[e, k]:
                gaps[e, k] = best


@njit(cache=True)
def event_frames(cf, oc, ocn, idx, actor, spl, tgt, origin, u, ok):
    """for envs idx: origin = actor's piece closest to mass spl, u = unit vector to the partner piece
    closest to mass tgt; ok[e]=0 if either piece is missing (>15% off)"""
    for n in range(idx.shape[0]):
        e = idx[n]
        a = actor[n]
        found = 0
        best = np.zeros(2, np.int64)
        for side in range(2):
            p = a if side == 0 else 1 - a
            m_want = spl if side == 0 else tgt
            bi = -1
            bd = 1e18
            for k in range(ocn[e, p]):
                s_ = oc[e, p, k]
                d = abs(cf[e, s_, SZ] * cf[e, s_, SZ] / 100 - m_want)
                if d < bd:
                    bd = d
                    bi = s_
            if bi >= 0 and bd <= 0.15 * m_want:
                best[side] = bi
                found += 1
        if found < 2:
            ok[e] = 0
            continue
        ok[e] = 1
        origin[e, 0] = cf[e, best[0], X]
        origin[e, 1] = cf[e, best[0], Y]
        dx = cf[e, best[1], X] - origin[e, 0]
        dy = cf[e, best[1], Y] - origin[e, 1]
        d = math.hypot(dx, dy)
        if d < 1:
            d = 1.0
            dx = 1.0
            dy = 0.0
        u[e, 0] = dx / d
        u[e, 1] = dy / d


class Technique:
    def __init__(self, t):
        self.t = t
        self.ma, self.mb = t["masses_a"], t["masses_b"]
        self.n = len(self.ma) + len(self.mb)
        self.seq = t["seq"]
        self.E = len(self.seq)
        self.mirror = t["mirror"]
        self.eats = t["eats"]
        self.npos = 2 * (self.n - 1)
        self.dim = self.npos + 7 * self.E + 1
        # primary eat of each event: the actor's launched half eats (kind 0) or is fed (kind 1)
        self.spl = np.zeros(self.E); self.tgt = np.zeros(self.E)
        for k, ((actor, kk), ev) in enumerate(zip(self.seq, self.eats)):
            pick = None
            for (eo, bo, m, em) in ev:
                if eo == actor:
                    pick = (em * 2 ** kk, m); break
                if bo == actor:
                    pick = (m * 2 ** kk, em); break
            if pick is None:
                pick = (ev[0][3], ev[0][2])
            self.spl[k], self.tgt[k] = pick[0] * TOTAL, pick[1] * TOTAL
        lo, hi = [], []
        lo += [-2800] * self.npos; hi += [2800] * self.npos
        for _ in range(self.E):
            lo += [-3000, -3000, 0, -3000, -3000, -3000, -3000]
            hi += [3000, 3000, MAXGAP, 3000, 3000, 3000, 3000]
        lo += [-1]; hi += [1]
        self.LO, self.HI = np.array(lo, float), np.array(hi, float)

    def rows(self, v):
        owners = [0] * len(self.ma) + [1] * len(self.mb)
        masses = list(self.ma) + list(self.mb)
        pos = [(0.0, 0.0)] + [(v[2 * i], v[2 * i + 1]) for i in range(self.n - 1)]
        return [[o, x, y, math.sqrt(m * TOTAL * 100), 100, 0, 0, 0] for o, (x, y), m in zip(owners, pos, masses)]

    def ev_params(self, V, k):
        b = self.npos + 7 * k
        return V[:, b:b + 2], np.round(V[:, b + 2]).astype(int), V[:, b + 3:b + 5], V[:, b + 5:b + 7]


LOCAL_FRAMES = False


def evaluate(tech, V, cycles=CYCLES, record=None, upto=None, close=True, ends=None, hook=None,
             init_state=None, cyc0=0, preroll=None):
    """init_state: full simulator arrays of one world to start every candidate from (instead of the
    candidate's own starting positions); cyc0: cycle index to start at (role parity for mirror)."""
    """Run each candidate for `cycles` cycles. Splits may share a tick (gap 0 = both players split
    at the same moment), so eats are checked per CYCLE against the technique's full list."""
    N = len(V)
    sim = Sim(N)
    for i in range(N):
        sim.load(i, tech.rows(V[i]), seed=i + 1)
    if init_state is not None:
        for kk, arr in init_state.items():
            getattr(sim, kk)[:] = arr[None]
    E = tech.E
    cur = [tech.ev_params(V, k) for k in range(E)]
    gaps = np.stack([c[1] for c in cur], 1)                      # (N, E) gap after event k
    gaps[:, E - 1] = np.maximum(gaps[:, E - 1], 1)               # the cycle itself has length >= 1
    tev = np.concatenate([np.zeros((N, 1), int), np.cumsum(gaps, 1)], 1)   # event k at tev[:,k]; period = tev[:,E]
    period = tev[:, E]
    pre = np.zeros(N, int) if preroll is None else np.round(preroll[:, 4]).astype(int)
    cost = np.zeros(N)
    alive = np.ones(N, bool)
    cyc = np.full(N, cyc0, int)
    tc = np.zeros(N, int)
    origin = np.zeros((N, 2)); u = np.zeros((N, 2))
    first_actor = np.full(N, tech.seq[0][0] ^ int(tech.mirror and cyc0 % 2 == 1), np.int64)
    frames_of(sim.cf, sim.oc, sim.ocn, first_actor, origin, u)
    world_fix(u)
    hand = np.ones(N)
    start_layout = [sim.cells(i) for i in range(N)]
    ref = [frame_layout(start_layout[i], 0, origin[i], u[i], 1.0) for i in range(N)]
    ref_pending = np.zeros(N, bool)
    cyc_eats = [[] for _ in range(N)]
    trans_t0 = np.full(N, -1); trans_c0 = np.zeros((N, 2))
    com_of = lambda cells: np.array([sum(c[2] * c[4] ** 2 for c in cells), sum(c[3] * c[4] ** 2 for c in cells)]) / sum(c[4] ** 2 for c in cells)
    rec = None
    if record is not None:
        from splitrun.replay import Recorder
        rec = Recorder(sim, record, "technique")
    T = int(period.max()) * cycles + 2 + int(pre.max())
    swap_of = lambda c: (c % 2 == 1) if tech.mirror else False
    req = tech.E if upto is None else upto
    all_eats = [e for ev in tech.eats[:req] for e in ev]
    later_eats = [e for ev in tech.eats[req:] for e in ev]
    MAXE = max(1, len(all_eats))
    near = np.full((N, MAXE), 1e9)
    tgt_owner = np.zeros((N, MAXE), np.int64)
    tgt_mass = np.array([[m for _, _, m, _ in all_eats]] * N)

    def set_targets(i, c):
        near[i] = 1e9
        tgt_owner[i] = [a ^ int(swap_of(c)) for a, _, _, _ in all_eats]

    def score_cycle(i, c):
        """per missing intended eat 5 + up to 5 for how close it came; 10 per unintended eat"""
        got = list(cyc_eats[i])
        miss = 0.0
        for q, (a, _, m, em) in enumerate(all_eats):
            a ^= int(swap_of(c))
            j = next((j for j, g in enumerate(got) if g[0] == a and abs(g[1] - m) < 0.03 * m + 0.001
                      and abs(g[2] - em) < 0.03 * em + 0.001), None)
            if j is None:
                miss += 0.5 + 0.5 * min(1.0, max(near[i, q], 0.0) / 1500.0)
            else:
                got.pop(j)
        for (a, _, m, em) in later_eats:              # eats of later events: allowed, not required
            a ^= int(swap_of(c))
            j = next((j for j, g in enumerate(got) if g[0] == a and abs(g[1] - m) < 0.03 * m + 0.001
                      and abs(g[2] - em) < 0.03 * em + 0.001), None)
            if j is not None:
                got.pop(j)
        return miss + len(got)

    for i in range(N):
        set_targets(i, cyc0)
    for t in range(T):
        mouse = np.zeros((N, 2, 2)); splits = np.zeros((N, 2), np.int64)
        w = np.stack([-u[:, 1], u[:, 0]], 1) * hand[:, None]
        if preroll is not None and (pre > 0).any():
            # repositioning phase before the first split (cycle clock does not run yet)
            ii = np.nonzero(pre > 0)[0]
            fr = lambda xy: origin[ii] + xy[:, :1] * u[ii] + xy[:, 1:2] * w[ii]
            ra = (cyc[ii] % 2) if tech.mirror else np.zeros(len(ii), int)
            mouse[ii, ra] = fr(preroll[ii, 0:2]); mouse[ii, 1 - ra] = fr(preroll[ii, 2:4])
            pre[ii] -= 1
            sim.active[:] = alive
            sim.step(mouse, splits)
            for i in np.nonzero(sim.evn)[0]:
                if pre[i] >= 0 and sim.evn[i] > 0 and i in set(ii.tolist()):
                    cost[i] += 10 * (cycles - (cyc[i] - cyc0)) + 5      # nothing may be eaten while repositioning
                    alive[i] = False
            if (pre > 0).any():
                continue
            frames_of(sim.cf, sim.oc, sim.ocn, np.array([tech.seq[0][0] ^ int(tech.mirror and c % 2 == 1) for c in cyc], np.int64), origin, u)
            world_fix(u)
            start_layout = [sim.cells(i) for i in range(N)]
            continue
        to_world = lambda xy: origin + xy[:, :1] * u + xy[:, 1:2] * (np.stack([-u[:, 1], u[:, 0]], 1) * hand[:, None])
        roleA = (cyc % 2) if tech.mirror else np.zeros(N, int)      # the player playing role A
        for k in range(E):
            in_seg = (tc >= tev[:, k]) & (tc < tev[:, k + 1])
            if in_seg.any():
                _, _, amv, bmv = cur[k]
                ii = np.nonzero(in_seg)[0]
                pa = to_world(amv)[ii]; pb = to_world(bmv)[ii]
                mouse[ii, roleA[ii]] = pa
                mouse[ii, 1 - roleA[ii]] = pb
        for k in range(E):                                        # split presses (several may share a tick)
            at = alive & (tc == tev[:, k])
            if at.any():
                ii = np.nonzero(at)[0]
                actor = tech.seq[k][0] ^ roleA[ii]
                if LOCAL_FRAMES:
                    # re-anchor on the real pieces: origin = splitter, x-axis -> its target
                    okf = np.ones(N, np.int64)
                    event_frames(sim.cf, sim.oc, sim.ocn, ii.astype(np.int64), actor.astype(np.int64),
                                 tech.spl[k], tech.tgt[k], origin, u, okf)
                    bad = ii[okf[ii] == 0]
                    if len(bad):
                        cost[bad] += 10 * (cycles - (cyc[bad] - cyc0)) + 5
                        alive[bad] = False
                    w = np.stack([-u[:, 1], u[:, 0]], 1) * hand[:, None]
                    ii = ii[okf[ii] == 1]
                    actor = tech.seq[k][0] ^ roleA[ii]
                mouse[ii, actor] = to_world(cur[k][0])[ii]
                splits[ii, actor] += tech.seq[k][1]
        sim.active[:] = alive
        sim.step(mouse, splits)
        if hook is not None:
            hook(sim, tc.copy(), cyc.copy(), tev, alive.copy(), splits.copy())
        eat_gaps(sim.cf, sim.oc, sim.ocn, tgt_owner, tgt_mass, sim.active, near)
        if rec is not None:
            rec.after_step(np.trunc(mouse[record]), splits[record])
        for i in np.nonzero(sim.evn)[0]:
            for ev in sim.ev[i, :sim.evn[i]]:
                if ev[EV_EATER_OWN] == ev[EV_EATEN_OWN]:
                    cost[i] += 10                        # own merge: not part of any technique
                else:
                    cyc_eats[i].append((int(ev[EV_EATER_OWN]), ev[EV_MASS] / TOTAL, ev[EV_EATER_MASS] / TOTAL))
        tc += 1
        done = alive & (tc == period)
        for i in np.nonzero(done)[0]:
            cost[i] += 10 * score_cycle(i, cyc[i])
            cyc_eats[i] = []
            if req < tech.E or not close:             # curriculum stage: eats only, no closing check
                cost[i] += TIME_W * period[i]
                alive[i] = False
                continue
            end = sim.cells(i)
            if ends is not None and cyc[i] == cyc0:
                ends[i] = end
            a = _descriptor(start_layout[i], 0)
            b = _descriptor(end, 1 if tech.mirror else 0)
            if a[0] != b[0]:
                cost[i] += 10 * (cycles - (cyc[i] - cyc0))
                alive[i] = False
                continue
            merr = np.abs(a[1] - b[1]).mean() / a[1].mean()
            if merr > 0.05:
                # wrong masses: fail, but say how wrong so the search can descend
                cost[i] += 10 * (cycles - (cyc[i] - cyc0)) - 5 + 5 * min(1.0, merr / 0.5)
                alive[i] = False
                continue
            # a stationary splitrun: the pair's centre must not travel from cycle to cycle
            drift = np.linalg.norm(com_of(end) - com_of(start_layout[i]))
            # the artificial start (nothing in flight) is not on the repeating orbit: cycle 0 is a
            # warm-up; periodicity is scored between consecutive cycle ENDS from then on.
            # A closed cycle must always beat a failed one (failure costs >= 5): cap the layout term below 4
            if cyc[i] - cyc0 >= (0 if ONORBIT else 1):
                lay = _layout_error(a, b)
                if WORLD_FRAME:
                    lay = max(lay, orient_error(a, b))
                if STRICT_LAY > 0 and lay > STRICT_LAY:
                    cost[i] += 10 * (cycles - (cyc[i] - cyc0)) - 5 + 5 * min(1.0, lay / 1500)
                    alive[i] = False
                    continue
                if TRANS_W > 0:
                    # NET travel: displacement since the first scored cycle start, per tick (a zig-zag earns nothing)
                    cost[i] += 4 * (1 - np.exp(-lay / 300))
                    if trans_t0[i] < 0:
                        trans_c0[i] = com_of(start_layout[i]); trans_t0[i] = t
                    elif cyc[i] - cyc0 == cycles - 1:
                        cost[i] -= TRANS_W * np.linalg.norm(com_of(end) - trans_c0[i]) / max(t - trans_t0[i], 1) * cycles
                else:
                    cost[i] += 4 * (1 - np.exp(-lay / 300)) + min(1.0, max(0.0, drift - 30.0) / 300)
            cost[i] += TIME_W * period[i]
            start_layout[i] = end
            if cyc[i] == cyc0:
                ref_pending[i] = True
            cyc[i] += 1
            tc[i] = 0
            set_targets(i, cyc[i])
            if cyc[i] - cyc0 >= cycles:
                alive[i] = False
        if done.any():
            nxt = np.array([tech.seq[0][0] ^ int(swap_of(c)) for c in cyc])
            o2 = origin.copy(); u2 = u.copy()
            frames_of(sim.cf, sim.oc, sim.ocn, nxt, o2, u2)
            world_fix(u2)
            origin[done] = o2[done]; u[done] = u2[done]
            # the layout check ignores reflections, so pick the handedness that matches the start
            for i in np.nonzero(done & alive)[0]:
                cells = sim.cells(i)
                sw = int(swap_of(cyc[i]))
                if ref_pending[i]:
                    # after the warm-up the repeating shape is known: use it as the handedness reference
                    ref[i] = frame_layout(cells, sw, origin[i], u[i], hand[i])
                    ref_pending[i] = False
                    continue
                e_p = layout_mismatch(frame_layout(cells, sw, origin[i], u[i], 1.0), ref[i])
                e_m = layout_mismatch(frame_layout(cells, sw, origin[i], u[i], -1.0), ref[i])
                hand[i] = 1.0 if e_p <= e_m else -1.0
        if not alive.any():
            break
    return cost, cyc, rec


def evolve(tech, V, fn, rng, max_gens, target, scale0=0.05, n_elite=128, log=None, label=""):
    """(mu + lambda) evolution with an adaptive step: keep the step while the best keeps improving,
    shrink it when it doesn't, kick it up when stuck; stop at `target` or after `max_gens`."""
    pop = len(V)
    cost = fn(V)
    o = np.argsort(cost)[:n_elite]
    el, ec = V[o], cost[o]
    scale, best, stuck = scale0, ec[0], 0
    for g in range(1, max_gens + 1):
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
            scale *= 0.8
            if stuck % 6 == 0:
                scale = min(scale0, scale * 4)     # stuck: kick the step size back up
        if log and g % 20 == 0:
            log(f"   {label} gen {g}: best cost {ec[0]:.3f} (step {scale:.4f})")
        if ec[0] <= target or stuck >= 24:
            break
    return el, ec


CURRICULUM_ONLY = False
ONORBIT = False       # True: the start itself must be on the orbit (score layout from the first cycle)
ROBUST_K = 1          # >1: score every candidate on K copies with jittered starting positions


def robust(tech, fn, K=None, sigma=15.0, seed=123):
    """wrap an evaluation so each candidate is scored on K copies (the first exact, the rest with the
    starting positions jittered by sigma units) -> mean cost. Pushes the search away from knife-edge
    solutions whose eats only just happen."""
    K = K or ROBUST_K
    if K <= 1:
        return fn
    rng = np.random.default_rng(seed)

    def f(V):
        R = np.repeat(V, K, 0)
        noise = rng.normal(0, sigma, (len(R), tech.npos))
        noise[::K] = 0
        R[:, :tech.npos] += noise
        return fn(R).reshape(len(V), K).mean(1)
    return f


INIT_POP = None


def solve(tech, pop=8192, gens=40, seed=0, log=print):
    """phase 1: make ONE cycle close on itself (continuous layout error to descend);
    phase 2: refine the survivors so that CYCLES consecutive re-framed cycles work."""
    rng = np.random.default_rng(seed)
    V = rng.uniform(tech.LO, tech.HI, (pop, tech.dim))
    hist_best = []
    # curriculum: make split 1's eats happen, then splits 1-2, ... (each stage starts from the last)
    if INIT_POP is not None:
        V = INIT_POP[rng.integers(0, len(INIT_POP), pop)] + rng.normal(0, 1, (pop, tech.dim)) * (tech.HI - tech.LO) * 0.005
        V = np.clip(V, tech.LO, tech.HI)
        V[:len(INIT_POP)] = np.clip(INIT_POP, tech.LO, tech.HI)
    for j in (range(1, tech.E + 1) if INIT_POP is None else []):
        fn = robust(tech, lambda X, j=j: evaluate(tech, X, cycles=1, upto=j, close=False)[0])
        cost = fn(V)
        el = V[np.argsort(cost)[:128]]; ec = np.sort(cost)[:128]
        for g in range(1, 31):                       # fixed step schedule: worked best for the eats
            kids = np.clip(el[rng.integers(0, 128, pop)] + rng.normal(0, 1, (pop, tech.dim)) * (tech.HI - tech.LO) * 0.08 * 0.95 ** g,
                           tech.LO, tech.HI)
            c = fn(kids)
            allV = np.concatenate([el, kids]); allc = np.concatenate([ec, c])
            o = np.argsort(allc)[:128]
            el, ec = allV[o], allc[o]
            if ec[0] < 0.5:
                break
        log(f"   curriculum: first {j} split(s) -> best cost {ec[0]:.2f}")
        V = el[rng.integers(0, len(el), pop)]
        V[:len(el)] = el
    if INIT_POP is not None:
        el, ec = V[:len(INIT_POP)], np.zeros(len(INIT_POP))
    tech.curriculum_elite = el.copy()
    if CURRICULUM_ONLY:
        return el, ec
    # closing: one cycle must come back to its own layout, then 3, then 10 chained cycles
    phases = ((1, 4 * gens, 0.5), (CYCLES, 2 * gens, 1.2), (10, gens, 4.0)) if ONORBIT else \
        ((CYCLES, 4 * gens, 1.2), (10, gens, 4.0))
    for cyc_n, n_gens, target in phases:
        p = pop if cyc_n < 10 else max(512, pop // 4)
        el, ec = evolve(tech, V[:p], robust(tech, lambda X, cyc_n=cyc_n: evaluate(tech, X, cycles=cyc_n)[0]), rng, n_gens,
                        target if TRANS_W <= 0 else -1e9,
                        scale0=0.03, log=log, label=f"closing ({cyc_n} cycle{'s' if cyc_n > 1 else ''})")
        log(f"   closing over {cyc_n} cycle(s): best cost {ec[0]:.3f}")
        if cyc_n == phases[0][0]:
            tech.phase1_elite = el.copy()
        V = el[rng.integers(0, len(el), pop)]
        V[:len(el)] = el
    elite, ecost = el, ec
    return elite, ecost


def refine(tech, elite, pop, gens, rng, log=print):
    """phases 2+3 only, starting from an existing elite (used by the gap-shrinking continuation)"""
    V = elite[rng.integers(0, len(elite), pop)] + rng.normal(0, 1, (pop, tech.dim)) * (tech.HI - tech.LO) * 0.01
    V = np.clip(V, tech.LO, tech.HI)
    V[:len(elite)] = np.clip(elite, tech.LO, tech.HI)
    hb = []
    for cyc_n in (CYCLES, 10):
        if cyc_n == 10:
            pop = max(512, pop // 4)
            V = V[:pop]
        cost, _, _ = evaluate(tech, V, cycles=cyc_n)
        el = V[np.argsort(cost)[:128]]; ec = np.sort(cost)[:128]
        for g in range(1, gens + 1):
            scale = 0.03 * (0.9 ** g)
            kids = np.clip(el[rng.integers(0, len(el), pop)] + rng.normal(0, 1, (pop, tech.dim)) * (tech.HI - tech.LO) * scale,
                           tech.LO, tech.HI)
            c, _, _ = evaluate(tech, kids, cycles=cyc_n)
            allV = np.concatenate([el, kids]); allc = np.concatenate([ec, c])
            o = np.argsort(allc)[:128]
            el, ec = allV[o], allc[o]
            hb.append(ec[0])
            if len(hb) > 5 and hb[-6] - hb[-1] < 0.005:
                break
        hb = []
        V = el[rng.integers(0, len(el), pop)]
        V[:len(el)] = el
    return el, ec


def best_chain(tech, elite, n=16, cycles=100):
    """chain-test the first n candidates in ONE batched run; best = most clean cycles, then lowest cost"""
    m = min(n, len(elite))
    c, cyc, _ = evaluate(tech, elite[:m], cycles=cycles)
    j = int(np.lexsort((c, -cyc))[0])
    return int(cyc[j]), float(c[j]), j


def shrink_gaps(tech, elite, pop=4096, gens=12, seed=0, log=print):
    """continuation: repeatedly lower the cap on every gap between splits (and on the cycle), re-optimise
    positions/angles from the previous solution, keep going while it still chains 100/100 cycles"""
    rng = np.random.default_rng(seed)
    gi = [tech.npos + 7 * e + 2 for e in range(tech.E)]
    best = best_chain(tech, elite)
    v0 = elite[best[2]]
    cur_total = sum(int(round(v0[g])) for g in gi)
    good = elite
    log(f"   start: {best[0]}/100 clean, cycle {cur_total} ticks")
    global TIME_W
    tw0 = TIME_W
    TIME_W = 0.02                          # stronger push towards short cycles while shrinking
    step = 0.85
    while cur_total > tech.E and step < 0.99:
        target = max(tech.E, min(cur_total - 1, int(cur_total * step)))
        # scale all gaps of the population down proportionally to hit the target cycle length
        E2 = good.copy()
        tot = np.maximum(E2[:, gi].round().sum(1, keepdims=True), 1)
        E2[:, gi] = np.floor(E2[:, gi] * target / tot)
        HI_save = tech.HI.copy()
        tech.HI[gi] = np.maximum(E2[:, gi].max(0), 1)
        el, ec = refine(tech, E2, pop, gens if step <= 0.85 else 2 * gens, rng, log)
        tech.HI = HI_save
        b = best_chain(tech, el)
        per = sum(int(round(el[b[2]][g])) for g in gi)
        log(f"   cap {target} ticks: chain {b[0]}/100, cycle {per} ticks")
        if b[0] < 100:
            step = 1 - (1 - step) / 2           # smaller step, longer refinement
            continue
        good = el[[b[2]] + [j for j in range(len(el)) if j != b[2]]]
        cur_total = per
        step = 0.85
    TIME_W = tw0
    return good


def chain_test(tech, v, cycles=100, record=None):
    """run the solution for many cycles; returns (clean cycles completed, replay recorder)"""
    c, cyc, rec = evaluate(tech, v[None], cycles=cycles, record=record)
    return int(cyc[0]), c[0], rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("catalogue")
    ap.add_argument("which", type=int, nargs="*")
    ap.add_argument("--pop", type=int, default=8192)
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--max-gap", type=int, default=MAXGAP, help="max ticks between consecutive splits")
    ap.add_argument("--shrink", action="store_true", help="start from the saved elite and shrink the gaps")
    ap.add_argument("--autoshrink", action="store_true", help="after a successful solve, shrink the gaps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--local", action="store_true", help="re-anchor every split on the real splitter/target")
    ap.add_argument("--robust", type=int, default=1, help="score each candidate on K jittered copies")
    ap.add_argument("--curriculum-only", action="store_true", help="stop after the per-split curriculum; save *_cur.npy")
    ap.add_argument("--onorbit", action="store_true", help="start must be on the orbit (long quiet gap at the cycle end)")
    ap.add_argument("--last-gap", type=int, default=0, help="allow the final gap (quiet repositioning) up to this many ticks")
    ap.add_argument("--init-cur", action="store_true", help="skip the curriculum: start from the saved *_cur.npy population")
    a = ap.parse_args()
    techs = json.load(open(a.catalogue))
    out = os.path.join(ROOT, "runs", "techniques")
    globals()["ROBUST_K"] = a.robust
    globals()["CURRICULUM_ONLY"] = a.curriculum_only
    globals()["ONORBIT"] = a.onorbit
    if a.local:
        LOCAL_FRAMES = True
        globals()["LOCAL_FRAMES"] = True
    for k in (a.which or range(len(techs))):
        tech = Technique(techs[k])
        tech.HI[[tech.npos + 7 * e + 2 for e in range(tech.E)]] = a.max_gap
        if a.last_gap:
            tech.HI[tech.npos + 7 * (tech.E - 1) + 2] = a.last_gap
        t0 = time.time()
        print(f"technique {k}: {'MIRROR' if tech.mirror else 'IDENTITY'} events {tech.seq} rho {techs[k]['rho']:.2f}", flush=True)
        if a.init_cur:
            ip = np.load(os.path.join(out, f"{os.path.basename(a.catalogue)[:-5]}_t{k}_cur.npy"))
            if a.onorbit and a.last_gap:
                # let every eat land and everything settle before the cycle closes (the start is then on the orbit)
                ip[:, tech.npos + 7 * (tech.E - 1) + 2] = int(a.last_gap * 0.85)
            globals()["INIT_POP"] = ip
        path = os.path.join(out, f"{os.path.basename(a.catalogue)[:-5]}_t{k}{'_s%d' % a.seed if a.seed else ''}{'_local' if a.local else ''}_elite.npy")
        if a.shrink:
            elite = shrink_gaps(tech, np.load(path), a.pop, a.gens)
            ecost = evaluate(tech, elite[:1], cycles=1)[0]
        else:
            elite, ecost = solve(tech, a.pop, a.gens, seed=a.seed)
            if a.curriculum_only:                   # only a curriculum-only run may write *_cur.npy
                np.save(path.replace("_elite.npy", "_cur.npy"), tech.curriculum_elite)
                print(f"   curriculum-only: saved {path.replace('_elite.npy', '_cur.npy')}", flush=True)
                continue
            np.save(path.replace("_elite.npy", "_p1.npy"), tech.phase1_elite)
        np.save(path, elite)
        best = best_chain(tech, elite)
        if a.autoshrink and not a.shrink and best[0] >= 100:
            elite = elite[[best[2]] + [j for j in range(len(elite)) if j != best[2]]]
            elite = shrink_gaps(tech, elite, a.pop, 12)
            np.save(path, elite)
            best = best_chain(tech, elite)
        v = elite[best[2]]
        per = sum(int(round(v[tech.npos + 7 * e + 2])) for e in range(tech.E))
        print(f"   best one-cycle cost {ecost[0]:.3f}; chain test: {best[0]}/100 clean cycles, cycle length {per} ticks "
              f"(elite #{best[2]})  ({time.time() - t0:.0f}s)", flush=True)
