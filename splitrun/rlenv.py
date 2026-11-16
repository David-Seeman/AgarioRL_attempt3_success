"""Vectorised RL environment: two players (controlled together) perform a target splitrun technique.

Everything the policy sees and does is in ROLE terms: role A = the player that plays A in the current
cycle (for mirror techniques the players swap roles every cycle), so a mirror technique only has to
be learned for one half.

Observation (float32, OBS_DIM): up to 8 pieces per role, sorted by mass, in a frame centred on the
pair's centre of mass with the x-axis pointing from role A's centre to role B's centre:
    x, y, mass, boost vector (x, y), age, exists, can-merge
plus: one-hot of the next expected split, ticks since the last split, ticks since progress, pending
intended eats, distance between the two players.

Action (per role): cursor offset (2 floats in [-1, 1] -> x3000 units, relative to that role's centre,
in the pair frame) and a split choice {none, single, double}. One decision every ACTION_REPEAT ticks
(splits are pressed on the first of those ticks).

Reward (technique-following):
    +1     an intended eat happens (right eater player, eaten mass and eater mass within 3%)
    (no reward for pressing split itself -- only for the eats it produces)
    +2*exp(-err/150)  at each cycle boundary from the 2nd one on: the layout (masses + all pairwise
           distances, role-ordered) repeats that of the previous boundary  -- tolerant, not exact
    -0.002 per tick (less downtime is better)
  terminal (episode ends): -2 unintended eat / own merge / wrong split press, -1 an intended eat
  never happened (60 ticks), no progress for 120 ticks, pair drifts > 3000 units, a player vanishes.
"""
import json
import math

import numpy as np
from numba import njit, prange

from .sim import (Sim, S, X, Y, SZ, BDX, BDY, BD, EX, OWN, BIRTH, BOOSTING, CANMERGE, G_TICK,
                  EV_EATER_OWN, EV_EATEN_OWN, EV_MASS, EV_EATER_MASS)

TOTAL = 37000.0
MAXPC = 8            # pieces per role in the observation
NFEAT = 8
MAXEV = 8            # max split events per cycle
MAXP = 16            # pending intended eats
ACTION_REPEAT = 2
OBS_DIM = 2 * MAXPC * NFEAT + MAXEV + 6
PEND_TIMEOUT = 90
SHAPE = 0.5          # reward per 1000 units an intended eater gets closer to eating its target
PROGRESS_TIMEOUT = 120
MAX_DRIFT = 3000.0

# tracker int fields
T_NEXT, T_CYC, T_BOUND, T_SINCE_SPLIT, T_SINCE_PROG, T_TICKS, T_PN, T_DONE = range(8)
NTI = 8
# tracker float fields
TF_COM0X, TF_COM0Y, TF_PREVN = range(3)
NTF = 3


class TechSpec:
    """numba-friendly arrays for one technique (catalogue format: seq, mirror, eats)"""

    def __init__(self, t, name=""):
        self.name = name
        self.t = t
        self.E = len(t["seq"])
        assert self.E <= MAXEV
        self.actors = np.array([a for a, _ in t["seq"]], np.int64)
        self.ks = np.array([k for _, k in t["seq"]], np.int64)
        self.mirror = 1 if t["mirror"] else 0
        ne = max(len(e) for e in t["eats"])
        self.ev_n = np.array([len(e) for e in t["eats"]], np.int64)
        self.ev_role = np.zeros((self.E, ne), np.int64)
        self.ev_m = np.zeros((self.E, ne))
        self.ev_em = np.zeros((self.E, ne))
        for k, ev in enumerate(t["eats"]):
            for j, (eo, bo, m, em) in enumerate(ev):
                self.ev_role[k, j] = eo
                self.ev_m[k, j] = m * TOTAL
                self.ev_em[k, j] = em * TOTAL
        self.masses = (list(t["masses_a"]), list(t["masses_b"]))
        # primary eat per split: splitter piece (actor's, mass = 2^k x launched half) and its target
        self.prim_spl = np.zeros(self.E); self.prim_tgt = np.zeros(self.E)
        for k, ((actor, kk), ev) in enumerate(zip(t["seq"], t["eats"])):
            pick = None
            for (eo, bo, m, em) in ev:
                if eo == actor:
                    pick = (em * 2 ** kk, m); break
                if bo == actor:
                    pick = (m * 2 ** kk, em); break
            if pick is None:
                pick = (ev[0][3], ev[0][2])
            self.prim_spl[k], self.prim_tgt[k] = pick[0] * TOTAL, pick[1] * TOTAL


@njit(cache=True)
def _com(cf, oc, ocn, e, p):
    x = 0.0
    y = 0.0
    m = 0.0
    for k in range(ocn[e, p]):
        s_ = oc[e, p, k]
        w = cf[e, s_, SZ] * cf[e, s_, SZ]
        x += cf[e, s_, X] * w
        y += cf[e, s_, Y] * w
        m += w
    if m == 0:
        return 0.0, 0.0, 0.0
    return x / m, y / m, m


@njit(cache=True)
def _frame(cf, oc, ocn, e, pa):
    """pair frame: origin = total centre of mass, x-axis from role A's centre to role B's centre"""
    ax, ay, am = _com(cf, oc, ocn, e, pa)
    bx, by, bm = _com(cf, oc, ocn, e, 1 - pa)
    ox = (ax * am + bx * bm) / max(am + bm, 1.0)
    oy = (ay * am + by * bm) / max(am + bm, 1.0)
    dx = bx - ax
    dy = by - ay
    d = math.hypot(dx, dy)
    if d < 1:
        dx = 1.0
        dy = 0.0
        d = 1.0
    return ox, oy, dx / d, dy / d, d, ax, ay, bx, by


@njit(cache=True)
def _sorted_pieces(cf, oc, ocn, e, p, out):
    """slots of player p sorted by mass descending; returns count"""
    n = ocn[e, p]
    for k in range(n):
        out[k] = oc[e, p, k]
    for a in range(n):
        for b in range(a + 1, n):
            if cf[e, out[b], SZ] > cf[e, out[a], SZ]:
                t = out[a]
                out[a] = out[b]
                out[b] = t
    return n


@njit(cache=True)
def build_obs(cf, ci, oc, ocn, g, tr, trf, mirror, E, pend_n, obs):
    N = cf.shape[0]
    tmp = np.empty(S, np.int64)
    for e in range(N):
        for j in range(OBS_DIM):
            obs[e, j] = 0.0
        sw = (tr[e, T_CYC] % 2) if mirror == 1 else 0
        pa = sw                                   # player playing role A
        ox, oy, ux, uy, dist, ax, ay, bx, by = _frame(cf, oc, ocn, e, pa)
        wx = -uy
        wy = ux
        tick = g[e, G_TICK]
        for r in range(2):
            p = pa if r == 0 else 1 - pa
            n = _sorted_pieces(cf, oc, ocn, e, p, tmp)
            for k in range(min(n, MAXPC)):
                s_ = tmp[k]
                rx = cf[e, s_, X] - ox
                ry = cf[e, s_, Y] - oy
                base = (r * MAXPC + k) * NFEAT
                obs[e, base + 0] = (rx * ux + ry * uy) / 2000.0
                obs[e, base + 1] = (rx * wx + ry * wy) / 2000.0
                obs[e, base + 2] = cf[e, s_, SZ] * cf[e, s_, SZ] / 100 / 10000.0
                if ci[e, s_, BOOSTING] == 1:
                    bdx = cf[e, s_, BDX] * cf[e, s_, BD]
                    bdy = cf[e, s_, BDY] * cf[e, s_, BD]
                    obs[e, base + 3] = (bdx * ux + bdy * uy) / 780.0
                    obs[e, base + 4] = (bdx * wx + bdy * wy) / 780.0
                obs[e, base + 5] = min(tick - ci[e, s_, BIRTH], 200) / 100.0
                obs[e, base + 6] = 1.0
                obs[e, base + 7] = 1.0 if ci[e, s_, CANMERGE] == 1 else 0.0
        b = 2 * MAXPC * NFEAT
        obs[e, b + tr[e, T_NEXT]] = 1.0
        b += MAXEV
        obs[e, b + 0] = min(tr[e, T_SINCE_SPLIT], 100) / 50.0
        obs[e, b + 1] = min(tr[e, T_SINCE_PROG], 120) / 60.0
        obs[e, b + 2] = pend_n[e] / 4.0
        obs[e, b + 3] = dist / 2000.0
        obs[e, b + 4] = 1.0 if tr[e, T_NEXT] == 0 else 0.0
        obs[e, b + 5] = min(tr[e, T_BOUND], 10) / 10.0


@njit(cache=True)
def _layout_desc(cf, oc, ocn, e, pa, pos, mass):
    """role-ordered (A then B, by mass desc) positions + masses; returns count"""
    tmp = np.empty(S, np.int64)
    c = 0
    for r in range(2):
        p = pa if r == 0 else 1 - pa
        n = _sorted_pieces(cf, oc, ocn, e, p, tmp)
        for k in range(n):
            s_ = tmp[k]
            pos[c, 0] = cf[e, s_, X]
            pos[c, 1] = cf[e, s_, Y]
            mass[c] = cf[e, s_, SZ] * cf[e, s_, SZ] / 100
            c += 1
    return c


@njit(cache=True)
def env_actions(cf, oc, ocn, tr, mirror, act, mouse, splits_first):
    """act (N, 2, 3): per role [cx, cy, split] with cx, cy in [-1, 1]"""
    N = cf.shape[0]
    for e in range(N):
        sw = (tr[e, T_CYC] % 2) if mirror == 1 else 0
        pa = sw
        ox, oy, ux, uy, dist, ax, ay, bx, by = _frame(cf, oc, ocn, e, pa)
        wx = -uy
        wy = ux
        for r in range(2):
            p = pa if r == 0 else 1 - pa
            cx = ax if r == 0 else bx
            cy = ay if r == 0 else by
            dx = act[e, r, 0] * 3000.0
            dy = act[e, r, 1] * 3000.0
            mouse[e, p, 0] = cx + dx * ux + dy * wx
            mouse[e, p, 1] = cy + dx * uy + dy * wy
            splits_first[e, p] = int(act[e, r, 2])


@njit(cache=True)
def env_post_split(cf, oc, ocn, tr, trf, mirror, E, actors, ks, ev_n, ev_role, ev_m, ev_em,
                   splits_first, pend_n, pend_p, pend_m, pend_em, pend_age, prev_pos, prev_mass, rew, pend_d, next_d):
    """validate the split presses of this step (in technique order) BEFORE the tick is simulated.
    Adds the expected eats of every valid split to the pending list; handles cycle boundaries."""
    N = cf.shape[0]
    pos = np.empty((S, 2))
    mass = np.empty(S)
    for e in range(N):
        if tr[e, T_DONE] != 0:
            continue
        sw = (tr[e, T_CYC] % 2) if mirror == 1 else 0
        pressed = np.zeros(2, np.int64)
        for p in range(2):
            pressed[p] = splits_first[e, p]
        # consume presses in technique order (both players may press in the same step)
        for _ in range(2):
            if pressed[0] == 0 and pressed[1] == 0:
                break
            k = tr[e, T_NEXT]
            p_exp = actors[k] ^ sw
            if pressed[p_exp] != ks[k]:
                break
            pressed[p_exp] = 0
            # cycle boundary: the first split of a cycle; compare with the previous boundary layout
            if k == 0:
                n = _layout_desc(cf, oc, ocn, e, sw, pos, mass)
                if tr[e, T_BOUND] >= 1 and trf[e, TF_PREVN] == n:
                    err = 0.0
                    merr = 0.0
                    cnt = 0
                    for a in range(n):
                        merr += abs(mass[a] - prev_mass[e, a]) / max(prev_mass[e, a], 1.0)
                        for b in range(a + 1, n):
                            d1 = math.hypot(pos[a, 0] - pos[b, 0], pos[a, 1] - pos[b, 1])
                            d0 = math.hypot(prev_pos[e, a, 0] - prev_pos[e, b, 0], prev_pos[e, a, 1] - prev_pos[e, b, 1])
                            err += abs(d1 - d0)
                            cnt += 1
                    err /= max(cnt, 1)
                    merr /= max(n, 1)
                    if merr < 0.05:
                        rew[e] += 2.0 * math.exp(-err / 150.0)
                for a in range(n):
                    prev_pos[e, a, 0] = pos[a, 0]
                    prev_pos[e, a, 1] = pos[a, 1]
                    prev_mass[e, a] = mass[a]
                trf[e, TF_PREVN] = n
                tr[e, T_BOUND] += 1
            # expected eats of this split -> pending (in PLAYER terms)
            for j in range(ev_n[k]):
                q = pend_n[e]
                if q < MAXP:
                    pend_p[e, q] = ev_role[k, j] ^ sw
                    pend_m[e, q] = ev_m[k, j]
                    pend_em[e, q] = ev_em[k, j]
                    pend_age[e, q] = 0
                    pend_d[e, q] = 1e9
                    pend_n[e] = q + 1
            # no reward for the press itself (that was exploited by spamming the right key);
            # only the eats it produces are rewarded
            tr[e, T_SINCE_SPLIT] = 0
            next_d[e] = 1e9                         # the next split's target changes
            tr[e, T_NEXT] = k + 1
            if tr[e, T_NEXT] == E:
                tr[e, T_NEXT] = 0
                tr[e, T_CYC] += 1
                sw = (tr[e, T_CYC] % 2) if mirror == 1 else 0
        if pressed[0] != 0 or pressed[1] != 0:
            rew[e] -= 2.0                           # a split that is not the technique's next split
            tr[e, T_DONE] = 2


@njit(cache=True)
def _eat_gap(cf, oc, ocn, e, p_eater, eater_mass, target_mass, reach):
    """smallest (centre distance - eat radius - reach) over eater pieces of player p_eater with mass
    ~eater_mass and target pieces of the other player with mass ~target_mass; 1e9 if none"""
    best = 1e9
    for a in range(ocn[e, p_eater]):
        i = oc[e, p_eater, a]
        mi = cf[e, i, SZ] * cf[e, i, SZ] / 100
        if abs(mi - eater_mass) > 0.06 * eater_mass + 40:
            continue
        for b in range(ocn[e, 1 - p_eater]):
            j = oc[e, 1 - p_eater, b]
            mj = cf[e, j, SZ] * cf[e, j, SZ] / 100
            if abs(mj - target_mass) > 0.06 * target_mass + 40:
                continue
            d = math.hypot(cf[e, i, X] - cf[e, j, X], cf[e, i, Y] - cf[e, j, Y]) - (cf[e, i, SZ] - cf[e, j, SZ] / 3) - reach
            if d < best:
                best = d
    return max(best, 0.0) if best < 1e8 else 1e9


@njit(cache=True)
def env_shaping(cf, oc, ocn, tr, mirror, actors, prim_spl, prim_tgt, pend_n, pend_p, pend_m, pend_em,
                pend_d, next_d, rew):
    """progress shaping: +SHAPE per 1000 units that (a) each pending intended eat's eater gets closer
    to its target, (b) the next split's splitter gets closer to split reach of its target.
    Measured per item (new items start from their current distance), so starting a split is never
    penalised."""
    N = cf.shape[0]
    for e in range(N):
        if tr[e, T_DONE] != 0:
            continue
        for j in range(pend_n[e]):
            d = _eat_gap(cf, oc, ocn, e, pend_p[e, j], pend_em[e, j], pend_m[e, j], 0.0)
            if d < 1e8 and pend_d[e, j] < 1e8:
                rew[e] += SHAPE * (pend_d[e, j] - d) / 1000.0
            pend_d[e, j] = d
        k = tr[e, T_NEXT]
        sw = (tr[e, T_CYC] % 2) if mirror == 1 else 0
        p = actors[k] ^ sw
        # the launched half flies ~820 units: being within that of its target is "in reach"
        d = _eat_gap(cf, oc, ocn, e, p, prim_spl[k], prim_tgt[k], 800.0)
        if d < 1e8 and next_d[e] < 1e8 and tr[e, T_SINCE_SPLIT] > 0:
            rew[e] += SHAPE * (next_d[e] - d) / 1000.0
        next_d[e] = d


@njit(cache=True)
def env_post_tick(cf, oc, ocn, ev, evn, tr, trf, pend_n, pend_p, pend_m, pend_em, pend_age, rew, pend_d):
    """score the eats of the tick just simulated; timeouts; drift"""
    N = cf.shape[0]
    for e in range(N):
        if tr[e, T_DONE] != 0:
            continue
        tr[e, T_TICKS] += 1
        tr[e, T_SINCE_SPLIT] += 1
        tr[e, T_SINCE_PROG] += 1
        rew[e] -= 0.002
        for q in range(evn[e]):
            eo = int(ev[e, q, EV_EATER_OWN])
            bo = int(ev[e, q, EV_EATEN_OWN])
            if eo == bo:
                rew[e] -= 2.0                       # own merge: never part of a technique
                tr[e, T_DONE] = 3
                break
            m = ev[e, q, EV_MASS]
            em = ev[e, q, EV_EATER_MASS]
            hit = -1
            for j in range(pend_n[e]):
                if pend_p[e, j] == eo and abs(pend_m[e, j] - m) < 0.03 * pend_m[e, j] + 40 \
                        and abs(pend_em[e, j] - em) < 0.03 * pend_em[e, j] + 40:
                    hit = j
                    break
            if hit < 0:
                rew[e] -= 2.0                       # an eat the technique does not have
                tr[e, T_DONE] = 4
                break
            rew[e] += 1.0
            tr[e, T_SINCE_PROG] = 0
            tr[e, T_PN] += 1                        # intended eats achieved this episode
            last = pend_n[e] - 1
            pend_p[e, hit] = pend_p[e, last]
            pend_m[e, hit] = pend_m[e, last]
            pend_em[e, hit] = pend_em[e, last]
            pend_age[e, hit] = pend_age[e, last]
            pend_d[e, hit] = pend_d[e, last]
            pend_n[e] = last
        if tr[e, T_DONE] != 0:
            continue
        for j in range(pend_n[e]):
            pend_age[e, j] += 1
            if pend_age[e, j] > PEND_TIMEOUT:
                rew[e] -= 1.0                       # an intended eat never happened
                tr[e, T_DONE] = 5
        if tr[e, T_SINCE_PROG] > PROGRESS_TIMEOUT and tr[e, T_DONE] == 0:
            rew[e] -= 1.0
            tr[e, T_DONE] = 6
        if (ocn[e, 0] == 0 or ocn[e, 1] == 0) and tr[e, T_DONE] == 0:
            rew[e] -= 1.0
            tr[e, T_DONE] = 7
        ax, ay, am = _com(cf, oc, ocn, e, 0)
        bx, by, bm = _com(cf, oc, ocn, e, 1)
        cx = (ax * am + bx * bm) / max(am + bm, 1.0)
        cy = (ay * am + by * bm) / max(am + bm, 1.0)
        if math.hypot(cx - trf[e, TF_COM0X], cy - trf[e, TF_COM0Y]) > MAX_DRIFT and tr[e, T_DONE] == 0:
            rew[e] -= 1.0
            tr[e, T_DONE] = 8


class SplitrunEnv:
    """N parallel worlds following one technique. Start states come from a StartPool."""

    SIM_ARRAYS = ("cf", "ci", "pc", "pcn", "bc", "bcn", "oc", "ocn", "nf", "ni", "nit", "g")

    def __init__(self, spec, pool, n_envs, max_ticks=1500, seed=0, shaping=False):
        self.spec, self.pool, self.N, self.max_ticks = spec, pool, n_envs, max_ticks
        self.shaping = shaping
        self.sim = Sim(n_envs)
        self.rng = np.random.default_rng(seed)
        N = n_envs
        self.tr = np.zeros((N, NTI), np.int64)
        self.trf = np.zeros((N, NTF))
        self.pend_n = np.zeros(N, np.int64)
        self.pend_p = np.zeros((N, MAXP), np.int64)
        self.pend_m = np.zeros((N, MAXP))
        self.pend_em = np.zeros((N, MAXP))
        self.pend_age = np.zeros((N, MAXP), np.int64)
        self.pend_d = np.full((N, MAXP), 1e9)
        self.next_d = np.full(N, 1e9)
        self.prev_pos = np.zeros((N, S, 2))
        self.prev_mass = np.zeros((N, S))
        self.obs = np.zeros((N, OBS_DIM), np.float32)
        self.rew = np.zeros(N)
        self.mouse = np.zeros((N, 2, 2))
        self.splits = np.zeros((N, 2), np.int64)
        self.no_splits = np.zeros((N, 2), np.int64)
        self.ep_ret = np.zeros(N)
        self.ep_cyc0 = np.zeros(N, np.int64)
        self.eats_per_cycle = float(spec.ev_n.sum())
        self.reset(np.arange(N))

    def reset(self, idx):
        for e in idx:
            self.pool.restore(self, e, self.rng)
        self.sim.active[idx] = 1
        self.ep_ret[idx] = 0
        self.ep_cyc0[idx] = self.tr[idx, T_CYC]

    def observe(self):
        build_obs(self.sim.cf, self.sim.ci, self.sim.oc, self.sim.ocn, self.sim.g, self.tr, self.trf,
                  self.spec.mirror, self.spec.E, self.pend_n, self.obs)
        return self.obs

    def step(self, act):
        """act (N, 2, 3) float: per role cursor x, y in [-1,1] and split in {0,1,2}.
        Returns obs, reward, terminated, truncated, info (finished-episode stats)"""
        env_actions(self.sim.cf, self.sim.oc, self.sim.ocn, self.tr, self.spec.mirror, act, self.mouse, self.splits)
        return self.advance(self.mouse, self.splits, ACTION_REPEAT)

    def advance(self, mouse, splits, repeat):
        """world-coordinate cursors (N,2,2) and split presses (N,2) per PLAYER, for `repeat` ticks"""
        sim, sp = self.sim, self.spec
        self.rew[:] = 0
        self.mouse[:] = mouse
        self.splits[:] = splits
        env_post_split(sim.cf, sim.oc, sim.ocn, self.tr, self.trf, sp.mirror, sp.E, sp.actors, sp.ks,
                       sp.ev_n, sp.ev_role, sp.ev_m, sp.ev_em, self.splits, self.pend_n, self.pend_p,
                       self.pend_m, self.pend_em, self.pend_age, self.prev_pos, self.prev_mass, self.rew,
                       self.pend_d, self.next_d)
        for r in range(repeat):
            sim.active[:] = self.tr[:, T_DONE] == 0
            sim.step(self.mouse, self.splits if r == 0 else self.no_splits)
            env_post_tick(sim.cf, sim.oc, sim.ocn, sim.ev, sim.evn, self.tr, self.trf, self.pend_n, self.pend_p,
                          self.pend_m, self.pend_em, self.pend_age, self.rew, self.pend_d)
            if self.shaping:
                env_shaping(sim.cf, sim.oc, sim.ocn, self.tr, sp.mirror, sp.actors, sp.prim_spl, sp.prim_tgt,
                            self.pend_n, self.pend_p, self.pend_m, self.pend_em, self.pend_d, self.next_d, self.rew)
        term = self.tr[:, T_DONE] != 0
        trunc = (~term) & (self.tr[:, T_TICKS] >= self.max_ticks)
        self.ep_ret += self.rew
        done = term | trunc
        info = None
        if done.any():
            idx = np.nonzero(done)[0]
            info = dict(ret=self.ep_ret[idx].copy(), ticks=self.tr[idx, T_TICKS].copy(),
                        reason=self.tr[idx, T_DONE].copy(),
                        cycles=self.tr[idx, T_PN] / self.eats_per_cycle,
                        splits=None, trunc=trunc[idx].copy())
            self.reset(idx)
        return self.observe(), self.rew.copy(), term, trunc, info


class StartPool:
    """Saved full simulator states (+ tracker state) to start episodes from."""

    def __init__(self, spec):
        self.spec = spec
        self.states = []

    def add_from_sim(self, sim, e, nxt, parity, pend):
        """pend: list of (player, m, em, age) intended eats still outstanding"""
        st = {k: getattr(sim, k)[e].copy() for k in SplitrunEnv.SIM_ARRAYS}
        self.states.append((st, int(nxt), int(parity), [tuple(p) for p in pend]))

    def add_from_env(self, env, e, cap=4000, rng=None):
        """snapshot a live env world (used to grow the pool from the agent's own progress)"""
        pend = [(int(env.pend_p[e, j]), float(env.pend_m[e, j]), float(env.pend_em[e, j]), int(env.pend_age[e, j]))
                for j in range(env.pend_n[e])]
        st = {k: getattr(env.sim, k)[e].copy() for k in SplitrunEnv.SIM_ARRAYS}
        item = (st, int(env.tr[e, T_NEXT]), int(env.tr[e, T_CYC] % 2), pend)
        if len(self.states) < cap:
            self.states.append(item)
        else:
            # keep the original (scripted) states; replace a random grown one
            lo = getattr(self, "n_seed", 0)
            j = lo + int((rng or np.random).integers(len(self.states) - lo))
            self.states[j] = item

    def restore(self, env, e, rng):
        st, nxt, parity, pend = self.states[rng.integers(len(self.states))]
        sim = env.sim
        for k, v in st.items():
            getattr(sim, k)[e] = v
        # episode bookkeeping restarts at tick 0 of the saved world time
        env.tr[e] = 0
        env.tr[e, T_NEXT] = nxt
        env.tr[e, T_CYC] = parity
        env.pend_n[e] = len(pend)
        for j, (p, m, em, age) in enumerate(pend[:MAXP]):
            env.pend_p[e, j], env.pend_m[e, j], env.pend_em[e, j], env.pend_age[e, j] = p, m, em, age
        env.trf[e] = 0
        env.pend_d[e] = 1e9
        env.next_d[e] = 1e9
        cells = sim.cells(e)
        M = sum(c[4] ** 2 for c in cells)
        env.trf[e, TF_COM0X] = sum(c[2] * c[4] ** 2 for c in cells) / M
        env.trf[e, TF_COM0Y] = sum(c[3] * c[4] ** 2 for c in cells) / M

    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.states, f)

    def load(self, path):
        import pickle
        with open(path, "rb") as f:
            self.states = pickle.load(f)
        return self
