"""Discovery environment: no target technique. Reward = clean, periodic splitrunning with many splits.

Rules (checked every tick, per world):
  * every piece a split launches must be USED (eat or be eaten by the other player) within
    USE_WINDOW ticks, otherwise the split was purposeless -> episode ends (-2).
    purpose mode: ONE used piece per split is enough (a big player's split necessarily launches
    halves of all its pieces; in imbalanced splitruns some of them simply stay)
  * every cross-player eat must involve a recently launched piece (age <= YOUNG) -> otherwise the
    eat is a slow absorption, not a splitrun move -> episode ends (-2)
  * a player below 15% of the total mass ends the episode (-2)
  * imbalance mode (imb_max < 0.5): the repeat reward only counts when, at the press, the smaller
    player holds at most imb_max of the total mass
  * own-piece merges end the episode (-2); the pair drifting > MAX_DRIFT ends it (-1); no split for
    IDLE ticks ends it (-1)
Rewards:
  +0.2               each split whose launched pieces were all used (small: +1 was exploited by
                     one-way feeding spam)
  +BONUS * min(k, 8) at a split press, if the layout (masses + all pairwise distances, players may
                     swap) matches the layout at the press k splits earlier -- a repeating orbit with
                     k splits per cycle; longer cycles pay more
  -0.002 per tick
Observation / action: same as splitrun/rlenv.py with roles fixed (role A = player 0).
"""
import math

import numpy as np
from numba import njit

from .sim import (Sim, S, X, Y, SZ, CID, OWN, BIRTH, G_TICK, EX, EV_EATER_OWN, EV_EATEN_OWN, EV_EATER_ID,
                  EV_EATEN_ID, EV_EATER_AGE, EV_EATEN_AGE)
from .rlenv import (build_obs, env_actions, _sorted_pieces, _com, OBS_DIM, ACTION_REPEAT, MAX_DRIFT,
                    T_NEXT, T_CYC, T_BOUND, T_SINCE_SPLIT, T_SINCE_PROG, T_TICKS, T_PN, T_DONE, NTI, NTF,
                    TF_COM0X, TF_COM0Y, StartPool, SplitrunEnv)

USE_WINDOW = 45
YOUNG = 60
IDLE = 100
BONUS = 1.0
RATE = 0.03        # per-tick reward per split of the verified cycle length (per_tick mode)
CLEAN_R = 0.5      # small: a clean split alone is not the goal (one-way feeding spam exploited +1)
MIN_SHARE = 0.15   # a player falling below this share of the mass ends the episode
MAXL = 24          # launched pieces tracked
MAXH = 12          # layouts remembered (one per split press)
NP_ = 16           # pieces per layout


@njit(cache=True)
def _layout(cf, oc, ocn, e, pos, mass, own):
    tmp = np.empty(S, np.int64)
    c = 0
    for p in range(2):
        n = _sorted_pieces(cf, oc, ocn, e, p, tmp)
        for k in range(n):
            if c < NP_:
                s_ = tmp[k]
                pos[c, 0] = cf[e, s_, X]; pos[c, 1] = cf[e, s_, Y]
                mass[c] = cf[e, s_, SZ] * cf[e, s_, SZ] / 100
                own[c] = p
                c += 1
    return c


@njit(cache=True)
def _match(pos, mass, own, n, hpos, hmass, hown, hn):
    """layout distance to a remembered layout (players may swap); 1e9 if masses/counts differ"""
    if n != hn:
        return 1e9
    best = 1e9
    for swap in range(2):
        # order both by (player^swap, mass desc): the stored one is (player, mass desc)
        idx = np.empty(n, np.int64)
        c = 0
        for p in range(2):
            for k in range(n):
                if (own[k] ^ swap) == p:
                    idx[c] = k
                    c += 1
        ok = True
        for a in range(n):
            if hown[a] != (own[idx[a]] ^ swap) or abs(mass[idx[a]] - hmass[a]) > 0.03 * hmass[a] + 30:
                ok = False
                break
        if not ok:
            continue
        err = 0.0
        cnt = 0
        for a in range(n):
            for b in range(a + 1, n):
                d1 = math.hypot(pos[idx[a], 0] - pos[idx[b], 0], pos[idx[a], 1] - pos[idx[b], 1])
                d0 = math.hypot(hpos[a, 0] - hpos[b, 0], hpos[a, 1] - hpos[b, 1])
                err += abs(d1 - d0)
                cnt += 1
        err /= max(cnt, 1)
        if err < best:
            best = err
    return best


@njit(cache=True)
def disc_post_split(cf, ci, oc, ocn, g, tr, splits_first, hist_pos, hist_mass, hist_own, hist_n, hist_k, rew, ban_unit,
                    per_tick, imb_max):
    """at a split press: periodicity bonus vs remembered layouts, then remember this one"""
    N = cf.shape[0]
    pos = np.empty((NP_, 2)); mass = np.empty(NP_); own = np.empty(NP_, np.int64)
    for e in range(N):
        if tr[e, T_DONE] != 0:
            continue
        presses = (1 if splits_first[e, 0] > 0 else 0) + (1 if splits_first[e, 1] > 0 else 0)
        if presses == 0:
            continue
        n = _layout(cf, oc, ocn, e, pos, mass, own)
        count = tr[e, T_PN]                         # split presses so far this episode
        best_k = 0
        for h in range(MAXH):
            if hist_n[e, h] == 0:
                continue
            k = count - hist_k[e, h]
            if k < 2:
                continue
            d = _match(pos, mass, own, n, hist_pos[e, h], hist_mass[e, h], hist_own[e, h], hist_n[e, h])
            if d < 80.0 and (best_k == 0 or k < best_k):
                best_k = k                          # the shortest period that repeats
        if best_k > 0 and ban_unit > 0:
            # novelty: no bonus for cycles on an already-known mass lattice (all pieces ~ multiples of ban_unit)
            on_lattice = True
            for a in range(n):
                q = mass[a] / ban_unit
                if abs(q - round(q)) > 0.03 * max(q, 1.0):
                    on_lattice = False
                    break
            if on_lattice:
                best_k = 0
        if best_k > 0 and imb_max < 0.5:
            # imbalance mode: only orbits where one player holds at most imb_max of the mass count
            ma = 0.0; mb = 0.0
            for a in range(n):
                if own[a] == 0:
                    ma += mass[a]
                else:
                    mb += mass[a]
            if min(ma, mb) > imb_max * (ma + mb):
                best_k = 0
        if per_tick:
            tr[e, T_NEXT] = min(best_k, 8)          # current verified cycle length (0 = not in an orbit)
            if best_k > 0:
                tr[e, T_BOUND] += 1
        elif best_k > 0:
            rew[e] += BONUS * min(best_k, 8)
            tr[e, T_BOUND] += 1                     # number of periodic repeats seen
        slot = count % MAXH
        for a in range(n):
            hist_pos[e, slot, a, 0] = pos[a, 0]; hist_pos[e, slot, a, 1] = pos[a, 1]
            hist_mass[e, slot, a] = mass[a]; hist_own[e, slot, a] = own[a]
        hist_n[e, slot] = n
        hist_k[e, slot] = count
        tr[e, T_PN] = count + presses
        tr[e, T_SINCE_SPLIT] = 0


@njit(cache=True)
def disc_post_tick(cf, ci, oc, ocn, g, ev, evn, tr, trf, split_player, lid, lsplit, lused, ln, sused, sneed, rew,
                   min_flight, per_tick, purpose, max_drift):
    """track launched pieces; score eats; end on messiness"""
    N = cf.shape[0]
    for e in range(N):
        if tr[e, T_DONE] != 0:
            continue
        tick = g[e, G_TICK]
        tr[e, T_TICKS] += 1
        tr[e, T_SINCE_SPLIT] += 1
        rew[e] -= 0.002
        if per_tick:
            rew[e] += RATE * tr[e, T_NEXT]          # longer verified cycles pay more per tick, speed does not
        # pieces launched this tick (born now, owned by a player who pressed split)
        for p in range(2):
            if split_player[e, p] < 0:
                continue
            sid = split_player[e, p]
            born = 0
            for k in range(ocn[e, p]):
                if ci[e, oc[e, p, k], BIRTH] == tick:
                    born += 1
            if born == 0:
                rew[e] -= 1.0; tr[e, T_DONE] = 2        # a split press that launched nothing
                break
            for k in range(ocn[e, p]):
                s_ = oc[e, p, k]
                if ci[e, s_, BIRTH] == tick and ln[e] < MAXL:
                    q = ln[e]
                    lid[e, q] = ci[e, s_, CID]; lsplit[e, q] = sid; lused[e, q] = -1
                    ln[e] = q + 1
                    sneed[e, sid % 16] += 1
        # eats
        for q in range(evn[e]):
            eo = int(ev[e, q, EV_EATER_OWN]); bo = int(ev[e, q, EV_EATEN_OWN])
            if eo == bo:
                rew[e] -= 1.0; tr[e, T_DONE] = 3
                break
            if ev[e, q, EV_EATER_AGE] > YOUNG and ev[e, q, EV_EATEN_AGE] > YOUNG:
                rew[e] -= 1.0; tr[e, T_DONE] = 4       # slow absorption, not a splitrun move
                break
            for j in range(ln[e]):
                if lused[e, j] < 0 and (lid[e, j] == int(ev[e, q, EV_EATER_ID]) or lid[e, j] == int(ev[e, q, EV_EATEN_ID])):
                    age = ev[e, q, EV_EATER_AGE] if lid[e, j] == int(ev[e, q, EV_EATER_ID]) else ev[e, q, EV_EATEN_AGE]
                    if age < min_flight:
                        rew[e] -= 1.0; tr[e, T_DONE] = 10   # point-blank: the piece never really flew
                        break
                    lused[e, j] = tick
                    sid = lsplit[e, j]
                    sused[e, sid % 16] += 1
                    if sused[e, sid % 16] == (1 if purpose else sneed[e, sid % 16]):
                        rew[e] += CLEAN_R           # every piece this split launched has been used (purpose: one)
                        tr[e, T_SINCE_PROG] = 0
        if tr[e, T_DONE] != 0:
            continue
        # compact used entries older than the window
        w = 0
        for j in range(ln[e]):
            keep = True
            if lused[e, j] >= 0 and tick - lused[e, j] > 5:
                keep = False
            if keep:
                lid[e, w] = lid[e, j]; lsplit[e, w] = lsplit[e, j]; lused[e, w] = lused[e, j]
                w += 1
        ln[e] = w
        if tr[e, T_SINCE_SPLIT] > IDLE:
            rew[e] -= 3.0; tr[e, T_DONE] = 6            # doing nothing must be the worst option
        if ocn[e, 0] == 0 or ocn[e, 1] == 0:
            rew[e] -= 1.0; tr[e, T_DONE] = 7
        ax, ay, am = _com(cf, oc, ocn, e, 0)
        bx, by, bm = _com(cf, oc, ocn, e, 1)
        if tr[e, T_DONE] == 0 and min(am, bm) < MIN_SHARE * (am + bm):
            rew[e] -= 1.0; tr[e, T_DONE] = 9            # one-sided: mass drained from one player
        cx = (ax * am + bx * bm) / max(am + bm, 1.0)
        cy = (ay * am + by * bm) / max(am + bm, 1.0)
        if math.hypot(cx - trf[e, TF_COM0X], cy - trf[e, TF_COM0Y]) > max_drift and tr[e, T_DONE] == 0:
            rew[e] -= 1.0; tr[e, T_DONE] = 8


@njit(cache=True)
def disc_unused(cf, ci, oc, ocn, g, tr, lid, lsplit, lused, ln, sused, rew, purpose):
    """a launched piece that is still unused USE_WINDOW ticks after birth ends the episode"""
    N = cf.shape[0]
    for e in range(N):
        if tr[e, T_DONE] != 0:
            continue
        tick = g[e, G_TICK]
        for j in range(ln[e]):
            if lused[e, j] >= 0:
                continue
            # find the piece: if it no longer exists it was used (eaten) or merged (caught elsewhere)
            found = False
            for p in range(2):
                for k in range(ocn[e, p]):
                    s_ = oc[e, p, k]
                    if ci[e, s_, CID] == lid[e, j]:
                        found = True
                        if tick - ci[e, s_, BIRTH] > USE_WINDOW:
                            if purpose and sused[e, lsplit[e, j] % 16] > 0:
                                lused[e, j] = tick      # its split had a purpose: this piece just stays
                            else:
                                rew[e] -= 1.0
                                tr[e, T_DONE] = 2
                        break
                if found:
                    break
            if tr[e, T_DONE] != 0:
                break


class DiscoverEnv(SplitrunEnv):
    """same start pools / observation / actions as SplitrunEnv, discovery reward"""

    def __init__(self, spec, pool, n_envs, max_ticks=1500, seed=0, min_flight=0, ban_unit=0.0, per_tick=False,
                 imb_max=0.5, purpose=False):
        N = n_envs
        self.purpose = purpose
        self.max_drift = MAX_DRIFT
        self.imb_max = imb_max
        self.per_tick = per_tick
        self.min_flight = min_flight
        self.ban_unit = ban_unit
        self.lid = np.zeros((N, MAXL), np.int64); self.lsplit = np.zeros((N, MAXL), np.int64)
        self.lused = np.zeros((N, MAXL), np.int64); self.ln = np.zeros(N, np.int64)
        self.sused = np.zeros((N, 16), np.int64); self.sneed = np.zeros((N, 16), np.int64)
        self.hist_pos = np.zeros((N, MAXH, NP_, 2)); self.hist_mass = np.zeros((N, MAXH, NP_))
        self.hist_own = np.zeros((N, MAXH, NP_), np.int64); self.hist_n = np.zeros((N, MAXH), np.int64)
        self.hist_k = np.zeros((N, MAXH), np.int64)
        self.split_player = np.full((N, 2), -1, np.int64)
        super().__init__(spec, pool, n_envs, max_ticks, seed)
        self.eats_per_cycle = 1.0

    def reset(self, idx):
        super().reset(idx)
        self.tr[idx, T_CYC] = 0              # roles fixed: role A = player 0
        self.ln[idx] = 0; self.sused[idx] = 0; self.sneed[idx] = 0
        self.hist_n[idx] = 0; self.tr[idx, T_PN] = 0; self.tr[idx, T_NEXT] = 0

    def advance(self, mouse, splits, repeat):
        sim = self.sim
        self.rew[:] = 0
        self.mouse[:] = mouse
        self.splits[:] = splits
        disc_post_split(sim.cf, sim.ci, sim.oc, sim.ocn, sim.g, self.tr, self.splits, self.hist_pos, self.hist_mass,
                        self.hist_own, self.hist_n, self.hist_k, self.rew, self.ban_unit, self.per_tick, self.imb_max)
        for r in range(repeat):
            sim.active[:] = self.tr[:, T_DONE] == 0
            if r == 0:
                # split ids: presses are numbered in order (player 0 first); T_PN already counts them
                p0 = self.splits[:, 0] > 0
                p1 = self.splits[:, 1] > 0
                base = self.tr[:, T_PN] - p0 - p1
                self.split_player[:, 0] = np.where(p0, base, -1)
                self.split_player[:, 1] = np.where(p1, base + p0, -1)
                for p in range(2):
                    for e in np.nonzero(self.split_player[:, p] >= 0)[0]:
                        sid = self.split_player[e, p] % 16
                        self.sneed[e, sid] = 0; self.sused[e, sid] = 0
            else:
                self.split_player[:] = -1
            sim.step(self.mouse, self.splits if r == 0 else self.no_splits)
            disc_post_tick(sim.cf, sim.ci, sim.oc, sim.ocn, sim.g, sim.ev, sim.evn, self.tr, self.trf, self.split_player,
                           self.lid, self.lsplit, self.lused, self.ln, self.sused, self.sneed, self.rew, self.min_flight,
                           self.per_tick, self.purpose, self.max_drift)
            disc_unused(sim.cf, sim.ci, sim.oc, sim.ocn, sim.g, self.tr, self.lid, self.lsplit, self.lused, self.ln, self.sused,
                        self.rew, self.purpose)
        term = self.tr[:, T_DONE] != 0
        trunc = (~term) & (self.tr[:, T_TICKS] >= self.max_ticks)
        self.ep_ret += self.rew
        done = term | trunc
        info = None
        if done.any():
            idx = np.nonzero(done)[0]
            info = dict(ret=self.ep_ret[idx].copy(), ticks=self.tr[idx, T_TICKS].copy(), reason=self.tr[idx, T_DONE].copy(),
                        cycles=self.tr[idx, T_BOUND].astype(float), splits=self.tr[idx, T_PN].copy(), trunc=trunc[idx].copy())
            self.reset(idx)
        return self.observe(), self.rew.copy(), term, trunc, info

    def step(self, act):
        env_actions(self.sim.cf, self.sim.oc, self.sim.ocn, self.tr, 0, act, self.mouse, self.splits)
        return self.advance(self.mouse, self.splits, ACTION_REPEAT)

    def observe(self):
        build_obs(self.sim.cf, self.sim.ci, self.sim.oc, self.sim.ocn, self.sim.g, self.tr, self.trf,
                  0, 1, self.pend_n, self.obs)
        return self.obs
