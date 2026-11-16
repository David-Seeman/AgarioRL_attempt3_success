"""Neutral balls for the feeding mode: K wandering balls of many sizes that player pieces can eat.

Deterministic (a hash of ball id, spawn count, tick and a per-world seed decides everything), so the
planner's copies of a world predict the balls exactly. Balls never eat players.
* Motion is smooth: the heading is interpolated (cosine) between hashed key headings every TURN ticks,
  and balls near the map edge steer back towards the middle instead of bouncing.
* Sizes: a fixed wide variety from the start: mass log-uniform in [MMIN, MMAX] (respawns use the same range).
* Eating works both ways with the normal rule (size >= 1.14 x the other's, centre within size - other/3):
  a player piece eats a smaller ball (the ball respawns elsewhere), and a big ball eats a player piece
  (the ball grows) -- the duo has to steer around balls bigger than its pieces.
  Eats of balls >= WORTHY x team mass count as 'big' (the goal); smaller ones are simply ignored.
"""
import math
import numpy as np
from numba import njit
from .sim import X, Y, SZ, CID, EX, OWN, G_TICK, update_cell, remove_cell

MAPR = 8800.0              # balls live inside +-MAPR
TURN = 60                  # ticks between key headings
MMIN, MMAX = 30.0, 30000.0
WORTHY = 1.0 / 3.0


@njit(cache=True)
def _h(a, b, c):
    """integer hash -> float in [0, 1)"""
    x = (a * 73856093) ^ (b * 19349663) ^ (c * 83492791) ^ 0x5bd1e995
    x = (x ^ (x >> 13)) * 1274126177
    x = x ^ (x >> 16)
    return (x & 0xFFFFFF) / 16777216.0


@njit(cache=True)
def _spawn(bx, by, bs, bt, e, k, n, seed, team, tick):
    bx[e, k] = (2 * _h(seed, k, n * 7 + 1) - 1) * MAPR
    by[e, k] = (2 * _h(seed, k, n * 7 + 2) - 1) * MAPR
    m = MMIN * math.exp(_h(seed, k, n * 7 + 3) * math.log(MMAX / MMIN))
    bs[e, k] = math.sqrt(m * 100)
    bt[e, k] = tick


@njit(cache=True)
def _heading(seed, k, n, tick):
    """smoothly varying heading: cosine interpolation between hashed key angles (shortest way round)"""
    i = tick // TURN
    f = (tick % TURN) / TURN
    a0 = 2 * math.pi * _h(seed, k * 131 + n, i)
    a1 = 2 * math.pi * _h(seed, k * 131 + n, i + 1)
    d = a1 - a0
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    w = (1 - math.cos(math.pi * f)) / 2
    return a0 + d * w


def init_balls(N, K, seeds, team=37000.0, tick=0, clear=3000.0):
    """spawn K balls per world; none within `clear` of the map centre (where the team starts)"""
    bx = np.zeros((N, K)); by = np.zeros((N, K)); bs = np.zeros((N, K)); bn = np.zeros((N, K), np.int64)
    bt = np.zeros((N, K), np.int64)
    for e in range(N):
        for k in range(K):
            _spawn(bx, by, bs, bt, e, k, 0, int(seeds[e]), team, tick)
            while math.hypot(bx[e, k], by[e, k]) < clear:
                bn[e, k] += 1
                _spawn(bx, by, bs, bt, e, k, bn[e, k], int(seeds[e]), team, tick)
    return bx, by, bs, bn, bt


@njit(cache=True)
def team_mass(cf, ci, e):
    t = 0.0
    for s in range(cf.shape[1]):
        if ci[e, s, EX] == 1:
            t += cf[e, s, SZ] * cf[e, s, SZ] / 100
    return t


@njit(cache=True)
def balls_tick(cf, ci, pc, pcn, bc, bcn, oc, ocn, nf, ni, nit, g, P, active, bx, by, bs, bn, bt, seeds,
               pids, pn, pdl, pok, score, gained, gained_small, lost, purpose_w):
    N = cf.shape[0]
    K = bx.shape[1]
    for e in range(N):
        if active[e] == 0:
            continue
        tick = g[e, G_TICK]
        seed = seeds[e]
        team = team_mass(cf, ci, e)
        for k in range(K):
            ang = _heading(seed, k, bn[e, k], tick)
            x, y = bx[e, k], by[e, k]
            r = math.hypot(x, y)
            if r > 0.9 * MAPR:
                # near the edge: turn smoothly back towards the middle
                w = min(1.0, (r - 0.9 * MAPR) / (0.1 * MAPR))
                ci_ = math.atan2(-y, -x)
                d = ci_ - ang
                while d > math.pi:
                    d -= 2 * math.pi
                while d < -math.pi:
                    d += 2 * math.pi
                ang += d * w
            v = 50.0 * bs[e, k] ** -0.4396754
            bx[e, k] = x + math.cos(ang) * v
            by[e, k] = y + math.sin(ang) * v
            # a big ball eats a player piece (the ball grows)
            eaten_piece = False
            for p in range(2):
                for j in range(ocn[e, p] - 1, -1, -1):
                    c = oc[e, p, j]
                    sc = cf[e, c, SZ]
                    if bs[e, k] >= 1.14 * sc and math.hypot(cf[e, c, X] - bx[e, k], cf[e, c, Y] - by[e, k]) < bs[e, k] - sc / 3:
                        lost[e] += sc * sc / 100
                        bs[e, k] = math.sqrt(bs[e, k] * bs[e, k] + sc * sc)
                        remove_cell(cf, ci, pc, pcn, bc, bcn, oc, ocn, ni, nit, e, c)
                        eaten_piece = True
                        break
                if eaten_piece:
                    break
            if eaten_piece:
                continue
            # eaten by a player piece?
            for p in range(2):
                done = False
                for j in range(ocn[e, p]):
                    c = oc[e, p, j]
                    sc = cf[e, c, SZ]
                    if sc < 1.14 * bs[e, k]:
                        continue
                    d = math.hypot(cf[e, c, X] - bx[e, k], cf[e, c, Y] - by[e, k])
                    if d < sc - bs[e, k] / 3:
                        m = bs[e, k] * bs[e, k] / 100
                        cf[e, c, SZ] = math.sqrt(sc * sc + bs[e, k] * bs[e, k])
                        update_cell(cf, ci, nf, ni, nit, g, P, e, c)
                        if m >= WORTHY * team:
                            gained[e] += m
                        else:
                            gained_small[e] += m
                        cid = ci[e, c, CID]
                        # eating a ball gives a pending split its purpose
                        for s in range(pdl.shape[1]):
                            if pdl[e, s] < 0 or pok[e, s] == 1:
                                continue
                            for q in range(pn[e, s]):
                                if pids[e, s, q] == cid:
                                    pok[e, s] = 1
                                    score[e] += purpose_w
                                    break
                        bn[e, k] += 1
                        _spawn(bx, by, bs, bt, e, k, bn[e, k], seed, team + m, tick)
                        done = True
                        break
                if done:
                    break


@njit(cache=True)
def ball_danger(cf, oc, ocn, bx, by, bs, out):
    """how close each world's pieces are to balls that could eat them (mass-weighted, 0 = safe)"""
    N = cf.shape[0]
    K = bx.shape[1]
    for e in range(N):
        out[e] = 0.0
        for p in range(2):
            for j in range(ocn[e, p]):
                c = oc[e, p, j]
                sc = cf[e, c, SZ]
                for k in range(K):
                    if bs[e, k] < 1.14 * sc:
                        continue
                    gap = math.hypot(cf[e, c, X] - bx[e, k], cf[e, c, Y] - by[e, k]) - (bs[e, k] - sc / 3)
                    if gap < 700:
                        out[e] += (700 - gap) / 700 * (sc * sc / 100) / 1000
