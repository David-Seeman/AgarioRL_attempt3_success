"""Batched, exact re-implementation of the imsolo (OgarII fork) tick for two players.

Every function below mirrors a function in reference_server/Megasplit-Beta/src and
performs the same floating-point operations in the same order, including the
quadtree that decides the order collisions are resolved in, and the list-order
quirks of World.liveUpdate().  tests/test_fidelity.py diffs this against the real
server code cell-by-cell.

Only what can happen in the splitrun environment is ported: player cells, moving,
splitting, rigid collisions, eating/merging, autosplit, borders.  No food,
viruses, ejecting, minions or decay (decay is ported but configured to 0).
"""
import json
import math
import os

import numpy as np
from numba import njit, prange

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

S = 32          # cell slots per env (2 players x 16 cells)
NNODES = 96     # quadtree node pool per env
MAXEV = 64      # eat events recorded per env per tick

# float cell fields
X, Y, SZ, BDX, BDY, BD, RX, RY, RS = range(9)
NCF = 9
# int cell fields
EX, OWN, CID, BIRTH, BOOSTING, CANMERGE, QNODE = range(7)
NCI = 7
# quadtree node int fields
N_LEVEL, N_PARENT, N_HASSPLIT, N_CH0, N_NITEMS, N_INUSE = 0, 1, 2, 3, 7, 8
NNI = 9
# per-env globals
G_TICK, G_NEXTID, G_RNG, G_OVERFLOW = range(4)
NG = 4
# event fields: eater owner, eaten owner, eaten mass, eater id, eaten id, eater age, eaten age, eater boost d
EV_EATER_OWN, EV_EATEN_OWN, EV_MASS, EV_EATER_ID, EV_EATEN_ID, EV_EATER_AGE, EV_EATEN_AGE, EV_EATER_BOOST, EV_EATER_MASS = range(9)
NEVF = 9

# settings vector indices
(P_MAPX, P_MAPY, P_MAPW, P_MAPH, P_FMAXLEVEL, P_FMAXITEMS, P_EATMULT, P_EATOVERLAPDIV,
 P_MINSIZE, P_MAXSIZE, P_MINSPLITSIZE, P_SPLITCAP, P_MAXCELLS, P_MOVEMULT, P_SPLITSIZEDIV,
 P_SPLITDIST, P_SPLITBOOST, P_NOCOLLIDE, P_NOMERGE, P_MERGEOLD, P_MERGETIME, P_MERGEINC,
 P_DECAY, P_STEPMULT) = range(24)
NP = 24


def load_settings(path=None):
    with open(path or os.path.join(ROOT, "env_settings.json")) as f:
        s = json.load(f)
    p = np.zeros(NP)
    p[P_MAPX], p[P_MAPY], p[P_MAPW], p[P_MAPH] = s["worldMapX"], s["worldMapY"], s["worldMapW"], s["worldMapH"]
    p[P_FMAXLEVEL], p[P_FMAXITEMS] = s["worldFinderMaxLevel"], s["worldFinderMaxItems"]
    p[P_EATMULT], p[P_EATOVERLAPDIV] = s["worldEatMult"], s["worldEatOverlapDiv"]
    p[P_MINSIZE], p[P_MAXSIZE], p[P_MINSPLITSIZE] = s["playerMinSize"], s["playerMaxSize"], s["playerMinSplitSize"]
    p[P_SPLITCAP], p[P_MAXCELLS], p[P_MOVEMULT] = s["playerSplitCap"], s["playerMaxCells"], s["playerMoveMult"]
    p[P_SPLITSIZEDIV], p[P_SPLITDIST], p[P_SPLITBOOST] = s["playerSplitSizeDiv"], s["playerSplitDistance"], s["playerSplitBoost"]
    p[P_NOCOLLIDE], p[P_NOMERGE] = s["playerNoCollideDelay"], s["playerNoMergeDelay"]
    p[P_MERGEOLD] = 1.0 if s["playerMergeVersion"] == "old" else 0.0
    p[P_MERGETIME], p[P_MERGEINC], p[P_DECAY] = s["playerMergeTime"], s["playerMergeTimeIncrease"], s["playerDecayMult"]
    p[P_STEPMULT] = (1000.0 / s["serverFrequency"]) / 40.0
    assert s["playerMaxCells"] * 2 <= S
    return p, s


# --------------------------------------------------------------------------- rng
@njit(cache=True, inline="always")
def _imul(a, b):
    return (a * b) & np.uint64(0xFFFFFFFF)


@njit(cache=True)
def rng_next(g, e):
    """mulberry32, bit-identical to the seeded Math.random in tools/trace_server.js"""
    M = np.uint64(0xFFFFFFFF)
    s = (np.uint64(g[e, G_RNG]) + np.uint64(0x6D2B79F5)) & M
    g[e, G_RNG] = np.int64(s)
    t = s
    t = _imul(t ^ (t >> np.uint64(15)), t | np.uint64(1))
    t = t ^ ((t + _imul(t ^ (t >> np.uint64(7)), t | np.uint64(61))) & M)
    return float((t ^ (t >> np.uint64(14))) & M) / 4294967296.0


# --------------------------------------------------------------------------- quadtree (primitives/QuadTree.js)
@njit(cache=True, inline="always")
def _quadrant(nf, e, n, rx, ry, rs):
    cx = nf[e, n, 0]
    cy = nf[e, n, 1]
    t = ry - rs < cy and ry + rs < cy
    b = ry - rs > cy and ry + rs > cy
    l = rx - rs < cx and rx + rs < cx
    r = rx - rs > cx and rx + rs > cx
    if t:
        if l:
            return 0
        if r:
            return 1
    if b:
        if l:
            return 2
        if r:
            return 3
    return -1


@njit(cache=True)
def _alloc_node(nf, ni, g, e, x, y, w, h, level, parent):
    for n in range(NNODES):
        if ni[e, n, N_INUSE] == 0:
            nf[e, n, 0] = x
            nf[e, n, 1] = y
            nf[e, n, 2] = w
            nf[e, n, 3] = h
            ni[e, n, N_LEVEL] = level
            ni[e, n, N_PARENT] = parent
            ni[e, n, N_HASSPLIT] = 0
            ni[e, n, N_NITEMS] = 0
            ni[e, n, N_INUSE] = 1
            return n
    g[e, G_OVERFLOW] = 1
    return 0


@njit(cache=True)
def _node_push(ni, nit, ci, e, n, slot):
    k = ni[e, n, N_NITEMS]
    nit[e, n, k] = slot
    ni[e, n, N_NITEMS] = k + 1
    ci[e, slot, QNODE] = n


@njit(cache=True)
def _node_remove(ni, nit, e, n, slot):
    k = ni[e, n, N_NITEMS]
    j = 0
    while nit[e, n, j] != slot:
        j += 1
    for q in range(j, k - 1):
        nit[e, n, q] = nit[e, n, q + 1]
    ni[e, n, N_NITEMS] = k - 1


@njit(cache=True)
def _qt_descend_push(nf, ni, nit, ci, e, quad, slot, rx, ry, rs):
    """first half of QuadTree.insert(): walk down from `quad`, push; returns the node pushed to"""
    while True:
        if ni[e, quad, N_HASSPLIT] == 0:
            break
        qd = _quadrant(nf, e, quad, rx, ry, rs)
        if qd == -1:
            break
        quad = ni[e, quad, N_CH0 + qd]
    _node_push(ni, nit, ci, e, quad, slot)
    return quad


@njit(cache=True)
def _qt_split(node, cf, ci, nf, ni, nit, g, P, e):
    """QuadTree.split() on `node`, including the nested split() calls that the JS
    makes from inside branch.insert() -- emulated with an explicit call stack so the
    order of operations is identical (numba cannot safely cache recursive code)."""
    fn = np.empty(64, np.int64)   # node of each frame
    fi = np.empty(64, np.int64)   # loop index i (-1 = frame not entered yet)
    fl = np.empty(64, np.int64)   # loop bound l
    fp = np.empty(64, np.int64)   # 1 = waiting for a child split before splicing items[i]
    sp = 0
    fn[0] = node
    fi[0] = -1
    fp[0] = 0
    sp = 1
    while sp > 0:
        f = sp - 1
        n = fn[f]
        if fi[f] == -1:
            if ni[e, n, N_HASSPLIT] == 1 or ni[e, n, N_LEVEL] > P[P_FMAXLEVEL] or ni[e, n, N_NITEMS] < P[P_FMAXITEMS]:
                sp -= 1
                _qt_after_child(fi, fl, fp, ni, nit, e, fn, sp)
                continue
            ni[e, n, N_HASSPLIT] = 1
            x = nf[e, n, 0]
            y = nf[e, n, 1]
            hw = nf[e, n, 2] / 2
            hh = nf[e, n, 3] / 2
            lv = ni[e, n, N_LEVEL] + 1
            ni[e, n, N_CH0 + 0] = _alloc_node(nf, ni, g, e, x - hw, y - hh, hw, hh, lv, n)
            ni[e, n, N_CH0 + 1] = _alloc_node(nf, ni, g, e, x + hw, y - hh, hw, hh, lv, n)
            ni[e, n, N_CH0 + 2] = _alloc_node(nf, ni, g, e, x - hw, y + hh, hw, hh, lv, n)
            ni[e, n, N_CH0 + 3] = _alloc_node(nf, ni, g, e, x + hw, y + hh, hw, hh, lv, n)
            fi[f] = 0
            fl[f] = ni[e, n, N_NITEMS]
        called = False
        while fi[f] < fl[f]:
            it = nit[e, n, fi[f]]
            qd = _quadrant(nf, e, n, cf[e, it, RX], cf[e, it, RY], cf[e, it, RS])
            if qd == -1:
                fi[f] += 1
                continue
            quad = _qt_descend_push(nf, ni, nit, ci, e, ni[e, n, N_CH0 + qd], it,
                                    cf[e, it, RX], cf[e, it, RY], cf[e, it, RS])
            fp[f] = 1
            if sp >= 64:
                g[e, G_OVERFLOW] = 3
                return
            fn[sp] = quad
            fi[sp] = -1
            fp[sp] = 0
            sp += 1
            called = True
            break
        if not called:
            sp -= 1
            _qt_after_child(fi, fl, fp, ni, nit, e, fn, sp)


@njit(cache=True)
def _qt_after_child(fi, fl, fp, ni, nit, e, fn, sp):
    """a frame returned: the parent frame (if any) finishes `items.splice(i, 1); i--; l--;`"""
    if sp == 0:
        return
    f = sp - 1
    if fp[f] == 0:
        return
    fp[f] = 0
    n = fn[f]
    i = fi[f]
    k = ni[e, n, N_NITEMS]
    for q in range(i, k - 1):
        nit[e, n, q] = nit[e, n, q + 1]
    ni[e, n, N_NITEMS] = k - 1
    fl[f] -= 1


@njit(cache=True)
def _qt_merge(ni, e, quad):
    while quad != -1:
        if ni[e, quad, N_HASSPLIT] == 0:
            quad = ni[e, quad, N_PARENT]
            continue
        for b in range(4):
            br = ni[e, quad, N_CH0 + b]
            if ni[e, br, N_HASSPLIT] == 1 or ni[e, br, N_NITEMS] > 0:
                return
        ni[e, quad, N_HASSPLIT] = 0
        for b in range(4):
            ni[e, ni[e, quad, N_CH0 + b], N_INUSE] = 0


@njit(cache=True)
def qt_insert(cf, ci, nf, ni, nit, g, P, e, slot):
    quad = _qt_descend_push(nf, ni, nit, ci, e, 0, slot, cf[e, slot, RX], cf[e, slot, RY], cf[e, slot, RS])
    _qt_split(quad, cf, ci, nf, ni, nit, g, P, e)


@njit(cache=True)
def qt_remove(ci, ni, nit, e, slot):
    quad = ci[e, slot, QNODE]
    _node_remove(ni, nit, e, quad, slot)
    ci[e, slot, QNODE] = -1
    _qt_merge(ni, e, quad)


@njit(cache=True)
def qt_update(cf, ci, nf, ni, nit, g, P, e, slot):
    old = ci[e, slot, QNODE]
    # climbs to the root: Misc.fullyIntersects() can never be true for positive sizes
    newq = old
    while ni[e, newq, N_PARENT] != -1:
        newq = ni[e, newq, N_PARENT]
    rx = cf[e, slot, RX]
    ry = cf[e, slot, RY]
    rs = cf[e, slot, RS]
    while True:
        if ni[e, newq, N_HASSPLIT] == 0:
            break
        qd = _quadrant(nf, e, newq, rx, ry, rs)
        if qd == -1:
            break
        newq = ni[e, newq, N_CH0 + qd]
    if old == newq:
        return
    _node_remove(ni, nit, e, old, slot)
    _node_push(ni, nit, ci, e, newq, slot)
    _qt_merge(ni, e, old)
    _qt_split(newq, cf, ci, nf, ni, nit, g, P, e)


@njit(cache=True)
def qt_search(cf, nf, ni, nit, e, x, y, w, out, stack):
    """QuadTree.search(range) in the exact callback order; returns count written to out."""
    cnt = 0
    sp = 0
    stack[0] = 0
    sp = 1
    while sp > 0:
        sp -= 1
        n = stack[sp]
        for j in range(ni[e, n, N_NITEMS]):
            it = nit[e, n, j]
            bx = cf[e, it, RX]
            by = cf[e, it, RY]
            bw = cf[e, it, RS]
            if x - w <= bx + bw and x + w >= bx - bw and y - w <= by + bw and y + w >= by - bw:
                out[cnt] = it
                cnt += 1
        if ni[e, n, N_HASSPLIT] == 0:
            continue
        cx = nf[e, n, 0]
        cy = nf[e, n, 1]
        t = y - w < cy or y + w < cy
        b = y - w > cy or y + w > cy
        l = x - w < cx or x + w < cx
        r = x - w > cx or x + w > cx
        # push in reverse so that branch 0 is visited first (pre-order DFS)
        if b and r:
            stack[sp] = ni[e, n, N_CH0 + 3]
            sp += 1
        if b and l:
            stack[sp] = ni[e, n, N_CH0 + 2]
            sp += 1
        if t and r:
            stack[sp] = ni[e, n, N_CH0 + 1]
            sp += 1
        if t and l:
            stack[sp] = ni[e, n, N_CH0 + 0]
            sp += 1
    return cnt


# --------------------------------------------------------------------------- world (worlds/World.js)
@njit(cache=True)
def _list_remove(arr, e, n, v):
    j = 0
    while arr[e, j] != v:
        j += 1
    for q in range(j, n - 1):
        arr[e, q] = arr[e, q + 1]


@njit(cache=True)
def update_cell(cf, ci, nf, ni, nit, g, P, e, slot):
    cf[e, slot, RX] = cf[e, slot, X]
    cf[e, slot, RY] = cf[e, slot, Y]
    cf[e, slot, RS] = cf[e, slot, SZ]
    qt_update(cf, ci, nf, ni, nit, g, P, e, slot)


@njit(cache=True)
def add_cell(cf, ci, pc, pcn, bc, bcn, oc, ocn, nf, ni, nit, g, P, e, owner, x, y, size):
    slot = -1
    for s_ in range(S):
        if ci[e, s_, EX] == 0:
            slot = s_
            break
    if slot == -1:
        g[e, G_OVERFLOW] = 2
        return -1
    ci[e, slot, EX] = 1
    ci[e, slot, OWN] = owner
    ci[e, slot, CID] = g[e, G_NEXTID]
    g[e, G_NEXTID] += 1
    ci[e, slot, BIRTH] = g[e, G_TICK]
    ci[e, slot, BOOSTING] = 0
    ci[e, slot, CANMERGE] = 0
    cf[e, slot, X] = x
    cf[e, slot, Y] = y
    cf[e, slot, SZ] = size
    cf[e, slot, BDX] = 0.0
    cf[e, slot, BDY] = 0.0
    cf[e, slot, BD] = 0.0
    cf[e, slot, RX] = x
    cf[e, slot, RY] = y
    cf[e, slot, RS] = size
    qt_insert(cf, ci, nf, ni, nit, g, P, e, slot)
    # PlayerCell.onSpawned: ownedCells.push, playerCells.unshift
    oc[e, owner, ocn[e, owner]] = slot
    ocn[e, owner] += 1
    for q in range(pcn[e], 0, -1):
        pc[e, q] = pc[e, q - 1]
    pc[e, 0] = slot
    pcn[e] += 1
    return slot


@njit(cache=True)
def set_boosting(ci, bc, bcn, e, slot):
    if ci[e, slot, BOOSTING] == 1:
        return
    ci[e, slot, BOOSTING] = 1
    bc[e, bcn[e]] = slot
    bcn[e] += 1


@njit(cache=True)
def set_not_boosting(ci, bc, bcn, e, slot):
    if ci[e, slot, BOOSTING] == 0:
        return
    ci[e, slot, BOOSTING] = 0
    _list_remove(bc, e, bcn[e], slot)
    bcn[e] -= 1


@njit(cache=True)
def remove_cell(cf, ci, pc, pcn, bc, bcn, oc, ocn, ni, nit, e, slot):
    _list_remove(pc, e, pcn[e], slot)
    pcn[e] -= 1
    o = ci[e, slot, OWN]
    n = ocn[e, o]
    j = 0
    while oc[e, o, j] != slot:
        j += 1
    for q in range(j, n - 1):
        oc[e, o, q] = oc[e, o, q + 1]
    ocn[e, o] = n - 1
    qt_remove(ci, ni, nit, e, slot)
    set_not_boosting(ci, bc, bcn, e, slot)
    ci[e, slot, EX] = 0


@njit(cache=True)
def bounce_cell(cf, P, e, slot, bounce):
    r = cf[e, slot, SZ] / 2
    bx = P[P_MAPX]
    by = P[P_MAPY]
    bw = P[P_MAPW]
    bh = P[P_MAPH]
    if cf[e, slot, X] <= bx - bw + r:
        cf[e, slot, X] = bx - bw + r
        if bounce:
            cf[e, slot, BDX] = -cf[e, slot, BDX]
    if cf[e, slot, X] >= bx + bw - r:
        cf[e, slot, X] = bx + bw - r
        if bounce:
            cf[e, slot, BDX] = -cf[e, slot, BDX]
    if cf[e, slot, Y] <= by - bh + r:
        cf[e, slot, Y] = by - bh + r
        if bounce:
            cf[e, slot, BDY] = -cf[e, slot, BDY]
    if cf[e, slot, Y] >= by + bh - r:
        cf[e, slot, Y] = by + bh - r
        if bounce:
            cf[e, slot, BDY] = -cf[e, slot, BDY]


@njit(cache=True)
def launch_cell(cf, ci, pc, pcn, bc, bcn, oc, ocn, nf, ni, nit, g, P, e, slot, size, dx, dy, d):
    sq = cf[e, slot, SZ] * cf[e, slot, SZ]
    cf[e, slot, SZ] = math.sqrt(sq - size * size)
    x = cf[e, slot, X] + P[P_SPLITDIST] * dx
    y = cf[e, slot, Y] + P[P_SPLITDIST] * dy
    ns = add_cell(cf, ci, pc, pcn, bc, bcn, oc, ocn, nf, ni, nit, g, P, e, ci[e, slot, OWN], x, y, size)
    if ns < 0:
        return
    cf[e, ns, BDX] = dx
    cf[e, ns, BDY] = dy
    cf[e, ns, BD] = d
    set_boosting(ci, bc, bcn, e, ns)


@njit(cache=True)
def eat_result(cf, ci, P, e, tick, a, b):
    """PlayerCell.getEatResult(a -> b): 0 none, 1 rigid, 2 eat"""
    if ci[e, a, OWN] == ci[e, b, OWN]:
        delay = P[P_NOCOLLIDE]
        if (tick - ci[e, b, BIRTH]) * P[P_STEPMULT] < delay or (tick - ci[e, a, BIRTH]) * P[P_STEPMULT] < delay:
            return 0
        if ci[e, a, CANMERGE] == 1 and ci[e, b, CANMERGE] == 1:
            return 2
        return 1
    if cf[e, b, SZ] * P[P_EATMULT] > cf[e, a, SZ]:
        return 0
    return 2


@njit(cache=True)
def _jsround(v):
    return math.floor(v + 0.5)


@njit(cache=True)
def world_tick(cf, ci, pc, pcn, bc, bcn, oc, ocn, nf, ni, nit, g, P, e, mouse, splits, ev, evn,
               rigid, eat, found, stack):
    g[e, G_TICK] += 1
    tick = g[e, G_TICK]
    sm = P[P_STEPMULT]
    evn[e] = 0

    # cells[i].onTick() -> PlayerCell._canMerge
    for s_ in range(S):
        if ci[e, s_, EX] == 1:
            delay = P[P_NOMERGE]
            if P[P_MERGETIME] > 0:
                initial = _jsround(25 * P[P_MERGETIME])
                increase = _jsround(25 * cf[e, s_, SZ] * P[P_MERGEINC])
                if P[P_MERGEOLD] == 1.0:
                    delay = max(delay, initial + increase)
                else:
                    delay = max(delay, max(initial, increase))
            ci[e, s_, CANMERGE] = 1 if (tick - ci[e, s_, BIRTH]) * sm >= delay else 0

    # boosting cells
    i = 0
    l = bcn[e]
    while i < l:
        c = bc[e, i]
        d = cf[e, c, BD] / 9 * sm
        cf[e, c, X] += cf[e, c, BDX] * d
        cf[e, c, Y] += cf[e, c, BDY] * d
        bounce_cell(cf, P, e, c, True)
        update_cell(cf, ci, nf, ni, nit, g, P, e, c)
        cf[e, c, BD] -= d
        if cf[e, c, BD] >= 1:
            i += 1
        else:
            set_not_boosting(ci, bc, bcn, e, c)
            l -= 1

    # player cells: move, decay, autosplit, bounce, update.  playerCells may be
    # unshifted by autosplit mid-loop; like the server we re-read by index.
    i = 0
    l = pcn[e]
    while i < l:
        c = pc[e, i]
        o = ci[e, c, OWN]
        # movePlayerCell
        dx = mouse[e, o, 0] - cf[e, c, X]
        dy = mouse[e, o, 1] - cf[e, c, Y]
        dd = math.sqrt(dx * dx + dy * dy)
        if dd >= 1:
            dx /= dd
            dy /= dd
            ms = 88 * math.pow(cf[e, c, SZ], -0.4396754) * P[P_MOVEMULT]
            m = min(ms, dd) * sm
            cf[e, c, X] += dx * m
            cf[e, c, Y] += dy * m
        # decayPlayerCell
        sz = cf[e, c, SZ]
        ns = sz - sz * P[P_DECAY] / 50 * sm
        cf[e, c, SZ] = max(ns, P[P_MINSIZE])
        # autosplitPlayerCell
        minsplit = P[P_MAXSIZE] * P[P_MAXSIZE]
        cells_left = 1 + P[P_MAXCELLS] - ocn[e, o]
        sq = cf[e, c, SZ] * cf[e, c, SZ]
        overflow = math.ceil(sq / minsplit)
        if not (overflow == 1 or cells_left <= 0):
            times = int(min(overflow, cells_left))
            ssize = min(math.sqrt(sq / times), P[P_MAXSIZE])
            for _k in range(1, times):
                ang = rng_next(g, e) * 2 * math.pi
                launch_cell(cf, ci, pc, pcn, bc, bcn, oc, ocn, nf, ni, nit, g, P, e, c, ssize,
                            math.sin(ang), math.cos(ang), P[P_SPLITBOOST])
            cf[e, c, SZ] = ssize
        bounce_cell(cf, P, e, c, False)
        update_cell(cf, ci, nf, ni, nit, g, P, e, c)
        i += 1

    # broad phase in playerCells order
    nr = 0
    ne = 0
    for i in range(pcn[e]):
        c = pc[e, i]
        nfound = qt_search(cf, nf, ni, nit, e, cf[e, c, RX], cf[e, c, RY], cf[e, c, RS], found, stack)
        for j in range(nfound):
            other = found[j]
            if other == c:
                continue
            r = eat_result(cf, ci, P, e, tick, c, other)
            if r == 1:
                rigid[nr, 0] = c
                rigid[nr, 1] = other
                nr += 1
            elif r == 2:
                eat[ne, 0] = c
                eat[ne, 1] = other
                ne += 1

    # resolveRigidCheck
    for k in range(nr):
        a = rigid[k, 0]
        b = rigid[k, 1]
        dx = cf[e, b, X] - cf[e, a, X]
        dy = cf[e, b, Y] - cf[e, a, Y]
        d = math.sqrt(dx * dx + dy * dy)
        m = cf[e, a, SZ] + cf[e, b, SZ] - d
        if m <= 0:
            continue
        if d == 0:
            d = 1.0
            dx = 1.0
            dy = 0.0
        else:
            dx /= d
            dy /= d
        asq = cf[e, a, SZ] * cf[e, a, SZ]
        bsq = cf[e, b, SZ] * cf[e, b, SZ]
        M = asq + bsq
        aM = bsq / M
        bM = asq / M
        cf[e, a, X] -= dx * m * aM
        cf[e, a, Y] -= dy * m * aM
        cf[e, b, X] += dx * m * bM
        cf[e, b, Y] += dy * m * bM
        bounce_cell(cf, P, e, a, False)
        bounce_cell(cf, P, e, b, False)
        update_cell(cf, ci, nf, ni, nit, g, P, e, a)
        update_cell(cf, ci, nf, ni, nit, g, P, e, b)

    # resolveEatCheck
    for k in range(ne):
        a = eat[k, 0]
        b = eat[k, 1]
        if ci[e, a, EX] == 0 or ci[e, b, EX] == 0:
            continue
        dx = cf[e, b, X] - cf[e, a, X]
        dy = cf[e, b, Y] - cf[e, a, Y]
        d = math.sqrt(dx * dx + dy * dy)
        if d > cf[e, a, SZ] - cf[e, b, SZ] / P[P_EATOVERLAPDIV]:
            continue
        if evn[e] < MAXEV:
            q = evn[e]
            ev[e, q, EV_EATER_OWN] = ci[e, a, OWN]
            ev[e, q, EV_EATEN_OWN] = ci[e, b, OWN]
            ev[e, q, EV_MASS] = cf[e, b, SZ] * cf[e, b, SZ] / 100
            ev[e, q, EV_EATER_ID] = ci[e, a, CID]
            ev[e, q, EV_EATEN_ID] = ci[e, b, CID]
            ev[e, q, EV_EATER_AGE] = tick - ci[e, a, BIRTH]
            ev[e, q, EV_EATEN_AGE] = tick - ci[e, b, BIRTH]
            ev[e, q, EV_EATER_BOOST] = cf[e, a, BD] if ci[e, a, BOOSTING] == 1 else 0.0
            ev[e, q, EV_EATER_MASS] = cf[e, a, SZ] * cf[e, a, SZ] / 100
            evn[e] = q + 1
        asq = cf[e, a, SZ] * cf[e, a, SZ]
        bsq = cf[e, b, SZ] * cf[e, b, SZ]
        cf[e, a, SZ] = math.sqrt(asq + bsq)
        remove_cell(cf, ci, pc, pcn, bc, bcn, oc, ocn, ni, nit, e, b)
        update_cell(cf, ci, nf, ni, nit, g, P, e, a)

    # players in join order: split attempts (ejecting is not part of this environment)
    for p in range(2):
        if ocn[e, p] == 0:
            continue
        att = splits[e, p]
        j = 0
        while j < P[P_SPLITCAP] and att > 0:
            l = ocn[e, p]
            for q in range(l):
                if ocn[e, p] >= P[P_MAXCELLS]:
                    break
                c = oc[e, p, q]
                if cf[e, c, SZ] < P[P_MINSPLITSIZE]:
                    continue
                dx = mouse[e, p, 0] - cf[e, c, X]
                dy = mouse[e, p, 1] - cf[e, c, Y]
                d = math.sqrt(dx * dx + dy * dy)
                if d < 1:
                    dx = 1.0
                    dy = 0.0
                    d = 1.0
                else:
                    dx /= d
                    dy /= d
                launch_cell(cf, ci, pc, pcn, bc, bcn, oc, ocn, nf, ni, nit, g, P, e, c,
                            cf[e, c, SZ] / P[P_SPLITSIZEDIV], dx, dy, P[P_SPLITBOOST])
            att -= 1
            j += 1


@njit(cache=True, parallel=True)
def step_batch(cf, ci, pc, pcn, bc, bcn, oc, ocn, nf, ni, nit, g, P, mouse, splits, ev, evn, active):
    E = cf.shape[0]
    for e in prange(E):
        if active[e] == 0:
            continue
        rigid = np.empty((S * S, 2), np.int64)
        eat = np.empty((S * S, 2), np.int64)
        found = np.empty(S, np.int64)
        stack = np.empty(4 * NNODES, np.int64)
        world_tick(cf, ci, pc, pcn, bc, bcn, oc, ocn, nf, ni, nit, g, P, e, mouse, splits, ev, evn,
                   rigid, eat, found, stack)


@njit(cache=True)
def clear_env(cf, ci, pcn, bcn, ocn, nf, ni, g, P, e, seed):
    for s_ in range(S):
        ci[e, s_, EX] = 0
        ci[e, s_, BOOSTING] = 0
        ci[e, s_, QNODE] = -1
    pcn[e] = 0
    bcn[e] = 0
    ocn[e, 0] = 0
    ocn[e, 1] = 0
    for n in range(NNODES):
        ni[e, n, N_INUSE] = 0
    _alloc_node(nf, ni, g, e, P[P_MAPX], P[P_MAPY], P[P_MAPW], P[P_MAPH], 1, -1)
    g[e, G_TICK] = 0
    g[e, G_NEXTID] = 1
    g[e, G_RNG] = seed if seed != 0 else 1
    g[e, G_OVERFLOW] = 0


@njit(cache=True)
def load_env(cf, ci, pc, pcn, bc, bcn, oc, ocn, nf, ni, nit, g, P, e, seed, cells):
    """cells rows: owner, x, y, size, age, boost_dx, boost_dy, boost_d (added in row order)"""
    clear_env(cf, ci, pcn, bcn, ocn, nf, ni, g, P, e, seed)
    for k in range(cells.shape[0]):
        s_ = add_cell(cf, ci, pc, pcn, bc, bcn, oc, ocn, nf, ni, nit, g, P, e,
                      int(cells[k, 0]), cells[k, 1], cells[k, 2], cells[k, 3])
        ci[e, s_, BIRTH] = g[e, G_TICK] - int(cells[k, 4])
        if cells[k, 7] > 0:
            cf[e, s_, BDX] = cells[k, 5]
            cf[e, s_, BDY] = cells[k, 6]
            cf[e, s_, BD] = cells[k, 7]
            set_boosting(ci, bc, bcn, e, s_)


class Sim:
    """N independent two-player worlds stepped together, one server tick per step()."""

    def __init__(self, n_envs, settings_path=None):
        self.P, self.settings = load_settings(settings_path)
        E = self.E = n_envs
        self.cf = np.zeros((E, S, NCF))
        self.ci = np.zeros((E, S, NCI), np.int64)
        self.pc = np.zeros((E, S), np.int64)
        self.pcn = np.zeros(E, np.int64)
        self.bc = np.zeros((E, S), np.int64)
        self.bcn = np.zeros(E, np.int64)
        self.oc = np.zeros((E, 2, S), np.int64)
        self.ocn = np.zeros((E, 2), np.int64)
        self.nf = np.zeros((E, NNODES, 4))
        self.ni = np.zeros((E, NNODES, NNI), np.int64)
        self.nit = np.zeros((E, NNODES, S), np.int64)
        self.g = np.zeros((E, NG), np.int64)
        self.ev = np.zeros((E, MAXEV, NEVF))
        self.evn = np.zeros(E, np.int64)
        self.mouse = np.zeros((E, 2, 2))
        self.splits = np.zeros((E, 2), np.int64)
        self.active = np.ones(E, np.int64)

    def _arrs(self):
        return (self.cf, self.ci, self.pc, self.pcn, self.bc, self.bcn, self.oc, self.ocn,
                self.nf, self.ni, self.nit, self.g, self.P)

    def load(self, e, cells, seed=1):
        """cells: array/list of rows (owner, x, y, size, age, boost_dx, boost_dy, boost_d)"""
        cells = np.asarray(cells, dtype=np.float64).reshape(-1, 8)
        load_env(*self._arrs(), e, seed, cells)

    def step(self, mouse, splits):
        """mouse: (E,2,2) cursor world coords per player; splits: (E,2) split presses this tick."""
        np.trunc(mouse, out=self.mouse)         # the wire protocol sends integer cursors
        self.splits[:] = splits
        self.evn[self.active == 0] = 0          # stopped worlds report no new events
        step_batch(*self._arrs(), self.mouse, self.splits, self.ev, self.evn, self.active)

    def cells(self, e):
        """list of (id, owner, x, y, size, boost_d) in server playerCells order"""
        out = []
        for i in range(self.pcn[e]):
            s_ = self.pc[e, i]
            out.append((int(self.ci[e, s_, CID]), int(self.ci[e, s_, OWN]), self.cf[e, s_, X], self.cf[e, s_, Y],
                        self.cf[e, s_, SZ], self.cf[e, s_, BD] if self.ci[e, s_, BOOSTING] else 0.0))
        return out

    def events(self, e):
        return self.ev[e, :self.evn[e]].copy()
