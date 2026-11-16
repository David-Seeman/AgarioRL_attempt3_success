"""Structured capture search for the feeding controller (no luck involved).

When a piece that can sit on the target ball safely (>= 0.77x its mass) is near it, try a grid of capture
plans in one batch of the exact simulator and return the first that swallows the target without losing mass:
  dasher  : one of the nearest candidate pieces (its owner = the dasher's player, the other = the feeder)
  style   : 'ram'  -- the dasher's player steers straight at the ball (no split)
            'dash' -- the dasher's player splits towards the ball; its launched half is the dasher
  feed    : the feeder splits (single / double / triple) at the dasher, `delay` ticks in, aimed `lead` units
            ahead of it towards the ball (the mass that tips it over 1.3x the ball)
Rules are those of tools/choreo.py (purpose, no slow absorption, no mass lost to balls).
"""
import math
import numpy as np
import tools.choreo as C
from splitrun.sim import X, Y, SZ, EX, OWN, CID, BIRTH, G_TICK

H = 75           # ticks simulated per plan


def plans_for(world, k):
    """list of plan dicts for target ball k"""
    w = world.sim
    bx, by = float(world.bx[0, k]), float(world.by[0, k]); tm = float(world.bs[0, k] ** 2 / 100)
    cands = []
    for p in (0, 1):
        for s_ in w.oc[0, p, :w.ocn[0, p]]:
            m = w.cf[0, s_, SZ] ** 2 / 100
            d = math.hypot(w.cf[0, s_, X] - bx, w.cf[0, s_, Y] - by)
            if m >= 0.77 * tm and d < 1400:
                cands.append((d, int(w.ci[0, s_, CID]), p, m))
    cands.sort()
    out = []
    for d, cid, p, m in cands[:4]:
        for style in ("ram", "dash"):
            if style == "dash" and m / 2 < 0.4 * tm:
                continue
            for kind in (1, 2, 3):
                for delay in (0, 4, 8, 14, 22):
                    for lead in (0, 150, 350):
                        out.append(dict(cid=cid, owner=p, style=style, kind=kind, delay=delay, lead=lead))
    return out


def piece_pos(sim, i, cid):
    for p in (0, 1):
        for s_ in sim.oc[i, p, :sim.ocn[i, p]]:
            if sim.ci[i, s_, CID] == cid:
                return np.array([sim.cf[i, s_, X], sim.cf[i, s_, Y]]), p
    return None, None


def run_plans(world, k, plans, rec=None, stop_after=None):
    """simulate every plan (world 0 of `world` copied into a batch); returns eat tick per plan (-1 = none).
    With rec (one plan), the plan is played IN `world` itself and recorded, until stop_after ticks."""
    N = len(plans)
    if rec is None:
        b = C.Batch(N)
        b.load_from([world.state(0)], np.zeros(N, int))
    else:
        b = world
        b.purp[:] = 0; b.gained[:] = 0; b.gained_small[:] = 0; b.lost[:] = 0
    sim = b.sim
    eaten_t = np.full(N, -1)
    bn0 = int(world.bn[0, k])
    dasher = np.array([pl["cid"] for pl in plans])
    fed = np.zeros(N, bool)
    lost = np.zeros(N)
    owner = np.array([pl["owner"] for pl in plans]); dashm = np.array([pl["style"] == "dash" for pl in plans])
    kind = np.array([pl["kind"] for pl in plans]); delay = np.array([pl["delay"] for pl in plans])
    lead = np.array([pl["lead"] for pl in plans], float)
    ar = np.arange(N)
    for t in range(H if stop_after is None else stop_after):
        mouse = np.zeros((N, 2, 2)); sp = np.zeros((N, 2), np.int64)
        bxy = np.stack([b.bx[:, k], b.by[:, k]], 1)
        # where is each plan's dasher (vectorised lookup by cell id)?
        hitm = (sim.ci[:, :, CID] == dasher[:, None]) & (sim.ci[:, :, EX] == 1)
        found = hitm.any(1)
        slot = hitm.argmax(1)
        dpos = np.stack([sim.cf[ar, slot, X], sim.cf[ar, slot, Y]], 1)
        act = ~b.dead & (eaten_t < 0)
        b.dead |= act & ~found
        act &= found
        mouse[ar, owner] = bxy                                         # the dasher's player drives at the ball
        # the feeder holds its formation (no contact = no slow absorption), leaning a little towards the dasher;
        # only its feed split is aimed at the dasher
        fc = np.stack([np.array(C.com(sim, i, 1 - owner[i])) for i in range(N)])
        tod = dpos - fc; tod = tod / np.maximum(np.linalg.norm(tod, axis=1, keepdims=True), 1e-6)
        mouse[ar, 1 - owner] = fc + tod * 150
        if t == 0:
            sp[ar[act & dashm], owner[act & dashm]] = 1                 # launch the dasher at the ball
        ft = act & ~fed & (t == delay + dashm.astype(int))
        dv = bxy - dpos; dv = dv / np.maximum(np.linalg.norm(dv, axis=1, keepdims=True), 1e-6)
        aimxy = dpos + dv * lead[:, None]
        mouse[ar[ft], 1 - owner[ft]] = aimxy[ft]
        sp[ar[ft], 1 - owner[ft]] = kind[ft]
        fed |= ft
        b.step(mouse, sp)
        if rec is not None:
            rec.after_step(np.trunc(mouse[0]), sp[0])
            rec.data["frames"][-1].extend([[1_000_000 + kk * 10_000 + int(b.bn[0, kk]) % 10_000, 2,
                                            round(float(b.bx[0, kk]), 1), round(float(b.by[0, kk]), 1),
                                            round(float(b.bs[0, kk]), 2), 0] for kk in range(C.BALLS)])
        if t == 0:
            # for 'dash' plans the dasher is now the launched half nearest the ball
            for i, pl in enumerate(plans):
                if pl["style"] == "dash":
                    o = pl["owner"]
                    born = [(sim.ci[i, s_, CID], sim.cf[i, s_, X], sim.cf[i, s_, Y]) for s_ in sim.oc[i, o, :sim.ocn[i, o]]
                            if sim.ci[i, s_, BIRTH] == sim.g[i, G_TICK]]
                    if born:
                        dasher[i] = min(born, key=lambda z: (z[1] - bxy[i, 0]) ** 2 + (z[2] - bxy[i, 1]) ** 2)[0]
        hit = (b.bn[:, k] != bn0) & (eaten_t < 0) & ~b.dead
        eaten_t[hit] = t
    lost = b.lost.copy()
    ok = (eaten_t >= 0) & ~b.dead
    team = np.array([(b.sim.cf[i, :, SZ] ** 2 / 100 * (b.sim.ci[i, :, EX] == 1)).sum() for i in range(N)])
    return eaten_t, ok, lost, team


def best_capture(world, k):
    plans = plans_for(world, k)
    if not plans:
        return None, 0
    eaten_t, ok, lost, team = run_plans(world, k, plans)
    T0 = float((world.sim.cf[0, :, SZ] ** 2 / 100 * (world.sim.ci[0, :, EX] == 1)).sum())
    good = [i for i in range(len(plans)) if ok[i] and lost[i] <= 0.02 * T0]
    if not good:
        return None, len(plans)
    i = min(good, key=lambda j: eaten_t[j])
    plans[i]["eat_t"] = int(eaten_t[i])
    return plans[i], len(plans)
