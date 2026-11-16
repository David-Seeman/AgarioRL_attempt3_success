"""Fast TRAVELLING splitruns: a repeating program of circle shots + forward trims, anchored to the WORLD
travel direction d (not to the rotating line between the players), tuned by evolution for sustained speed.

A program is a cycle of K events, each fired by player A or B (mirror option: every other cycle the roles swap
and the sideways axis flips, so the pair alternates sides like a human circle):
  C (circle)  the actor splits at a partner piece picked by a rule (biggest / nearest / frontmost / rearmost)
              plus an offset in the (d, d-perp) frame; single or double split
  T (trim)    the actor splits along d turned by an angle (cursor far = all its cells launch in parallel)
After each event, gap ticks until the next one; meanwhile each player's cursor = own centre + an offset in
the (d, d-perp) frame, blended towards the partner.

Feedback comes from picking the real pieces every event. All the splitrun rules of tools/choreo.py apply
(every split's pieces used in an eat within 50 ticks, no slow old-old absorption, no no-op split, both players
>= 12% of the mass, <= 10 cells each, <= 2 tiny pieces, a split at least every 50 ticks); a world that breaks
one stops (keeps the distance it made). Fitness = distance travelled along d in the horizon.

    python tools/travel_search.py --rhythm CT --K 4 --secs 30 --gens 40
"""
import argparse, json, math, os, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import tools.choreo as C  # noqa
from splitrun.sim import X, Y, SZ, EX, OWN, G_TICK  # noqa

# per-event parameter layout
PN = ["actor", "sel", "ox", "oy", "kind", "ang", "logR", "gap", "acx", "acy", "abeta", "bcx", "bcy", "bbeta"]
LO = np.array([0, 0, -700, -700, 0, -110, math.log(300), 2, -2500, -2500, -0.5, -2500, -2500, -0.5])
HI = np.array([5.999, 4.999, 700, 700, 1, 110, math.log(30000), 16, 2500, 2500, 1.0, 2500, 2500, 1.0])
NPE = len(PN)
IX = {k: i for i, k in enumerate(PN)}
TOT = 37000.0
SURV = 150.0


def start_rows(total=TOT, which=0, sep=1.0):
    starts = ["runs/techniques/technique_0.json", "runs/techniques/technique_6.json", "runs/techniques/technique_2plus2.json"]
    return C.rows_from_replay(os.path.join(ROOT, starts[which % len(starts)]), total=total)


class Program:
    def __init__(self, rhythm, K, mirror=True):
        self.types = [rhythm[k % len(rhythm)] for k in range(K)]
        self.K, self.mirror = K, mirror
        self.dim = K * NPE
        self.LO = np.tile(LO, K); self.HI = np.tile(HI, K)

    def get(self, V, k, name):
        return V[:, k * NPE + IX[name]]


def pick(cf, mask, val):
    """index of the slot with the largest val among mask (per world); -1 if none"""
    v = np.where(mask, val, -np.inf)
    j = v.argmax(1)
    return np.where(mask.any(1), j, -1)


def simulate(prog, V, secs=30.0, rows=None, total=TOT, record=None, d=(1.0, 0.0)):
    """run each candidate program; returns (distance along d, seconds survived, lateral drift, stats)"""
    N = len(V)
    d = np.asarray(d, float); dp = np.array([-d[1], d[0]])
    b = C.Batch(N)
    rows = rows or start_rows(total)
    for i in range(N):
        b.sim.load(i, rows, seed=i + 1)
    b.last_split[:] = b.sim.g[:, G_TICK]
    s = b.sim
    T = int(secs * 25)
    k = np.zeros(N, int)                  # next event index
    timer = np.full(N, 4)                 # ticks until it fires
    cyc = np.zeros(N, int)
    last = np.full(N, prog.K - 1)         # last fired event (its gap-phase cursors apply)
    ar = np.arange(N)

    def centres():
        ex = s.ci[:, :, EX] == 1; own = s.ci[:, :, OWN]; m = s.cf[:, :, SZ] ** 2 * ex
        out = []
        for p in (0, 1):
            mp = m * (own == p); M = np.maximum(mp.sum(1), 1)
            out.append(np.stack([(s.cf[:, :, X] * mp).sum(1) / M, (s.cf[:, :, Y] * mp).sum(1) / M], 1))
        mt = np.maximum(m.sum(1), 1)
        com = np.stack([(s.cf[:, :, X] * m).sum(1) / mt, (s.cf[:, :, Y] * m).sum(1) / mt], 1)
        return out[0], out[1], com, ex, own, m

    _, _, com0, _, _, _ = centres()
    alive_t = np.zeros(N)
    end_com = com0.copy()
    rec = None
    if record is not None:
        from splitrun.replay import Recorder
        rec = Recorder(s, record, "travel")
    nsplit = np.zeros(N, int)
    for t in range(T):
        cA, cB, com, ex, own, m = centres()
        sw = (cyc % 2 == 1) & prog.mirror             # roles swapped, sideways axis flipped
        sg = np.where(sw, -1.0, 1.0)[:, None]
        cen = [cA, cB]
        mouse = np.zeros((N, 2, 2)); sp = np.zeros((N, 2), np.int64)
        # gap-phase cursors of the last fired event (role A/B = player 0/1, swapped on mirrored cycles)
        for role, (cxn, cyn, bn) in enumerate((("acx", "acy", "abeta"), ("bcx", "bcy", "bbeta"))):
            cx = np.choose(last, [prog.get(V, kk, cxn) for kk in range(prog.K)])
            cy = np.choose(last, [prog.get(V, kk, cyn) for kk in range(prog.K)])
            be = np.choose(last, [prog.get(V, kk, bn) for kk in range(prog.K)])
            pl = np.where(sw, 1 - role, role)
            own_c = np.where((pl == 0)[:, None], cA, cB); oth = np.where((pl == 0)[:, None], cB, cA)
            tgt = own_c + cx[:, None] * d + cy[:, None] * sg * dp + be[:, None] * (oth - own_c)
            mouse[ar, pl] = tgt
        # events firing now
        fire = (timer <= 0) & ~b.dead
        if fire.any():
            ii = np.nonzero(fire)[0]
            kk = k[ii]
            g = lambda name: V[ii, kk * NPE + IX[name]]
            # actor rule: 0 role A, 1 role B, 2 the rear player (along d), 3 the front player, 4 heavier, 5 lighter
            ar_ = g("actor").astype(int)
            role = np.where(ar_ == 1, 1, 0)
            actor = np.where(sw[ii], 1 - role, role)
            pA = cA[ii] @ d; pB = cB[ii] @ d
            mAt = (m[ii] * (own[ii] == 0)).sum(1); mBt = (m[ii] * (own[ii] == 1)).sum(1)
            actor = np.where(ar_ == 2, (pB < pA).astype(int), actor)
            actor = np.where(ar_ == 3, (pB > pA).astype(int), actor)
            actor = np.where(ar_ == 4, (mBt > mAt).astype(int), actor)
            actor = np.where(ar_ == 5, (mBt < mAt).astype(int), actor)
            q = 1 - actor
            exi, owni, mi = ex[ii], own[ii], m[ii]
            xs, ys = s.cf[ii, :, X], s.cf[ii, :, Y]
            proj = xs * d[0] + ys * d[1]
            mA_ = exi & (owni == actor[:, None]); mB_ = exi & (owni == q[:, None])
            sel = g("sel").astype(int)
            circ = np.array([prog.types[x] == "C" for x in kk])
            # source piece: circle -> the actor's biggest; trim -> by rule (biggest / front / rear)
            sval = np.where((~circ & (sel == 2))[:, None], proj, np.where((~circ & (sel == 3))[:, None], -proj, mi))
            src = pick(None, mA_, sval)
            acx = np.where(src >= 0, xs[np.arange(len(ii)), np.maximum(src, 0)], 0)
            acy = np.where(src >= 0, ys[np.arange(len(ii)), np.maximum(src, 0)], 0)
            dist = np.hypot(xs - acx[:, None], ys - acy[:, None])
            land = np.hypot(xs - (acx + 860 * d[0])[:, None], ys - (acy + 860 * d[1])[:, None])
            # circle target: 0 biggest, 1 nearest, 2 frontmost, 3 rearmost, 4 nearest to a forward shot's landing point
            val = np.where((sel == 0)[:, None], mi, np.where((sel == 1)[:, None], -dist,
                           np.where((sel == 2)[:, None], proj, np.where((sel == 3)[:, None], -proj, -land))))
            tg = pick(None, mB_, val)
            ok = (src >= 0) & (tg >= 0)
            tx = xs[np.arange(len(ii)), np.maximum(tg, 0)]; ty = ys[np.arange(len(ii)), np.maximum(tg, 0)]
            sgi = sg[ii]
            aimC = np.stack([tx, ty], 1) + g("ox")[:, None] * d + g("oy")[:, None] * sgi * dp
            a_ = np.radians(g("ang"))[:, None]
            dirT = np.cos(a_) * d + np.sin(a_) * sgi * dp
            aimT = np.stack([acx, acy], 1) + dirT * np.exp(g("logR"))[:, None]
            aim = np.where(circ[:, None], aimC, aimT)
            kind = np.where(circ & (g("kind") > 0.5), 2, 1)
            for n_, i in enumerate(ii):
                if not ok[n_]:
                    b.dead[i] = True; continue
                mouse[i, actor[n_]] = aim[n_]; sp[i, actor[n_]] = kind[n_]
            nsplit[ii] += 1
            timer[ii] = np.round(g("gap")).astype(int)
            last[ii] = kk
            k[ii] = (kk + 1) % prog.K
            cyc[ii] += (k[ii] == 0)
        timer -= 1
        was = ~b.dead
        b.step(mouse, sp)
        # tidiness limits (as in choreo.score_event)
        ex2 = s.ci[:, :, EX] == 1; own2 = s.ci[:, :, OWN]; m2 = s.cf[:, :, SZ] ** 2 / 100 * ex2
        sc_ = (m2.sum(1) / 37000.0)[:, None]
        nA = (ex2 & (own2 == 0)).sum(1); nB = (ex2 & (own2 == 1)).sum(1)
        b.dead |= (nA > 10) | (nB > 10) | ((ex2 & (m2 < 500 * sc_)).sum(1) > 2)
        if rec is not None and not b.dead[record]:
            rec.after_step(np.trunc(mouse[record]), sp[record])
        still = was & ~b.dead
        alive_t[still] = t + 1
        if still.any():
            _, _, comn, _, _, _ = centres()
            end_com[still] = comn[still]
        if not (~b.dead).any():
            break
    mv = end_com - com0
    prog_d = mv @ d; lat = np.abs(mv @ dp)
    return prog_d, alive_t / 25, lat, rec, nsplit


def fitness(prog, V, secs, rows):
    pd, st, lat, _, _ = simulate(prog, V, secs, rows)
    # staying alive is worth SURV units/s (so the search first learns to splitrun, then to travel)
    return pd - 0.3 * lat + SURV * st, pd, st


def evolve(prog, pop, gens, secs, seed=0, log=print, rows_list=None, init=None):
    rng = np.random.default_rng(seed)
    V = rng.uniform(prog.LO, prog.HI, (pop, prog.dim))
    if init is not None:
        V[:len(init)] = init
    rows_list = rows_list or [start_rows(which=0)]

    def fit(X_):
        f = np.zeros(len(X_)); pd = np.zeros(len(X_)); st = np.zeros(len(X_))
        for r in rows_list:
            a, b_, c_ = fitness(prog, X_, secs, r)
            f += a / len(rows_list); pd += b_ / len(rows_list); st += c_ / len(rows_list)
        return f, pd, st
    f, pd, st = fit(V)
    ne = max(16, pop // 8)
    o = np.argsort(-f)[:ne]; el, ef, epd, est = V[o], f[o], pd[o], st[o]
    scale = 0.08
    best = ef[0]; stuck = 0
    t0 = time.time()
    for gen in range(1, gens + 1):
        par = el[rng.integers(0, len(el), pop)]
        kids = par + rng.normal(0, 1, par.shape) * (prog.HI - prog.LO) * scale
        # occasional crossover of whole events between two elites
        cx = rng.random(pop) < 0.3
        other = el[rng.integers(0, len(el), pop)]
        evm = np.repeat(rng.random((pop, prog.K)) < 0.5, NPE, 1)
        kids[cx] = np.where(evm[cx], other[cx], kids[cx])
        kids = np.clip(kids, prog.LO, prog.HI)
        f, pd, st = fit(kids)
        allV = np.concatenate([el, kids]); allf = np.concatenate([ef, f]); allpd = np.concatenate([epd, pd]); allst = np.concatenate([est, st])
        o = np.argsort(-allf)[:ne]
        el, ef, epd, est = allV[o], allf[o], allpd[o], allst[o]
        if ef[0] > best + 1:
            best, stuck = ef[0], 0
        else:
            stuck += 1; scale = max(0.005, scale * 0.85)
            if stuck % 8 == 0:
                scale = 0.06
        log(f"   gen {gen}: best {epd[0] / secs:.0f} u/s over {est[0]:.1f} s (fitness {ef[0]:.0f}), "
            f"elite median {np.median(epd) / secs:.0f} u/s, step {scale:.3f} ({time.time() - t0:.0f}s)")
    return el, ef, epd, est


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rhythm", default="CT"); ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--pop", type=int, default=512); ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--secs", type=float, default=30); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--total", type=float, default=TOT)
    ap.add_argument("--name", default="")
    a = ap.parse_args()
    prog = Program(a.rhythm, a.K, mirror=not a.no_mirror)
    name = a.name or f"travel_{a.rhythm}_K{a.K}{'_nm' if a.no_mirror else ''}_s{a.seed}"
    out = os.path.join(ROOT, "runs", "travel"); os.makedirs(out, exist_ok=True)
    print(f"{name}: types {prog.types}, mirror {prog.mirror}, dim {prog.dim}", flush=True)
    rows = [start_rows(a.total, which=0)]
    el, ef, epd, est = evolve(prog, a.pop, a.gens, a.secs, a.seed, rows_list=rows)
    np.save(os.path.join(out, name + ".npy"), el)
    # verify the best few over twice the horizon
    pd, st, lat, _, ns = simulate(prog, el[:16], a.secs * 2, rows[0])
    j = int(np.argmax(pd))
    print(f"VERIFY {a.secs * 2:.0f} s: best {pd[j] / (a.secs * 2):.0f} u/s (survived {st[j]:.1f} s, lateral {lat[j]:.0f}); "
          f"top-16 survivals {np.round(st, 1).tolist()}", flush=True)
    json.dump(dict(name=name, rhythm=a.rhythm, K=a.K, mirror=prog.mirror, params=el[j].tolist(), speed=float(pd[j] / (a.secs * 2)),
                   survived=float(st[j])), open(os.path.join(out, name + ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
