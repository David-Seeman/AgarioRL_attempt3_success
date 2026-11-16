"""Live splitrun controller: receding-horizon beam search over designed moves (tools/choreo.py), streamed.

Every step the beam (the best BEAM partial futures, M sampled moves each) is extended by one event in the
exact simulator. Once the beam reaches LOOKAHEAD events beyond what has been played, the first event of the
best future is COMMITTED: played in the real world and streamed. Only futures that share that event are
kept. If every future dies, the search backs up within the uncommitted part; if nothing from the committed
state survives, the run is over and a new one starts immediately from a fresh layout.

The viewer ("Live controller" tab) reads runs/live/live.json: the rolling last WINDOW ticks of the current
run, plus the lengths of all runs so far. Finished runs longer than 60 s are also saved as replays.

    python tools/live_controller.py            # runs until stopped
"""
import argparse, json, math, os, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import tools.choreo as C  # noqa
import tools.capture as CP  # noqa
from splitrun.sim import X, Y, SZ, EX, OWN, G_TICK  # noqa
from splitrun.replay import Recorder  # noqa

OUT = os.path.join(ROOT, "runs", "live")
WINDOW = 1500          # ticks kept in the stream file (60 s)
STARTS = ["runs/techniques/technique_0.json", "runs/techniques/technique_6.json", "runs/techniques/technique_2plus2.json"]
STYLE = {"double": 1.5, "trim": 1.0, "cross": 0.5, "drift_w": 0.02}
DYN = dict(path="dir", dir=(1.0, 0.0), speed_w=0.05, lat_w=0.02, chase_p=0.5, dash=2.0, double=1.5, trim=0.5,
           cross=0.5, drift_w=0.01)
FEED = dict(feed=3.0, dash=2.0, capture=4.0, swallow=3.0, max_loss=0.04, halve_w=60.0, feeder_w=0.01, drift_w=0.0, food_w=0.02, lost_w=0.3, danger_w=15.0, seek_w=0.02, chase_p=0.35, seek_k=-1, share_min=0.05, wait=0.0, waitmax=30,
            max_cells=8, hard_cells=16, double=1.0, trim=0.5, cross=0.3)
START_MASS = 3500.0
WATCHDOG = 8.0         # s of wall clock without a committed move -> restart
MAXFAIL = 60           # failed expansions without a committed move -> restart


class Live:
    def __init__(self, beam=24, M=64, lookahead=10, seed=0, pace=True, mode="stationary", out=None):
        self.beam, self.M, self.L, self.pace = beam, M, lookahead, pace
        self.mode = mode
        self.style = dict(STYLE)
        if mode == "dynamic":
            # fast translation towards a waypoint the viewer sets (runs/live_dynamic/control.json)
            self.style.update(DYN)
            self.waypoint = np.array([6000.0, 0.0]); self.ctl_mtime = 0.0
        if mode == "feeding":
            # start below 4k, hunt balls worth >= 1/3 of the team's mass, grow to 60k (then restart);
            # when no good move exists, pause (reorient / let pieces remerge) instead of restarting
            C.BALLS = 80
            C.IDLE_MAX = 250
            C.USE_WINDOW = 100
            C.MINSHARE = 0.03          # feeding the eater is the point: a player may drop to 3% of the team
            self.style.update(FEED)
        global OUT
        if out:
            OUT = out
        self.rng = np.random.default_rng(seed)
        self.runs = []                     # finished run lengths (s)
        self.run_id = 0
        os.makedirs(OUT, exist_ok=True)

    def new_run(self):
        self.run_id += 1
        start = STARTS[(self.run_id - 1) % len(STARTS)]
        rows = C.rows_from_replay(os.path.join(ROOT, start), total=START_MASS if self.mode == "feeding" else getattr(self, "total", None))
        # a little variety: jitter the start layout
        rows = [(o, x + self.rng.normal(0, 40), y + self.rng.normal(0, 40), s, a, bx, by, bd) for o, x, y, s, a, bx, by, bd in rows]
        if self.mode == "feeding":
            m_ = sum(r_[3] ** 2 for r_ in rows)
            cx_ = sum(r_[1] * r_[3] ** 2 for r_ in rows) / m_; cy_ = sum(r_[2] * r_[3] ** 2 for r_ in rows) / m_
            rows = [(o, x - cx_, y - cy_, s, a, bx, by, bd) for o, x, y, s, a, bx, by, bd in rows]
        if self.mode == "dynamic":
            # start where the previous run ended (or the map centre), so the travel continues
            m_ = sum(r_[3] ** 2 for r_ in rows)
            cx_ = sum(r_[1] * r_[3] ** 2 for r_ in rows) / m_; cy_ = sum(r_[2] * r_[3] ** 2 for r_ in rows) / m_
            sx, sy = getattr(self, "start_at", (0.0, 0.0))
            rows = [(o, x - cx_ + sx, y - cy_ + sy, s, a, bx, by, bd) for o, x, y, s, a, bx, by, bd in rows]
        self.world = C.Batch(1)
        self.world.sim.load(0, rows, seed=int(self.rng.integers(1, 1 << 30)))
        self.world.last_split[:] = self.world.sim.g[:, G_TICK]
        if C.BALLS:
            from splitrun import balls as BL
            self.world.bseed[0] = int(self.rng.integers(1, 1 << 30))
            self.world.bx, self.world.by, self.world.bs, self.world.bn, self.world.bt = BL.init_balls(
                1, C.BALLS, self.world.bseed, team=START_MASS)
            self.style["seek_k"] = -1; self.seek_n = -1
            self.pick_target()
        self.pauses = 0
        self.rec = Recorder(self.world.sim, 0, f"live_run_{self.run_id}")
        m0 = self.world.sim.cf[0, :, SZ] ** 2 * (self.world.sim.ci[0, :, EX] == 1)
        self.center0 = np.array([(self.world.sim.cf[0, :, X] * m0).sum(), (self.world.sim.cf[0, :, Y] * m0).sum()]) / m0.sum()
        self.t_run = time.time()
        self.hold = 0.0
        self.moves = 0
        # beam: (state, score, uncommitted actions)
        self.nodes = [(self.world.state(0), 0.0, [])]
        self.history = []
        self.start_name = os.path.basename(start)[:-5]

    def team(self):
        w = self.world.sim
        ex = w.ci[0, :, EX] == 1; m = w.cf[0, :, SZ] ** 2 / 100 * ex
        return float(m.sum()), np.array([(w.cf[0, :, X] * m).sum(), (w.cf[0, :, Y] * m).sum()]) / max(m.sum(), 1)

    def pick_target(self):
        """feeding: chase the worthy ball (>= 1/3 of the team's mass, small enough to be eaten by a consolidated
        piece) with the best mass / distance; keep it until it is eaten, respawns or stops being worthy"""
        T, c = self.team()
        W = self.world
        k = self.style.get("seek_k", -1)
        if k >= 0:
            m = W.bs[0, k] ** 2 / 100
            if W.bn[0, k] == self.seek_n and T / 3 <= m <= 0.7 * T:
                C.CHASE = (float(W.bx[0, k]), float(W.by[0, k]))
                return
        m = W.bs[0] ** 2 / 100
        w = self.world.sim
        ex = w.ci[0, :, EX] == 1; pm = w.cf[0, :, SZ] ** 2 / 100 * ex
        bi = int(np.argmax(pm)); bxy = np.array([w.cf[0, bi, X], w.cf[0, bi, Y]])     # the team's biggest piece
        # worthy (>= 1/3 of the team) and swallowable after ONE feed: some piece plus half of the partner's
        # biggest piece reaches 1.3x the ball
        own = w.ci[0, :, OWN]
        best = 0.0
        for p_ in (0, 1):
            mine = pm[ex & (own == p_)]; other = pm[ex & (own == 1 - p_)]
            if len(mine) and len(other):
                best = max(best, mine.max() + 0.5 * other.max())
        ok = (m >= T / 3) & (m <= best / 1.3)
        if not ok.any():
            self.style["seek_k"] = -1; C.CHASE = None; return
        d = np.hypot(W.bx[0] - bxy[0], W.by[0] - bxy[1])
        val = np.where(ok, -d, -1e18)
        k = int(np.argmax(val))
        self.style["seek_k"] = k; self.seek_n = int(W.bn[0, k])
        C.CHASE = (float(W.bx[0, k]), float(W.by[0, k]))

    def try_capture(self):
        """feeding: if a piece can get onto the target ball, search the structured capture plans; commit the
        first that swallows it without losing mass (the splitrun beam resumes afterwards)"""
        k = self.style.get("seek_k", -1)
        if k < 0:
            return False
        plan, n = CP.best_capture(self.world, k)
        self.capture_tries = getattr(self, "capture_tries", 0) + 1
        if plan is None:
            return False
        CP.run_plans(self.world, k, [plan], rec=self.rec, stop_after=plan["eat_t"] + 3)
        self.captures = getattr(self, "captures", 0) + 1
        self.moves += 1
        self.nodes = [(self.world.state(0), 0.0, [])]; self.history = []
        self.style["seek_k"] = -1
        self.pick_target()
        return True

    def pause(self):
        """no good move: spend some downtime (no split, drift together) and plan again from there"""
        T, c = self.team()
        act = dict(move=C.MOVES.index("wait"), actor=0, wait=20, j=np.zeros(2, np.int64), off=np.zeros(2), rho=0.0, phi=0.0,
                   c1=np.zeros(4), c2=np.zeros(4), chase=0, fkind=1, dt=1, lead=0.0, chase_m=0.0,
                   chase_xy=np.zeros(2))
        C.play(self.world, {k: np.array([v]) for k, v in act.items()}, rec=self.rec)
        self.pauses += 1
        self.nodes = [(self.world.state(0), 0.0, [])]
        self.history = []

    def steer(self):
        """dynamic mode: head for the waypoint (read from control.json when it changes); hold once there"""
        cp = os.path.join(OUT, "control.json")
        try:
            mt = os.path.getmtime(cp)
            if mt != self.ctl_mtime:
                self.ctl_mtime = mt
                wp = json.load(open(cp)).get("waypoint")
                if wp:
                    new = np.clip(np.array(wp, float), -8200, 8200)
                    if np.hypot(*(new - self.waypoint)) > 1:
                        self.waypoint = new
                        self.nodes = [(self.world.state(0), 0.0, [])]; self.history = []      # re-plan now
        except (OSError, ValueError):
            pass
        _, c = self.team()
        v = self.waypoint - c; dist = float(np.hypot(*v))
        if dist < 1500:
            self.style["path"] = "hold"; self.style["waypoint"] = tuple(self.waypoint); C.CHASE = None
        else:
            d = v / dist
            self.style["path"] = "dir"; self.style["dir"] = (float(d[0]), float(d[1])); C.DIR = d.copy()
            C.CHASE = (float(c[0] + d[0] * 5000), float(c[1] + d[1] * 5000))

    def expand(self):
        if self.mode == "dynamic":
            self.steer()
        if self.mode == "feeding" and self.style.get("seek_k", -1) >= 0:
            k = self.style["seek_k"]; C.CHASE = (float(self.world.bx[0, k]), float(self.world.by[0, k]))
            C.CHASE_M = float(self.world.bs[0, k] ** 2 / 100)
        nodes, M = self.nodes, self.M
        b = C.Batch(len(nodes) * M)
        b.load_from([n[0] for n in nodes], np.repeat(np.arange(len(nodes)), M))
        cand = C.sample(self.rng, b.N, self.style)
        rh = self.style.get("rhythm")
        slotT = None
        if rh:
            # rhythm: 'C' = a circle shot (single or double) at the partner, 'T' = a forward trim. Most candidates
            # follow it; the rest are free (a bonus rewards following it, so it breaks only when it must)
            k = np.repeat(np.array([self.moves + len(n[2]) for n in nodes]), M) % len(rh)
            slotT = np.array([rh[x] == "T" for x in k])
            force = self.rng.random(b.N) < self.style.get("rhythm_p", 0.8)
            circ = np.array([C.MOVES.index(x) for x in ("circle", "double")])
            forced = np.where(slotT, C.MOVES.index("ftrim"), np.where(np.isin(cand["move"], circ), cand["move"], self.rng.choice(circ, b.N)))
            cand["move"] = np.where(force, forced, cand["move"])
        C.play(b, cand)
        ev = C.score_event(b, cand, self.style, self.center0)
        if slotT is not None:
            isT = cand["move"] == C.MOVES.index("ftrim")
            isC = np.isin(cand["move"], [C.MOVES.index("circle"), C.MOVES.index("double")])
            ev = ev + self.style.get("rhythm_w", 8.0) * np.where(slotT, isT, isC)
        parent = np.repeat(np.arange(len(nodes)), M)
        total = np.array([nodes[p][1] for p in parent]) + ev
        alive = np.nonzero(~b.dead)[0]
        if len(alive) == 0:
            return False
        order = alive[np.argsort(-total[alive])]
        chosen, per = [], {}
        for i in order:
            if per.get(parent[i], 0) >= 3:
                continue
            per[parent[i]] = per.get(parent[i], 0) + 1
            chosen.append(i)
            if len(chosen) >= self.beam:
                break
        new = []
        for i in chosen:
            act = {k: (v[i].copy() if isinstance(v, np.ndarray) and v.ndim > 1 else v[i].item()) for k, v in cand.items()}
            new.append((b.state(i), float(total[i]), nodes[parent[i]][2] + [act]))
        self.history.append(self.nodes)
        self.nodes = new
        return True

    def commit(self):
        """play the best future's first event for real; keep only the futures that begin with it"""
        first = self.nodes[0][2][0]
        C.play(self.world, {k: np.array([v]) for k, v in first.items()}, rec=self.rec)
        self.moves += 1
        self.nodes = [(s, sc, acts[1:]) for s, sc, acts in self.nodes if acts[0] is first]
        self.history = [[(s, sc, acts[1:]) for s, sc, acts in h if acts and acts[0] is first] for h in self.history[1:]]
        self.history = [h for h in self.history if h]
        return not self.world.dead[0]

    def backtrack(self):
        for _ in range(4):
            if not self.history:
                return False
            self.nodes = self.history.pop()
        return bool(self.nodes)

    def write(self, final=False):
        d = self.rec.data
        n = len(d["frames"])
        lo = max(0, n - WINDOW)
        w_ = self.world.sim
        mass = float((w_.cf[0, :, SZ] ** 2 / 100 * (w_.ci[0, :, EX] == 1)).sum())
        tgt = None
        if self.mode == "feeding" and self.style.get("seek_k", -1) >= 0:
            k = self.style["seek_k"]
            tgt = [round(float(self.world.bx[0, k])), round(float(self.world.by[0, k])), round(float(self.world.bs[0, k]))]
        speed = None
        if self.mode == "dynamic" and n > 130:
            def cm(fr):
                P = [c for c in fr if c[1] < 2]; M = sum(c[4] ** 2 for c in P)
                return np.array([sum(c[2] * c[4] ** 2 for c in P), sum(c[3] * c[4] ** 2 for c in P)]) / M
            speed = float(np.hypot(*(cm(d["frames"][-1]) - cm(d["frames"][-126]))) / 5.0)
        live = dict(run_id=self.run_id, start=self.start_name, mode=self.mode, mass=round(mass), target=tgt, speed=speed,
                    waypoint=[float(v) for v in self.waypoint] if self.mode == "dynamic" else None, base_tick=lo, total_ticks=n - 1,
                    tick_rate=25, map=d["map"], frames=d["frames"][lo:], inputs=d["inputs"][max(0, lo - 1):],
                    input_base=max(0, lo - 1), events=[e for e in d["events"] if e[0] >= lo],
                    runs=self.runs[-50:], running_s=(n - 1) / 25, updated=time.time(), final=final)
        tmp = os.path.join(OUT, f"live.json.{os.getpid()}.tmp")         # per-process: writers never collide
        with open(tmp, "w") as f:
            json.dump(live, f, separators=(",", ":"))
        os.replace(tmp, os.path.join(OUT, "live.json"))

    def finish(self, why):
        if self.mode == "dynamic":
            _, c = self.team(); self.start_at = (float(c[0]), float(c[1]))
        secs = (len(self.rec.data["frames"]) - 1) / 25
        self.runs.append(dict(run=self.run_id, seconds=round(secs, 1), moves=self.moves, ended=why))
        self.write(final=True)
        if secs >= 60 or (self.mode == "feeding" and secs >= 8):      # feeding: keep every attempt for review
            extra = ""
            if self.mode == "feeding":
                w = self.world.sim; T = float((w.cf[0, :, SZ] ** 2 / 100 * (w.ci[0, :, EX] == 1)).sum())
                extra = f" Feeding: {getattr(self, 'captures', 0)} captures, team {START_MASS:,.0f} -> {T:,.0f}."
            self.rec.data["description"] = (f"Live controller run {self.run_id} ({self.mode}): {secs:.1f} s, {self.moves} moves, "
                                            f"ended: {why}.{extra} Receding-horizon beam search over designed moves.")
            self.rec.save(os.path.join(OUT, f"run_{time.strftime('%m%d_%H%M%S')}_{self.run_id:03d}.json"))
        print(f"run {self.run_id}: {secs:.1f} s, {self.moves} moves, ended: {why}", flush=True)

    def loop(self, max_runs=10 ** 9, max_seconds=None):
        self.new_run()
        last_write = 0
        t0 = time.time()
        last_commit, fails = time.time(), 0
        while self.run_id <= max_runs:
            if max_seconds and time.time() - t0 > max_seconds:
                self.finish("stopped"); return
            # watchdog: a hard state can make the search go round in circles without ever committing a
            # move (that froze the stream); give it WATCHDOG s / MAXFAIL failed expansions, then restart
            if time.time() - last_commit > WATCHDOG or fails > MAXFAIL:
                if self.mode == "feeding" and self.pauses < 10 and not self.world.dead[0]:
                    self.pause(); last_commit, fails = time.time(), 0; continue
                self.finish("stuck (watchdog)"); self.new_run(); last_commit, fails = time.time(), 0; continue
            ok = self.expand()
            if not ok:
                fails += 1
                if not self.backtrack():
                    if self.mode == "feeding" and self.pauses < 10 and not self.world.dead[0]:
                        self.pause(); last_commit, fails = time.time(), 0; continue
                    self.finish("no continuation"); self.new_run(); last_commit, fails = time.time(), 0; continue
                continue
            if len(self.nodes[0][2]) >= self.L:
                if not self.commit():
                    self.finish("rule broken"); self.new_run(); last_commit, fails = time.time(), 0; continue
                last_commit, fails = time.time(), 0
                if self.mode == "feeding":
                    self.pauses = 0
                    if self.try_capture():
                        last_commit = time.time()
                        continue
                    tot, _ = self.team()
                    if tot >= 60000:
                        self.finish("reached 60k mass"); self.new_run(); continue
                    old = self.style.get("seek_k", -1)
                    self.pick_target()
                    if self.style.get("seek_k", -1) != old:        # new target: re-plan from here
                        self.nodes = [(self.world.state(0), 0.0, [])]; self.history = []
                if not self.nodes:
                    if self.mode == "feeding":
                        self.pause(); continue
                    self.finish("no continuation"); self.new_run(); continue
            played = (len(self.rec.data["frames"]) - 1) / 25
            if time.time() - last_write > 0.5:
                self.write(); last_write = time.time()
            if self.pace:
                # stay at most ~8 s ahead of the wall clock so the viewer can follow in real time
                ahead = played - (time.time() - self.t_run)
                if ahead > 8:
                    time.sleep(min(ahead - 8, 1.0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--beam", type=int, default=24); ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--lookahead", type=int, default=16); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-pace", action="store_true"); ap.add_argument("--runs", type=int, default=10 ** 9)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--mode", default="stationary", choices=["stationary", "dynamic", "feeding"])
    ap.add_argument("--out", default=None, help="stream directory (default runs/live)")
    a = ap.parse_args()
    Live(a.beam, a.m, a.lookahead, a.seed, pace=not a.no_pace, mode=a.mode, out=a.out).loop(a.runs, a.seconds)
