# Hand-off: two-player splitrun controller in imsolo physics

This is for the next AI model working in this folder. It covers four things:
- what has been built;
- what I (the previous model) understand about the imsolo environment and about splitruns;
- where I got stuck;
- open directions.

The README (`README.md`, about 800 lines) is the detailed stage-by-stage log, with numbers. This file is the
condensed version plus my opinions. **Treat my diagnoses as hypotheses, not facts.** Several of my
"conclusions" came from short searches, and you may well see something I missed.

---

## 1. The user and how to work with them

- The user plays agar.io / imsolo.pro and knows splitrunning very well. They judge results **by watching
  them** in the viewer, not by numbers. Give them something to watch and a link.
- They want **staged work**. Finish a step, show it, then get feedback. Ask clarifying questions when a
  tactic or goal is ambiguous. Past models failed by misunderstanding what a splitrun is, then building
  fake "splitruns".
- **Keep searches short and report progress.** They have complained about 30–90 minute silent runs. If an
  approach stalls, say so early and change approach instead of tuning it into a rabbit hole.
- **Never edit `reference_server/`.** It is a pristine clone of the imsolo server
  (github.com/jimboy3100/imsolo-servers, `Megasplit-Beta/`).
- They like double splits and straight triple splits best. Creativity is welcome, but **functionality comes
  first**: low downtime, every split purposeful.

---

## 2. The environment (what I'm confident about)

### Simulator
- `splitrun/sim.py` is a batched Numba port of the server's `World.liveUpdate()`, one server tick per step.
- `tests/test_fidelity.py` compares it with the real server JS (`tools/trace_server.js`): 60/60 random
  aggressive scenarios are **bit-identical in cells, ids and order** for up to 1,000 ticks. Even the
  quadtree collision order is reproduced.
- Speed is about 550k env-ticks/s on an M2 across 512+ parallel worlds.
- Settings live in `env_settings.json`:
  - 25 ticks/s;
  - map 18,142 × 18,142, centred on 0 (±9,071);
  - no food, viruses or decay;
  - 16 cells max per player;
  - autosplit on.

### Physics constants (25 ticks/s)

| quantity | value |
|---|---|
| mass | size² / 100 (18.5k mass = size 1360) |
| eat (between players) | eater size ≥ 1.14 × victim size (mass ≥ 1.3×), and centre distance < eater − victim/3 |
| split | every cell of the player above min size halves; the new half starts 40 units toward the cursor with a 780 boost decaying ×8/9 per tick: **~820 units in ~1 s, independent of mass** |
| movement | speed = 88 · size^-0.4397 · 1.4 per tick toward the cursor: 18.5k ≈ 129 u/s, 6k ≈ 164 u/s, 3k ≈ 190 u/s |
| own cells | pass through each other for 13 ticks after a split, then push apart rigidly (this spring-apart shove is what the user calls a push-split's "recoil") |
| merge (own) | after round(0.75 · size) ticks: ~29 s for a 9.25k half. **Merging is effectively unavailable during a splitrun**, so the only way to reduce cell count is for the partner to eat pieces |
| autosplit | any cell above 22,500 mass splits in a random direction (seeded RNG, reproduced) |
| cell cap | 16 per player; at the cap the OLDEST cells split first ("selective split at the cap") |

- Double split = two splits in the same tick. The user chose this; the simulator also supports one tick
  apart.
- A **straight triple split**, the imsolo "n then triple split" line of 8, needs two conditions:
  - it must be fired from one cell, or from cells already lined up with the aim;
  - the cursor must be held far along the line for about 22 ticks (`HOLD` in `tools/choreo.py`).

  Otherwise the triple comes out messy. The user rejected messy triples labelled as straight.

### Viewer
- `tools/serve.py` serves `viewer/index.html`. Run it with `.claude/launch.json` "viewer", port 8765.
- Tabs:
  - **All replays** (everything under `runs/`, indexed by `splitrun/replay.write_index`);
  - **Non-periodic catalogue**;
  - **Live controller**, with three streams: Stationary, Moving and Feeding.
- Moving stream: click the map, or use the arrow keys / WASD, to steer. This posts `/api/steer`, which writes
  `runs/live_dynamic/control.json`.
- Rendering:
  - smooth interpolation, with eaten pieces sliding into their eater;
  - a calm follow camera;
  - balls drawn in various colours and skins.

---

## 3. What a splitrun is (the user's definition, as I understood it)

Two players, A and B, keep feeding each other's freshly launched pieces. Mass flows A→B→A forever, and the
pair visually circles around each other. The project's clean rules are enforced by `tools/freerun.py` and
`tools/choreo.py`:
1. **Every split is purposeful.** Within 50 ticks (2 s), some piece of that split (a launched piece or its
   parent) must take part in a cross-player eat.
2. **No slow absorption.** A cross eat needs at least one piece younger than 60 ticks. An old piece
   sitting on another old piece and slowly eating it looks like downtime.
3. **No no-op split presses.** Both players keep at least 12% of the team's mass, and there is at most
   50 ticks between splits.
4. **Tidiness.** At most 10 cells per player and at most 2 tiny crumbs. Thresholds scale with team mass.
5. **Low downtime.** The catalogue averages 0.36–0.47 s between splits. Human-level splitrunning is fast.

The user's tactic glossary, which matters for understanding their feedback:
- **Circle shot:** the basic split at the partner's pieces.
- **Trim:** an off-circle split, usually sideways and slightly backwards from the circle (sometimes two:
  one sideways, one back). It shrinks your big pieces, and the partner soon eats the launched piece. For
  travel, the user says humans circle around each other **with trims in the desired direction**.
- **Split between:** your launched piece flies through the gap between two of the partner's pieces.
- **Double split:** mainly for reach. This is the user's favourite to watch.
- **Push-split:** using the spring-apart shove to push your own backpiece into the partner. It is rare
  and shouldn't be relied on.
- **My ideas the user liked:** fly-through (the launched piece passes over a same-size partner piece to
  reach another), selective split at the 16-cell cap, and crossing splits (both players split at once and
  the shots cross).
- **Not relevant:** tricksplits and popsplits (popsplits need viruses).
- **Merges** mean too much downtime.
- Human splitruns often have **mass imbalance above 2×** between the players.

### Theory I worked out
- **Mass bookkeeping (Stage 2–4 in the README).** In a periodic run, launched pieces = eats per cycle.
  Trims therefore require double eats.
  - The clean periodic mass patterns can be enumerated by LP (`tools/mass_cycles.py`,
    `tools/technique_catalogue.py`).
  - Their stability factor ρ < 1 means mass errors die out by themselves.
  - A plain 3-piece circle can hold about 1.6× imbalance.
- **Open-loop scripts are fragile.** A fixed sequence of aims and timings breaks within 3–10 turns at ±1–2%
  mass error. Anything robust needs feedback: re-aim at the real pieces every split.
- **Why sustainable patterns are hard.** Each split adds as many cells as the splitter has, and only
  cross eats remove cells (merges are too slow). So a sustainable pattern must eat at the same rate it
  launches. Formations that fragment burn out within seconds.

---

## 4. What was completed (each was shown to the user and accepted unless noted)

| stage | result | where |
|---|---|---|
| 1 | exact simulator + fidelity test + viewer | `splitrun/sim.py`, `tests/`, `viewer/` |
| 2–2d | periodic "techniques": a catalogue of 8 mass patterns, 4 realised and chained 100+ clean cycles; downtime shrunk to 10–39 ticks per cycle; a 2+2 technique | `tools/solve_technique.py`, `runs/techniques/` |
| 3 | PPO on a technique (t4 learned 28 clean cycles), plus discovery RL (it found variants of technique 0, which the user judged as minor) | `rl/ppo.py`, `splitrun/rlenv*.py` |
| 4 | imbalanced techniques: mass patterns exist, but no working imbalanced orbit was found | README Stage 4a/4b |
| 5 | **non-periodic creative catalogue**: 18 runs of 30 s each, average gap 0.36–0.47 s, at masses 3.5k / 10k / 37k / 45k+ (autosplits), with trims, between-splits, doubles, push, cross, fly-through, cap-splits, and a **straight-triple run at 37k** | `tools/choreo.py`, `tools/build_catalogue.py`, `runs/freeruns/` |
| 5b–c | **live controller**: stationary runs up to 908 s, auto-restart, watchdog against freezes | `tools/live_controller.py`, `runs/live/` |
| 5c/6 | **moving (dynamic) controller**, steerable from the viewer: runs of 1.5–3 min at ~200 u/s | same, `--mode dynamic` |
| 5d–e | feeding controller: random wandering balls of 30–30k mass that obey real agar eat rules, plus capture search. **The user rejected it: the team "completely lost the ability to splitrun"**, so it is paused | `splitrun/balls.py`, `tools/capture.py` |
| 6–6c | fast-travel attempts (see §5) | README Stage 6, 6b, 6c |

### How the main controller works (`tools/choreo.py` + `tools/live_controller.py`)
- **Designed moves:** circle, trim, double, between, push, cross, triple, striple (straight triple),
  fly-through, dash (a split toward a world point), ftrim (a trim along the travel direction), wait,
  feed, capture, swallow.
- Each move has sampled parameters: wait, target piece, aim offset, cursors before and after.
- **Receding-horizon beam search:**
  - beam 24, 64 samples per node;
  - the exact simulator plays every candidate, and the rules above prune the dead ones;
  - an event is committed once the best future is `lookahead` events (10–16) long;
  - it backtracks on dead ends.
- **Scoring:** purpose rewards, minus downtime, crumbs, excess cells and imbalance, plus style bonuses and
  path objectives (stationary drift, "dir" progress along a direction, hold at a waypoint).

---

## 5. Where I got stuck: fast sustained travel (the current task)

The user's request: "optimise for speed in the dynamic splitrun no feed version. It should still have low
downtime but also quickly move in any given direction." Then: "human splitruns involve the two players
generally moving in a circle direction around each other but with trim splits in the desired direction. It
can go much faster than you have now." They also said to try both "trim after every circle split" and "trim
every two circle splits"; humans do either. Once fast travel works, it is meant to power the feeder.

Measured on `tools/speed_bench.py` (start at x=-7,000, head to +8,000, 30 s; team 37k unless stated):

| approach | sustained speed |
|---|---|
| walking only, no splits | ~130–150 u/s |
| beam planner, "dir" objective (the live Moving controller) | 140–280 depending on seed; earlier long live runs ~200–230 |
| beam allowed a fine formation (≤16 cells) | 350–445 u/s, but only for 2–8 s: the structure burns |
| beam + world-anchored forward trim (`ftrim`), rhythm "circle, trim" or "circle, circle, trim" | 150–175, dies in 5–16 s (strict rhythm: every branch dies in 4 events) |
| evolved open-loop programs, world-anchored (`tools/travel_search.py`) | die in 4–9 s whatever the rhythm |
| RL (PPO, `splitrun/rlenv_translate.py`, progress reward) | 135–146, survives 40–60 s |
| periodic solver + travel reward (`tools/translate_technique.py`) | 3–8 net: the cycle frame flips each mirror cycle, so the travel cancels |
| periodic solver, world-fixed frame (`tools/travel_solve.py`, `WORLD_FRAME`) | technique 0: 3-cycle sprints at 200–500, never 10 cycles; with a strict layout repeat not even 1 cycle; techniques 4 and 6 never close |
| beam, old-old eats allowed (trail sweeping) | no gain |
| **beam at 10k / 3.5k total mass** | **275–330 sustained for 30 s** |

### My diagnosis (hypotheses)
- **The rhythm isn't the bottleneck.** About half the beam's splits are already forward "dash" splits, so it
  rediscovers "circle + forward trim" on its own.
- **The bottleneck is the clean-rule budget.**
  - A split moves half of each splitting cell about 820 units, so splitting fast and forward would be
    fast. A team of two 18.5k cells splitting forward every 10 ticks would move about 500 u/s.
  - But forward splits leave parents behind and multiply cells.
  - Those pieces must be eaten young, within 2 s, by the partner, while the cell count stays ≤10.
  - The partner's eats happen at the eater's position. A forward-launched piece eaten by a piece behind it
    gives the progress back.
  - So the useful eats are the ones where the forward piece is the eater, or where the eater is already
    ahead.
- **The beam is myopic.** A 10–16 event lookahead with sampled moves finds bursts, but not a sustainable
  structure.
- **Mass matters less than expected beyond 10k.** The 820-unit launch is fixed, so smaller teams leap
  further relative to their size and walk faster.
- **Benchmark noise is large:** the same settings give 140 or 280 on different seeds. Use several seeds
  and long horizons (≥60 s) before believing a number.

---

## 6. Other open problems (lower priority, but real)
- **Feeding version (paused).** The goal: start below 4k, splitrun while hunting balls ≥ 1/3 of the team's
  mass, avoid bigger balls, never stop splitrunning, grow to 60k.
  - The hard part is a **capture while splitrunning**: a dasher piece reaches the ball, and the partner
    throws mass into it mid-flight (a single, double or triple split, or several splits) so it passes
    1.3× and swallows.
  - My versions either stopped splitrunning or lost mass.
  - The intended order is fast travel first, then feeding.
- **4+-split periodic techniques.** There are 132 four-split and 66 five-split mass patterns, but their
  geometry was never solved. Open-loop plans are knife-edge and need the start already on the orbit.
- **Imbalanced (>2×) periodic orbits.** None found yet.

---

## 7. Practical notes
- **Python setup:**
  - `.venv` has torch, used for the RL;
  - the tools run with `python3` (numba, numpy).
  - The first run of a numba file compiles (`cache=True`), so expect a few seconds of delay.
- **Not a git repository.** Back up files before large rewrites.
- **Useful entry points:**
  - `python3 tools/live_controller.py --mode dynamic --out runs/live_dynamic --lookahead 10`: live moving
    controller; watch it at `http://localhost:8765/viewer/?tab=live&live=live_dynamic`;
  - `python3 tools/speed_bench.py --seeds 0 1 2 --secs 60 [--total 10000] [--style '{...}']`;
  - `python3 tools/choreo.py --name X --style '{"double": 3}' [--total 10000]`: an offline catalogue run.
- **Latest code changes (Stage 6c), all optional:**
  - `ftrim` move and global `DIR` in `choreo.py`;
  - the `rhythm` style ("CT", "CCT") in `live_controller.expand`;
  - `YOUNG` override in `choreo.py`, passed to `freerun.score_tick(young=...)`;
  - `WORLD_FRAME` / `STRICT_LAY` in `solve_technique.py`;
  - `--total` / `--young` in `speed_bench.py`.

  All defaults keep the old behaviour.
- **This folder was moved** from `~/Desktop/Claude_Projects/AgarioRL_attempt3`. No code hard-codes the old
  path, but some old logs mention it.
- **Background jobs:** the user has seen the CPU swamped by many parallel jobs. Run few, and keep them short.

---

## 8. Directions to consider (yours to choose, these are only my ideas)

I deliberately left this open. Here are the threads I would pull, in rough order of my confidence:

- **Reason about the fast pattern from first principles before searching.**
  - Work out the geometry of a "rolling" circle: which shots go forward, which eats keep mass forward,
    and how cell count is balanced.
  - The user preferred maths and geometry over brute force in the past (Stage 4b).
  - A hand-designed travelling cycle that is then refined could beat everything above.
- **Closed-loop tracking of a reference.** If any translating pattern exists, even a 3-cycle sprint like
  `runs/dynamic/wtravel_t0_*.json`, a feedback controller that re-aims at real pieces every event might
  sustain it. Scripts fail because they are open loop.
- **A better planner objective or horizon.**
  - Value the long-term "structure health" (cell count, piece ages, mass balance, formation shape)
    instead of only short-term progress.
  - Or learn a value function from the beam's own rollouts.
- **RL with world-frame actions.** The policy currently acts in the rotating pair frame. A world-frame
  split direction, or a "split along the requested direction" action, might lift it above 146 u/s.

Good luck. The simulator is trustworthy. The open question is the strategy.
