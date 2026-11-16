# Splitrun RL

Goal: an RL agent that controls **two players** and performs a **stationary splitrun** in
imsolo.pro physics. The players circle a fixed point by repeatedly eating each other's
backpieces, and they keep it up indefinitely. Later: discover new splitrun patterns.

The environment is a small, empty server: no food, viruses, minions or other players, and
no mass decay. It holds two players of 18.5k mass each, with at most 16 cells per player.
Single and double split are allowed, ejecting is disabled, and autosplit is on.

The project is done in stages, and each stage is signed off before the next one starts.

| stage | what | status |
|---|---|---|
| 0 | agree on the definition and the physics | done |
| 1 | environment: exact simulator + fidelity test + replay viewer (imsolo-style rendering) | done |
| 2 | splitrun detector (periodicity) + scripted clean splitruns (technique catalogue, 4 realised) | **awaiting sign-off** |
| 3 | first RL: algorithm and reward written down explicitly before training | – |
| 4 | longer and more robust runs, then discovery of new patterns | – |

## Layout

```
env_settings.json        the physics constants (imsolo GitHub settings + the overrides below)
reference_server/        pristine clone of github.com/jimboy3100/imsolo-servers, NEVER edited
splitrun/sim.py          batched Numba port of World.liveUpdate(), one server tick per step
splitrun/replay.py       record episodes for the viewer
splitrun/metrics.py      splitrun detector: splitrun_verdict() (periodicity-based)
splitrun/pattern23.py    the "2+3" clean splitrun pattern
tools/mass_cycles.py     enumerates which clean periodic mass patterns can exist (linear programs)
tools/solve_turn23.py    solves the geometry of one 2+3 turn for a mirror-periodic orbit
tools/run_pattern23.py   chains the solved turn (re-framing on the real pieces every turn)
tools/technique_catalogue.py  all clean periodic mass patterns up to a size (+ mass stability rho)
tools/solve_technique.py      fast, chainable geometry for any catalogue technique
tools/record_techniques.py    replays + plain-language descriptions -> runs/techniques/
tools/find_openings.py        split/eat sequences from two 18.5k cells towards a technique's start
tools/trace_server.js    runs the real server code headlessly on an input tape (ground truth)
tools/bench.py           simulator throughput
tools/make_stage1_demos.py   scripted scenarios for visual checks -> runs/stage1/
tests/test_fidelity.py   simulator vs real server, every cell, every tick
viewer/index.html        replay viewer (lists everything under runs/)
```

## Physics source and settings

The physics are taken from the imsolo server source (OgarII fork), `Megasplit-Beta/`, with the
constants from its `cli/settings.json`. The user confirmed that the live game differs only in
its interface, and that merge time is the same. The following are overridden in
`env_settings.json`:

- `playerMaxCells` 16 (the GitHub file says 128)
- `playerDecayMult` 0
- no pellets, viruses, boosters, mothercells, bots or minions
- the map stays at imsolo's 18,142 × 18,142, and the pair starts near the centre, so walls are
  never involved

Key numbers that follow from these settings (25 ticks/s):

| quantity | value |
|---|---|
| mass | size² / 100 (18.5k mass = size 1360) |
| eat rule | eater size ≥ 1.14 × victim size (mass ≥ 1.3×), and centres within `eater − victim/3` |
| split | halves the mass; the new cell starts 40 units ahead with a boost of 780 (it covers ~820 units in ~1 s) |
| own cells | pass through each other for 13 ticks after a split, then push apart |
| merge | a cell may re-merge `round(0.75 × size)` ticks after it was born: ~29 s for a 9,250 half, ~20 s for a 4,625 quarter |
| autosplit | any single cell above 22,500 mass splits in a random direction |

**Open question: double-split timing.** The live client sends double split as its own packet
(opcode 28), which the GitHub server does not have. So we do not know whether the live server
performs both splits in the same tick or one tick apart. The simulator supports both.
`runs/stage1/02a` and `02b` show the difference.

## Fidelity

`python3 tests/test_fidelity.py 60 1000` runs 60 random, aggressive two-player split
scenarios (up to 1,000 ticks and 32 cells each) through both the real server code and the
simulator:

- **60/60 are identical**: same cells, same ids, same order, every tick.
- The largest difference in position or size is about 1e-9, which is floating-point noise
  (`Math.pow` vs C `pow`).
- The quadtree collision order is reproduced exactly, and the test catches it if it isn't:
  with a flat collision order, 29 of 30 scenarios diverge.
- Autosplit uses a seeded `Math.random` in both, so it is covered as well.

Speed on an M2: about 550,000 env-ticks/s across 512+ parallel worlds, compared with about
90 steps/s against the websocket server in an earlier attempt.

## Viewing replays

```bash
python3 -m http.server 8765        # from the project root
# open http://localhost:8765/viewer/
```

The viewer controls:

- **Playback:** Space plays and pauses; ←/→ step one tick.
- **Camera:** `fixed` frames the whole run, which makes drift obvious; `follow` tracks the pair.
- **Centre trail:** the combined centre of mass, to judge whether the pair stays stationary.
- **Eats panel:** lists each cross-player eat and merge as it happens.
- **Overlay:** draws a second replay as dashed outlines, e.g. real server vs simulator.

## Stage 2: what a splitrun is, measured

**Definition used (from the user):** a proper splitrun is *clean and periodic*. Every split has a
purpose, and the relative positions of the balls repeat, or nearly repeat, every cycle, for many
cycles. The pair circles and stays in place.

**Detector** (`splitrun.metrics.splitrun_verdict`). A replay counts as a splitrun when all of these hold:

- **The layout repeats.** Each layout is described by every piece's mass plus all pairwise
  distances, which ignores where the pair is and which way it faces. The layout must match
  itself one period later, optionally with A and B swapping roles, to within 60 units. A few
  ticks of timing jitter are allowed.
- **It repeats for at least 10 consecutive cycles.**
- **Every period contains at least one cross-player eat**, so a pair that just sits still does not count.
- **Mass crosses in both directions.**
- **The combined centre drifts at most 3,000 units, and both players survive.**

An earlier detector counted alternating eats. It was fooled by messy exchanges and by
"ping-pong" of tiny pieces. The greedy-search example in `runs/stage2_rejected/` is kept as a
negative example.

**Which clean patterns can exist.** `tools/mass_cycles.py` treats one turn as bookkeeping:
- a split halves every one of the acting player's pieces;
- an eat needs the eater to have at least 1.3× the mass of what it eats;
- every launched piece must either eat or be eaten.

Checking whether masses can come back to the same values, or to the mirror image, is then a
linear program. With up to 4 pieces per player there is exactly **one** clean mirror pattern:

    before the turn   actor: 1/3, 1/6            partner: 1/4, 1/6, 1/12      (of total mass)
    actor single-splits:  its big piece's half (1/6) eats the partner's 1/12 backpiece
                          its small piece's half (1/12) flies into the partner's 1/4 and is eaten
    after the turn    actor: 1/6, 1/4, 1/12      partner: 1/3, 1/6            = mirror image

With 37k total, each player holds exactly 18,500 mass at every turn boundary.

**The 2+3 splitrun in the simulator.** `tools/solve_turn23.py` solves the geometry of one turn:
where the 5 pieces sit, where to aim the split, and how to move until the partner's split
62 ticks later. It is constrained to produce exactly the two intended eats, and the next turn's
layout must be the mirror image of this one. The solution matches to about 2 units.
`tools/run_pattern23.py` then chains turns, rebuilding the frame from the real pieces each turn.

- 400 of 400 turns clean (16.5 min of game time).
- Layout error at each split: median 5.5 units, max 20.
- Centre of mass drift: at most 181 units.
- The pattern rotates about 21° per turn, so a full circle takes about 17 turns (43 s).

Replay: `runs/stage2_pattern23/splitrun_2plus3.json`.

## Stage 2b: more techniques, less downtime, and inexact repetition

### A technique catalogue (`tools/technique_catalogue.py`)

The mass-bookkeeping search is generalized. A cycle is up to N split events by either player,
single or double splits. Every eat must involve a *flying* piece: one launched by this event or
the previous one, so a partner can split straight back into a backpiece. A launched piece may
eat up to two pieces, and every split must produce at least one eat. The cycle can return to
itself or to its mirror image.

With up to 3 starting pieces per player and 2 events per cycle, it finds 8 distinct techniques.
None uses a double split inside the cycle.

**Mass stability ρ.** A technique's mass bookkeeping is linear, so an error in the piece masses
maps to A·error after one cycle. ρ is the size of A's largest eigenvalue (ignoring the one that
belongs to the pattern itself).

- ρ < 1 means mass errors shrink every cycle, so the technique pulls itself back to its exact
  masses as long as its eats keep happening.
- All 8 techniques have ρ between 0.50 and 0.82. For example, the 2+3 technique has ρ = 0.82.

### Techniques realised in the simulator

Replays are in `runs/techniques/`. Every technique below is clean for 60/60 cycles and passes
`splitrun_verdict`. Masses are for 37k total.

| # | cycle | start | what happens each cycle | ρ |
|---|---|---|---|---|
| 0 | 20 ticks, 1 split | A 12,333 + 6,167 · B 9,250 + 6,167 + 3,083 | A splits: its big half eats B's 3,083 backpiece, and its small half is fed to B's 9,250. The result is the mirror image. (the "2+3") | 0.82 |
| 6 | 33 ticks, 2 splits | A 11,385 + 5,692 · B 11,385 + 5,692 + 2,846 | A splits: its big half eats B's 2,846 backpiece. B splits right back: B's launched pieces eat A's backpiece and one of A's launched pieces, and A's launched piece eats one of B's. Repeats itself. | 0.75 |
| 4 | 56 ticks, 2 splits | A 17,412 (one cell) · B 10,882 + 4,353 + 4,353 | A splits twice in a row. The 1st half eats B's 4,353; the 2nd split's half eats the other 4,353, while B's big piece eats A's 6,529 launched piece. The result is the mirror image. | 0.61 |
| 7 | 62 ticks, 2 splits | A 9,652 + 6,435 · B 12,870 + 4,826 + 3,217 | A splits: one launched half eats both of B's small backpieces. B splits: B's piece eats A's 3,217 launched piece. The result is the mirror image. | 0.76 |

Techniques 1–3 (5 pieces; one launched piece must eat two pieces while an old piece eats
another) are mass-feasible, but no geometry was found in the search budget. Technique 5 is
technique 0 written out over two turns.

Not covered yet:
- more than 3 starting pieces per player, or more than 2 splits per cycle (exhaustive
  enumeration there needs pruning; it ran for more than 20 minutes without finishing);
- merges inside a cycle;
- tricks based on the 16-cell cap.

### Downtime (`tools/solve_technique.py`)

Geometry is solved per technique. The search covers starting positions, split times, split aim
and movement cursors. It is scored over 1, then 3, then 10 chained cycles, re-read from the real
pieces at every cycle start, with a small penalty per tick of cycle length.

The 2+3 technique needs no downtime: it re-solves to a **19–20-tick cycle (0.8 s)** instead of 62,
and runs 400 clean turns. That is `runs/stage2_pattern23/splitrun_2plus3_fast.json`.

### Inexact repetition (the concern for RL)

Measured:

- **The scripted controllers are geometrically fragile.** Their moves are computed once and
  fired at fixed ticks. Start the fast 2+3 with piece masses off by just ±1–2%, and it usually
  breaks within 3–10 turns: slightly different sizes lead to one extra eat.
- **So exact mirror/identity layouts will not occur in RL.** Anything that rewards exact
  repetition would never fire.

Proposed fix, for Stage 3:

1. **Reward similarity, not equality.** Compare the layout at each split with the layout one
   split earlier (mirror) and two splits earlier (identity). The layout is described by every
   piece's mass plus all pairwise distances, so position and orientation don't matter. Give a
   smooth reward, `exp(-error / σ)`, that allows ±a few ticks of timing and some % of mass.
   Near-repeats earn nearly full reward.
2. **Reward clean splits, independent of technique.** A split earns reward only if its launched
   pieces are used: each launched piece eats or is eaten within about 1 s, and nothing else is
   eaten. Splits with no purpose earn nothing, or a small penalty.
3. **Prefer techniques with ρ < 1.** Their mass errors die out by themselves, so the policy only
   has to correct geometry. A closed-loop policy does this naturally, because it sees the real
   pieces every tick; the scripted ones cannot.
4. **No exact openings exist.** Techniques need masses like 1/3 of the total, while splits only
   make powers-of-2 fractions of 18.5k. Openings must therefore get *close* and rely on (3).
   Within 3 splits nothing gets within 15%, so openings need longer sequences, probably with
   merges. This is deferred until techniques are chosen.

## Stage 2c: less downtime, and the search for 4+-split techniques

**Downtime.** The gap between splits may now be 0 ticks, so both players can split at once,
and eats are checked per cycle. `solve_technique.py --shrink` then works like this: start from a
working solution, lower the cap on the cycle length step by step, re-optimise positions and angles
each time, and keep going while it still chains 100/100 cycles. Results (all clean 60/60 in the
replays):

| technique | before | after | per split |
|---|---|---|---|
| 0 (2+3) | 20 ticks | **10 ticks** | 0.40 s |
| 6 | 33 | **19** | 0.38 s |
| 4 | 56 | **33** | 0.66 s |
| 7 | 62 | **39** | 0.78 s |

Each one hit a wall just below these numbers: the next step down broke the chain.

**New 2-split technique** (`technique_2plus2`, 2+2 pieces, 22 ticks per 2 splits). It was found by
the sampler below and missed by the first catalogue, because one piece eats twice in a row.

**4+-split techniques**

The mass patterns come from `tools/random_techniques.py`:

- It draws random split/eat programs, choosing only eats that are legal for the masses at that
  moment, and runs their bookkeeping cycle after cycle.
- Real techniques are self-correcting, so their masses converge.
- Programs that repeat after fewer splits are rejected. The first "4-split" hit was really a
  2-split technique done twice.
- Result: **132 plausible genuine 4-split and 66 5-split techniques** (at most 3 eats per split,
  and no piece eating more than 2 at once).

Geometry for these is **not solved yet**:

- The solver has a per-split curriculum: get split 1's eats right, then splits 1–2, and so on.
  It gets every split's eats right for several techniques, and the masses close exactly.
- The last step, the layout coming back to itself, stalls at about 300 units of error. So the
  next cycle fails.

**Speed-ups made** (after the user flagged slow runs):

- Chain tests are batched: 1.9 s instead of about 40 s.
- Per-tick bookkeeping is vectorised: 1.3 s instead of 3.0 s per 4,096-candidate evaluation.
- The 10-cycle refinement uses a quarter of the population.
- Search phases stop early once they stop improving.
- Techniques are solved in parallel processes.

Bugs fixed along the way:

- For mirror techniques, the sampler had logged A and B swapped.
- A correctly closed cycle with a large layout error used to score worse than a failed cycle.
- Eat matching was too loose (±8%). It is now ±3%.

## Stage 2d: distinct families, and why 4+-split geometry is hard

**Families** (`tools/technique_families.py`) group mass patterns by the order players split in,
using a canonical form under A/B relabelling and cyclic re-starts. Simple alternation (ABAB...)
is the "splits in a circle" family. Of 43 families, 24 are *not* simple alternation: AABB, AABAB,
AABABBAB, AABBaB (with a double split), and others. Technique 4 is the AA-mirror family (full cycle
AABB).

`random_techniques.py --actors AAA` forces a split order. It found 15 `AAA` techniques (one player
attacks three times in a row, a direct extension of technique 4), 108 `AAB` and 99 `AABB`.
It found no `AAAA` in 2M samples.

**What stops the 4+-split geometry**, established by measurement:

1. **Knife-edge solutions.** The search found cycles whose eats only just happen: changing one
   parameter by 1 unit broke 19 of 28 directions. Robust scoring (`--robust K`) fixes this by
   scoring each candidate on K jittered copies. All eats then work under jitter.
2. **The artificial start is off the orbit.** Scripted plans fire fixed aims at fixed times, so
   every cycle must begin in the same state, including pieces in flight. A start with nothing
   in flight is only on the orbit if the cycle has a quiet gap at its boundary. That is why the
   2-split solutions were first found with downtime and then shrunk by continuation. Once they
   were shrunk, they converge from the artificial start: Newton's closing residual on shrunk
   technique 4 is 640 units, yet it chains perfectly.
3. **For 4+ splits no such quiet moment exists.** No open-loop plan works for both the warm-up
   cycle and the steady cycles. The answer is feedback within the cycle: splits triggered by the
   geometry and aimed at the real pieces. That is exactly what an RL policy provides, and what a
   closed-loop scripted controller would need to do with the knife-edge problem solved.

Tools added: `solve_closedloop.py` (a closed-loop controller designed from scratch; not yet
successful), `newton_close.py` (Newton closing of the starting layout; fails for reason 2), and
`--robust` / warm-up scoring in `solve_technique.py`.

## Stage 3: RL (running overnight)

**Algorithm: PPO** (`rl/ppo.py`). Clipped surrogate, GAE (γ 0.995, λ 0.95), 256 parallel worlds,
64-step rollouts, 4 epochs, Adam 3e-4.

- **One network controls both players.** An MLP (256-256, tanh) takes the observation (normalised
  with a running mean/std) and outputs, per role, a 2-D Gaussian cursor and a {none, single,
  double} split choice. A separate MLP gives the value.
- **One decision every 2 ticks.**

**Environment** (`splitrun/rlenv.py`, numba). Everything is in *role* terms, so a mirror
technique only has to be learned for half a cycle.

- **Observation:** up to 8 pieces per role, in the pair frame: position, mass, momentum, age,
  can-merge. Plus the next expected split, the timers and pending eats.

**Reward** (technique-following):

- +1 per intended eat.
- A tolerant repeat bonus up to +2 at each cycle start: layout similarity to the previous
  cycle start, `exp(-err/150)`.
- −0.002 per tick.
- The episode ends with a penalty on anything that breaks the technique:
  - an unintended eat or own merge;
  - a wrong split press;
  - an intended eat that never happens within 60 ticks;
  - no progress for 120 ticks;
  - drift of more than 3,000 units.
- A first version also rewarded +1 per correct split press. The agent exploited that by
  spamming the key, so pressing now earns nothing; only the resulting eats do.

**Start states** (`tools/make_startpool.py`) are full world snapshots, including pieces in
flight. They are taken from scripted solutions: re-simulated replays for the 2-split techniques,
and the first clean cycle of curriculum solutions for the multi-split families. With
`--grow-pool`, states from the agent's own episodes that completed at least one clean cycle are
added over time (Go-Explore style), so the frontier expands.

**Validation before training:** replaying the scripted technique 4 and 7 solutions through the
environment gives 60/60 cycles with no false terminations, about 6.9 reward per cycle.

**Runs** (`.venv/bin/python rl/status.py` for a one-screen summary; eval replays appear in the
viewer under `rl/<run>/`):

- `t4_ppo`: technique 4, as a proof.
- `AAB1_ppo`: `AAB` mirror family, 3 splits per half, `AAABBB` over a full cycle.
- `AABB0_ppo`: `AABB`, 4 splits per cycle.

**Early result:** `t4_ppo` learned technique 4 within about 10M ticks. Deterministic
evaluation completes 28 clean cycles in 60 s. Its layouts repeat (median 8 units between
consecutive half-cycle starts), and it runs **faster than the script** (26 vs 33 ticks per
half-cycle). The fixed-period detector rejects it because the cycle length varies by a few
ticks. The detector should compare phase-aligned layouts at cycle starts instead.

### Discovery RL (no target technique): `splitrun/rlenv_discover.py`

**Reward:**

- +0.5 for each clean split: all of its launched pieces are used within 45 ticks.
- A repeat bonus of +min(k, 8) whenever the layout (masses + all pairwise distances, with A and B
  allowed to swap) matches the layout k splits earlier.

**The episode ends on:**

- an unused launched piece;
- a split press that launched nothing;
- an eat between two pieces that weren't recently launched;
- an own-piece merge;
- a player below 15% of the total mass;
- drift of more than 3,000 units;
- 100 ticks without a split (−3, so doing nothing is the worst option).

`--min-flight 3`: a launched piece must fly at least 3 ticks before its eat. This is the largest
value all known techniques satisfy; the fast 2+3 and technique 6 use short-range eats.

**Episodes start** from the four known orbits (techniques 0, 4, 6, 7).

**Loopholes found and closed along the way:**

1. One-way feeding spam, which earned the old +1 per clean split. Fixed by lowering the clean-split
   reward and adding the 15% mass rule.
2. Doing nothing, when idling cost less than a risky split. Fixed with the −3 idle penalty.

**Results** (`runs/rl/discoveries/`):

- `discover_s0_best`: a **6-split cycle** (B, A, B, A, B, A). It repeats for 25 consecutive cycles
  with a median layout error of 0.9 units, at 29 ticks per cycle. 15 rotations per minute, drift
  133 units. Launched pieces fly a median of 11 ticks before their eat. From other starts it is
  less stable (12/23 cycles in one test).
- `discover_s1_flight3_best`: a **2-split cycle** where A and B split 1 tick apart. It repeats for
  78 consecutive cycles, at 9 ticks per cycle.

Both use only pieces of 12,333, 6,167 and 3,083, the 2+3 technique's mass lattice, so the agent
builds on that family.

## Stage 4a: imbalanced splitruns (in progress)

**Which imbalances can work** (mass bookkeeping, `tools/random_techniques.py --max-share 0.35`, mostly
single splits). "Plausible" means at most 3 eats per split and no piece eating more than 2 at once.

| smaller player's share | big player vs small | 2-split cycles found / plausible | 3-split cycles found / plausible |
|---|---|---|---|
| 0.30–0.35 | 1.9–2.3× | 55 / 10 | 90 / 19 |
| 0.25–0.30 | 2.3–3.0× | 46 / 4 | 61 / 12 |
| 0.20–0.25 | 3.0–4.0× | 98 / 2 | 47 / 10 |
| 0.15–0.20 | 4.0–5.7× | 39 / 0 | 50 / 14 |
| < 0.15 | > 5.7× | 41 / 0 | 72 / 7 |

- Why: an eat needs a 1.3× mass ratio. For mass to flow **big → small**, the big player's split must
  cut pieces below the small player's pieces / 1.3. Every split halves **all** of the big player's
  pieces, so it also launches pieces that nobody eats; they simply stay part of it.
- 2-split cycles run out at about 4:1. 3-split cycles reach beyond 5:1, because the big player can
  halve twice before it is fed back.

**Split-direction measure** (`tools/split_directions.py`): the angle between each split and the line
to the partner (0° = straight at the partner).

| technique | split angles | spread |
|---|---|---|
| 0 | 30° every split | 0.2° |
| 6 | about 29° | 4.7° |
| 7 | about 55° | 2.6° |
| 2+2 | about 26° | 3.2° |
| RL 6-split | about 33° | 4.8° |
| RL 5-split | about 26° | 3.1° |
| **4** | **alternates 9° and 47°** | **19°** |

This matches the user's visual judgement: everything except technique 4 is a "circle".

**Geometry so far**

- 40 imbalanced patterns screened, 10 have one full cycle that can be performed (shares 0.27–0.35).
  None closes into a repeating orbit with the open-loop solver.
- The solver regressed: from scratch it no longer closes technique 0 either (3/100 cycles).

**RL attempts** (all stopped; `splitrun/rlenv_discover.py` gained `imb_max` and `purpose`)

1. Discovery from imbalanced start states (`runs/rl/pool_imb_f.pkl`, 1,766 states): 0 cycles. The
   small player drains below 15%.
   - Bug found and fixed: the "every launched piece must be used" rule forbids imbalanced
     splitruns. `--purpose` now requires only one used piece per split.
2. Continuation from the balanced discovery policies (`--init-from`, `--imb-schedule
   0.45,0.20,0.03,30`): the policy keeps splitrunning, but never once reaches an orbit with share
   ≤ 0.45. Balanced techniques self-correct back to 50/50, so the sparse reward gives no gradient.
3. The per-cycle re-planner (`build_chain.py`) chains at most 1–2 cycles.

Lesson: RL learns a technique well only from start states on a **working** orbit (t4: 31 cycles/min).
The missing piece for imbalance is one working seed.

## Stage 4b: imbalance, reasoned from the geometry

User guidance (2026-09-24): human imbalanced splitruns are broadly circles. The big player adds a few
off-circle **trim** splits from its biggest piece so that its backpieces get small enough.

**Mathematical results**

1. **Pieces must balance.** If each player's piece count repeats every cycle, then
   launched pieces per cycle = eats per cycle.
   - In a plain circle, every launched piece takes part in exactly one eat.
   - A trim launches pieces with no eats. So elsewhere in the cycle, some launched piece must take
     part in two eats: it flies through two backpieces, or eats one and is then swallowed.
2. **The big player always has ≥ 2 cells at > 2x.** 2x means over 24,700 mass, and a single cell
   autosplits above 22,500. So every trim launches ≥ 2 pieces and needs ≥ 2 double eats.
3. **Aimable splits are not enough.** Take splits aimed in closed form (≤ 2 cells; one shot per
   launched piece, aimed at the crossing of two lines). Among all 48,024 two-split programs, only
   technique 0 is periodic. Imbalance therefore needs volleys with multiple eats.
4. **A plain circle can hold 1.6x with no trims** (`tools/circle_trim.py`,
   `runs/techniques/circle_imb.json` #0):
   - A 12,870 + 6,435 + 3,217 | B 6,435 + 4,826 + 3,217.
   - This is technique 0 plus one extra rung: B's 4,826 swallows A's extra launched 1,609.
   - A's pieces form a "conveyor ladder" (8 → 4 → 2 → eaten).
5. **One trim per 2-split cycle (tAB) gives no imbalanced solution** under these volley rules.

**Geometry so far**

- The 1.6x circle's geometry was seeded from technique 0's solved layout plus a translated extra
  pair: in a parallel volley, the extra shot lines up automatically.
- All eats of one full cycle happen and the masses return exactly
  (`runs/techniques/imbalanced_circle_1p6x_one_cycle.json`).
- The second cycle does not repeat. The ladder has to lie on a rotating ring, and own-piece
  collisions push its pieces apart. Neither the open-loop solver nor the per-cycle re-planner holds
  it beyond 1 cycle.

Earlier this stage (all failed, stopped): blind controller-program search, a per-split MPC
controller, and the closed-loop parametric controller on sampled multi-eat patterns.

## Stage 5: non-periodic creative splitruns (catalogue)

New goal (2026-09-29): periodicity dropped. Runs must:
1. last at least 20 s with as little downtime as possible (the priority);
2. contain many creative splits.

**Method: choreography with beam search** (`tools/choreo.py`), exact simulator, no RL.

A run is a sequence of split events. Each event is one designed move:

| move | what it does |
|---|---|
| circle | shot at a chosen partner piece |
| trim | aimed 70-180° away from the partner |
| double | a double split |
| between | aimed at the gap between two partner pieces |
| push | aimed into the splitter's own piece |
| both | both players split in the same tick |

Search:
- Each event also has a wait (at most 12 ticks), an aim offset, and cursors for both players.
- The best 64 partial runs are kept, with 128 random candidates per step.
- When every branch dies, the search jumps back about 10 events and resamples with less style
  pressure.

Rules; a branch dies on any of:
- a split whose pieces take part in no cross-player eat within 50 ticks;
- a slow absorption between two old pieces;
- a split press at the 16-cell cap;
- a player below 12% of the mass;
- more than 50 ticks (2 s) without a split;
- more than 10 cells, or more than 2 pieces under 500 mass.

Ranking: + purposeful splits (launched piece used), − downtime, − crumbs and excess cells,
− mass imbalance outside the style's band, + a style bonus per creative move.

**Analysis** (`tools/analyze_freerun.py`) is independent of the planner and measures moves from the replay:
- off-circle splits (more than 60°), double splits, simultaneous splits;
- between-gap splits: a launched piece's flight crosses the open gap between two partner pieces;
- pure trims: launched pieces never used;
- push-assisted eats: an older own piece is shoved by its launched piece faster than its cursor could
  move it, then eaten. Plain technique 0 has these too, because rigid separation pushes the backpiece
  backwards.

**Catalogue** (`runs/freeruns/`, viewer tab "Non-periodic catalogue"): all 8 runs last 30-32 s.

| run | average gap | longest gap | stands out |
|---|---|---|---|
| tempo | 0.37 s | 0.52 s | least downtime |
| trims | 0.47 s | 0.68 s | 27 off-circle splits; widest angle spread (70°) |
| doubles | 0.44 s | 0.68 s | 30 double splits |
| between | 0.38 s | 0.68 s | 38 between-gap splits |
| push | 0.45 s | 0.68 s | 24 push-assisted eats (tempo has 39 as a side effect) |
| both | 0.44 s | 0.68 s | 20 simultaneous splits |
| mixed | 0.40 s | 0.68 s | all moves |
| imbalanced | 0.42 s | 0.68 s | smaller player at 27% of the mass on average (other player ~2.7x) |

Tidiness: 6-7 cells per player and 1-2 crumbs on average, with short spikes up to 16 cells mid-event.

What did not work first:
- A generic planner with no designed moves shattered the players into crumbs or drained one of them.
- The trained discovery policy used as a base only recovers from states on its own orbit: every
  trim, double split and push-split killed it.

## Stage 5b: user feedback, mass variants, live controller (2026-09-29)

**Definitions from the user**
- **Push-split:** uses the "recoil", which is the spring-apart shove after the 13-tick no-collision
  window. It is rare; sometimes used to shove your own backpiece into your partner.
- **Trim:** aimed sideways and slightly backwards away from the circle (or two trims: one sideways, one
  backwards). The trim's launched pieces are soon eaten by the partner.
- **Split between the pieces:** your launched piece flies through the gap between two of the partner's
  pieces.
- **Double splits:** mainly for reach. The user likes them; use more.
- **Straight triple split:** "n" + triple split in imsolo. Modelled as 3 presses in one tick with the
  cursor 30,000 units away, so every launch is parallel.
- Tricksplits and popsplits are irrelevant here.
- **My ideas the user approved:** fly-through, selective split at the cell cap, crossing splits.
- Merges imply too much downtime.

**New moves** in `tools/choreo.py`:
- triple (straight);
- flythrough: aim at a partner piece behind another partner piece of similar size, which the launched
  half can neither eat nor be eaten by;
- cross: both players split at each other in the same tick;
- capsplit: style bonus when the splitter has 14 or more cells.

Trims are now 80-150° off the partner direction. `--total M` scales the start layout: masses by M,
distances by sqrt(M). Crumb thresholds scale with it. A drift penalty keeps the pair stationary.

**Mass-varied catalogue**: all 30-31 s, average gap 0.36-0.47 s, longest gap 0.68 s.
- **3.5k:** doubles (39 double splits); straight triples (5); crossing + fly-through (14 verified
  crossings, 16 fly-throughs).
- **10k:** doubles + trims (31 double splits); triples + trims; fly-through (19 fly-throughs, shortest
  downtime at 0.36 s).
- **45k/60k:** 45k never quite reaches the autosplit limit (pieces peak at 22,098). 60k has 2
  autosplits; the many-cells 60k run has 8 autosplits, but only 1 true cap-selective split.

The analyser also checks crossing splits (both launch lines actually cross) and fly-throughs (the
launched piece's path covers a same-size partner piece, and a smaller partner piece behind it is eaten
within 30 ticks). `tools/build_catalogue.py` rebuilds the viewer catalogue.

**Live controller** (`tools/live_controller.py`): receding-horizon beam search over the same moves.
- Beam of 24, 64 candidates per node, look-ahead of 10 events. The first event of the best future is
  committed and played in the real world.
- If every future dies it backtracks within the look-ahead; if nothing survives, it restarts immediately.
- It streams the rolling last 60 s to `runs/live/live.json` (viewer tab "Live controller", ~3 s behind the
  frontier). Runs of 60 s or more are saved as `runs/live/run_NNN.json`.
- Test: 217.8 s without breaking a rule (stopped by the time limit). 512 splits, 0.45 s average gap,
  0.68 s longest, 6.6 cells per player, drift under 1,700 units (`runs/live/test_run_218s.json`).

**Viewer**
- Agar-style client interpolation: pieces ease over about 120 ms, and eaten pieces slide into their
  eater.
- Calm follow camera; its tightest view scales with total mass.
- Recordings auto-play and loop.

## Stage 5c: straight triples, controller freeze, moving and feeding controllers (2026-09-29)

**Straight triple splits.** The catalogue's earlier triples were not straight. Measured in the exact
simulator:
- From ONE cell with the cursor far along the aim, a triple split is perfectly straight at any angle
  and any press timing.
- It goes messy when (a) the splitter has several cells side by side, so each fires its own line, or
  (b) the cursor moves on during the ~13 ticks of flight: pieces drift sideways at different speeds and
  then spring apart from a skewed stack.

The new `striple` move in `tools/choreo.py`:
- only allowed with 1 cell, or 2 cells lined up toward the target (within 60°);
- the cursor goes 30,000 units along the line and stays locked there for 22 ticks (HOLD), like "n" in
  imsolo.

The analyser verifies straightness: principal-axis fit of the splitter's pieces 12 ticks later, or
just before its next split.

Result: `runs/freeruns/straight_triples.json`, 30.5 s at 0.40 s average gap, 2 verified straight
triples: A's opening from a single cell, and B's at 12 s after it got back down to 1 cell. The
triple's pieces are eaten within about 0.5 s, so the line is visible only briefly.

**Controller freezes.** Two causes:
1. In a hard state the search went round in circles (extend, back up, extend) without ever
   committing or giving up, so the stream stopped. Fixed with a watchdog: after 8 s or 60 failed
   expansions without a commit, restart. The look-ahead is now 16 events.
2. Two controllers writing the same temp file crashed one. Temp files are now per process.

The viewer plays 5 s behind the frontier.

**Moving controller** (`--mode dynamic`): the pair's centre follows a target on a loop of radius 4,500
around the map centre at 150 units/s. Test: runs of 185 s (then the watchdog) and over 61 s (stopped
by the test limit), staying 3,400-5,000 from the centre.

**Feeding controller** (`--mode feeding`, `splitrun/balls.py`):
- 80 neutral balls, mass 20-3,000 (log-uniform), wandering (new heading every 40 ticks; smaller is
  faster). Deterministic hashing, so the planner's copies predict them exactly.
- A player piece eats a ball by the normal rule and grows; the ball respawns elsewhere. Balls never eat
  and there is no decay.
- A ball eat counts as a split's purpose. Eaten mass is rewarded (food_w).
- The pair tours a loop of radius 5,000; a run restarts at 60k combined mass.
- Test: 143 s, growing about 5k mass per 40 s.

**Viewer:** the "Live controller" tab has a Stationary / Moving / Feeding selector
(`runs/live`, `runs/live_dynamic`, `runs/live_feeding`). Balls are drawn green and ignored by the camera
and centre trail. Long runs are saved as `run_<time>_<n>.json`. Start a controller with:

    python tools/live_controller.py [--mode dynamic --out runs/live_dynamic | --mode feeding --out runs/live_feeding]

## Stage 5d: feeding version rework (in progress, 2026-09-29)

**Done**
- Start at 3,500 total mass; restart at 60k.
- Targets are balls of at least 1/3 of the team's mass, swallowable after one feed (at most 1.3x the
  biggest piece). Smaller balls are ignored: no reward, no penalty.
- No restart when stuck: the controller commits a pause (no split, drift together) and replans. It
  restarts only after 10 pauses in a row.
- Balls (`splitrun/balls.py`):
  - fixed variety of sizes from the start, log-uniform 30-30,000;
  - they EAT player pieces they are 1.14x bigger than (and grow);
  - smooth motion: cosine-interpolated headings, and they steer back from the map edge;
  - the viewer draws random colours and procedural skins.
- New moves:
  - feed: the partner splits into the teammate's piece nearest the target;
  - dash: a split towards the target;
  - strike: no split, both cursors on the target;
  - wait: a pause.
- Purpose window 100 ticks in feeding mode; a danger term for balls that can eat us; strong penalty for
  mass lost to balls. The chase point is stored in each move, which fixed replay divergence.

**Open problem.** A ball of at least 1/3 of the team is bigger than almost all of the team's pieces,
so it preys on them. Controlled test (worthy ball 700 units from the team): the team collapses from
3,500 to 132-2,060 because small pieces race ahead into the ball. No worthy swallow happened in free
runs either.

**Background controllers at this point:** stationary runs of 461, 273, 516, 579 and 908 s; moving
runs of 180, 128, 111 and 88 s.

## Stage 5e: capture search (2026-09-30, in progress)

User direction: real agar rules for balls; prune tree-search branches that lose mass; the capture should
be a partner throwing mass (single / double / triple split, or two splits) into a frontmost piece as it
travels toward the target, while splitrunning.

**Built**
- Loss pruning: a branch dies if more than 4% of the team's mass is lost to balls in one move. In
  controlled tests the team no longer collapses (3,500 kept, previously 132-2,060).
- Lean cursors: the formation drifts toward the target instead of every piece being steered onto it.
- Dasher protection: a penalty when the dasher's own player splits, since that halves the dasher.
- `tools/capture.py`: a structured capture search, run after every committed move when a candidate
  eater (at least 0.77x the ball) is within 1,400 units of the target. It simulates a grid of plans
  in one batch (about 0.1 s) and commits the first that swallows the target without losing mass:
  - which dasher (up to 4);
  - ram, or dash-split toward the ball;
  - feed with a single, double or triple split from the partner;
  - feed delay 0-22 ticks, and a lead of 0/150/350 units.
- Found and fixed along the way:
  - the feeder steered onto the dasher, causing slow absorptions;
  - the "drained player" floor of 12% forbade feeding. It is now 3% in feeding mode.

**Results**
- Controlled test (worthy 1,200 ball, 1,500 units away): 2 of 4 seeds capture within 1.3-2.2 s
  (3,500 to 4,700).
- Free 5-minute run: 9 runs ended after 15-35 s ("no continuation": the splitrun planner runs out of
  moves under the feeding-mode rules), with 1 capture (to 4,832).
- The planner is slow in hard states: many backtracks, about 0.16 s per expansion.

## Stage 6: fast translation + user steering (2026-09-30)

The feeding work is paused: the team lost its splitrun under the feeding rules (user feedback).

**Objective** (`tools/choreo.py`, `path="dir"`):
- speed_w x (how far the team's centre advanced along the wanted direction during the move),
  minus lat_w x (sideways drift);
- once within 1,500 units of the waypoint, hold there (stationary objective);
- extra options: lean the formation toward the travel direction, and dash splits along it.

**Benchmark** (`tools/speed_bench.py`: start at x=-7,000, head for x=+8,000):
- Every sustainable setting lands at about 200-230 units/s, with a split every 0.5-0.6 s (longest gap
  0.7-0.9 s).
- For reference, steering alone (no splits) moves this team about 150 units/s: a 12,333 piece moves
  141 units/s and a 6,167 piece 164.
- Allowing a fine formation (up to 16 cells, few crumb penalties) reaches 350-445 units/s, but only
  for 2-8 s. It burns its structure: pieces fragment and nothing recombines them (merges take 10-20 s),
  so the planner runs out of legal moves. A sustainable pattern must launch and eat at the same rate.

**Steering:**
- `tools/serve.py` replaces `python -m http.server` (see `.claude/launch.json`) and accepts
  `POST /api/steer`, which writes `runs/live_dynamic/control.json`.
- In the viewer's Live > Moving stream, click the map to set a destination, or use the arrow keys /
  WASD to go 6,000 units in that direction. The controller re-plans immediately and holds once there.
- Live runs so far: 192 s and 182 s, but also several that ended after 6-31 s.

**Stage 6b: speed optimisation attempts** (sustained travel, units/s)

| approach | result |
|---|---|
| beam planner + progress objective | about 200-230 sustained; the live Moving controller uses it |
| beam + finer formation (up to 16 cells) | 350-445 for 2-8 s, then it runs out of moves (structure burns) |
| beam + faster cadence / double forward trims | 135-275, often dying early |
| beam + projected centre of mass (credit trims at launch) | no gain |
| RL (PPO), progress reward, warm-started from the discovery policy | 135-146 sustained over 40-60 s |
| periodic technique solver + net-travel reward (techniques 0, 6, 4) | 3-8 net: split aims are in the rotating pair frame, so a "forward" bias loops round |

Bug found on the way: the discovery environment ended episodes when the pair drifted more than 3,000
units from its start. That is now a parameter and off for travel.

Conclusion: the human pattern (circle + trims in a fixed map direction) needs trims anchored to the
world direction while the circle continues in the pair frame. Next step: a parametric circle+trim
controller with world-anchored trims, optimised for sustained speed.

**Stage 6c: circle + world-anchored forward trims** (37k unless stated; same 30 s benchmark, `tools/speed_bench.py`)

| approach | result |
|---|---|
| beam planner + new `ftrim` move (split along the map direction), free choice | 200 for 12 s, then no continuation |
| beam + rhythm "circle, trim" (trim after every circle), soft-enforced | 159 for 5 s; strict rhythm: every branch dies within 4 events |
| beam + rhythm "circle, circle, trim" | 153-175 for 12-16 s |
| evolved open-loop programs (`tools/travel_search.py`), rhythms C / CT / CCT, with feedback rules | survive only 4-9 s (cell limit or one player drained) |
| periodic solver with a world-fixed frame (`tools/travel_solve.py`, techniques 0/4/6) | technique 0: 3-cycle sprints at 200-500, never 10 cycles; with a strict layout repeat not even 1 cycle; 4 and 6 never close |
| beam, old-old eats allowed (sweeping the trail) | no gain (134-192) |
| beam at 10k / 3.5k total mass | 275-307 sustained 30 s (37k: 140-280 depending on seed) |

Findings: about half the beam's splits are already forward "dash" splits, i.e. it discovers circle +
forward trim by itself, so the rhythm is not the bottleneck. Periodic translation of the known
techniques does not close. Open-loop programs are too fragile for the splitrun rules.
