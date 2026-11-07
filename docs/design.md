# How Agent1 works

This is the long version of the design. The [README](../README.md) has the short one.
All numbers below are the values in [`agent/config.py`](../agent/config.py) and the code;
code references are to the modules in [`agent/`](../agent/).

## Contents

1. [Setting](#1-setting)
2. [The race model](#2-the-race-model)
3. [Valuing positions](#3-valuing-positions)
4. [Evaluating actions](#4-evaluating-actions)
5. [On the ball](#5-on-the-ball)
6. [Off the ball](#6-off-the-ball)
7. [Defensive assignment](#7-defensive-assignment)
8. [Stamina](#8-stamina)
9. [CPU time](#9-cpu-time)
10. [Configuration and disabled alternatives](#10-configuration-and-disabled-alternatives)
11. [Known issues](#11-known-issues)
12. [Code map](#12-code-map)

---

## 1. Setting

The game is RoboCup 2D simulated soccer, 5-a-side, two halves of 750 cycles (10 cycles
per second). The course framework gives every player an exact world model (full state,
no noise) and calls one of two methods per player per cycle, only in normal play:

- `on_ball(agent)` when the player can kick the ball;
- `off_ball(agent)` otherwise.

Set pieces, kick-offs, penalties, the goalkeeper, neck and view control are handled by
the framework and are the same for every team. Our goalkeeper simply clears if `on_ball`
is ever called for him.

There is no communication. The players coordinate because every one of them runs the
same deterministic computation on the same exact world model, so they agree on who
chases, who blocks and where everyone else should go.

Both entry points are wrapped in `try/except`: a bug falls back to a clearance or to
walking home instead of crashing the player.

## 2. The race model

Almost every decision reduces to one question: **for this ball trajectory, at which cycle
can each player first control the ball?** `_race` in `physics.py` answers it for many
trajectories and all players at once, as one numpy computation.

**Ball.** A kicked ball's speed decays by a factor of 0.94 per cycle, so after `k` cycles
it has travelled `s0 * G[k]` with `G[k] = (1 - 0.94^k) / 0.06`. A whole trajectory is one
vectorised line (`_kick_traj`), predicted up to 40 cycles ahead.

**Players.** `_Phys` precomputes `DASH[n]`, the distance a player covers in `n` full
dashes from rest, and variants for dashing sideways or backwards (`DASH_DIR`) and for a
tired player who can only sprint for `j` more cycles before dropping to the stamina-floor
pace (`DASH_TIRED`).

A player reaches the ball at cycle `t` if

    distance(ball at t, player) <= control radius + DASH[t - reaction - turn]

- **Reaction:** one cycle in the search.
- **Turn:** 0 cycles if the ball is within 25° of his body direction, otherwise 1–3
  depending on the angle and his current speed (the server's inertia rule: a moving
  player can turn at most 180 / (1 + 5 · speed) degrees per cycle).
- **Drift:** a moving player first coasts along his current velocity.

**Opponents get the benefit of the doubt.** Their control radius is 0.3 m larger than
ours, they may dash sideways or backwards without turning first, and their keeper uses
his catch radius inside his box. **Our tired teammates do not:** their sprint budget
from the stamina model ([§8](#8-stamina)) limits how far they get.

The difference between their first arrival and ours, in cycles, is the **margin**. It is
what turns into probabilities everywhere below.

## 3. Valuing positions

`_Phys.threat(x, y)` is the chance that our possession starting at `(x, y)` ends in a
goal. It is about 5% deep in our half and rises along a sigmoid towards their box, is
lower out wide, and has a bonus for the central zone in front of goal. From it:

| Function | Meaning |
|---|---|
| `v_keep(x, y, space)` | we have the ball at `(x, y)`: threat, plus a little for space around it |
| `v_lose(x, y)` | they win it at `(x, y)`: minus their threat (`threat` mirrored) |
| `v_out(x, y)` | ball goes out at `(x, y)`: a smaller loss |

So the main evaluator puts every option on one scale: roughly the chance that the
possession ends in our goal minus the chance that it ends in theirs.

The chain search ([§5](#5-on-the-ball)) uses a second scale, `_field_eval`: metres of
progress plus 100 × threat plus a bonus for the central slot.

## 4. Evaluating actions

### Passes (`EvaluatorMixin._eval_kicks`)

**Candidates** (`_pass_targets`): each receiver's feet and points on rings of 3, 6 and
9 m around him (8, 8 and 12 points); kicks into space in a fan in front of the ball
(9 directions × 8–25 m); through balls 7 and 13 m ahead of forward receivers. Each
target is tried at arrival speeds of 1.0, 1.4 and 1.8 m/cycle. Receivers exclude anyone
offside. If one kick cannot reach the needed speed, the pass takes two kicks (one cycle
later).

**Success probability** from the margin `m` (clipped to ±6 cycles, against the earlier of
the opponents' arrival and the ball going out):

    p = min( sig((m - 0.8) / 0.6),  sig((m + 1.9) / 2.8) )

then multiplied by 0.95–1.0 depending on how quickly an opponent can close down the
receiver, recalibrated with a logistic fitted to our logs (`PASS_FIT`, which also looks
at the margin against opponents with a 2 m tackle reach), and for two-kick passes
scaled down by how close the nearest opponent is (`_multi_keep`).

**Value:**

    value = p · 0.985^t · v_keep(reception point) + (1 - p) · v_lose(where they get it)

A pass that our own keeper would collect in his box is an illegal back-pass and is
valued at −0.85.

### Shots (`_eval_shots`)

13 aim points across the goal mouth, each scored as

    P(goal) = P(on target) × P(no field player intercepts) × P(keeper doesn't save)

- **On target:** the kick's error window grows with distance (0.03 · d + 0.3 m).
- **Intercepts:** from the race, `sig((t_them - t_goal + 1) / 0.45)`.
- **Keeper:** a logistic fitted to our logs on his distance from the shot line and the
  cycles until the ball passes him (`GK_A`, `GK_P`, `GK_N`), combined with a check of
  whether he can physically get there.

### Dribbles (`OnBallMixin._dribble_cands`)

A dribble is simulated as the ball being carried at 0.72 m/cycle (after a one-cycle turn
if the direction is more than 35° off the body). Opponents race the carry path. Success
is 0.97 if nobody can reach it, otherwise `sig((t_them - K + 0.5) / 0.8)`, and for
dribbles of up to 10 m it is recalibrated with a logistic fitted to our logs
(`CARRY_FIT`: distance to the nearest opponent, number of opponents within 4 m).
Dribbles that leave the pitch or pass next to our keeper in our box are dropped.

### Hold and clear

Holding the ball is allowed only with nobody within 2.2 m; its value falls as the nearest
opponent gets closer. Clearing (30 m upfield and wide) is an option only behind
x = −32.

## 5. On the ball

`OnBallMixin._on_ball`, each cycle:

1. **Committed kick.** If last cycle started a two-kick pass or shot, continue it, unless
   the player is now pressed and the direction is blocked, or the pass lane has dropped
   below p = 0.55.
2. **Keeper reset.** Deep in our half, pressed, with our keeper more than 8 m off his
   line: kick the ball out over the touchline so he can get back.
3. **Shot decision** (`SearchMixin._on_ball_chain`). Within 30 m of goal, evaluate the
   best one-kick shot (or a two-kick full-speed shot). Shoot if `P(goal)` clears the
   threshold for the situation:

   | Situation | Threshold |
   |---|---|
   | default | 0.68 |
   | inside the box | 0.64 |
   | x > 32 with an opponent within 2.2 m | 0.35 |
   | more than 24 m out, not pressed | 0.72 |

   Before shooting, try a short dribble that would make the shot clearly better
   (`_carry_in`). Shots below 0.86 that are not under pressure are not taken yet: they
   go on to be compared with the alternatives below.
4. **Candidates:** shots, one-kick passes in a 15° fan, dribbles of 2.5, 4.5 and 8 m,
   hold, clear.
5. **Expectimax beam search** over passes and dribbles (`SearchMixin._beam_exp`).
6. **Last resort,** if nothing is viable: a channel kick down the wing, a bounce pass to a
   nearby teammate, an evasive dribble, or a clearance.
7. Pick the option with the highest value and execute it. One exception: when pressed in
   our half with two or more tired teammates and a risky best option, kick the ball out
   for a throw-in instead (`_touch_out`).

The score affects risk: after cycle 1150 the agent plays safer when leading and takes
more risks and shoots more when behind (`_risk`).

### The expectimax beam

Every pass or dribble is a **chance node**: with probability `p` we keep the ball and
play continues from the child state, otherwise we get the loss value. Each node carries
two accumulators, `A` (expected value banked from failures along the path) and `Q`
(probability of still having the ball, times discounts), and is worth

    A + Q · V(current ball position)

Expanding a node with a pass of success `p'` arriving at cycle `t`:

    A' = A + q · (1 - p') · V(loss)
    Q' = q · p' · 0.985^t          with q = 0.92 · Q (distrust of deep predictions)

A shot is a leaf worth `A + Q · 0.96 · shot value`. Each layer keeps the 24 best nodes,
at most 3 per (parent, receiver) pair so the beam does not fill with near-identical
passes, for up to 4 layers. The value of each **root** action is the best value found
anywhere below it. The search is *anytime*: it stops at a deadline from the CPU budget
([§9](#9-cpu-time)) and uses the best values found so far.

**Child states** (`_State.child`) imagine the world when the pass is received:

- the nearest opponent presses the receiver;
- the other opponents shift to cover the space between the ball and their goal, and
  their keeper guards it;
- every opponent gets a larger control radius the further we extrapolated him
  (uncertainty);
- our players coast on their velocity and push up (more on a counter-attack), staying
  onside;
- the passer makes a give-and-go run.

### Executing

Two-kick passes and shots are *committed*: the plan is stored and continued next cycle.
After a pass beyond x = −5, the passer starts a give-and-go run for up to 16 cycles.

## 6. Off the ball

`OffBallMixin._off_ball`: each cycle a player takes the **first** rule that applies.

1. **Shot block.** The ball is heading into our goal before we can reach it, and the
   tackle succeeds with probability ≥ 0.40: tackle.
2. **Fair tackle.** The opponents have the ball, or will get it at least two cycles
   before us. Tackle if the probability is ≥ 0.52 (≥ 0.40 deep, x < −18, or wide,
   |y| > 18), aiming the ball upfield.
3. **Foul.** The opponent has the ball, the player has committed fewer than 2 fouls and
   has no card, and the foul succeeds with probability ≥ 0.78. Two cases:
   - *last-ditch:* 16.5–32 m from our goal, outside the box, situation serious;
   - *midfield:* x between −25 and 25 with a fair-tackle chance below 0.52 and a
     teammate behind the ball.
4. **Chase.** The chaser from the shared context (first to the ball, with a little
   stickiness so the role doesn't flip every cycle) intercepts, if we are attacking or
   he is at most 2 cycles behind the first opponent.
5. **Block.** When defending, `_block_plan` predicts the opponent carrier's dribble
   towards our goal (0.82 m/cycle, bent towards his current running direction). The
   teammate who can reach the earliest point of that path in time runs there,
   preferring someone already goal-side. When within 14 m of the ball (16 m if the
   carrier is standing still) he engages: he runs straight through the ball if it is
   within 4.2 m and 30° of his heading, sprints to where the ball will be next cycle if
   he is close or caught upfield of it, and otherwise keeps heading for the block point.
6. **Give-and-go.** Continue the run started after our own pass.
7. **Position.**
   - *Attacking:* `_support_point` gives special roles first. The deepest player stays
     back as rest defence; near their box, the two most advanced take the far post and
     the cut-back spot; forwards make runs into the box or diagonals ahead of the ball;
     the two nearest come short as outlets when the carrier is pressed. Otherwise
     candidate points around the player's (ball-adjusted) home position are scored by
     how open they are for a pass (the race again), the value of receiving there,
     distance and crowding.
   - *Defending:* the spot from the defensive assignment ([§7](#7-defensive-assignment)).

## 7. Defensive assignment

`DefenseMixin._defense_plan` decides where the free defenders (everyone except the chaser
or blocker) stand. It is a small **minimax**: we choose spots to minimise the best thing
the opponents can do next.

1. **Their options** (`_their_actions`): passes to each of the carrier's teammates (to
   his feet, towards our goal, and a through ball), dribbles of 6, 12 and 18 m in several
   directions including straight down the wing, and shots if he is within 28 m. Each is
   valued from their point of view (our field evaluation, mirrored). When the ball is
   not too far upfield and time allows, the four best are extended by one more step of
   look-ahead, including explicit follow-up shots (`_their_followups`, which reuses our
   own pass and shot evaluators on the mirrored world).
2. **Candidate spots**, with weights:

   | Spot | Weight |
   |---|---|
   | cover spot 6.5–11 m goal-side of the ball | 3.2 |
   | wing cut-off ahead of a wide carrier | 2.6 |
   | goal-side marks on the 3 most dangerous receivers | 2.2, 1.5, 1.1 |
   | a spot in each of those passing lanes | 0.8 × the mark's weight |
   | a deep screen 14 m behind the ball | 1.0 |
   | players' current positions, as filler | 0.5 |

3. **Which spot stops which option.** Each spot is raced against each option's ball path.
   A defender at the spot *kills* the option if he gets to the path before it completes
   (one cycle early for passes, two for dribbles), counting the time he needs to reach
   the spot. Our keeper and the chaser/blocker kill options too.
4. **Assignment.** Every permutation of free players to spots is tried (at most 4 field
   players, so this is cheap). An assignment costs:
   - the opponents' **soft-max** value over the options still open, including the carrier
     simply keeping the ball (temperature 4);
   - plus 0.08 × weight × distance for each player's run.

   The cover spot must be filled when the ball is within 45 m of our goal.

Plain man-marking (`_man_marks`) is kept as a fallback for when there are no options to
model.

## 8. Stamina

**Estimating everyone's stamina** (`stamina.py`). The world model only reports a
player's own stamina, so every agent replays the server's stamina rules for all 10
players. Each cycle it infers the dash power that explains a player's observed change of
velocity (inverting the server's update for decay, effort and dash direction, with a
special case for collisions), charges it, and adds the recovery. Stoppages, when cycles
pass unseen, are bridged with linear recovery tables fitted to our match logs by gap
length, stoppage cause and how far the player moved; our own players' table charges
backward, forward and sideways movement separately. If a player is seen dashing harder
than his estimate allows, the estimate is raised. A teammate who should be sprinting flat
out but is seen dashing at exactly 45 (the recovery rate) must be at the stamina floor,
and his estimate is set there.

**Using it.** A teammate's *sprint budget* is (estimate − floor) / 55 full-power dashes,
since each costs 100 and recovers 45. The race model uses it, so passes and dribbles avoid
teammates who could not make the sprint.

**Dash power** for positioning (`_dash_power`), first matching row:

| Situation | Power |
|---|---|
| serious danger | 100 |
| stamina within 600 of the floor | 45 (costs nothing net) |
| the last 100 cycles of a half | 100 |
| urgent | 100 / 80 / 55 by remaining stamina |
| otherwise | 100 / 75 / 45 by remaining stamina |

Power is then capped for short distances, for runs far from the ball, and so that a
defender arrives just in time rather than early. Chasing, blocking and tackling are not
throttled.

## 9. CPU time

The harness gives each team a CPU budget. `_allow_ms` spreads what is left over the
remaining cycles. An on-ball decision may use 10× the average share, capped at 400 ms
and never eating into the harness's per-decision limit. Off-ball decisions are cheap and
have no deadline of their own, except that the defensive follow-up look-ahead only runs
with more than 1.2 s of budget left. The search is anytime, so running out of time costs
depth, not correctness.

## 10. Configuration and disabled alternatives

[`agent/config.py`](../agent/config.py) documents every constant. Many boolean flags
(`FIX_*`, `IMP_*`, ...) switch on a fix that was compared against the version without
it. These alternatives are implemented but **switched off** in the final configuration:

| Flag | What it does |
|---|---|
| `EXP_BEAM = 2` | the chain beam search (`_beam_chain`, field-evaluation scores) only decides shots; everything else uses the expectimax beam |
| `STAMINA_MODE = 0` | hysteresis recovery mode (walk below 50% stamina until back above 70%) is off |
| `STAM_SIGNAL = False` | stamina signalling through dash power is off |
| `MAN_MARK`, `STOPPER`, `LOW_BLOCK`, `MIDBLOCK` | alternative defensive schemes |
| `KEEPER_RACE` | pure-race keeper model (the fitted logistic is used instead) |
| `STAMINA_SAVE` | extra power caps per run type |

The constants marked *unused* in `config.py` (`DRAIN_*`, `RUN_*`, `OFFSIDE_*` and a few
others) are not read anywhere: they are left over from features that were removed.

## 11. Known issues

- `SearchMixin._chain_value` refers to an undefined name `C` in its dribble branch when
  `light=False`. The current settings never reach that line (`BEAM` is on, and the
  defence always calls it with `light=True`), but turning `BEAM` off would raise a
  `NameError`, which `on_ball` would catch and answer with a clearance.
- `ENGAGE_CONE` is defined both at module level and in `AgentConfig` (the class attribute
  is the one used).
- `Agent2` is an empty stub kept from the starter template.

## 12. Code map

| Module | Contents |
|---|---|
| `config.py` | constants and feature flags, `AgentConfig` |
| `physics.py` | `_Phys` (server tables, value maps), `_kick_traj`, `_race`, `_go_point` |
| `stamina.py` | `StaminaPredictor` |
| `state.py` | `_State` (search node, `child()`), `_KickResult` |
| `evaluators.py` | `EvaluatorMixin`: pass targets, `_eval_kicks`, `_eval_shots`, stamina bookkeeping |
| `search.py` | `SearchMixin`: shot decision, chain beam, expectimax beam |
| `on_ball.py` | `OnBallMixin`: `on_ball`, shared `_context`, dribbles, hold/clear, execution |
| `defense.py` | `DefenseMixin`: opponents' options, `_defense_plan`, `_man_marks` |
| `off_ball.py` | `OffBallMixin`: `off_ball` rules, blocker, support positions, dash power |
| `player.py` | `Agent1`, the mixins combined |

`agent.py` at the repository root is these modules concatenated by
[`tools/bundle.py`](../tools/bundle.py), because the harness loads exactly one file.
