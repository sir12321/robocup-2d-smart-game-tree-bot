# How Agent1 works

This is the long version of the design. The [README](../README.md) has the short one.
All numbers are the values in [`agent.py`](../agent.py), which is a single file because
the course harness loads exactly one. Method and constant names below refer to that file.

## Contents

1. [Setting](#1-setting)
2. [The race model](#2-the-race-model)
3. [Valuing positions](#3-valuing-positions)
4. [Evaluating actions](#4-evaluating-actions)
5. [On the ball: the game-tree search](#5-on-the-ball-the-game-tree-search)
6. [Off the ball](#6-off-the-ball)
7. [Defensive marking](#7-defensive-marking)
8. [Stamina](#8-stamina)
9. [CPU time](#9-cpu-time)
10. [Switched-off and leftover code](#10-switched-off-and-leftover-code)
11. [Code map](#11-code-map)

---

## 1. Setting

The game is RoboCup 2D simulated soccer, 5-a-side, two halves of 750 cycles (10 cycles
per second). The course framework gives every player an exact world model (full state,
no noise) and calls one of two methods per player per cycle, only in normal play:

- `on_ball(agent)` when the player can kick the ball;
- `off_ball(agent)` otherwise.

Set pieces, kick-offs, penalties, the goalkeeper, and neck and view control are handled
by the framework and are the same for every team. Our goalkeeper simply clears if
`on_ball` is ever called for him.

There is no communication. The players coordinate because every one of them runs the
same deterministic computation on the same exact world model, so they agree on who
chases, who blocks and where everyone else should go.

Both entry points are wrapped in `try/except`: a bug falls back to a clearance or to
walking home instead of crashing the player.

## 2. The race model

Almost every decision reduces to one question: **for this ball trajectory, at which cycle
can each player first control the ball?** `_race` answers it for many trajectories and
all players at once, as one numpy computation.

**Ball.** A kicked ball's speed decays by a factor of 0.94 per cycle, so after `k` cycles
it has travelled `s0 · G[k]` with `G[k] = (1 − 0.94^k) / 0.06`. A whole trajectory is one
vectorised line (`_kick_traj`), predicted up to 40 cycles ahead.

**Players.** `_Phys` precomputes `DASH[n]`, the distance a player covers in `n` full
dashes from rest, and variants for dashing sideways or backwards (`DASH_DIR`) and for a
tired player who can only sprint for `j` more cycles before dropping to the stamina-floor
pace (`DASH_TIRED`).

A player reaches the ball at cycle `t` if

    distance(ball at t, player) <= control radius + DASH[t − reaction − turn]

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
goal: about 5% deep in our half, rising along a sigmoid towards their box (centred at
x = 22), and up to 40% lower out wide. From it:

| Function | Meaning |
|---|---|
| `v_keep(x, y, space)` | we have the ball at `(x, y)`: threat, plus a little for space around it |
| `v_lose(x, y)` | they win it at `(x, y)`: minus their threat (`threat` mirrored) |
| `v_out(x, y)` | ball goes out at `(x, y)`: a full loss in their half, a smaller one in ours |

The search ([§5](#5-on-the-ball-the-game-tree-search)) scores positions with
`_field_eval` instead:

    _field_eval(x, y) = x + 100 · (threat(x, y) − 0.05) + 15 · slot(x, y)

that is, metres of progress plus the scoring threat, plus a bonus for the central slot in
front of their goal (x > 31, |y| < 12).

## 4. Evaluating actions

### Passes (`_eval_kicks1`)

**Candidates** (`_pass_targets`): each receiver's feet and points on rings of 3, 6 and
9 m around him (8, 8 and 12 points), kicks into space in a fan in front of the ball
(9 directions × 8–25 m), and through balls 8 and 14 m ahead of receivers beyond
x = −15. Each target is tried at arrival speeds of 1.0, 1.4 and 1.8 m/cycle. Receivers
exclude anyone offside. If one kick cannot reach the needed speed, the pass takes two
kicks (one cycle later), and under pressure two-kick passes lose success probability
(`_multi_keep`), because an opponent may steal the ball between the kicks.

**Success probability** from the margin `m` (clipped to ±6 cycles):

    p = min( sig((m − 0.8) / 0.6),  sig((m + 1.9) / 2.8) )

then multiplied by 0.95–1.0 depending on how quickly an opponent can close down the
receiver after he controls the ball.

**Value:**

    value = p · 0.985^t · v_keep(reception point) + (1 − p) · v_lose(where they get it)

A pass that our own keeper would collect in his box is an illegal back-pass and is
valued at −0.85.

### Shots (`_eval_shots`)

13 aim points across the goal mouth, each scored as

    P(goal) = P(on target) × P(no field player intercepts) × P(keeper arrives too late)

- **On target:** the kick's error window grows with distance (0.03 · d + 0.3 m).
- **Intercepts:** from the race, `sig((t_them − t_goal + 1) / 0.45)`.
- **Keeper:** the race again, with his catch radius: `sig((t_keeper − t_goal + 1) / 0.5)`.

### Dribbles and self-passes (`_dribs`)

A dribble is a kick followed by a run, simulated with the server's own player physics
(turning with inertia, dash acceleration, top speed, the sprint budget):

- **Short dribbles:** a touch in one of 16 directions around the body, collected again
  after up to two turns and 2–4 dashes.
- **Self-passes:** a longer kick into space along the current heading (or one turn away),
  run onto after 5–20 dashes. Longer further upfield, never more than the player's
  sprint budget.

Each candidate must be physically kickable with this cycle's kick power and ball
velocity. Opponents race the ball along its path. A dribble is kept only if every
opponent who isn't left behind (more than 4 m behind the carrier) gets there at least one
cycle after our player collects the ball, with an extra safety allowance of up to two
cycles. Its success is 0.88 + 0.03 × that margin, capped at 0.97. Below x = 16 a dribble
may not head backwards (more than 100° from straight ahead).

### Hold and clear

Holding the ball is allowed only with nobody within 2.2 m (or close to their goal line or
touchline in their third, where losing it costs less); its value falls as the nearest
opponent gets closer. Clearing (30 m upfield and wide) is an option only behind
x = −30.

## 5. On the ball: the game-tree search

`_on_ball`, each cycle:

1. **Committed kick.** If last cycle started a two-kick pass or shot, continue it, unless
   the player is now pressed and the direction is blocked, or the pass lane has dropped
   below p = 0.5.
2. **Shot.** Within 30 m of goal, evaluate the best one-kick shot (or a two-kick
   full-speed shot when the one-kick shot is weak and there is time). Shoot if `P(goal)`
   clears the threshold:

   | Situation | Threshold |
   |---|---|
   | default, and inside the box | 0.70 |
   | x > 32 with an opponent within 2.2 m | 0.30 |

   Before shooting, try a short dribble towards goal that would make the shot clearly
   better (`_carry_in`).
3. **Game-tree search** over chains of passes and dribbles (`_on_ball_chain`, below).
4. **Fallback,** only if the search found nothing: shots, one-kick passes, dribbles,
   hold or clear, valued as in [§4](#4-evaluating-actions) with a 2–3 level look-ahead
   (`_deep_value`); then a channel kick down the wing or a bounce pass; then a clearance.
5. One exception when executing: pressed in our half, with two or more teammates below
   3000 stamina and a risky chosen action (success below 0.8), kick the ball out for a
   throw-in instead (`_touch_out`).

The score affects risk: after cycle 1150 the agent plays safer when leading and takes
more risks and shoots more when behind (`_risk`).

### The tree

**Nodes** are game states (`_State`): who has the ball, where, and where everyone stands.
**Our moves** are the passes and dribbles from [§4](#4-evaluating-actions). **The
opponents' reply** is modelled pessimistically: in the child state (`_State.child`)
every opponent has run towards the new ball position, covering 80% of the distance he
could sprint in that time (the keeper 50%), and gets a larger control radius the further
he was extrapolated. Our players coast on their velocity and push up, staying onside.

**Scores.** A pass is only expanded if it is safe enough: success at least 0.80, or 0.65
in attack (ball beyond x = 15 or reception beyond x = 22). Each step is scored with the field evaluation at the reception
point, minus costs for:

- risk: 30 per unit of failure probability in attack, 18 elsewhere;
- time: 0.05 per cycle until reception;
- going backwards: 0.3 per metre, plus 1 more per metre (beyond 4 m) when not pressed;
- each extra pass in the chain: 0.5;
- for dribbles, circling away from goal instead of going at it.

Shots inside the tree are worth 82 + 40 × P(goal) when P(goal) ≥ 0.7.

**Beam search** (`_beam_chain`, `_beam_run`). The root's safe passes and best dribbles
form the first layer. Each layer expands every node and keeps the best 10 children, at
most 3 per (parent, receiver) pair so the beam does not fill with near-identical passes,
for up to 3 layers, with at most 2 dribbles along one line. Every descendant's score,
minus the costs accumulated on its path, raises the value of the root action it started
from. So **each first pass is worth the best chain that can follow it**, and the player
plays the first move of the best chain. That is tiki-taka as a search problem: keep the
ball moving through safe passes until a chain opens up a shot.

Holding the ball is a root option too, when no opponent can reach it within 4 cycles.
The search is *anytime*: it stops at a deadline ([§9](#9-cpu-time)) and uses the best
values found so far.

### Executing

Two-kick passes and shots are *committed*: the plan is stored and continued next cycle.
Dribbles are executed as a planned kick to the computed point.

## 6. Off the ball

`_off_ball`: each cycle a player takes the **first** rule that applies.

1. **Zone tackle.** An opponent has the ball (not right on a goal line). Tackle if the
   fair tackle succeeds with probability ≥ 0.8 (0.6 in our own box), or with a deliberate
   foul if that succeeds with ≥ 0.8, the player has no card, it is outside our box and
   the ball is at x ≤ 20. The direction is chosen to keep the ball in play and, within
   30 m of our goal, never towards our own goal (a back pass to our keeper).
2. **Chase.** The chaser (first to the ball in the race, with a half-cycle bonus for last
   cycle's chaser so the role doesn't flip) intercepts if we are attacking or he is at
   most one cycle behind. For 12 cycles after we lose the ball he also contests it from up
   to three cycles behind (`_contest`).
3. **Block.** When defending, `_block_plan` predicts the opponent carrier's dribble
   towards our goal at 0.73 m/cycle, bent halfway towards his current running direction.
   The teammate who can reach the earliest point of that path in time runs there. Within
   12 m of the ball (15 m if the carrier is standing still) he engages: running straight
   through the ball if it is within 3.5 m and 25° of his heading, otherwise intercepting.
4. **Position.**
   - *Attacking* (`_support_point`): when the carrier is pressed (an opponent within
     4 m), the two nearest teammates come short as outlets. The two most advanced
     players make runs: into the box once the ball is beyond x = 20 (the most advanced
     one already from halfway), and diagonally ahead of the ball before that. Everyone else picks the best point around his ball-adjusted home
     position, scored by how open it is for a pass (the race again), the value of
     receiving there, distance and crowding.
   - *Defending:* the marking spot from [§7](#7-defensive-marking).

   Dash power is chosen as in [§8](#8-stamina).

## 7. Defensive marking

`_defense_plan` gives every free player (everyone except the chaser or blocker) a spot.

**Man-marking** (`_man_marks`). Every opponent field player except the carrier is
marked from 1.5 m, on the ball side of him (75% towards the ball, 25% towards our goal),
using his position 2 cycles ahead. An opponent more than 8 m upfield of the ball is not
marked; that spot becomes a screen 14 m behind the ball instead. Spare players stand on
the line from the ball to our goal.

**Assignment.** Every permutation of free players to spots is tried (at most 4 field
players, so this is cheap), and the one with the smallest

    longest run + 0.3 × total run

wins: a min-max assignment, so nobody is left with a long run to his man.

**Set-piece covers** (`_sp_covers`). For 30 cycles after a stoppage, if the ball is within
35 m of our goal, opponents within 25 m of our goal are covered instead. They are ranked
by how much of our goal they can see, and each is covered from the ball side.

**Serious danger** (`_serious`). When the carrier is within 30 m of our goal with nobody
goal-side, or within 15 m with at most one of us covering, everyone runs at full power.

## 8. Stamina

**Estimating everyone's stamina** (`StaminaPredictor`). The world model only reports a
player's own stamina, so every agent replays the server's stamina rules for all 10
players. Each cycle it infers the dash power that explains a player's observed change of
velocity (inverting the server's update for decay, effort and dash direction, with a
special case for collisions), charges it, and adds the recovery. Stoppages, when cycles
pass unseen, are bridged with linear recovery tables fitted to our match logs by gap
length, stoppage cause and how far the player moved; our own players' table charges
backward, forward and sideways movement separately. If a player is seen dashing harder
than his estimate allows, the estimate is raised.

**Sprint budgets.** A teammate's budget is (estimate − 600 safety margin − floor) / 55
full-power dashes, since each costs 100 and recovers 45. The race model and the dribble
simulation use it, so passes and dribbles avoid teammates who could not make the sprint.

**Dash power** for positioning (`_dash_power`), first matching row:

| Situation | Power |
|---|---|
| serious danger | 100 |
| stamina within 600 of the floor | 45 (costs nothing net) |
| the last 100 cycles of a half | 100 |
| urgent | 100 / 75 / 50 by remaining stamina |
| otherwise | 100 / 70 / 45 by remaining stamina |

Power is then capped for short distances, for runs more than 30 m from the ball, and so
that a defender arrives just in time rather than early.

**Recovery mode.** Below 50% stamina (4000) a player walks to his positions at the power
that regains 25 stamina per cycle, until he is back above 70% (5600). It never applies
to chasing, blocking or tackling, in serious danger, with the ball within 25 m of our
goal, to the deepest defender with the ball near him, or in the last 100 cycles of a
half.

**Reading teammates' recovery** (`_rec_signal`). Every teammate runs the same rule, so a
teammate seen dashing at exactly the recovery power must be below 5600, and his estimate
is capped there. A teammate who should be sprinting flat out but is seen dashing at
exactly 45 must be at the stamina floor. Both only ever lower an estimate.

## 9. CPU time

The harness gives each team a CPU budget. `_allow_ms` spreads what is left over the
remaining cycles: an on-ball decision may use 10× the average share, capped at 400 ms
and never eating into the harness's per-decision limit, and the search is capped at
300 ms. The search is anytime, so running out of time costs depth, not correctness.
Off-ball decisions are cheap and have no deadline of their own.

## 10. Switched-off and leftover code

`agent.py` is the exact file that played the [400 matches in the results](../README.md#results),
so it was not tidied up. Some of it never runs:

- **Switched off:** counter-attack mode (`COUNTER = False`) and the "guard" logic, which
  is never assigned (the context always sets `guard` to `None`).
- **Never read:** 57 class constants, from features that were tried and removed: kick
  scenarios (`KICK_SCEN`, `KS_*`, `KICK2_*`), defensive roles and shape (`ROLE_*`,
  `DEF_ROLES`, `SHAPE_*`, `REST_*`, `FRONT_*`, `JOCKEY_*`), a dedicated outlet
  (`OUTLET*`), and others.
- `_eval_kicks` is a thin wrapper around `_eval_kicks1`, kept from an experiment.

## 11. Code map

| Part of `agent.py` | Contents |
|---|---|
| top | module constants and the stamina recovery tables (`GAP_TABLE_*`, `US_GAP_TABLE`) |
| `StaminaPredictor` | stamina estimates for all players |
| `_Phys`, `_kick_traj`, `_race` | server physics tables, value maps, the race model |
| `_State`, `_KickResult` | search nodes and the opponents' reply (`child`), kick results |
| `Agent1`, constants | all tuning constants |
| `Agent1`, `_context` … `_race_budgets` | shared per-cycle context, risk, pass targets, shot evaluation, stamina bookkeeping |
| `Agent1.on_ball` … `_by_their_line` | on-ball decision: game-tree search, dribble simulation, execution |
| `Agent1.off_ball` … `_counter_state` | off-ball rules: tackles, chasing, blocking, support positions, dash power, recovery mode |
| `Agent1._defend_point` … `_lane_spot` | defensive marking and set-piece covers |
