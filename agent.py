"""
Agent1: a search-based 5-a-side RoboCup 2D player (COL333 Assignment 3).

GENERATED FILE - do not edit by hand. The readable source is the agent/
package; rebuild this file with `python3 tools/bundle.py`.
"""

import math
import time
from itertools import permutations

import numpy as np
from pyrusgeom.angle_deg import AngleDeg
from pyrusgeom.geom_2d import Vector2D
from src.student.interface import Tools, Action, note, StarterAgent


# ==========================================================================
# SECTION: CONFIG & HYPERPARAMETERS (agent/config.py)
# ==========================================================================


# --- Time and prediction horizon ---------------------------------------------
HORIZON = 40          # ball trajectories and races are predicted this many cycles ahead
MATCH_CYCLES = 1500   # cycles in a full match (two halves); used for CPU-time budgeting
HALF_CYCLES = 750     # cycles in one half
LATE_CYCLES = 100     # in the last 100 cycles of a half nobody saves stamina

# --- Stamina management -------------------------------------------------------
RESERVE = 600.0       # safety margin kept above the server's stamina floor
FREE_POWER = 45.0     # dash power that costs no net stamina (= stamina_inc_max)
PACE_SLACK = 3.0      # cycles of slack when pacing a run to arrive "just in time"
SPRINT_NET = 55.0     # net stamina per full-power dash (100 spent - 45 recovered);
                      # sprint budget = (stamina - floor) / SPRINT_NET cycles
SIG_BASE = 5.0        # stamina signal (STAM_SIGNAL, off): a resting player dashes at
SIG_LO = 2000.0       # power SIG_BASE + (stamina - SIG_LO) / SIG_STEP, so teammates
SIG_STEP = 120.0      # can read his stamina from his observed acceleration

# --- Search ------------------------------------------------------------------
NEVER = 999           # "never reaches the ball" sentinel in race results
GAMMA = 0.985         # per-cycle discount on the value of a future reception
OPP_MARGIN = 0.3      # opponents get a control radius 0.3 m larger than ours (pessimistic)
INERTIA = 5.0         # server inertia_moment: max turn per cycle = 180 / (1 + INERTIA * speed)

# How players move inside imagined child states of the search (agent/state.py)
CHILD_DRIFT = True       # players coast on their current (decaying) velocity, pursuers
                         # cover DASH[t - 1] metres instead of t metres
CHILD_LEAD_FIX = True    # don't double-count the 0.6-cycle velocity lead of root opponents
CHILD_OPP_VEL = True     # child-state opponents keep a velocity towards their target
CHILD_OPP_SPEED = 0.45   # ...of this speed
OPP_PURSUIT = 0.85       # opponents cover this fraction of their full-dash reach
OPP_UNC = 0.22           # opponent control radius grows by 0.22 m per metre we
OPP_UNC_MAX = 2.0        # extrapolated him (uncertainty), capped at 2 m
GK_REACH = 0.55          # the opposing keeper moves 0.55x as far as a field player
GIVE_GO = True           # after a pass the passer runs forward (give-and-go), both in
                         # the search model and in real play
GO_PURSUIT = 0.85        # fraction of his reach the passer covers on that run (model)
GO_X0 = -5.0             # give-and-go only when the ball is beyond this x
ENGAGE_CONE = 30.0       # same as AgentConfig.ENGAGE_CONE (the class attribute is used)


def _sig(z):
    """Numerically safe logistic function."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _norm_deg(a):
    """Wrap an angle in degrees to [-180, 180)."""
    return (a + 180.0) % 360.0 - 180.0


# --- Stamina recovery across stoppages (agent/stamina.py) ---------------------
# While play is stopped the server keeps running but we see no cycles, so a
# player's stamina after the gap is predicted from tables fitted to our match
# logs. Key: (gap bucket, goal scored?, ball went out?), where the gap bucket
# is 0: <4, 1: <15, 2: <40, 3: <70, 4: >=70 cycles. Coefficients (a, b, c, d[, e]):
#   recovered cycles n = max(a * gap + b * goal + c, 0)
#   stamina += min(stamina_inc_max * recovery * n, capacity) - d * displacement
#   (v3 only: d += e * stamina / stamina_max).
# GAP_GLOBAL* is the fallback for keys missing from the table. V2/V3/unversioned
# are successive fits: field players use V3, goalkeepers V2.
GAP_GLOBAL_V2 = (0.573, 50.614, 1.468, 8.332)
GAP_TABLE_V2 = {
    (0, 0, 0): (-1.883, 641.233, 4.987, -2.993),
    (0, 0, 1): (-0.1, 315.079, 0.937, -3.313),
    (1, 0, 0): (1.055, 54.311, 30.316, 99.826),
    (2, 0, 0): (0.972, -356271.016, 30.507, 98.71),
    (2, 0, 1): (0.979, -377.939, 1.515, 104.925),
    (3, 0, 0): (-9.473, 14986.795, 502.163, 100.597),
    (3, 0, 1): (1.145, -297.649, -5.615, 102.916),
    (3, 1, 1): (-0.642, 92.613, 0.009, -13.025),
}

GAP_GLOBAL_V3 = (0.583, 56.152, 3.223, 18.063, 0.002)
GAP_TABLE_V3 = {
    (0, 0, 0): (-25.108, 4259.872, 53.594, 90.008, -0.051),
    (0, 0, 1): (0.399, 7.041, 0.019, 1.398, -0.03),
    (1, 0, 0): (0.936, 759.454, 30.587, 97.876, 0.001),
    (2, 0, 0): (0.954, 48.029, 30.343, 98.811, 0.002),
    (2, 0, 1): (0.953, 46.351, 2.012, 104.896, 0.001),
    (3, 0, 0): (-9.434, -11899.287, 498.607, 94.994, 1.965),
    (3, 0, 1): (1.125, -123.614, -4.501, 104.365, 0.016),
    (3, 1, 1): (1.0, 0.0, 18.442, 17.947, -59.518),
}

GAP_GLOBAL = (0.581, 50.354, 1.848, 10.463)
GAP_TABLE = {
    (0, 0, 0): (-1.096, -25003.615, 3.414, -2.991),
    (1, 0, 0): (1.055, 54.311, 30.316, 99.826),
    (2, 0, 0): (-0.688, -667.566, 54.77, 97.291),
    (2, 0, 1): (0.974, -470.989, 1.747, 105.696),
    (3, 0, 0): (0.959, 36.924, 2.892, 100.037),
    (3, 0, 1): (1.188, 1977.387, -7.266, 105.734),
    (3, 1, 0): (0.744, 19.033, 9.341, -13.133),
    (3, 1, 1): (-8.536, 456.691, 0.079, -13.658),
}

# Same idea for our own field players, with direction-aware displacement costs.
# Coefficients (a, c, kb, kf, kl, e): n = max(a * gap + c, 0);
#   stamina += recovery(n) - (kb * back + kf * forward + kl * lateral metres)
#              - e * displacement * stamina / stamina_max
# 'taker' is used for the player nearest the ball (the set-piece taker).
US_GAP_GLOBAL = None   # no global fallback: unknown keys use the generic table
US_GAP_TABLE = {
    (0, 0, 0): (0.489, -0.000, 9.238, 24.130, 0.001, 0.007),
    (0, 0, 1): (0.612, 0.001, 54.153, 24.954, 0.000, 0.011),
    (3, 0, 0): (1.588, 0.003, 39.307, 77.636, 83.611, -0.004),
    (3, 1, 1): (1.059, 18.894, -0.000, 12.061, 0.321, -23.924),
    'taker': (0.174, 32.342, 36.515, 39.887, 41.534, 63.557),
}

GAP_SD = 309.0        # std dev (stamina units) added to an estimate per stoppage
DRIFT_SD2 = 0.0       # per-cycle variance growth (disabled)
SURE_MARGIN = 300.0   # a stoppage that predicts >= max + 300 surely refilled stamina


class AgentConfig:
    """Per-agent tuning knobs, mixed into Agent1 (read as self.NAME)."""

    # Pass model. Success p = min of two logistics of the race margin
    # (opponent arrival - our arrival, cycles): sig((m - M0) / S) and the
    # log-fitted sig((m - FIT_M0) / FIT_S).
    PASS_M0 = 0.8
    PASS_S = 0.6
    FIT_M0 = -1.9
    FIT_S = 2.8
    RINGS = ((3.0, 8), (6.0, 8), (9.0, 12))   # pass targets: (radius m, points) around each receiver
    END_SPEEDS = (1.0, 1.4, 1.8)              # ball speed on arrival at the target
    CHILD_RINGS = ((3.0, 8), (6.0, 8))        # smaller candidate set at deeper search nodes
    CHILD_SPEEDS = (1.0, 1.8)
    MULTI_DELAY = 1                           # extra cycle for passes needing two kicks
    KICK_LATE = 2                             # a planned pass may arrive up to 2 cycles late
    DRIB_DISTS = (2.5, 4.5, 8.0)              # dribble lengths tried by the fallback evaluator
    P_PASS = 0.80        # min pass success to enter the chain search (midfield)
    P_DRIB = 0.88        # min dribble success (own half)
    P_ATT = 0.68         # min pass success in the attacking third
    P_DRIB_ATT = 0.76    # min dribble success in the opponent half
    LOSS_ATT = 32.0      # chain-score penalty per unit of failure probability in attack
    BACK_W = 1.0         # chain-score penalty per metre played backwards (when not pressed)
    SLOT_W = 16.0        # _field_eval bonus for the central zone in front of goal
    SLOT_X = 38.0        # dribbles may aim at (SLOT_X, +-7), the edges of that zone
    DRIB_K = 6           # root dribbles expanded by the chain search
    DRIB_DEPTH = 1       # (non-beam chain) depth to which dribbles are expanded
    CHANNEL_V = 1.7      # min one-kick speed for a channel kick (last resort)
    CONE = 35.0          # a direction is blocked if an opponent within 3 m is within 35 deg of it
    BACK_X = -10.0       # backward-pass penalty and long dribbles only beyond this x
    OPEN_D = 6.0         # "open ahead": nobody within 6 m in front -> 15 m dribbles allowed
    DIAG_X0 = -15.0      # forward / diagonal support runs only when the ball is beyond this x
    ENGAGE_D = 4.2       # the blocker runs straight through the ball when this close...
    ENGAGE_CONE = 30.0   # ...and the ball is within 30 deg of his body direction
    HOLD_ENGAGE = 16.0   # engage distance when the carrier is almost standing still
    CARRY_FIT = (0.51, 0.31, -1.36, 0.23)  # dribble success recalibration (fitted):
                         # sig(a + b * nearest-opponent dist + c * #opps within 4 m + d * logit(p))
    CARRY_FIT_THEIRS = (1.71, 0.09, -0.87, 0.17)  # unused (opponent values are inlined)
    CARRY_ALL = True         # unused
    FIX_OURS_ONLY = True     # unused
    PASS_TACKLE_R = 0.0      # unused
    PASS_OUT_MARGIN = True   # pass margin is against min(opponent arrival, ball going out)
    PASS_FIT = (1.11, 0.16, 0.10)  # pass success recalibration (fitted):
                         # sig(a + b * logit(p) + c * margin vs. a 2 m tackle reach)
    HOLD_V = 0.15        # carrier slower than this counts as "holding" (see HOLD_ENGAGE)
    CARRY_IN_P = 0.82    # if the shot chance is below this and nobody is within
    CARRY_IN_D = 3.0     # 3 m, try a short dribble that improves the shot first
    V_BACKPASS = -0.85   # value of a pass our keeper would pick up in his box (illegal)
    SHADE = 0.6          # blocker's target is shifted 0.6 m towards the centre

    # Shot Selection & Keeper Race
    KEEPER_RACE = False  # alternative keeper model (pure race); off, the fitted GK_* logistic is used
    KR_P1 = 0.67         # its save-miss probability when the keeper is 1 cycle late
    KR_P2 = 0.33         # ...and when he is 2+ cycles late
    SHOT_P = 0.68        # min goal probability to shoot / to value a shot inside the search
    BOX_SHOT_P = 0.64    # lower threshold inside the box
    SHOT_SURE = 0.86     # above this, shoot without comparing against passes
    SHOT_COMPARE = True  # below SHOT_SURE (and not pressed), weigh the shot against alternatives
    PER_AIM_V1 = True    # one-kick shot speed computed per aim point
    SHOT_RAY_FIX = True  # aim at the goal-line point of the evaluated shot ray
    FIX_SHOT_BOTH = True     # also try a two-kick full-speed shot when the one-kick shot is weak
    SHOT_BOTH_STRICT = True  # ...but only when not pressed
    FIX_CONT_SHOT = True     # when continuing a two-kick shot, queue a one-kick shot as fallback
    FIX_SHOT = True          # two-kick shots are committed multi-cycle kicks at max speed
    KIN_SHOT = True          # keeper model also checks whether he can physically reach the ray
    SHOT_V0_US = 82.0    # shot value in chain-score units: V0 + V1 * P(goal)
    SHOT_V0 = 60.0       # (SHOT_V0_US at our root / beam, SHOT_V0 deeper and for opponents)
    SHOT_V1 = 40.0
    SHOT_P_PRESSED = 0.35  # threshold near goal (x > 32) with an opponent within HOLD_BAN_D
    SHOT_KICK_LATE = 2
    FAR_SHOT_D = 24.0    # from more than 24 m out (and not pressed)...
    FAR_SHOT_P = 0.72    # ...require at least this goal probability

    # Expectimax Search
    CHAIN_PLAN = ((1, 24, True), (1, 6, False), (2, 4, False))  # (depth, k, light) per level; only if BEAM is off
    CHILD_K = 4          # children expanded per node in the depth-limited searches
    DEEP_LIGHT = False   # use a reduced candidate set below the first layer
    BEAM = True          # chain search as a beam search...
    BEAM_W = 12          # ...of width 12
    BEAM_D = 3           # ...and depth 3
    BEAM_DIV = 3         # at most 3 beam nodes per (parent, receiver): keeps the beam diverse
    EXP_BEAM = 2         # 0: old depth-limited fallback; 1: chain search, then expectimax beam;
                         # 2: chain only decides shots, everything else by the expectimax beam
    EXP_W = 24           # expectimax beam width
    EXP_D = 4            # expectimax beam depth
    EXP_DEEP = 0.92      # value multiplier per extra layer (distrust of deep predictions)
    EXP_SHOT = 0.96      # multiplier on a shot's value at a search node
    CHILD_ADV = 0.38     # in child states teammates advance 0.38x their reach (x1.25 on counters)
    BEAM_CARRIES = 2     # dribbles allowed along one beam path
    BEAM_ROOT_DRIB = True  # root dribbles always enter the beam
    HOP_COST = 0.45      # chain-score cost per extra pass

    # Defense, Tackling & Fouls
    MAN_MARK = False     # True: plain man-marking instead of the threat-based assignment
    REST_DEFENSE = True  # in attack, the deepest non-chaser stays back
    FORCE_COVER = True   # the assignment must fill the cover spot when the ball is within 45 m of our goal
    STOPPER = False      # dedicated marker on the deepest opponent...
    STOP_GAP = 2.5       # ...standing 2.5 m goal-side of him
    PRESS_ALL = True     # the blocker uses the high engage distance on the whole pitch
    THROUGH = True       # also consider through balls 7 and 13 m ahead of receivers
    LOW_BLOCK = False    # alternative compact low block
    STAMINA_MODE = 0     # 0: _dash_power; 1: hysteresis recovery mode (_dash_power_h); 2: mode 1 in attack only
    MARK_DEEP_GS = True  # man-marking (MAN_MARK / fallback only): goal-side mark on the deepest opponent
    MARK_DEEP_GAP = 2.2  # ...from 2.2 m
    MARK_BEHIND = 8.0    # opponents more than 8 m upfield of the ball are screened, not marked
    PRESS_ALL_FROM = -60.0  # PRESS_ALL applies beyond this x (i.e. everywhere)
    MARK_GS_ALL = True   # every mark is taken goal-side
    STAMINA_SAVE = 0     # extra power caps per run type (0 = off)
    FINISH = True        # when the ball is beyond FINISH_X, the two most advanced players
    FINISH_X = 26.0      # attack the far post and the cut-back spot
    MIDBLOCK = False     # mid-block variant (off) and its parameters:
    MB_T = 30            # cycles of opponent possession before it kicks in
    MB_X = 0.0           # ball x beyond which it applies
    MB_LINE = 0.0        # line the block holds
    MB_POWER = 60.0      # dash power cap while holding it

    IMP_TACKLE = True    # tackle to block a ball that is heading into our goal
    SHOTBLOCK_P = 0.40   # ...when the tackle succeeds with at least this probability
    TACKLE_OPEN_P = 0.52 # min tackle probability in open play
    TACKLE_DEF_P = 0.40  # min tackle probability deep (x < -18) or wide (|y| > 18)
    FOULS = True         # allow deliberate foul tackles
    FOUL_MAX = 2         # at most 2 per player per match (and only without a card)
    FOUL_P = 0.78        # min foul success probability
    LASTDITCH = True     # foul when the ball is 16.5-32 m from our goal, outside the box,
    LD_MIN = 16.5        # and the situation is serious
    LD_MAX = 32.0
    MID_FOUL = True      # foul in midfield (x in [MID_X0, MID_X1]) when a fair tackle
    MID_X0 = -25.0       # would succeed with less than MID_TACKLE_MAX
    MID_X1 = 25.0
    MID_TACKLE_MAX = 0.52
    FOUL_COVER = True    # ...and only if a teammate is behind the ball

    # Touch-out, Offside & Breakaway (DRAIN)
    TOUCH_OUT = True     # kick the ball out for a throw-in when, in our half,
    TOUCH_ST = 3000.0    # at least TOUCH_N teammates are below 3000 stamina,
    TOUCH_N = 2          # an opponent is within TOUCH_PRESS m and the chosen
    TOUCH_PRESS = 2.0    # action would succeed with p < TOUCH_P
    TOUCH_P = 0.8
    OFFSIDE_AWARE = True     # unused
    OFFSIDE_MARGIN = 0.2     # unused
    OFFSIDE_KEEP = 3.0       # unused
    OFFSIDE_BACK = 1.5       # unused
    DRAIN = True             # unused (DRAIN_*: abandoned breakaway feature)
    DRAIN_X_MIN = -35.0      # unused
    DRAIN_RUN_V = 0.85       # unused
    DRAIN_LEAD = 1.0         # unused
    DRAIN_CATCH_R = 2.0      # unused
    DRAIN_PATH_M = 3         # unused
    DRAIN_RACE_T = 200       # unused
    DRAIN_ZONE = 25.0        # unused
    DRAIN_CLOSE = 6.0        # unused
    DRAIN_GRID = 2.0         # unused
    DRAIN_GK_D = (4.0, 5.5, 6.8)  # unused
    DRAIN_GK_ANG = 1.4       # unused
    DRAIN_GK_R = 7.0         # unused
    DRAIN_MAX_ANG = 50.0     # unused
    DRAIN_SHOT_DELAY = 2     # unused
    DRAIN_PLAN_MIN = 0.67    # unused
    RUN_KEEPER = True        # unused
    DRAIN_V1 = 2.55          # unused
    DRAIN_ST_RESERVE = 300.0 # unused
    DRAIN_AT = 1.5           # unused
    DRAIN_MAX_T = 200        # unused
    DRAIN_GAP = 40           # unused
    RUN_TIRE = True          # unused (RUN_*: abandoned support-run feature)
    RUN_SUP = True           # unused
    RUN_LANE = 9.0           # unused
    RUN_SUP_REACH = 25       # unused
    RUN_COMMIT_D = 4.0       # unused
    RUN_PASS_P = 0.6         # unused
    RUN_JOG = 45.0           # unused
    RUN_RACE = False         # unused
    RUN_FINISH = True        # unused
    RUN_DRIBBLE = False      # unused
    RUN_MIN_LEN = 8.0        # unused
    RUN_STALL = 15           # unused
    RUN_TOUCH_CAP = True     # unused
    RUN_TOUCH_K = 0.6        # unused

    # Stamina tracking
    STAM_TRACK = True    # run the StaminaPredictor for every player each cycle
    STAM_RACE = True     # teammates' sprint budgets limit them in the race model
    STAM_SIGNAL = False  # stamina signalling through dash power (see SIG_*)
    SIG_AT = 500         # ...only after this many cycles into a half
    STAM_FLOOR = True    # a flat-out teammate seen dashing at exactly 45 is at the stamina floor

    # Time budget, search breadth, assorted fixes
    MULTI_PRESS = True     # two-kick passes lose success probability under pressure
    CHAIN_MS = 280.0       # time cap for the chain search (ms)
    ONBALL_CAP_MS = 400.0  # time cap for one on-ball decision (ms)
    ONBALL_SHARE = 10.0    # an on-ball decision may use 10x the average per-cycle budget
    CHAIN_DRIB = (3.0, 6.0, 10.0)  # root dribble lengths in the chain search (+15 m when open)
    CHILD_DRIB = (3.0, 6.0)        # dribble lengths at deeper nodes...
    CHILD_DRIB_K = 2               # ...keeping the best 2
    THEIR_K = 4            # defence: best opponent actions extended by one follow-up
    THEIR_W = 0.92         # discount on the value of that follow-up
    THEIR_MIN_MS = 1200.0  # ...only with more than 1.2 s of CPU budget left
    THEIR_X = 15.0         # ...and only when their ball is at x < 15
    THEIR_SHOT2 = True     # model their follow-up shots explicitly
    GO_RUN = 11.0          # give-and-go run length (m)
    GO_CYCLES = 16         # ...and how long the passer keeps running (cycles)
    FIX_FALLBACK = True    # queue "hold" as a fallback before the final "clear"
    FIX_COMMIT = True      # abort a committed two-kick action if its direction is blocked
    FIX_GK_BACK = True     # ignore our keeper for balls that end up behind him
    IMP_CHASER = True      # chaser choice also weighs distance, run direction and last chaser
    IMP_KEEPER_RESET = True  # kick out when our keeper is off his line and the ball is deep
    FIX_CHAIN_RISK = True  # chain search charges each pass/dribble step for its risk...
    MID_LOSS = 18.0        # ...at MID_LOSS per unit of failure probability outside attack
    FIX_LANE = True        # re-check a committed pass's lane; abort if p < 0.55
    FIX_REACH = True       # unused
    FIX_MIRROR_VEL = True  # mirrored (opponents' view) state leads positions by velocity
    FIX_BLOCK = True       # blocker: prefer goal-side teammates, use BLOCK_ENGAGE_HIGH
    BLOCK_ENGAGE_HIGH = 14.0  # engage distance higher up the pitch
    REACH_REACT1 = False   # unused
    REACH2 = True          # unused (only mentioned in a comment)
    FIX_RISK_SYM = True    # symmetric step cost in the chain search
    FIX_SPOTS = True       # marking spots are clamped goal-side of the ball
    RUN_SPLIT_X = 18.0     # beyond this x both forwards run into the box
    ATTACK_MARGIN1 = True  # in their half we "attack" if we reach the ball 1 cycle first
    CHAIN_FALLBACK = True  # if the chain's best is much worse than holding, use the fallback evaluator
    FIX_DEF_PACE = True    # defenders pace runs against a fast assumed threat...
    DEF_PACE_V = 2.2       # ...of 2.2 m/cycle (so they hurry)
    CLEAR_X = -32.0        # clearing is an option only when the ball is behind this x
    PRESS_D = 1.8          # "pressed": an opponent within 1.8 m
    HOLD_BAN_D = 2.2       # no holding the ball with an opponent within 2.2 m
    OUTLET_D = 4.2         # carrier pressed within 4.2 m -> the two nearest come short
    GK_A, GK_P, GK_N = 2.88, 1.22, -1.31  # P(beat keeper) = sig(A + P * keeper's distance
                                          # from the shot line + N * cycles he has) (fitted)
    DRIBBLE_V = 0.82       # assumed opponent dribble speed when predicting their carrier
    COMPACT = 0.72         # home positions pulled towards the ball: x = bx + 0.72 * (home - bx)
    BLOCK_ENGAGE = 6.5     # engage distance deep in our half
    PRESS_X = -15.0        # ball beyond this x -> high press settings


# ==========================================================================
# SECTION: STAMINA PREDICTOR (agent/stamina.py)
# ==========================================================================


class StaminaPredictor:
    """Tracks [stamina, effort, recovery, capacity] for every player.

    Players are keyed ("us" | "them", uniform number). Call `update(agent)`
    once per cycle, then read `estimate(key)`, `uncertainty(key)` (std dev
    that grows across stoppages) or `perfect_dashes(key)` (full-power dashes
    left before the effort/recovery threshold).

    gap        which stoppage table to use ("v3", "v2" or the original).
    levels     "frozen": effort and recovery are never decreased; otherwise
               their server decay below the thresholds is modelled.
    floor      "bound": if a player is seen dashing harder than his estimate
               allows, raise the estimate to the lowest value consistent
               with that dash ("anchor" also resets the uncertainty).
    us_gap     use the direction-aware table for our own field players.
    """

    def __init__(self, sp, gap="v3", levels="frozen", margin=0.0, coast_fix=True, floor="bound",
                 us_gap=True, log_gaps=False):
        self.gap, self.levels, self.margin, self.coast_fix = gap, levels, margin, coast_fix
        self.floor = floor
        self.us_gap = us_gap
        self.gap_log = [] if log_gaps else None
        self.last_need = {}
        self.gap_global, self.gap_table = {"v2": (GAP_GLOBAL_V2, GAP_TABLE_V2),
                                           "v3": (GAP_GLOBAL_V3, GAP_TABLE_V3)}.get(gap, (GAP_GLOBAL, GAP_TABLE))
        self.gk_global, self.gk_table = (GAP_GLOBAL_V2, GAP_TABLE_V2) if gap in ("v2", "v3") else (GAP_GLOBAL, GAP_TABLE)
        g = lambda k, d: float(sp.get(k, d)) if isinstance(sp, dict) else float(getattr(sp, k, d) or d)
        self.st_max = g("stamina_max", 8000.0)
        self.inc_max = g("stamina_inc_max", 45.0)
        self.extra = g("extra_stamina", 50.0)
        self.capacity = g("stamina_capacity", 130600.0)
        self.dpr = g("dash_power_rate", 0.006)
        self.decay = g("player_decay", 0.4)
        self.side_rate = g("side_dash_rate", 0.4)
        self.back_rate = g("back_dash_rate", 0.7)
        self.max_power = g("max_dash_power", 100.0)
        self.eff_min = g("effort_min", 0.6)
        self.eff_dec = g("effort_dec", 0.005)
        self.eff_inc = g("effort_inc", 0.01)
        self.eff_dec_thr = g("effort_dec_thr", 0.3) * self.st_max
        self.eff_inc_thr = g("effort_inc_thr", 0.6) * self.st_max
        self.rec_min = g("recover_min", 0.5)
        self.rec_dec = g("recover_dec", 0.002)
        self.rec_dec_thr = g("recover_dec_thr", 0.3) * self.st_max
        half = g("half_time", 75.0)
        self.half_cycles = half * 10.0 if half < 1000 else half
        self.touch_ball = g("player_size", 0.3) + g("ball_size", 0.085) + 0.03
        self.touch_player = 2.0 * g("player_size", 0.3) + 0.03
        self.half_x = g("pitch_half_length", 52.5)
        self.half_y = g("pitch_half_width", 34.0)
        self.state = {}
        self.var = {}
        self.prev = {}
        self.last = self.score = self.ball = None

    def update(self, agent):
        wm = agent.wm
        obs = {}
        for key, p in ([(("us", wm.self.uniform_number), wm.self)]
                       + [(("us", p.uniform_number), p) for p in wm.teammates]
                       + [(("them", p.uniform_number), p) for p in wm.opponents]):
            if p.uniform_number <= 0:
                continue
            obs[key] = (p.position.x(), p.position.y(), p.velocity.x(), p.velocity.y(),
                        p.body_direction, bool(p.is_goalie))
        b = wm.ball
        ball = (b.position.x(), b.position.y(), b.velocity.x(), b.velocity.y())
        self.observe(wm.cycle, (wm.our_team_score, wm.their_team_score), ball, obs)

    def observe(self, cycle, score, ball, obs):
        """Advance every estimate to `cycle`.

        obs maps player key -> (x, y, vx, vy, body, is_goalie). After a gap
        (stoppage) the fitted tables are applied; otherwise the dash power
        each player must have used is inferred and charged. Everything
        resets at half time.
        """
        if cycle == self.last:
            return
        self.last_need = {}
        half = self.last is not None and self.last < self.half_cycles <= cycle
        if half:
            self.state = {k: self._fresh() for k in self.state}
            self.var = {k: 0.0 for k in self.var}
            self.prev = {}
        gap = 0 if self.last is None else cycle - self.last - 1
        goal = self.score is not None and score != self.score
        out = self.ball is not None and (abs(self.ball[0]) > self.half_x or abs(self.ball[1]) > self.half_y)
        if self.ball is not None and self.gap in ("v2", "v3"):
            bx, by = self.ball[0] + self.ball[2], self.ball[1] + self.ball[3]
            out = out or abs(bx) > self.half_x or abs(by) > self.half_y
        taker = None
        if gap > 0 and not half and ball is not None and obs:
            taker = min(obs, key=lambda k: math.hypot(obs[k][0] - ball[0], obs[k][1] - ball[1]))
        for k, o in obs.items():
            s = self.state.get(k)
            if s is None:
                s = self.state[k] = self._fresh()
            o0 = self.prev.get(k)
            if gap > 0 and not half:
                disp = math.hypot(o[0] - o0[0], o[1] - o0[1]) if o0 else 0.0
                if self.us_gap and k[0] == "us" and not o[5] and o0:
                    dx = o[0] - o0[0]
                    if self.gap_log is not None:
                        self.gap_log.append((cycle, k, s[0], s[3], gap, goal, out, max(-dx, 0.0), max(dx, 0.0),
                                             abs(o[1] - o0[1]), disp, k == taker))
                    raw = self._gap_us(s, gap, goal, out, max(-dx, 0.0), max(dx, 0.0), abs(o[1] - o0[1]), disp,
                                       k == taker)
                else:
                    raw = self._gap(s, gap, goal, disp, out, o[5])
                sure = raw >= self.st_max + SURE_MARGIN
                self.var[k] = 0.0 if sure else self.var.get(k, 0.0) + GAP_SD * GAP_SD
                o0 = None
            if o0 is not None:
                need = self._need(o0, o, s[1], self._contact(k, obs, ball))
                self.last_need[k] = need
                if self.floor and need > 0.0:
                    low = need + self.dash_threshold()
                    if s[0] < low:
                        s[0] = min(low, self.st_max)
                        if self.floor == "anchor":
                            self.var[k] = 0.0
                self._step(s, need)
                self.var[k] = self.var.get(k, 0.0) + DRIFT_SD2
            self.prev[k] = o
        self.last, self.score, self.ball = cycle, score, ball

    def estimate(self, key):
        s = self.state.get(key)
        return s[0] if s else self.st_max

    def effort(self, key):
        s = self.state.get(key)
        return s[1] if s else 1.0

    def uncertainty(self, key):
        return math.sqrt(self.var.get(key, 0.0))

    def dash_threshold(self):
        return max(self.eff_dec_thr, self.rec_dec_thr)

    def perfect_dashes(self, key, z=0.0, power=None):
        power = self.max_power if power is None else power
        s = self.state.get(key) or self._fresh()
        st = self.estimate(key) - z * self.uncertainty(key)
        room = st - power - self.dash_threshold()
        if room <= 0.0:
            return 0
        net = power - min(self.inc_max * s[2], s[3])
        return 999 if net <= 0.0 else int(math.ceil(room / net))

    def safe_power(self, key, z=0.0):
        st = self.estimate(key) - z * self.uncertainty(key)
        return max(0.0, min(self.max_power, st - self.dash_threshold() - 1e-3))

    def _fresh(self):
        return [self.st_max, 1.0, 1.0, self.capacity]

    def _dir_rate(self, rel):
        a = abs((rel + 180.0) % 360.0 - 180.0)
        if a > 90.0:
            return self.back_rate - (self.back_rate - self.side_rate) * (1.0 - (a - 90.0) / 90.0)
        return self.side_rate + (1.0 - self.side_rate) * (1.0 - a / 90.0)

    def _power(self, ax, ay, body, effort):
        acc = math.hypot(ax, ay)
        if acc < 0.004:
            return 0.0
        rel = math.degrees(math.atan2(ay, ax)) - body
        return acc / (self.dpr * max(effort, 1e-6) * self._dir_rate(rel))

    def _need(self, o0, o1, effort, contact):
        """Dash power that explains the velocity change from o0 to o1.

        Inverts the server update v1 = decay * (v0 + accel), where
        accel = power * dash_power_rate * effort * direction rate. When the
        player touched the ball or another player (collision), velocity is
        scaled by -0.1 instead, and that case is tried first.
        """
        if contact and self.coast_fix and math.hypot(o1[2] - o0[2] * self.decay, o1[3] - o0[3] * self.decay) < 0.004 * self.decay:
            return 0.0
        if contact:
            k = -0.1 * self.decay
            p = self._power(o1[2] / k - o0[2], o1[3] / k - o0[3], o0[4], effort)
            if p <= self.max_power * 1.02:
                return min(p, self.max_power)
        p = self._power(o1[2] / self.decay - o0[2], o1[3] / self.decay - o0[3], o0[4], effort)
        return min(p, self.max_power)

    def _contact(self, key, obs, ball):
        x, y = obs[key][0], obs[key][1]
        if ball is not None and math.hypot(ball[0] - x, ball[1] - y) < self.touch_ball:
            return True
        lim = self.touch_player
        return any(k != key and abs(o[0] - x) < lim and math.hypot(o[0] - x, o[1] - y) < lim
                   for k, o in obs.items())

    def _step(self, s, need):
        """One server stamina update: spend `need`, then recover."""
        st, eff, rec, cap = s
        st = max(st - min(need, st + self.extra), 0.0)
        m = self.margin if self.levels == "margin" else 0.0
        if self.levels != "frozen":
            if st <= self.rec_dec_thr - m and rec > self.rec_min:
                rec = max(rec - self.rec_dec, self.rec_min)
            if st <= self.eff_dec_thr - m and eff > self.eff_min:
                eff = max(eff - self.eff_dec, self.eff_min)
        if st >= self.eff_inc_thr and eff < 1.0:
            eff = min(eff + self.eff_inc, 1.0)
        inc = min(self.inc_max * rec, self.st_max - st, cap)
        s[:] = [st + inc, eff, rec, max(cap - inc, 0.0)]

    def _gap(self, s, gap, goal, disp, out, goalie=False):
        """Bridge a stoppage of `gap` unseen cycles with the fitted table.
        Returns the raw (unclipped) prediction."""
        gb = 0 if gap < 4 else 1 if gap < 15 else 2 if gap < 40 else 3 if gap < 70 else 4
        table, default = (self.gk_table, self.gk_global) if goalie else (self.gap_table, self.gap_global)
        th = table.get((gb, int(goal), int(out)), default)
        a, b, c, d = th[:4]
        if len(th) > 4:
            d += th[4] * s[0] / self.st_max
        n = max(a * gap + b * goal + c, 0.0)
        inc = min(self.inc_max * s[2] * n, s[3])
        raw = s[0] + inc - d * disp
        s[0] = max(min(self.st_max, raw), 0.0)
        s[3] -= max(inc, 0.0)
        if s[0] >= self.eff_inc_thr:
            s[1] = min(1.0, s[1] + self.eff_inc * n)
        return raw

    def _gap_us(self, s, gap, goal, out, back, fwd, lat, disp, taker):
        """Like `_gap` for our own field players, charging backward, forward
        and lateral movement during the stoppage separately."""
        gb = 0 if gap < 4 else 1 if gap < 15 else 2 if gap < 40 else 3 if gap < 70 else 4
        th = US_GAP_TABLE.get("taker") if taker and not goal else None
        if th is None:
            th = US_GAP_TABLE.get((gb, int(goal), int(out)), US_GAP_GLOBAL)
        if th is None:
            return self._gap(s, gap, goal, disp, out, False)
        a, c, kb, kf, kl, e = th
        n = max(a * gap + c, 0.0)
        inc = min(self.inc_max * s[2] * n, s[3])
        raw = s[0] + inc - (kb * back + kf * fwd + kl * lat) - e * disp * s[0] / self.st_max
        s[0] = max(min(self.st_max, raw), 0.0)
        s[3] -= max(inc, 0.0)
        if s[0] >= self.eff_inc_thr:
            s[1] = min(1.0, s[1] + self.eff_inc * n)
        return raw


# ==========================================================================
# SECTION: PHYSICS & TRAJECTORY ENGINE (agent/physics.py)
# ==========================================================================


class _Phys:
    """Server parameters plus precomputed motion tables.

    G[k]            distance travelled in k cycles by a ball kicked at speed 1
                    (geometric series of the ball decay), so a kicked ball is
                    at origin + direction * s0 * G[k].
    DASH[n]         distance a player starting from rest covers in n full dashes.
    DASH_DIR[a, n]  same, dashing at 15*a degrees off his body direction
                    (side/back dashes are slower).
    DASH_TIRED[j, n]  same, but only the first j dashes at full power and the
                    rest at the stamina-floor pace (used for tired teammates).

    It also holds the position value maps (`threat`, `v_keep`, ...) that turn
    "where does the ball end up, and who has it" into a number.
    """

    def __init__(self, sp):
        self.decay = float(sp.ball_decay) or 0.94
        self.bmax = float(sp.ball_speed_max) or 3.0
        self.hl = float(sp.pitch_half_length) or 52.5
        self.hw = float(sp.pitch_half_width) or 34.0
        self.gh = (float(sp.goal_width) or 14.02) / 2.0
        self.pen_x = abs(float(sp.their_penalty_area_line_x)) or 36.0
        self.pen_hw = float(sp.penalty_area_half_width) or 20.16
        self.kick_r = float(sp.kickable_area) or 1.085
        self.catch_r = float(sp.catchable_area) or 1.3
        self.stamina_max = float(sp.stamina_max) or 8000.0
        self.st_floor = (float(getattr(sp, "recover_dec_thr", 0.0)) or 0.3) * self.stamina_max + 46.0

        n = HORIZON + 4
        self.G = np.array([(1.0 - self.decay ** k) / (1.0 - self.decay)
                           for k in range(n + 1)])
        accel = (float(sp.max_dash_power) or 100.0) * (float(sp.dash_power_rate) or 0.006)
        vmax = float(sp.player_speed_max) or 1.05
        pdecay = float(sp.player_decay) or 0.4
        dash, v, d = [0.0], 0.0, 0.0
        for _ in range(n + 10):
            v = min(v + accel, vmax)
            d += v
            dash.append(d)
            v *= pdecay
        self.DASH = np.array(dash)
        self.pdecay = pdecay
        side = float(getattr(sp, "side_dash_rate", 0.0)) or 0.4
        back = float(getattr(sp, "back_dash_rate", 0.0)) or 0.7
        rows = []
        for k in range(13):
            a = 15.0 * k
            rate = (side + (1.0 - side) * (1.0 - a / 90.0) if a <= 90.0
                    else back - (back - side) * (1.0 - (a - 90.0) / 90.0))
            row, v, d = [0.0], 0.0, 0.0
            for _ in range(n + 10):
                v = min(v + accel * rate, vmax)
                d += v
                row.append(d)
                v *= pdecay
            rows.append(row)
        self.DASH_DIR = np.array(rows)
        self.J = n + 10
        tired_acc = (float(getattr(sp, "stamina_inc_max", 0.0)) or 45.0) * (float(sp.dash_power_rate) or 0.006)
        rows = []
        for j in range(self.J + 1):
            row, v, d = [0.0], 0.0, 0.0
            for k in range(n + 10):
                v = min(v + (accel if k < j else tired_acc), vmax)
                d += v
                row.append(d)
                v *= pdecay
            rows.append(row)
        self.DASH_TIRED = np.array(rows)

    def threat(self, x, y):
        """Chance that our possession starting at (x, y) ends in a goal: a
        sigmoid in x (about 5% deep in our half, rising towards their box),
        lower out wide, plus a bonus for the central zone in front of goal."""
        wide = 1.0 - 0.4 * np.clip((np.abs(y) - 10.0) / 15.0, 0.0, 1.0)
        slot_bonus = 0.08 * _sig((x - 32.0) / 3.0) * _sig((11.0 - np.abs(y)) / 2.5)
        return 0.05 + 0.30 * wide / (1.0 + np.exp(-(x - 22.0) / 9.0)) + slot_bonus

    def danger(self, x, y):
        """The opponents' threat from (x, y): `threat` mirrored."""
        return self.threat(-x, y)

    def v_keep(self, x, y, space):
        """Value of us having the ball at (x, y), with a small bonus for
        `space` (distance to the nearest opponent, capped at 8 m)."""
        return self.threat(x, y) + 0.018 * np.clip(space, 0.0, 8.0) / 8.0

    def v_lose(self, x, y):
        """Value of the opponents winning the ball at (x, y)."""
        return -(self.danger(x, y) + 0.04)

    def v_out(self, x, y):
        """Value of the ball going out of play at (x, y): a smaller loss."""
        return -(0.35 * self.danger(x, y) + 0.015)


def _kick_traj(ph, ox, oy, ux, uy, s0, delay, T):
    """Ball positions for C kicks from (ox, oy), shape (C, T + 1, 2).

    Kick c leaves in unit direction (ux[c], uy[c]) at speed s0[c] after
    delay[c] cycles (0 for a one-kick pass, 1 if a second kick is needed).
    """
    t = np.arange(T + 1)[None, :]
    k = np.clip(t - delay[:, None], 0, ph.G.size - 1)
    g = ph.G[k] * s0[:, None]
    ox = np.broadcast_to(ox, s0.shape)[:, None]
    oy = np.broadcast_to(oy, s0.shape)[:, None]
    return np.stack([ox + ux[:, None] * g, oy + uy[:, None] * g], axis=2)


def _race(ph, B, P, body, ctrl, gk, react, vel=None, omni=None, tired=None):
    """First cycle at which each player can control the ball on each path.

    B      (C, T+1, 2) ball paths, e.g. from `_kick_traj`.
    P      (N, 2) player positions; body (N,) body directions in degrees.
    ctrl   (N,) control radius (kickable area; larger for opponents).
    gk     (N,) 1 = their keeper, -1 = ours: inside his own box a keeper
           uses the catch radius instead.
    react  (N,) reaction delay in cycles before a player starts moving.
    vel    optional (N, 2) velocities: players drift along them, and the
           turn cost depends on their speed (inertia).
    omni   optional (N,) mask of players who may dash sideways/backwards
           without turning first (we give opponents this benefit of the doubt).
    tired  optional (N,) sprint budgets: index into DASH_TIRED.

    A player reaches the ball at cycle t if the ball is within his control
    radius plus the distance he can cover in t - react - turn cycles.
    Returns a (C, N) int array of cycles, NEVER where he cannot make it
    within the horizon.
    """
    C, T1, _ = B.shape
    N = P.shape[0]
    if N == 0:
        return np.full((C, 0), NEVER, dtype=np.int64)
    if vel is not None:
        drift = (1.0 - ph.pdecay ** np.arange(T1)) / (1.0 - ph.pdecay)
        dx = B[:, :, None, 0] - (P[None, None, :, 0] + vel[None, None, :, 0] * drift[None, :, None])
        dy = B[:, :, None, 1] - (P[None, None, :, 1] + vel[None, None, :, 1] * drift[None, :, None])
    else:
        dx = B[:, :, None, 0] - P[None, None, :, 0]
        dy = B[:, :, None, 1] - P[None, None, :, 1]
    d = np.sqrt(dx * dx + dy * dy)
    c = np.broadcast_to(ctrl, d.shape)
    if np.any(gk):
        bx = B[:, :, 0]
        by = np.abs(B[:, :, 1])
        in_their = (bx > ph.pen_x) & (by < ph.pen_hw)
        in_our = (bx < -ph.pen_x) & (by < ph.pen_hw)
        c = np.where((gk == 1)[None, None, :] & in_their[:, :, None],
                     ph.catch_r + OPP_MARGIN, c)
        c = np.where((gk == -1)[None, None, :] & in_our[:, :, None],
                     ph.catch_r, c)
    ang = np.degrees(np.arctan2(dy, dx))
    diff = np.abs((ang - body + 180.0) % 360.0 - 180.0)
    if vel is not None:
        sp = np.hypot(vel[:, 0], vel[:, 1])
        a1 = 180.0 / (1.0 + INERTIA * sp)
        a2 = a1 + 180.0 / (1.0 + INERTIA * sp * ph.pdecay)
        turn = (diff > 25.0).astype(np.int64) * (1 + (diff > a1[None, None, :])
                                                 + (diff > a2[None, None, :])) * (d > c)
    else:
        turn = ((diff > 25.0).astype(np.int64) + (diff > 90.0)) * (d > c)
    t = np.arange(T1)[None, :, None]
    avail = np.clip(t - react[None, None, :] - turn, 0, ph.DASH.size - 1)
    if tired is None:
        ok = d <= c + ph.DASH[avail]
    else:
        ok = d <= c + ph.DASH_TIRED[np.asarray(tired, dtype=np.int64)[None, None, :], avail]
    if omni is not None and np.any(omni):
        k = np.clip(np.rint(diff / 15.0).astype(np.int64), 0, 12)
        now = np.clip(t - react[None, None, :], 0, ph.DASH.size - 1)
        ok |= omni[None, None, :] & (d <= c + ph.DASH_DIR[k, now])
    hit = ok.any(axis=1)
    return np.where(hit, ok.argmax(axis=1), NEVER)


def _go_point(ph, x, y, offside_x, dist):
    """Where a passer at (x, y) runs on a give-and-go: up to `dist` metres
    towards a point 10 m in front of their goal, kept onside.
    Returns (gx, gy, ux, uy): the target and the run direction."""
    tx, ty = ph.hl - 10.0, 0.45 * y
    d = math.hypot(tx - x, ty - y)
    if d < 1e-6:
        return x, y, 0.0, 0.0
    ux, uy = (tx - x) / d, (ty - y) / d
    k = min(dist, d)
    gx, gy = x + ux * k, y + uy * k
    if gx > 0.0:
        gx = max(min(gx, offside_x - 0.5), min(x, gx))
    return gx, gy, ux, uy


def _first_true(mask):
    """Index of the first True in each row of a (C, T) mask, NEVER if none."""
    hit = mask.any(axis=1)
    return np.where(hit, mask.argmax(axis=1), NEVER)


# ==========================================================================
# SECTION: STATE & KICK RESULT (agent/state.py)
# ==========================================================================


class _State:
    """One node of the on-ball search.

    ox, oy     ball position; `kicker` is the uniform number on the ball.
    mates      [(unum, x, y, body)] of our field players.
    opps       [(unum, x, y, body, is_goalie)].
    recv       teammates who can receive: everyone but the kicker, minus
               anyone offside.
    P, body, ctrl, gk, react, vel, omni, tired
               receivers then opponents, packed as the arrays `_race` takes.
    lead       opponents are raced from position + lead * velocity.
    adv        how far teammates push up in child states (CHILD_ADV).
    theirs     True when the state is mirrored to model the opponents'
               attack (see DefenseMixin._their_root).
    """

    def __init__(self, ph, ox, oy, kicker, mates, opps, offside_x, opp_vel=None, mate_vel=None,
                 opp_unc=None, lead=0.0, tired_map=None):
        self.ph = ph
        self.ox, self.oy = float(ox), float(oy)
        self.kicker = kicker
        self.mates = mates
        self.opps = opps
        self.offside_x = offside_x
        self.kbody = next((m[3] for m in mates if m[0] == kicker), 0.0)
        self.lead = lead
        self.adv = 0.0
        self.theirs = False
        line = max(offside_x, ox)
        self.recv = [m for m in mates
                     if m[0] != kicker and not (m[1] > line and m[1] > 0.0)]

        nus, nop = len(self.recv), len(opps)
        self.nus, self.nop = nus, nop
        self.op_pos = (np.array([[o[1], o[2]] for o in opps])
                       if nop else np.zeros((0, 2)))
        op_body = np.array([o[3] for o in opps]) if nop else np.zeros(0)
        op_gk = np.array([1 if o[4] else 0 for o in opps], dtype=np.int64)
        us_pos = (np.array([[m[1], m[2]] for m in self.recv])
                  if nus else np.zeros((0, 2)))
        us_body = np.array([m[3] for m in self.recv]) if nus else np.zeros(0)

        unc = np.zeros(nop) if opp_unc is None else np.asarray(opp_unc, dtype=float)
        self.P = np.vstack([us_pos, self.op_pos])
        self.body = np.concatenate([us_body, op_body])
        self.ctrl = np.concatenate([np.full(nus, ph.kick_r),
                                    np.full(nop, ph.kick_r + OPP_MARGIN) + unc])
        self.gk = np.concatenate([np.zeros(nus, dtype=np.int64), op_gk])
        self.react = np.ones(nus + nop, dtype=np.int64)
        self.op_ctrl = np.full(nop, ph.kick_r + OPP_MARGIN) + unc
        self.op_gk = op_gk
        self.op_body = op_body
        self.own_gk = None
        ov = np.asarray(opp_vel, dtype=float).reshape(-1, 2) if opp_vel is not None else np.zeros((nop, 2))
        mv = mate_vel or {}
        self.ov, self.mv = ov, mv
        uv = np.array([mv.get(m[0], (0.0, 0.0)) for m in self.recv], dtype=float).reshape(-1, 2)
        self.vel = np.vstack([uv, ov])
        self.P_race = self.P - np.vstack([np.zeros((nus, 2)), self.lead * ov])
        self.omni = np.concatenate([np.zeros(nus, dtype=bool), np.ones(nop, dtype=bool)])
        self.tired_map = tired_map
        self.tired = (None if tired_map is None else
                      np.array([tired_map.get(m[0], ph.J) for m in self.recv] + [ph.J] * nop,
                               dtype=np.int64))

    def space(self, x, y):
        """Distance from (x, y) to the nearest opponent (20 if there are none)."""
        if self.nop == 0:
            return np.full(np.shape(x), 20.0)
        d = np.hypot(np.asarray(x)[..., None] - self.op_pos[:, 0],
                     np.asarray(y)[..., None] - self.op_pos[:, 1])
        return d.min(axis=-1)

    def child(self, receiver_unum, lx, ly, t, body=None):
        """The state after `receiver_unum` controls the ball at (lx, ly) at cycle t.

        Opponents are the opponents' reply: the one nearest the reception
        point presses it, the others move to cover the space between it and
        the goal they defend, the keeper guards that goal, and each gets extra control
        radius for the uncertainty of the extrapolation. Teammates coast on
        their velocity and push up by `adv`; the passer runs a give-and-go if
        GIVE_GO applies. Passing the kicker's own number models a dribble.
        """
        ph = self.ph
        drift = (1.0 - ph.pdecay ** t) / (1.0 - ph.pdecay) if CHILD_DRIFT else 0.0
        reach = (float(ph.DASH[min(max(t - 1, 0), ph.DASH.size - 1)]) if CHILD_DRIFT
                 else float(t))

        # Identify the primary pressing opponent (closest field player to (lx, ly))
        field_dists = []
        for k, (u, x, y, b, g) in enumerate(self.opps):
            if not g:
                coast = drift - (self.lead if CHILD_LEAD_FIX else 0.0)
                cx, cy = x + self.ov[k, 0] * coast, y + self.ov[k, 1] * coast
                field_dists.append((math.hypot(lx - cx, ly - cy), u))
        presser_u = min(field_dists)[1] if field_dists else None

        opps, unc, opp_vel = [], [], []
        for k, (u, x, y, b, g) in enumerate(self.opps):
            coast = drift - (self.lead if CHILD_LEAD_FIX else 0.0)
            x, y = x + self.ov[k, 0] * coast, y + self.ov[k, 1] * coast
            r = reach * (GK_REACH if g else 1.0)
            if g:
                tx = max(ph.hl - 1.5, lx)
                ty = float(np.clip(ly * 0.35, -ph.gh + 1.0, ph.gh - 1.0))
            elif u == presser_u or getattr(self, "theirs", False):
                tx, ty = lx, ly
            else:
                gx_goal = ph.hl
                d_goal = math.hypot(gx_goal - lx, -ly) or 1.0
                slot_x = lx + (gx_goal - lx) / d_goal * min(7.0, 0.35 * d_goal)
                slot_y = ly * 0.65
                tx = 0.55 * slot_x + 0.45 * max(x, lx + 2.0)
                ty = 0.55 * slot_y + 0.45 * y

            dx, dy = tx - x, ty - y
            d = math.hypot(dx, dy)
            step = min(max(d - (0.8 if u == presser_u else 0.0), 0.0), OPP_PURSUIT * r)
            v = (0.0, 0.0)
            if d > 1e-6:
                x, y = x + dx / d * step, y + dy / d * step
                b = math.degrees(math.atan2(ly - y, lx - x))
                if step > 0.5:
                    v = (CHILD_OPP_SPEED * dx / d, CHILD_OPP_SPEED * dy / d)
            opps.append((u, x, y, b, g))
            opp_vel.append(v)
            unc.append(min(OPP_UNC * r, OPP_UNC_MAX))

        mates, mate_vel = [], {}
        keep = ph.pdecay ** t
        cap = max(self.offside_x, lx) - 0.5
        for m in self.mates:
            if m[0] == receiver_unum:
                mates.append((m[0], lx, ly, m[3] if body is None else body))
                continue
            if m[0] == self.kicker and GIVE_GO and self.ox > GO_X0:
                gx, gy, ux, uy = _go_point(ph, self.ox, self.oy, self.offside_x,
                                           GO_PURSUIT * reach)
                mates.append((m[0], gx, gy, math.degrees(math.atan2(uy, ux))))
                mate_vel[m[0]] = (0.8 * ux, 0.8 * uy)
                continue
            if m[0] == self.kicker:
                x, y = self.ox, self.oy
            else:
                vx, vy = self.mv.get(m[0], (0.0, 0.0))
                x, y = m[1] + vx * drift, m[2] + vy * drift
                mate_vel[m[0]] = (vx * keep, vy * keep)
            if self.adv:
                x = max(x, min(x + self.adv * reach, cap))
            mates.append((m[0], x, y, m[3]))

        offside_x = self.offside_x
        if self.adv and len(opps) > 1:
            offside_x = max(offside_x, sorted((o[1] for o in opps), reverse=True)[1])

        s = _State(ph, lx, ly, receiver_unum, mates, opps, offside_x,
                   opp_vel=opp_vel if CHILD_OPP_VEL else None,
                   mate_vel=mate_vel, opp_unc=unc, lead=0.0,
                   tired_map=self.tired_map)
        s.own_gk = self.own_gk
        s.adv = self.adv
        s.theirs = getattr(self, "theirs", False)
        return s


class _KickResult:
    """Vectorised evaluation of C candidate kicks (see EvaluatorMixin._eval_kicks).

    value   expected value of each kick; p: success probability.
    t_us / t_th   first cycle we / they reach the ball; who: index into recv.
    lx, ly  where we receive it; succ: we get there before it goes out.
    v_fail  value if the pass fails; vs: value if it succeeds.
    tx, ty, s0, delay   the kick itself (target, speed, extra kick cycles).
    """
    __slots__ = ("value", "p", "t_us", "t_th", "who", "lx", "ly", "succ", "v_fail",
                 "vs", "tx", "ty", "s0", "delay")


# ==========================================================================
# SECTION: EVALUATOR MIXIN (agent/evaluators.py)
# ==========================================================================


class EvaluatorMixin:
    """Action evaluation shared by the on-ball search and the defence model."""

    def _pass_targets(self, S, rings=None, speeds=None, space=True):
        """Candidate pass kicks from state S as arrays (tx, ty, s0).

        Targets: each receiver's feet plus points on `rings` around him, kicks
        into space in a fan in front of the ball (if `space`), and through
        balls ahead of forward receivers (THROUGH). Each target is tried at
        every arrival speed in `speeds`; s0 is the initial ball speed that
        arrives at the target with that speed.
        """
        ph = S.ph
        rings = self.RINGS if rings is None else rings
        speeds = self.END_SPEEDS if speeds is None else speeds
        pts = []
        for (_, x, y, _) in S.recv:
            pts.append((x, y))
            for r, n in rings:
                for i in range(n):
                    a = 2.0 * math.pi * i / n
                    pts.append((x + r * math.cos(a), y + r * math.sin(a)))
        for a in ((-80, -60, -40, -20, 0, 20, 40, 60, 80) if space else ()):
            for dist in (8.0, 12.0, 16.0, 20.0, 25.0):
                ra = math.radians(a)
                pts.append((S.ox + dist * math.cos(ra), S.oy + dist * math.sin(ra)))
        if not pts:
            return np.zeros(0), np.zeros(0), np.zeros(0)
        pts = np.array(pts)
        tx, ty = pts[:, 0], pts[:, 1]
        D = np.hypot(tx - S.ox, ty - S.oy)
        keep = ((np.abs(tx) < ph.hl - 1.0) & (np.abs(ty) < ph.hw - 1.0)
                & (D > 3.0) & (D < 40.0))
        tx, ty, D = tx[keep], ty[keep], D[keep]
        TX, TY, S0 = [], [], []
        for e in speeds:
            TX.append(tx)
            TY.append(ty)
            S0.append(np.minimum(ph.bmax, D * (1.0 - ph.decay) + e))
        if self.THROUGH:
            th = np.array([(x + dx, y + dy) for (_, x, y, _) in S.recv if x > -18.0
                           for dx in (7.0, 13.0) for dy in (-4.5, 0.0, 4.5)]).reshape(-1, 2)
            if th.size:
                Dt = np.hypot(th[:, 0] - S.ox, th[:, 1] - S.oy)
                ok = ((np.abs(th[:, 0]) < ph.hl - 1.0) & (np.abs(th[:, 1]) < ph.hw - 1.0)
                      & (Dt > 3.0) & (Dt < 40.0))
                for e in (0.3, 0.65):
                    TX.append(th[ok, 0])
                    TY.append(th[ok, 1])
                    S0.append(np.minimum(ph.bmax, Dt[ok] * (1.0 - ph.decay) + e))
        return np.concatenate(TX), np.concatenate(TY), np.concatenate(S0)

    def _eval_kicks(self, S, tx, ty, s0, delay, m0=None, T=HORIZON, k_multi=1.0):
        """Race and value C candidate kicks at once; returns a _KickResult.

        1. Simulate every ball path and race it against all players.
        2. Success probability from the margin (their arrival - ours): two
           logistics (hand-set and fitted), times a penalty if an opponent
           can close down the receiver on arrival, then recalibrated with
           PASS_FIT. Two-kick passes are scaled by `k_multi` under pressure.
        3. Value = p * GAMMA^t * v_keep(reception) + (1 - p) * v_lose. A
           pass our own keeper would collect in his box is a back-pass foul
           and gets V_BACKPASS.
        m0 shifts the first logistic (more/less risk late in the match).
        """
        ph = S.ph
        m0 = self.PASS_M0 if m0 is None else m0
        C = tx.size
        dx, dy = tx - S.ox, ty - S.oy
        D = np.maximum(np.hypot(dx, dy), 1e-6)
        ux, uy = dx / D, dy / D
        B = _kick_traj(ph, S.ox, S.oy, ux, uy, s0, delay, T)
        first = _race(ph, B, S.P_race, S.body, S.ctrl, S.gk, S.react, vel=S.vel, omni=S.omni,
                      tired=S.tired)
        nus = S.nus
        if nus:
            t_us = first[:, :nus].min(axis=1)
            who = first[:, :nus].argmin(axis=1)
        else:
            t_us = np.full(C, NEVER)
            who = np.zeros(C, dtype=np.int64)
        t_th = first[:, nus:].min(axis=1) if S.nop else np.full(C, NEVER)
        out = _first_true((np.abs(B[:, :, 0]) > ph.hl) | (np.abs(B[:, :, 1]) > ph.hw))
        if S.own_gk is not None:
            gd = np.hypot(B[:, :, 0] - S.own_gk[0], B[:, :, 1] - S.own_gk[1])
            reach = ph.catch_r + ph.DASH[np.clip(np.arange(T + 1) - 1, 0, ph.DASH.size - 1)]
            inbox = (B[:, :, 0] < -ph.pen_x) & (np.abs(B[:, :, 1]) < ph.pen_hw)
            t_gk = _first_true(inbox & (gd <= reach[None, :]))
            foul = (t_gk < t_us + 3) & (t_gk <= t_th) & (t_gk < out)
            t_us = np.where(foul, NEVER, t_us)
        else:
            foul = np.zeros(C, dtype=bool)

        idx = np.arange(C)
        lu = B[idx, np.minimum(t_us, T)]
        lt = B[idx, np.minimum(t_th, T)]
        lo = B[idx, np.minimum(out, T)]
        le = B[:, T]

        vs = ph.v_keep(lu[:, 0], lu[:, 1], S.space(lu[:, 0], lu[:, 1]))
        succ = (t_us <= T) & (t_us < out)
        contested = t_th <= T
        v_fail = np.where(contested, ph.v_lose(lt[:, 0], lt[:, 1]),
                          ph.v_lose(lu[:, 0], lu[:, 1]))

        mates_behind = sum(1 for m in S.mates if m[0] != S.kicker and m[1] < S.ox - 2.0)
        if mates_behind == 0 and S.ox < 15.0:
            v_fail = v_fail - 0.08

        out_margin = getattr(self, "PASS_OUT_MARGIN", True)
        margin = np.clip((np.minimum(t_th, out) if out_margin else t_th) - t_us, -6, 6).astype(float)
        shift = m0 - self.PASS_M0
        p = np.minimum(_sig((margin - m0) / self.PASS_S),
                       _sig((margin - self.FIT_M0 - shift) / self.FIT_S))
        if S.nop:
            od = np.hypot(lu[:, 0, None] - S.op_pos[None, :, 0], lu[:, 1, None] - S.op_pos[None, :, 1])
            t_ctrl = np.searchsorted(ph.DASH, np.maximum(od - ph.kick_r, 0.0)).min(axis=1) + 1
            p_ctrl = np.clip(0.95 + 0.017 * (t_ctrl - t_us), 0.95, 1.0)
            p_ctrl = np.where(np.hypot(ph.hl - lu[:, 0], lu[:, 1]) < 20.0, 1.0, p_ctrl)
            p = p * p_ctrl
        if k_multi < 1.0:
            p = np.where(delay > 0, p * k_multi, p)

        # Calibrate pass survival against opponent 2.0m tackle reach (from RN8p.py)
        pass_fit = getattr(self, "PASS_FIT", (1.11, 0.16, 0.10))
        if pass_fit is not None and S.nop and not getattr(S, "theirs", False):
            first2 = _race(ph, B, S.P_race[nus:], S.body[nus:], np.maximum(S.ctrl[nus:], 2.0),
                           S.gk[nus:], S.react[nus:], vel=S.vel[nus:], omni=S.omni[nus:])
            m2 = np.clip(first2.min(axis=1) - t_us, -6, 8).astype(float)
            a, b, c = pass_fit
            pc = np.clip(p, 1e-3, 1.0 - 1e-3)
            p = _sig(a + b * np.log(pc / (1.0 - pc)) + c * m2)

        p = np.where(succ, p, 0.0)
        disc = GAMMA ** np.minimum(t_us, T)

        v_nosucc = np.where(
            (t_th <= T) & (t_th < out), ph.v_lose(lt[:, 0], lt[:, 1]),
            np.where(out <= T, ph.v_out(lo[:, 0], lo[:, 1]),
                     0.5 * (ph.v_keep(le[:, 0], le[:, 1], 0.0)
                            + ph.v_lose(le[:, 0], le[:, 1]))))
        value = np.where(succ, p * disc * vs + (1.0 - p) * v_fail, v_nosucc)
        value = np.where(foul, self.V_BACKPASS, value)

        r = _KickResult()
        r.value, r.p, r.t_us, r.t_th, r.who = value, p, t_us, t_th, who
        r.lx, r.ly, r.succ, r.v_fail, r.vs = lu[:, 0], lu[:, 1], succ, v_fail, vs
        r.tx, r.ty, r.s0, r.delay = tx, ty, s0, delay
        return r

    def _eval_shots(self, S, s0, delay, n=13, T=HORIZON):
        """Evaluate shots at n points across the goal mouth.

        P(goal) = P(on target, given kick noise growing with distance)
                  x P(no field player intercepts first, from the race)
                  x P(the keeper does not save, fitted GK_* logistic of his
                    distance from the shot line and the time he has).
        Returns (value, P(goal), aim y) arrays.
        """
        ph = S.ph
        gy = np.linspace(-ph.gh + 0.55, ph.gh - 0.55, n)
        tx = np.full(n, ph.hl + 2.0)
        dx, dy = tx - S.ox, gy - S.oy
        D = np.maximum(np.hypot(dx, dy), 1e-6)
        s0 = np.broadcast_to(np.asarray(s0, dtype=float), (n,)).copy()
        delay = np.broadcast_to(np.asarray(delay, dtype=np.int64), (n,)).copy()
        B = _kick_traj(ph, S.ox, S.oy, dx / D, dy / D, s0, delay, T)
        t_goal = _first_true(B[:, :, 0] >= ph.hl)
        field = S.op_gk == 0
        gk = np.nonzero(~field)[0]
        if gk.size:
            first = _race(ph, B, S.op_pos[field], S.op_body[field], S.op_ctrl[field],
                          S.op_gk[field], np.ones(int(field.sum()), dtype=np.int64))
        else:
            first = _race(ph, B, S.op_pos, S.op_body, S.op_ctrl, S.op_gk,
                          np.ones(S.nop, dtype=np.int64))
        t_th = first.min(axis=1) if first.shape[1] else np.full(n, NEVER)
        dist = np.hypot(ph.hl - S.ox, gy - S.oy)
        e = 0.03 * dist + 0.3
        lo = np.maximum(gy - e, -ph.gh + 0.15)
        hi = np.minimum(gy + e, ph.gh - 0.15)
        p_on = np.clip((hi - lo) / (2.0 * e), 0.0, 1.0)
        rel = np.clip(t_th - t_goal, -6, 6).astype(float)
        p_beat = np.where(t_goal > T, 0.0, _sig((rel + 1.0) / 0.45))
        if gk.size and getattr(self, "KEEPER_RACE", False):
            k = int(gk[0])
            t_k = _race(ph, B, S.P_race[S.nus + k][None], S.op_body[k:k + 1], np.array([ph.kick_r]),
                        np.ones(1, dtype=np.int64), np.zeros(1, dtype=np.int64),
                        vel=S.ov[k:k + 1], omni=np.ones(1, dtype=bool))[:, 0]
            m = t_goal - t_k
            kr_p1 = getattr(self, "KR_P1", 0.67)
            kr_p2 = getattr(self, "KR_P2", 0.33)
            p_beat = p_beat * np.where(m <= 0, 1.0, np.where(m == 1, kr_p1, kr_p2))
        elif gk.size:
            kx, ky = S.op_pos[gk[0]]
            ux, uy = dx / D, dy / D
            rx, ry = kx - S.ox, ky - S.oy
            along = rx * ux + ry * uy
            perp = np.where(along > 0, np.abs(rx * uy - ry * ux), math.hypot(rx, ry))
            frac = np.clip(np.maximum(along, 0.1) * (1.0 - ph.decay) / s0, 0.0, 0.999)
            n_keep = np.log(1.0 - frac) / math.log(ph.decay) + delay
            p_gk = _sig(self.GK_A + self.GK_P * perp + self.GK_N * n_keep)
            if self.KIN_SHOT:
                avail = np.clip(n_keep.astype(np.int64) - 2, 0, ph.DASH.size - 1)
                p_gk = np.maximum(p_gk, _sig((perp - ph.catch_r - ph.DASH[avail]) / 0.45))
            p_beat = p_beat * p_gk
        pg = p_on * p_beat
        return pg - (1.0 - pg) * 0.03, pg, gy

    def _deep_value(self, S, res, i, depth, deadline):
        """Depth-limited look-ahead after pass i (used only when EXP_BEAM is 0)."""
        best = float(res.vs[i])
        if depth <= 1 or time.perf_counter() > deadline:
            return best
        recv = S.recv[int(res.who[i])][0]
        child = S.child(recv, float(res.lx[i]), float(res.ly[i]), int(res.t_us[i]))
        if child.ox > 10.0:
            sv, _, _ = self._eval_shots(child, self.ph.bmax, 1)
            best = max(best, 0.94 * float(sv.max()))
        tx, ty, s0 = self._pass_targets(child, rings=self.CHILD_RINGS,
                                        speeds=self.CHILD_SPEEDS)
        if tx.size:
            cres = self._eval_kicks(child, tx, ty, s0, np.ones(tx.size, dtype=np.int64))
            cv = cres.value.copy()
            if depth > 2 and time.perf_counter() < deadline:
                for j in np.argsort(-cv)[:self.CHILD_K]:
                    if not cres.succ[j]:
                        continue
                    v2 = self._deep_value(child, cres, j, depth - 1, deadline)
                    cv[j] = (cres.p[j] * GAMMA ** cres.t_us[j] * v2
                             + (1.0 - cres.p[j]) * cres.v_fail[j])
            best = max(best, 0.90 * float(cv.max()))
        return best

    def _kick_delay(self, agent, S, tx, ty, s0):
        """0 if one kick can reach speed s0 in that direction, else MULTI_DELAY."""
        ang = np.degrees(np.arctan2(ty - S.oy, tx - S.ox))
        vmax = np.array([self._vmax(agent, a) for a in ang])
        return np.where(vmax >= s0 * 0.98, 0, self.MULTI_DELAY).astype(np.int64)

    def _shot1_speeds(self, agent, S, n=13):
        ph = self.ph
        gy = np.linspace(-ph.gh + 0.55, ph.gh - 0.55, n)
        return np.array([min(self._vmax(agent, math.degrees(math.atan2(y - S.oy, ph.hl + 2.0 - S.ox))),
                             ph.bmax) for y in gy])

    def _vmax(self, agent, angle_deg):
        """Max ball speed one kick can give in this direction (cached per cycle)."""
        key = (agent.wm.cycle, int(round(angle_deg / 10.0)))
        v = self._vmax_cache.get(key)
        if v is None:
            if len(self._vmax_cache) > 200:
                self._vmax_cache.clear()
            wm = agent.wm
            vel = Tools.calc_max_velocity(AngleDeg(float(angle_deg)), wm.self.kick_rate,
                                          wm.ball.velocity, agent.server_params)
            v = vel.r()
            self._vmax_cache[key] = v
        return v

    def _stam_cycle_start(self, agent, on_ball):
        """Once per cycle: update the stamina model and teammates' sprint budgets.

        The sprint budget (full-power dashes left before the stamina floor)
        feeds the race model, so tired teammates are raced at the slower
        floor pace. With STAM_FLOOR, a teammate who should be running flat
        out but is seen dashing at exactly 45 must be at the floor. Returns
        True if this player spent the cycle sending a stamina signal.
        """
        if not (self.STAM_TRACK or self.STAM_RACE or self.STAM_SIGNAL or self.STAM_FLOOR):
            return False
        wm = agent.wm
        if wm.self.is_goalie:
            return False
        ph = self._phys(agent)
        if wm.cycle != self._stam_cycle:
            self._stam_cycle = wm.cycle
            if self._stam is None:
                self._stam = StaminaPredictor(agent.server_params)
                self._stam.half_cycles = float(HALF_CYCLES)
            P = self._stam
            prev = P.last
            P.update(agent)
            if self.STAM_FLOOR and self._floor_roles[0] == prev and wm.cycle == (prev or -9) + 1:
                for u in self._floor_roles[1]:
                    need = P.last_need.get(("us", u))
                    if (need is not None and abs(need - 45.0) <= 0.3 and ("us", u) in P.state
                            and P.state[("us", u)][0] > ph.st_floor + 1.0):
                        P.state[("us", u)][0] = ph.st_floor
                        P.var[("us", u)] = 0.0
                        note(agent, "floor_hit", u=u)
            self._budget = {k[1]: int(min(max((P.estimate(k) - ph.st_floor) / SPRINT_NET, 0.0), ph.J))
                            for k in P.state if k[0] == "us"}
            if wm.cycle % 25 == 0:
                note(agent, "stam", **{f"u{k[1]}": round(P.estimate(k)) for k in sorted(P.state)
                                       if k[0] == "us"})
        if self.STAM_FLOOR and self._floor_roles[0] != wm.cycle:
            self._floor_roles = (wm.cycle, self._flat_out(self._context(agent)))
        if not self.STAM_SIGNAL:
            return False
        ctx = self._context(agent)
        senders = self._sig_senders(agent, ctx)
        if self.STAM_FLOOR:
            self._floor_roles = (wm.cycle, self._floor_roles[1] - set(senders))
        for u in senders:
            self._sig_pending[u] = wm.cycle
        if not on_ball and wm.self.uniform_number in senders:
            self._send_signal(agent)
            return True
        return False

    def _sig_senders(self, agent, ctx):
        c = agent.wm.cycle
        h, hc = (0, c) if c < HALF_CYCLES else (1, c - HALF_CYCLES)
        if hc < self.SIG_AT:
            return []
        busy = set(self._busy(ctx)) | {ctx["chaser"]}
        bx, by = ctx["ball"][0], ctx["ball"][1]
        return [m[0] for m in ctx["mates"]
                if (h, m[0]) not in self._sig_done and m[0] not in self._sig_pending
                and m[0] not in busy and math.hypot(m[1] - bx, m[2] - by) >= 3.0]

    def _send_signal(self, agent):
        wm = agent.wm
        ph = self.ph
        st, eff, rec = wm.self.stamina, max(wm.self.effort, 0.05), wm.self.recovery
        inc = float(getattr(agent.server_params, "stamina_inc_max", 0.0)) or 45.0
        top = SIG_BASE + (ph.stamina_max - SIG_LO) / SIG_STEP
        p = SIG_BASE
        for _ in range(6):
            cmd = min(p / eff, 100.0)
            nxt = min(st - cmd + inc * rec, ph.stamina_max)
            p = min(max(SIG_BASE + (nxt - SIG_LO) / SIG_STEP, SIG_BASE), top)
        cmd = min(p / eff, 100.0)
        ahead = wm.self.position + Vector2D.polar2vector(30.0, wm.self.body_direction)
        note(agent, "sig", p=round(p, 3), cmd=round(cmd, 3), s=round(st))
        Action.go_to_point(agent, ahead, max_dash_power=cmd, distance_threshold=1.0)

    def _flat_out(self, ctx):
        """Teammates who should be sprinting this cycle (chaser, or everyone
        free when the situation is serious)."""
        out = set()
        ch = ctx["chaser"]
        if ch is not None and not ctx["kickable_mate"] and (
                ctx["attack"] or ctx["mate_t"].get(ch, NEVER) <= ctx["opp_min"] + 1):
            out.add(ch)
        if not ctx["attack"] and self._serious(ctx):
            busy = set(self._busy(ctx)) | {ch}
            out |= {m[0] for m in ctx["mates"] if m[0] not in busy}
        return out

    def _race_budgets(self, ph, mates, goalie, opps):
        """Per-player sprint budgets in `_race` order (None if STAM_RACE is off)."""
        if not self.STAM_RACE:
            return None
        return np.array([self._budget.get(m[0], ph.J) for m in mates]
                        + ([ph.J] if goalie else []) + [ph.J] * len(opps), dtype=np.int64)


# ==========================================================================
# SECTION: SEARCH & EXPECTIMAX MIXIN (agent/search.py)
# ==========================================================================


class SearchMixin:
    """Look-ahead search over action chains for the player on the ball."""

    def _field_eval(self, x, y):
        """Chain-search score of having the ball at (x, y): progress (x),
        plus 100x the scoring threat, plus a bonus for the central slot."""
        slot = self.SLOT_W * _sig((x - 31.0) / 2.0) * _sig((12.0 - np.abs(y)) / 2.0)
        return x + 100.0 * (self.ph.threat(x, y) - 0.05) + slot

    def _chain_scores(self, S, r, pressed):
        """Score kicks for the chain search; kicks below the success
        threshold (P_PASS, or P_ATT in attack) are marked unsafe (-inf)."""
        att = (S.ox > 15.0) | (r.lx > 22.0)
        safe = r.succ & (r.p >= np.where(att, self.P_ATT, self.P_PASS))
        mid = self.MID_LOSS if self.FIX_CHAIN_RISK and self.FIX_RISK_SYM else 0.0
        score = (self._field_eval(r.lx, r.ly) - 0.05 * r.t_us
                 - np.where(att, self.LOSS_ATT, mid) * (1.0 - r.p))
        if S.ox > self.BACK_X and not pressed:
            score = score - self.BACK_W * np.maximum(0.0, S.ox - 4.0 - r.lx)
        return safe, np.where(safe, score, -np.inf)

    def _step_cost(self, S, r, base):
        att = (S.ox > 15.0) | (r.lx > 22.0)
        if self.FIX_RISK_SYM:
            return 0.05 * r.t_us + np.where(att, self.LOSS_ATT, self.MID_LOSS) * (1.0 - r.p)
        return ((self._field_eval(r.lx, r.ly) - base)
                + np.where(att, 0.0, self.MID_LOSS * (1.0 - r.p)))

    def _on_ball_chain(self, agent, S, deadline, shot_w):
        """Shot decision first, then (if EXP_BEAM < 2) the chain search.

        Within 30 m of goal, take the best one- or two-kick shot if its goal
        probability clears the threshold for the situation (SHOT_P, lower in
        the box or under pressure, higher from far out), unless a short
        dribble would give a clearly better shot (`_carry_in`). Returns True
        if an action was executed, False to fall through to the main
        evaluator in OnBallMixin._on_ball.
        """
        ph = self.ph
        opts = []

        if math.hypot(ph.hl - S.ox, S.oy) < 30.0:
            v1 = self._vmax(agent, math.degrees(math.atan2(-S.oy, ph.hl - S.ox)))
            v1s = self._shot1_speeds(agent, S) if self.PER_AIM_V1 else None
            if (v1s.max() if v1s is not None else v1) >= 1.8:
                sv, pg, gy = self._eval_shots(S, v1s if v1s is not None else min(v1, ph.bmax), 0)
                kind = "shot1"
                d_opp, t_opp = self._pressure(S)
                if self.SHOT_BOTH_STRICT:
                    two = (pg.max() * shot_w < self.SHOT_P and d_opp >= self.HOLD_BAN_D
                           and t_opp > 2)
                else:
                    two = d_opp >= self.PRESS_D
                if self.FIX_SHOT_BOTH and two:
                    sv2, pg2, gy2 = self._eval_shots(S, ph.bmax, 1)
                    if pg2.max() > pg.max():
                        sv, pg, gy, kind = sv2, pg2, gy2, "shot"
            else:
                sv, pg, gy = self._eval_shots(S, ph.bmax, 1 if v1 > 1.2 else 2)
                kind = "shot"
            i = int(np.argmax(pg))
            note(agent, "sp", pg=round(float(pg[i]), 2), k=kind)
            p_need = (self.SHOT_P_PRESSED if S.ox > 32.0 and self._pressure(S)[0] < self.HOLD_BAN_D
                      else self.SHOT_P)
            if S.ox > 33.0 and abs(S.oy) < 16.0:
                p_need = min(p_need, self.BOX_SHOT_P)
            if (self.FAR_SHOT_D and math.hypot(ph.hl - S.ox, S.oy) > self.FAR_SHOT_D
                    and self._pressure(S)[0] >= self.HOLD_BAN_D):
                p_need = max(p_need, self.FAR_SHOT_P)
            if pg[i] * shot_w >= p_need:
                carry = self._carry_in(agent, S, float(pg[i]))
                if carry is not None:
                    note(agent, "carry_in", pg=round(float(pg[i]), 2), pg2=round(carry[2], 2))
                    self._execute(agent, S, (carry[2], "dribble", (carry[0], carry[1])), None)
                    return True
                shot = (float(sv[i]), kind, (float(gy[i]), float(pg[i])))
                if (not self.SHOT_COMPARE or pg[i] * shot_w >= self.SHOT_SURE
                        or self._pressure(S)[0] < self.HOLD_BAN_D):
                    self._execute(agent, S, shot, None)
                    return True
                opts.append((self.SHOT_V0_US + self.SHOT_V1 * float(pg[i]) * shot_w,
                             kind, shot[2]))
        if self.EXP_BEAM == 2:
            return False

        t0 = time.perf_counter()
        self._drib_wins = 0
        pressed = self._pressure(S)[0] < self.PRESS_D
        tx, ty, s0 = self._pass_targets(S)
        res = None
        if tx.size:
            delay = self._kick_delay(agent, S, tx, ty, s0)
            res = self._eval_kicks(S, tx, ty, s0, delay, k_multi=self._multi_keep(S))
            safe, base = self._chain_scores(S, res, pressed)
            score = base.copy()
        goal_ang = math.degrees(math.atan2(-S.oy, ph.hl - S.ox))
        angs = np.concatenate([np.arange(-157.5, 180.0, 22.5), [goal_ang], self._slot_angles(S),
                                self._evasive_angles(S)])
        long = S.ox > 0.0 or (S.ox > self.BACK_X and self._open_ahead(S, self.OPEN_D))
        dists = self.CHAIN_DRIB + ((15.0,) if long else ())
        drib = self._drib_options(S, dists, angs, pressed)
        level = 0
        cost = (self._step_cost(S, res, base) if res is not None and self.FIX_CHAIN_RISK
                else np.zeros(res.tx.size) if res is not None else None)
        dcost = [float(self._field_eval(d[1], d[2])) - d[0] if self.FIX_CHAIN_RISK else 0.0
                 for d in drib]
        if self.BEAM:
            level = self._beam_chain(S, res, safe if res is not None else None,
                                     base if res is not None else None, cost,
                                     score if res is not None else None, drib, dcost, long,
                                     deadline)
        for n_lv, (depth, k, light) in enumerate(() if self.BEAM else self.CHAIN_PLAN, 1):
            if time.perf_counter() > deadline:
                break
            level = n_lv
            if res is not None:
                for i in np.argsort(-score)[:k]:
                    if not safe[i] or time.perf_counter() > deadline:
                        break
                    child = self._child(S, res, i)
                    score[i] = max(score[i], self._chain_value(child, depth, deadline, light=light)
                                   - cost[i])
            if long and depth <= self.DRIB_DEPTH:
                for n_d, d in enumerate(drib[:self.DRIB_K]):
                    if time.perf_counter() > deadline:
                        break
                    child = S.child(S.kicker, d[1], d[2], d[3], body=d[4])
                    d[0] = max(d[0], self._chain_value(child, depth, deadline, drib=False,
                                                       light=light) - dcost[n_d])
        if res is not None:
            j = int(np.argmax(score))
            if np.isfinite(score[j]):
                opts.append((float(score[j]), "pass", j))
        for score_d, ex, ey, _, _ in drib:
            opts.append((score_d, "dribble", (ex, ey)))

        T = 4
        B = np.tile(np.array([S.ox, S.oy]), (1, T + 1, 1))
        first = _race(ph, B, S.op_pos, S.op_body, np.full(S.nop, ph.kick_r + 0.5),
                      S.op_gk, np.zeros(S.nop, dtype=np.int64))
        t_th = float(first.min()) if S.nop else NEVER
        if t_th > T:
            opts.append((float(self._field_eval(S.ox, S.oy)) - 1.5, "hold", None))

        if not opts:
            return False
        best = max(opts, key=lambda o: o[0])
        if (self.CHAIN_FALLBACK and S.ox > 0.0
                and best[0] < float(self._field_eval(S.ox, S.oy)) - 12.0):
            note(agent, "chain_fallback", v=round(best[0], 1))
            return False
        note(agent, "chain", k=best[1], dw=self._drib_wins, lv=level,
             ms=round(1000.0 * (time.perf_counter() - t0), 1),
             cut=int(time.perf_counter() > deadline))
        self._execute(agent, S, best, res)
        return True

    def _beam_chain(self, S, res, safe, base, cost, score, drib, dcost, long, deadline):
        """Seed the chain beam with the safe root passes and dribbles; values
        found deeper are written back into `score` / `drib` of their root."""
        nodes = []
        if res is not None:
            for i in np.flatnonzero(safe):
                nodes.append((float(base[i]), float(cost[i]), int(i),
                              lambda i=i: self._child(S, res, i), self.BEAM_CARRIES,
                              (-1, int(res.who[i]))))
        if long or self.BEAM_ROOT_DRIB:
            for n_d, d in enumerate(drib[:self.DRIB_K]):
                nodes.append((d[0], dcost[n_d], -1 - n_d,
                              lambda d=d: S.child(S.kicker, d[1], d[2], d[3], body=d[4]),
                              self.BEAM_CARRIES - 1,
                              (-1, -1 - n_d) if self.BEAM_ROOT_DRIB else (-1, -1)))
        def raise_root(root, v):
            if root >= 0:
                score[root] = max(score[root], v)
            else:
                drib[-1 - root][0] = max(drib[-1 - root][0], v)

        return self._beam_run(nodes, self._beam_expand, raise_root, self.BEAM_W, self.BEAM_D,
                              deadline)

    def _beam_expand(self, C, g, light, may_drib, b):
        ph = self.ph
        out = []
        if C.ox > 20.0:
            _, pg, _ = self._eval_shots(C, ph.bmax, 1)
            if pg.max() >= self.SHOT_P:
                out.append((self.SHOT_V0_US + self.SHOT_V1 * float(pg.max()) - g, 0.0, None,
                            False, None))
        pressed = self._pressure(C)[0] < self.PRESS_D
        tx, ty, s0 = self._pass_targets(C, rings=() if light else self.CHILD_RINGS,
                                        speeds=self.CHILD_SPEEDS, space=not light)
        if tx.size:
            r = self._eval_kicks(C, tx, ty, s0, np.ones(tx.size, dtype=np.int64))
            safe, sc = self._chain_scores(C, r, pressed)
            if safe.any():
                step = (self._step_cost(C, r, sc) if self.FIX_CHAIN_RISK
                        else np.zeros(sc.size))
                for j in np.flatnonzero(safe):
                    out.append((float(sc[j]) - self.HOP_COST - g,
                                g + self.HOP_COST + float(step[j]),
                                lambda j=j: self._child(C, r, j), self.BEAM_CARRIES,
                                (b, C.recv[int(r.who[j])][0])))
        if may_drib > 0:
            goal_ang = math.degrees(math.atan2(-C.oy, ph.hl - C.ox))
            angs = goal_ang + np.array([-30.0, 0.0, 30.0])
            if not light:
                angs = np.concatenate([angs, self._slot_angles(C), self._evasive_angles(C)[:3]])
            for d in self._drib_options(C, self.CHILD_DRIB, angs, pressed)[:self.CHILD_DRIB_K]:
                step = float(self._field_eval(d[1], d[2])) - d[0] if self.FIX_CHAIN_RISK else 0.0
                out.append((d[0] - 1.0 - g, g + 1.0 + step,
                            lambda d=d: C.child(C.kicker, d[1], d[2], d[3], body=d[4]),
                            may_drib - 1, (b, -1)))
        return out

    def _beam_run(self, nodes, expand, raise_root, width, depth, deadline):
        """Generic anytime beam search.

        nodes: (value, accumulator, root index, make_state, dribbles left,
        diversity key). Each layer expands every kept node with `expand`,
        reports each child's value to its root action via `raise_root`
        (a root's value is the best line found below it), and keeps the
        best `width` children. Stops after `depth` layers or at `deadline`.
        Returns the number of layers completed.
        """
        beam = self._beam_keep(nodes, width)
        layer = 0
        while beam and layer < depth and time.perf_counter() < deadline:
            layer += 1
            light = self.DEEP_LIGHT and layer > 1
            nxt = []
            for b, (_, g, root, make, may_drib, _) in enumerate(beam):
                if time.perf_counter() > deadline:
                    break
                for v, g2, make2, may2, key in expand(make(), g, light, may_drib, b):
                    raise_root(root, v)
                    if make2 is not None:
                        nxt.append((v, g2, root, make2, may2, key))
            beam = self._beam_keep(nxt, width)
        return layer

    def _beam_keep(self, nodes, width):
        """Best `width` nodes, at most BEAM_DIV per (parent, receiver) key so
        the beam does not fill up with near-identical passes."""
        nodes.sort(key=lambda n: -n[0])
        kept, per = [], {}
        for n in nodes:
            if len(kept) >= width:
                break
            if self.BEAM_DIV:
                c = per.get(n[5], 0)
                if c >= self.BEAM_DIV:
                    continue
                per[n[5]] = c + 1
            kept.append(n)
        return kept

    def _beam_exp(self, S, res, values, drib, deadline):
        """Expectimax beam search from the root (the search Agent1 uses).

        Root nodes are every feasible pass in `res` and every dribble in
        `drib`; each carries its accumulator (A, Q) = ((1 - p) * V(loss),
        p * GAMMA^t). Deeper values found by `_beam_expand_exp` raise the
        root's value in place (`values` for passes). Returns the dribble
        options with their backed-up values.
        """
        ph = self.ph
        nodes = []
        if values is not None:
            for i in np.flatnonzero(np.isfinite(values) & res.succ & (res.p > 0.0)):
                p = float(res.p[i])
                nodes.append((float(values[i]),
                              ((1.0 - p) * float(res.v_fail[i]), p * GAMMA ** int(res.t_us[i])),
                              int(i), lambda i=i: self._child(S, res, i), True,
                              (-1, int(res.who[i]))))
        dv = [o[0] for o in drib]
        for k, (_, _, (ex, ey, p, K, lx, ly)) in enumerate(drib):
            nodes.append((dv[k], ((1.0 - p) * float(ph.v_lose(lx, ly)), p * GAMMA ** K), -1 - k,
                          lambda ex=ex, ey=ey, K=K: S.child(
                              S.kicker, ex, ey, K,
                              body=math.degrees(math.atan2(ey - S.oy, ex - S.ox))),
                          False, (-1, -1)))

        def raise_root(root, v):
            if root >= 0:
                values[root] = max(values[root], v)
            else:
                dv[-1 - root] = max(dv[-1 - root], v)

        self._beam_run(nodes, self._beam_expand_exp, raise_root, self.EXP_W, self.EXP_D, deadline)
        return [(dv[k], "dribble", o[2]) for k, o in enumerate(drib)]

    def _beam_expand_exp(self, C, g, light, may_drib, b):
        """Children of state C for the expectimax beam.

        With g = (A, Q), each option at C is valued A + Q' * V where Q' folds
        in EXP_DEEP, the option's success probability and its discount; a
        shot is a leaf. Passes and dribbles both expand further, but a
        dribble may not directly follow another dribble (`may_drib`).
        """
        ph = self.ph
        A, Q = g
        q = Q * self.EXP_DEEP
        out = []
        if C.ox > 12.0:
            sv, _, _ = self._eval_shots(C, ph.bmax, 1)
            out.append((A + Q * self.EXP_SHOT * float(sv.max()), None, None, False, None))
        tx, ty, s0 = self._pass_targets(C, rings=() if light else self.CHILD_RINGS,
                                        speeds=self.CHILD_SPEEDS, space=not light)
        if tx.size:
            r = self._eval_kicks(C, tx, ty, s0, np.ones(tx.size, dtype=np.int64))
            for j in np.flatnonzero(r.succ & (r.p > 0.0)):
                A2 = A + q * (1.0 - float(r.p[j])) * float(r.v_fail[j])
                Q2 = q * float(r.p[j]) * GAMMA ** int(r.t_us[j])
                out.append((A2 + Q2 * float(r.vs[j]), (A2, Q2),
                            lambda j=j: self._child(C, r, j), True,
                            (b, C.recv[int(r.who[j])][0])))
        if may_drib:
            goal_ang = math.degrees(math.atan2(-C.oy, ph.hl - C.ox))
            angs = goal_ang + np.array([-30.0, 0.0, 30.0])
            if not light:
                angs = np.concatenate([angs, self._slot_angles(C), self._evasive_angles(C)[:3]])
            carries = []
            for p, ex, ey, K, lx, ly in self._dribble_cands(C, self.CHILD_DRIB, angs):
                A2 = A + q * (1.0 - p) * float(ph.v_lose(lx, ly))
                Q2 = q * p * GAMMA ** K
                carries.append((A2 + Q2 * float(ph.v_keep(ex, ey, C.space(ex, ey))), (A2, Q2),
                                lambda ex=ex, ey=ey, K=K: C.child(
                                    C.kicker, ex, ey, K,
                                    body=math.degrees(math.atan2(ey - C.oy, ex - C.ox))),
                                False, (b, -1)))
            carries.sort(key=lambda c: -c[0])
            out.extend(carries[:self.CHILD_DRIB_K])
        return out

    def _carry_in(self, agent, S, pg):
        """A short, safe dribble towards goal that improves on a shot with
        goal probability pg by at least 0.05, or None."""
        ph = self.ph
        if pg >= self.CARRY_IN_P or S.ox > ph.hl - 12.0 or S.nop == 0:
            return None
        if self._pressure(S)[0] < self.CARRY_IN_D:
            return None
        base = np.array([math.degrees(math.atan2(gy - S.oy, ph.hl - 2.0 - S.ox))
                         for gy in (-4.0, 0.0, 4.0)])
        best = None
        for p, ex, ey, K, _, _ in self._dribble_cands(S, (3.0, 6.0), base):
            if p < 0.9:
                continue
            _, pg2, _ = self._eval_shots(S.child(S.kicker, ex, ey, K), ph.bmax, 1)
            v = p * float(pg2.max())
            if v > pg + 0.05 and (best is None or v > best[2]):
                best = (ex, ey, v)
        return best

    def _child(self, S, res, i):
        """The state after candidate kick i of `res` is received."""
        recv = S.recv[int(res.who[i])][0]
        return S.child(recv, float(res.lx[i]), float(res.ly[i]), int(res.t_us[i]))

    def _chain_value(self, S, depth, deadline, drib=True, light=False, shoot=True):
        """Depth-limited chain value of state S (field-evaluation units).

        Used to look one step into the opponents' follow-ups in defence
        (always with light=True). The dribble branch below refers to an
        undefined name `C` when light=False; the current settings never get
        there (BEAM is on, so the non-beam chain loop never runs).
        """
        best = -np.inf
        if shoot and S.ox > 20.0:
            _, pg, _ = self._eval_shots(S, self.ph.bmax, 1)
            if pg.max() >= self.SHOT_P:
                best = self.SHOT_V0 + self.SHOT_V1 * float(pg.max())
        if depth <= 0 or time.perf_counter() > deadline:
            return best
        pressed = self._pressure(S)[0] < self.PRESS_D
        tx, ty, s0 = self._pass_targets(S, rings=() if light else self.CHILD_RINGS,
                                        speeds=self.CHILD_SPEEDS, space=not light)
        if tx.size:
            r = self._eval_kicks(S, tx, ty, s0, np.ones(tx.size, dtype=np.int64))
            safe, sc = self._chain_scores(S, r, pressed)
            if safe.any():
                if depth > 1:
                    cost = (self._step_cost(S, r, sc) if self.FIX_CHAIN_RISK
                            else np.zeros(sc.size))
                    for j in np.argsort(-sc)[:self.CHILD_K]:
                        if not safe[j] or time.perf_counter() > deadline:
                            break
                        sc[j] = max(sc[j], self._chain_value(self._child(S, r, j), depth - 1,
                                                             deadline,
                                                             light=light or self.DEEP_LIGHT)
                                    - cost[j])
                best = max(best, float(sc.max()) - 2.0)
        if drib:
            goal_ang = math.degrees(math.atan2(-S.oy, self.ph.hl - S.ox))
            angs = goal_ang + np.array([-30.0, 0.0, 30.0])
            if not light:
                angs = np.concatenate([angs, self._slot_angles(S), self._evasive_angles(C)[:3]])
            for d in self._drib_options(S, self.CHILD_DRIB, angs, pressed)[:self.CHILD_DRIB_K]:
                if time.perf_counter() > deadline:
                    break
                child = S.child(S.kicker, d[1], d[2], d[3], body=d[4])
                step = float(self._field_eval(d[1], d[2])) - d[0] if self.FIX_CHAIN_RISK else 0.0
                v = max(d[0], self._chain_value(child, depth, deadline, drib=False,
                                                light=light) - step) - 1.0
                if v > best:
                    best = v
                    self._drib_wins += 1
        return best


# ==========================================================================
# SECTION: ON-BALL ACTIONS MIXIN (agent/on_ball.py)
# ==========================================================================


class OnBallMixin:
    """The on_ball entry point, shared context, and action execution."""

    def _phys(self, agent):
        """The _Phys tables, built once from the server parameters."""
        if self.ph is None:
            self.ph = _Phys(agent.server_params)
        return self.ph

    def _allow_ms(self, agent, on_ball):
        """Milliseconds this decision may take: a share of the remaining CPU
        budget spread over the remaining cycles (on-ball decisions get
        ONBALL_SHARE times the average, capped), never eating into the
        harness's per-decision limit."""
        rem = agent.budget.remaining_ms
        if rem == float("inf"):
            rem = 60000.0
        left = max(60, MATCH_CYCLES - agent.wm.cycle)
        per = rem / left
        cap = self.ONBALL_CAP_MS if on_ball else 30.0
        allow = min(per * (self.ONBALL_SHARE if on_ball else 0.8), cap,
                    agent.budget.decision_remaining_ms - 280.0)
        return max(allow, 1.5)

    def _context(self, agent):
        """Shared picture of this cycle, computed once and cached.

        Races the ball's current path against all 10 players and derives:
        the chaser (our first arrival, with stickiness to last cycle's
        chaser; none if our keeper will catch it), first arrival cycles for
        each side and where, whether we are attacking (we reach the ball
        clearly first), and whether it is a counter-attack. Every player
        computes the same thing from the same world model, which is how the
        team coordinates without communicating.
        """
        wm = agent.wm
        key = (wm.cycle, wm.ball.position.x(), wm.ball.position.y())
        if self._ctx is not None and self._ctx["key"] == key:
            return self._ctx
        ph = self._phys(agent)

        me = wm.self
        mates, goalie = [], None
        for p in wm.teammates:
            if p.uniform_number <= 0:
                continue
            row = (p.uniform_number, p.position.x(), p.position.y(),
                   p.body_direction, p.velocity.x(), p.velocity.y())
            if p.is_goalie:
                goalie = row
            else:
                mates.append(row)
        if not me.is_goalie:
            mates.append((me.uniform_number, me.position.x(), me.position.y(),
                          me.body_direction, me.velocity.x(), me.velocity.y()))
        mates.sort(key=lambda r: r[0])
        opps = []
        for p in wm.opponents:
            if p.uniform_number <= 0:
                continue
            opps.append((p.uniform_number, p.position.x(), p.position.y(),
                         p.body_direction, p.is_goalie,
                         p.velocity.x(), p.velocity.y()))
        opps.sort(key=lambda r: r[0])

        bx, by = wm.ball.position.x(), wm.ball.position.y()
        bvx, bvy = wm.ball.velocity.x(), wm.ball.velocity.y()

        rows = mates + ([goalie] if goalie else [])
        P = [[r[1] + r[4] * 0.6, r[2] + r[5] * 0.6] for r in rows]
        P += [[o[1] + o[5] * 0.6, o[2] + o[6] * 0.6] for o in opps]
        body = [r[3] for r in rows] + [o[3] for o in opps]
        gk = [0] * len(mates) + ([-1] if goalie else []) + [1 if o[4] else 0 for o in opps]
        ctrl = [ph.kick_r] * (len(rows) + len(opps))
        T = HORIZON
        t = np.arange(T + 1)
        B = np.stack([bx + bvx * ph.G[t], by + bvy * ph.G[t]], axis=1)[None]
        first = _race(ph, B, np.array(P, dtype=float).reshape(-1, 2),
                      np.array(body, dtype=float), np.array(ctrl, dtype=float),
                      np.array(gk, dtype=np.int64),
                      np.zeros(len(P), dtype=np.int64),
                      tired=self._race_budgets(ph, mates, goalie, opps))[0]
        nm = len(mates)
        mate_t = first[:nm]
        gk_t = first[nm] if goalie else NEVER
        opp_t = first[len(rows):]
        opp_min = int(opp_t.min()) if opp_t.size else NEVER

        chaser = None
        if nm:
            if self.IMP_CHASER:
                score = []
                for i in range(nm):
                    ix, iy = B[0, min(int(mate_t[i]), T)]
                    dx, dy = ix - P[i][0], iy - P[i][1]
                    d = math.hypot(dx, dy)
                    sp = math.hypot(mates[i][4], mates[i][5])
                    align = ((mates[i][4] * dx + mates[i][5] * dy) / (sp * d)
                             if sp > 0.2 and d > 0.1 else 0.0)
                    stick = 0.35 if mates[i][0] == self._prev_chaser else 0.0
                    score.append((mate_t[i] + 0.01 * d - 0.3 * max(align, 0.0) - stick,
                                  mates[i][0]))
            else:
                score = [(mate_t[i] - (0.5 if mates[i][0] == self._prev_chaser else 0.0),
                          mates[i][0]) for i in range(nm)]
            best = min(score)
            chaser = best[1]
            best_t = int(mate_t[[m[0] for m in mates].index(chaser)])
            if gk_t < best_t:
                gi = min(int(gk_t), T)
                behind = self.FIX_GK_BACK and B[0, gi, 0] < goalie[1] - 0.5
                if B[0, gi, 0] < -ph.pen_x and abs(B[0, gi, 1]) < ph.pen_hw and not behind:
                    chaser = None
        self._prev_chaser = chaser
        our_min = min(int(mate_t.min()) if nm else NEVER, int(gk_t))

        oi = min(opp_min, T)
        opp_first = None
        if opp_t.size and opp_min < NEVER:
            opp_first = opps[int(opp_t.argmin())][0]
        ui = min(our_min, T)

        attack = (bool(wm.kickable_teammate_existance)
                  or (not wm.kickable_opponent_existance and nm > 0
                      and int(mate_t.min()) + (1 if self.ATTACK_MARGIN1 and bx > 0.0
                                               else 2 if bx > -25.0 else 5) <= opp_min))
        opps_ahead = sum(1 for o in opps if not o[4] and o[1] > bx + 2.0)
        counter = attack and (self._opp_poss > 3 or opps_ahead <= 1)

        self._ctx = {
            "key": key, "mates": mates, "goalie": goalie, "opps": opps,
            "mate_t": {mates[i][0]: int(mate_t[i]) for i in range(nm)},
            "ball": (bx, by, bvx, bvy), "chaser": chaser,
            "our_min": our_min, "opp_min": opp_min, "opp_first": opp_first,
            "our_loc": (float(B[0, ui, 0]), float(B[0, ui, 1])),
            "opp_loc": (float(B[0, oi, 0]), float(B[0, oi, 1])),
            "attack": attack,
            "counter": counter,
            "offside_x": wm.offside_line_x,
            "kickable_mate": bool(wm.kickable_teammate_existance),
            "kickable_opp": bool(wm.kickable_opponent_existance),
        }
        if attack:
            self._opp_poss = 0
        elif wm.cycle != self._opp_poss_cycle:
            self._opp_poss += 1
        self._opp_poss_cycle = wm.cycle
        self._ctx["midblock"] = (self.MIDBLOCK and not attack
                                 and self._opp_poss > self.MB_T and bx > self.MB_X)
        return self._ctx

    def _root_state(self, agent, ctx):
        """The search root: the current world as a _State."""
        wm = agent.wm
        mates = [(m[0], m[1], m[2], m[3]) for m in ctx["mates"]]
        opps = [(o[0], o[1] + o[5] * 0.6, o[2] + o[6] * 0.6, o[3], o[4])
                for o in ctx["opps"]]
        bx, by = ctx["ball"][0], ctx["ball"][1]
        S = _State(self.ph, bx, by, wm.self.uniform_number, mates, opps,
                   ctx["offside_x"], opp_vel=[(o[5], o[6]) for o in ctx["opps"]],
                   mate_vel={m[0]: (m[4], m[5]) for m in ctx["mates"]}, lead=0.6,
                   tired_map=self._budget if self.STAM_RACE else None)
        S.adv = self.CHILD_ADV * (1.25 if ctx.get("counter") else 1.0)
        g = ctx.get("goalie")
        if g:
            S.own_gk = (g[1], g[2])
        return S

    def _risk(self, agent):
        """(pass margin m0, shot weight) for the score situation: after cycle
        1150 play safer when leading, riskier and shoot more when behind."""
        wm = agent.wm
        lead = wm.our_team_score - wm.their_team_score
        late = wm.cycle > 1150
        if late and lead > 0:
            return self.PASS_M0 + 0.35, 1.0
        if late and lead < 0:
            return self.PASS_M0 - 0.20, 1.12
        return self.PASS_M0, 1.0

    def on_ball(self, agent) -> None:
        """Framework entry point when this player can kick the ball. Any
        exception falls back to a clearance instead of crashing."""
        try:
            self._stam_cycle_start(agent, True)
            self._on_ball(agent)
        except Exception as e:
            note(agent, "on_ball_error", err=str(e)[:120])
            Action.clear(agent)

    def _on_ball(self, agent):
        """The on-ball decision (see the module docstring for the order)."""
        wm = agent.wm
        ph = self._phys(agent)
        if wm.self.is_goalie:
            Action.clear(agent)
            return
        start = time.perf_counter()
        deadline = start + self._allow_ms(agent, True) / 1000.0
        ctx = self._context(agent)
        S = self._root_state(agent, ctx)
        m0, shot_w = self._risk(agent)

        c = self._commit
        if (c is not None and self.FIX_COMMIT and wm.cycle - c["cycle"] == 1
                and self._pressure(S)[0] < 1.5):
            ang = math.degrees(math.atan2(c["ty"] - S.oy, c["tx"] - S.ox))
            if not self._cone_mask(S, [ang])[0]:
                note(agent, "abort_" + c["kind"])
                c = None
        if (c is not None and self.FIX_LANE and c["kind"] == "pass"
                and wm.cycle - c["cycle"] == 1):
            D = math.hypot(c["tx"] - S.ox, c["ty"] - S.oy)
            n_left = max(c["arrive"] - 1, 1)
            s0 = min(ph.bmax, D * (1.0 - ph.decay) / (1.0 - ph.decay ** n_left))
            r = self._eval_kicks(S, np.array([c["tx"]]), np.array([c["ty"]]),
                                 np.array([s0]), np.zeros(1, dtype=np.int64))
            if not r.succ[0] or r.p[0] < 0.55:
                note(agent, "abort_lane", p=round(float(r.p[0]), 2))
                c = None
        if (c is not None and wm.cycle - c["cycle"] == 1 and c["age"] < 2
                and wm.last_kicker_uniform_number == wm.self.uniform_number):
            c["cycle"], c["age"] = wm.cycle, c["age"] + 1
            note(agent, "continue_" + c["kind"])
            Action.kick(agent, Vector2D(c["tx"], c["ty"]), c["arrive"],
                        accept_late=self.SHOT_KICK_LATE if c["kind"] == "shot" else self.KICK_LATE)
            if self.FIX_CONT_SHOT and c["kind"] == "shot":
                Action.kick_one_step(agent, Vector2D(c["tx"], c["ty"]))
            Action.hold(agent)
            return
        self._commit = None

        g = ctx["goalie"]
        if (self.IMP_KEEPER_RESET and g is not None and S.ox < -25.0
                and g[1] > -ph.hl + 8.0 and self._pressure(S)[0] < 3.0):
            note(agent, "keeper_reset", gx=round(g[1], 1))
            side = 1.0 if S.oy >= 0.0 else -1.0
            Action.kick_one_step(agent, Vector2D(S.ox + 8.0, side * (ph.hw + 5.0)))
            Action.clear(agent)
            return

        chain_deadline = min(deadline, start + self.CHAIN_MS / 1000.0)
        if self._on_ball_chain(agent, S, chain_deadline, shot_w):
            return

        options = []
        d_opp, t_opp = self._pressure(S)
        heavy = d_opp < self.PRESS_D

        tx, ty, s0 = self._pass_targets(S)
        res = None
        if tx.size:
            delay = self._kick_delay(agent, S, tx, ty, s0)
            res = self._eval_kicks(S, tx, ty, s0, delay, m0=m0, k_multi=self._multi_keep(S))

        bx, by = S.ox, S.oy
        if math.hypot(ph.hl - bx, by) < 30.0:
            best_shot = None
            v1 = self._vmax(agent, math.degrees(math.atan2(-by, ph.hl - bx)))
            v1s = self._shot1_speeds(agent, S) if self.PER_AIM_V1 else None
            if (v1s.max() if v1s is not None else v1) >= 1.8:
                sv, pg, gy = self._eval_shots(S, v1s if v1s is not None else min(v1, ph.bmax), 0)
                i = int(np.argmax(sv))
                best_shot = (float(sv[i]) * shot_w, "shot1", (float(gy[i]), float(pg[i])))
            sv, pg, gy = self._eval_shots(S, ph.bmax, 1 if v1 > 1.2 else 2)
            i = int(np.argmax(sv))
            cand = (float(sv[i]) * shot_w, "shot", (float(gy[i]), float(pg[i])))
            if best_shot is None or cand[0] > best_shot[0]:
                best_shot = cand
            min_pg = 0.22 if (heavy and bx > 35.0) else (0.55 if math.hypot(ph.hl - bx, by) > 20.0 else 0.38)
            if best_shot[2][1] >= min_pg:
                options.append(best_shot)

        ang1 = np.arange(-105.0, 106.0, 15.0)
        v1 = np.array([min(self._vmax(agent, a), ph.bmax) for a in ang1])
        ok = v1 > 0.8
        if heavy:
            ok &= self._cone_mask(S, ang1)
        if ok.any():
            ra, s1 = np.radians(ang1[ok]), v1[ok]
            tx1, ty1 = S.ox + 10.0 * np.cos(ra), S.oy + 10.0 * np.sin(ra)
            r1 = self._eval_kicks(S, tx1, ty1, s1, np.zeros(s1.size, dtype=np.int64), m0=m0)
            v = np.where(self._live(r1), r1.value, -np.inf) if heavy else r1.value
            j = int(np.argmax(v))
            if np.isfinite(v[j]):
                options.append((float(v[j]), "pass1", (r1, j)))

        drib = self._eval_dribbles(agent, S)
        if heavy and drib:
            dang = [math.degrees(math.atan2(o[2][1] - S.oy, o[2][0] - S.ox)) for o in drib]
            clear = self._cone_mask(S, dang)
            drib = [o for o, c in zip(drib, clear) if c and o[2][2] >= 0.55]
        options.extend(drib)

        if d_opp >= self.HOLD_BAN_D and t_opp > 2:
            options.append(self._eval_hold(agent, S))
        if S.ox < self.CLEAR_X:
            options.append(self._eval_clear(S))

        if self.EXP_BEAM:
            values = None
            if res is not None:
                values = res.value.copy()
                if heavy:
                    values = np.where(self._live(res), values, -np.inf)
            options = [o for o in options if o[1] != "dribble"]
            options.extend(self._beam_exp(S, res, values, drib, deadline))
            if values is not None:
                i = int(np.argmax(values))
                if np.isfinite(values[i]):
                    options.append((float(values[i]), "pass", i))
        elif res is not None:
            values = res.value.copy()
            if heavy:
                values = np.where(self._live(res), values, -np.inf)
            order = np.argsort(-values)
            for depth in (2, 3):
                if time.perf_counter() > deadline:
                    break
                k = 10 if depth == 2 else 4
                for i in order[:k]:
                    if time.perf_counter() > deadline:
                        break
                    if not np.isfinite(values[i]) or not res.succ[i]:
                        continue
                    v2 = self._deep_value(S, res, i, depth, deadline)
                    values[i] = (res.p[i] * GAMMA ** res.t_us[i] * v2
                                 + (1.0 - res.p[i]) * res.v_fail[i])
                order = np.argsort(-values)
            i = int(order[0])
            if np.isfinite(values[i]):
                options.append((float(values[i]), "pass", i))

        cornered = not options
        if cornered:
            last = [o for o in (self._channel_kick(agent, S), self._bounce_pass(agent, S))
                    if o is not None]
            away = self._evasive_angles(S)[:1]
            if away.size:
                base = away[0] + np.array([0.0, 30.0, -30.0])
                for p, ex, ey, K, lx, ly in self._dribble_cands(S, (2.5,), base):
                    v = (p * GAMMA ** K * float(ph.v_keep(ex, ey, S.space(ex, ey)))
                         + (1.0 - p) * float(ph.v_lose(lx, ly)))
                    last.append((v, "dribble", (ex, ey, p)))
            options.extend(last or [(-1.0, "clear", None)])

        tops = {}
        for o in options:
            tops[o[1]] = max(tops.get(o[1], -9.0), round(o[0], 3))
        note(agent, "fb", d=round(d_opp, 1), last=int(cornered), **tops)
        best = max(options, key=lambda o: o[0])
        self._execute(agent, S, best, res)

    def _pressure(self, S):
        """(distance to the nearest opponent, cycles until one can reach the ball)."""
        if S.nop == 0:
            return 99.0, NEVER
        d = float(np.hypot(S.op_pos[:, 0] - S.ox, S.op_pos[:, 1] - S.oy).min())
        T = 4
        B = np.tile(np.array([S.ox, S.oy]), (1, T + 1, 1))
        first = _race(S.ph, B, S.op_pos, S.op_body, np.full(S.nop, S.ph.kick_r + 0.5),
                      S.op_gk, np.zeros(S.nop, dtype=np.int64))
        return d, int(first.min())

    def _multi_keep(self, S):
        """Success multiplier for two-kick passes, shrinking as an opponent
        gets closer (he may steal the ball between the kicks)."""
        if not self.MULTI_PRESS:
            return 1.0
        d = self._pressure(S)[0]
        return (0.32 if d < 1.8 else 0.48 if d < 2.2 else 0.72 if d < 2.6
                else 0.87 if d < 3.5 else 0.94 if d < 4.0 else 1.0)

    @staticmethod
    def _live(r):
        return r.succ & (r.t_th > r.t_us)

    def _channel_kick(self, agent, S):
        """Last resort: a one-kick ball 20 m down a wing channel."""
        ph = S.ph
        angs = []
        for s in (1.0, -1.0):
            ac = math.degrees(math.atan2(s * ph.hw - S.oy, ph.hl - S.ox))
            angs += [ac - 30.0, ac - 15.0, ac, ac + 15.0, ac + 30.0,
                     s * 35.0, s * 50.0, s * 65.0, s * 80.0]
        angs = np.array(angs)
        v = np.array([min(self._vmax(agent, a), ph.bmax) for a in angs])
        ok = (v >= self.CHANNEL_V) & self._cone_mask(S, angs)
        if not ok.any():
            return None
        ra, s0 = np.radians(angs[ok]), v[ok]
        tx, ty = S.ox + 20.0 * np.cos(ra), S.oy + 20.0 * np.sin(ra)
        r = self._eval_kicks(S, tx, ty, s0, np.zeros(s0.size, dtype=np.int64))
        j = int(np.argmax(r.value))
        return (float(r.value[j]), "channel", (r, j))

    def _cone_mask(self, S, angs):
        """Which kick directions are not blocked by an opponent standing
        close (within 3 m and CONE degrees) to the ball."""
        angs = np.asarray(angs, dtype=float)
        if S.nop == 0:
            return np.ones(angs.size, dtype=bool)
        d = np.hypot(S.op_pos[:, 0] - S.ox, S.op_pos[:, 1] - S.oy)
        a = np.degrees(np.arctan2(S.op_pos[:, 1] - S.oy, S.op_pos[:, 0] - S.ox))
        off = np.abs(_norm_deg(angs[:, None] - a[None, :]))
        ok = ((off > self.CONE) | (d[None, :] >= 3.0)).all(axis=1)
        wide = ok & ((off > 90.0) | (d[None, :] >= S.ph.kick_r + 0.3)).all(axis=1)
        return wide if wide.any() else ok

    def _open_ahead(self, S, dist):
        if S.nop == 0:
            return True
        dx, dy = S.op_pos[:, 0] - S.ox, S.op_pos[:, 1] - S.oy
        d = np.hypot(dx, dy)
        ahead = np.abs(np.degrees(np.arctan2(dy, dx))) < 45.0
        return not bool(np.any(ahead & (d < dist)))

    def _bounce_pass(self, agent, S):
        """Last resort: a one-kick pass straight to a teammate 5-11 m away."""
        ph = S.ph
        tx, ty = [], []
        for m in S.recv:
            if 5.0 <= math.hypot(m[1] - S.ox, m[2] - S.oy) <= 11.0:
                tx.append(m[1])
                ty.append(m[2])
        if not tx:
            return None
        tx, ty = np.array(tx), np.array(ty)
        angs = np.degrees(np.arctan2(ty - S.oy, tx - S.ox))
        v = np.array([min(self._vmax(agent, a), ph.bmax) for a in angs])
        ok = (v >= 1.0) & self._cone_mask(S, angs)
        if not ok.any():
            return None
        r = self._eval_kicks(S, tx[ok], ty[ok], v[ok], np.zeros(int(ok.sum()), dtype=np.int64))
        j = int(np.argmax(r.value))
        return (float(r.value[j]), "pass1", (r, j))

    def _slot_angles(self, S):
        if S.ox >= self.SLOT_X - 3.0:
            return []
        return [math.degrees(math.atan2(sy - S.oy, self.SLOT_X - S.ox)) for sy in (-7.0, 7.0)]

    def _drib_options(self, S, dists, angs, pressed):
        """Dribbles that clear the success threshold, scored for the chain
        search, best first: [score, x, y, cycles, direction]."""
        attack = S.ox > 0.0
        p_min = self.P_DRIB_ATT if attack else self.P_DRIB
        out = []
        for p, ex, ey, K, _, _ in self._dribble_cands(S, dists, angs):
            if p < p_min:
                continue
            score = float(self._field_eval(ex, ey)) - 0.05 * K
            if attack:
                score -= self.LOSS_ATT * (1.0 - p)
            if S.ox > self.BACK_X and not pressed:
                score -= self.BACK_W * max(0.0, S.ox - 4.0 - ex)
            out.append([score, ex, ey, K, math.degrees(math.atan2(ey - S.oy, ex - S.ox))])
        out.sort(key=lambda d: -d[0])
        return out

    def _evasive_angles(self, S):
        if S.nop == 0:
            return np.zeros(0)
        d = np.hypot(S.op_pos[:, 0] - S.ox, S.op_pos[:, 1] - S.oy)
        k = int(np.argmin(d))
        away = math.degrees(math.atan2(S.oy - S.op_pos[k, 1], S.ox - S.op_pos[k, 0]))
        return away + np.array([0.0, 45.0, -45.0, 90.0, -90.0])

    def _dribble_cands(self, S, dists, base=None):
        """Simulate dribbles of each length in `dists` along each direction in `base`.

        The ball is carried at 0.72 m/cycle (after a turn if the direction
        is more than 35 deg off the body); opponents race the carry path.
        Success comes from the race margin, recalibrated with CARRY_FIT for
        short dribbles. Dribbles leaving the pitch or passing next to our own
        keeper in our box are dropped.
        Returns [(p, end x, end y, cycles, loss x, loss y)].
        """
        ph = S.ph
        body = S.kbody
        if base is None:
            goal_ang = math.degrees(math.atan2(-S.oy, ph.hl - S.ox))
            base = np.array([-90.0, -60.0, -30.0, -15.0, 0.0, 15.0, 30.0, 60.0, 90.0, goal_ang])
        speed = 0.72
        out = []
        for dist in dists:
            T = int(math.ceil(dist / speed)) + 5
            ra = np.radians(base)
            ux, uy = np.cos(ra), np.sin(ra)
            ex, ey = S.ox + ux * dist, S.oy + uy * dist
            turn = (np.abs(_norm_deg(base - body)) > 35.0).astype(np.int64)
            t = np.arange(T + 1)[None, :]
            travel = np.clip((t - turn[:, None]) * speed, 0.0, dist)
            B = np.stack([S.ox + ux[:, None] * travel, S.oy + uy[:, None] * travel], axis=2)
            first = _race(ph, B, S.op_pos, S.op_body,
                          np.full(S.nop, ph.kick_r + 0.5), S.op_gk,
                          np.zeros(S.nop, dtype=np.int64))
            t_th = first.min(axis=1) if S.nop else np.full(base.size, NEVER)
            K = turn + int(math.ceil(dist / speed))
            p = np.where(t_th > T, 0.97, _sig((t_th - K + 0.5) / 0.8))
            theirs = getattr(S, "theirs", False)
            if S.nop and dist <= 10.0 and (not theirs or -10.0 <= S.ox <= 35.0):
                od = np.hypot(S.op_pos[:, 0] - S.ox, S.op_pos[:, 1] - S.oy)
                a, b_d, b_n, b_r = (1.71, 0.09, -0.87, 0.17) if theirs else self.CARRY_FIT
                pr = np.clip(p, 0.01, 0.99)
                p = _sig(a + b_d * min(float(od.min()), 8.0) + b_n * float((od < 4.0).sum())
                         + b_r * np.log(pr / (1.0 - pr)))
            lt = B[np.arange(base.size), np.minimum(t_th, T)]
            inside = (np.abs(ex) < ph.hl - 1.0) & (np.abs(ey) < ph.hw - 1.0)
            if S.own_gk is not None:
                gd = np.hypot(B[:, :, 0] - S.own_gk[0], B[:, :, 1] - S.own_gk[1])
                box = (B[:, :, 0] < -ph.pen_x) & (np.abs(B[:, :, 1]) < ph.pen_hw)
                inside &= ~(box & (gd < ph.catch_r + 2.0)).any(axis=1)
            for i in range(base.size):
                if inside[i]:
                    out.append((float(p[i]), float(ex[i]), float(ey[i]), int(K[i]),
                                float(lt[i, 0]), float(lt[i, 1])))
        return out

    def _eval_dribbles(self, agent, S):
        """Dribble options valued like passes: p * GAMMA^K * V(end) + (1 - p) * V(loss)."""
        ph = S.ph
        goal_ang = math.degrees(math.atan2(-S.oy, ph.hl - S.ox))
        base = np.concatenate([[-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0, goal_ang],
                               self._evasive_angles(S)])
        out = []
        for p, ex, ey, K, lx, ly in self._dribble_cands(S, self.DRIB_DISTS, base):
            value = (p * GAMMA ** K * float(ph.v_keep(ex, ey, S.space(ex, ey)))
                     + (1.0 - p) * float(ph.v_lose(lx, ly)))
            out.append((value, "dribble", (ex, ey, p, K, lx, ly)))
        return out

    def _eval_hold(self, agent, S):
        """Value of keeping the ball this cycle (riskier the closer the nearest opponent)."""
        ph = S.ph
        d, _ = self._pressure(S)
        p = 0.0 if d < 1.0 else 0.45 if d < 2.0 else 0.8 if d < 3.0 else 0.95 if d < 5.0 else 0.98
        v = (p * (float(ph.v_keep(S.ox, S.oy, S.space(S.ox, S.oy))) - 0.012)
             + (1.0 - p) * float(ph.v_lose(S.ox, S.oy)))
        return (v, "hold", None)

    def _eval_clear(self, S):
        """Value of a long clearance upfield and wide."""
        ph = S.ph
        lx = min(ph.hl - 5.0, S.ox + 30.0)
        ly = math.copysign(min(ph.hw - 3.0, abs(S.oy) + 12.0), S.oy if S.oy else 1.0)
        v = 0.35 * float(ph.v_keep(lx, ly, 4.0)) + 0.65 * float(ph.v_out(lx, ly))
        return (v, "clear", None)

    def _start_go(self, wm, S):
        """After passing, remember a give-and-go run target for off_ball."""
        if GIVE_GO and S.ox > GO_X0:
            gx, gy, _, _ = _go_point(S.ph, S.ox, S.oy, S.offside_x, self.GO_RUN)
            self._go = (wm.cycle, gx, gy)

    def _touch_out(self, agent, S, best, res):
        """When heavily pressed in our own half and 2+ teammates are low on stamina,
        kick the ball out over the nearest touchline to prevent a dangerous turnover."""
        ph = self.ph
        if S.ox >= 0.0 or self._pressure(S)[0] > getattr(self, "TOUCH_PRESS", 2.0):
            return False
        value, kind, payload = best
        if kind in ("shot", "shot1", "clear"):
            return False
        if kind == "pass":
            p = float(res.p[payload])
        elif kind in ("pass1", "channel"):
            p = float(getattr(payload[0], "p")[payload[1]])
        elif kind == "dribble":
            p = float(payload[2]) if len(payload) > 2 else 0.5
        else:
            p = 0.5
        if p >= getattr(self, "TOUCH_P", 0.80):
            return False
        me = agent.wm.self.uniform_number
        st = {m[0]: (agent.wm.self.stamina if m[0] == me else
                     (self._stam.estimate(("us", m[0])) if self._stam is not None else ph.stamina_max))
              for m in self._context(agent)["mates"]}
        tired = sum(1 for v in st.values() if v < getattr(self, "TOUCH_ST", 3000.0))
        if tired < getattr(self, "TOUCH_N", 2):
            return False
        side = 1.0 if S.oy >= 0.0 else -1.0
        tx = min(S.ox + 10.0, ph.hl - 2.0)
        note(agent, "touch_out", k=kind, p=round(p, 2), tired=tired, x=round(S.ox, 1))
        Action.kick_one_step(agent, Vector2D(tx, side * (ph.hw + 5.0)))
        return True

    def _execute(self, agent, S, best, res):
        """Carry out the chosen (value, kind, payload) option."""
        if self.TOUCH_OUT and self._touch_out(agent, S, best, res):
            return
        self._execute_base(agent, S, best, res)

    def _execute_base(self, agent, S, best, res):
        """Issue the kick/dribble/hold actions. Passes and shots that need
        two kicks are stored in self._commit so the next cycle continues
        them. Fallback actions (hold, then clear) are queued after the main
        one."""
        value, kind, payload = best
        wm = agent.wm
        ph = S.ph
        if kind == "pass":
            i = payload
            tx, ty, s0 = float(res.tx[i]), float(res.ty[i]), float(res.s0[i])
            D = math.hypot(tx - S.ox, ty - S.oy)
            arrive = Tools.steps_required(s0, D, ph.decay, ph.bmax) or 1
            while (res.delay[i] > 0 and 1 < arrive < 60
                   and D * (1.0 - ph.decay) / (1.0 - ph.decay ** arrive) >= ph.bmax):
                arrive += 1
            note(agent, "pass", v=round(value, 3), p=round(float(res.p[i]), 2),
                 t=int(res.t_us[i]), m=int(min(res.t_th[i], 99)) - int(res.t_us[i]),
                 d=int(res.delay[i]), s0=round(s0, 2), r=int(S.recv[int(res.who[i])][0]))
            self._commit = {"cycle": wm.cycle, "age": 0, "kind": "pass",
                            "tx": tx, "ty": ty, "arrive": arrive}
            self._start_go(wm, S)
            Action.kick(agent, Vector2D(tx, ty), arrive, accept_late=self.KICK_LATE)
            Action.hold(agent)
        elif kind == "pass1":
            r, i = payload
            note(agent, "pass", v=round(value, 3), p=round(float(r.p[i]), 2),
                 t=int(r.t_us[i]), m=int(min(r.t_th[i], 99)) - int(r.t_us[i]),
                 d=0, s0=round(float(r.s0[i]), 2), one=1)
            self._start_go(wm, S)
            Action.kick_one_step(agent, Vector2D(float(r.tx[i]), float(r.ty[i])))
        elif kind == "channel":
            r, i = payload
            note(agent, "channel", v=round(value, 3), s0=round(float(r.s0[i]), 2))
            Action.kick_one_step(agent, Vector2D(float(r.tx[i]), float(r.ty[i])))
        elif kind in ("shot", "shot1"):
            gy, pg = payload
            if self.SHOT_RAY_FIX:
                gy = S.oy + (gy - S.oy) * (ph.hl - S.ox) / max(ph.hl + 2.0 - S.ox, 1.0)
            target = Vector2D(ph.hl, gy)
            note(agent, kind, v=round(value, 3), pg=round(pg, 2))
            if kind == "shot1":
                Action.kick_one_step(agent, target)
            elif self.FIX_SHOT:
                D = math.hypot(ph.hl - S.ox, gy - S.oy)
                arrive = 2
                while arrive < 60 and D * (1.0 - ph.decay) / (1.0 - ph.decay ** arrive) >= ph.bmax:
                    arrive += 1
                self._commit = {"cycle": wm.cycle, "age": 0, "kind": "shot",
                                "tx": ph.hl, "ty": gy, "arrive": arrive}
                Action.kick(agent, target, arrive, accept_late=self.SHOT_KICK_LATE)
                Action.kick_one_step(agent, target)
            else:
                self._commit = {"cycle": wm.cycle, "age": 0, "kind": "shot",
                                "tx": ph.hl, "ty": gy, "arrive": 1}
                Action.kick(agent, target, 1)
        elif kind == "dribble":
            note(agent, "dribble", v=round(value, 3))
            Action.dribble(agent, Vector2D(payload[0], payload[1]))
        elif kind == "hold":
            note(agent, "hold", v=round(value, 3))
            Action.hold(agent)
        else:
            note(agent, "clear", v=round(value, 3))
        if self.FIX_FALLBACK and kind != "clear" and S.ox > self.CLEAR_X:
            Action.hold(agent)
        Action.clear(agent)


# ==========================================================================
# SECTION: DEFENSE & MAN-MARKING MIXIN (agent/defense.py)
# ==========================================================================


class DefenseMixin:
    """Where our free defenders stand when the opponents have the ball."""

    def _their_eval(self, x, y):
        """Value to the opponents of having the ball at (x, y)."""
        return self._field_eval(-x, y)

    def _their_actions(self, ctx, ox, oy, T=30):
        """The opponent carrier's candidate actions from (ox, oy).

        Returns (paths B, cycle each action completes, their value, owner,
        receivers), where owner is the receiving opponent's number for
        passes, 0 for dribbles and -1 for shots; None if there are none.
        """
        ph = self.ph
        gx = -ph.hl
        holder = ctx["opp_first"]
        recv = [o for o in ctx["opps"] if not o[4] and o[0] != holder]
        paths, done, value, owner = [], [], [], []
        t = np.arange(T + 1)[None, :]

        if recv:
            pts = []
            for o in recv:
                fx, fy = o[1] + o[5] * 3.0, o[2] + o[6] * 3.0
                d = math.hypot(gx - fx, fy) or 1.0
                pts.append((fx, fy))
                pts.append((fx + (gx - fx) / d * 7.0, fy - fy / d * 7.0))
                # Also model straight wing/channel through-balls ahead of receivers
                pts.append((max(fx - 9.0, -ph.hl + 2.0), fy))
            P = np.array(pts)
            tx = np.clip(P[:, 0], -ph.hl + 1.0, ph.hl - 1.0)
            ty = np.clip(P[:, 1], -ph.hw + 1.0, ph.hw - 1.0)
            dx, dy = tx - ox, ty - oy
            D = np.maximum(np.hypot(dx, dy), 1e-6)
            s0 = np.minimum(ph.bmax, D * (1.0 - ph.decay) + 1.6)
            B = _kick_traj(ph, ox, oy, dx / D, dy / D, s0, np.ones(tx.size, dtype=np.int64), T)
            OP = np.array([[o[1] + o[5] * 0.6, o[2] + o[6] * 0.6] for o in recv])
            first = _race(ph, B, OP, np.array([o[3] for o in recv]),
                          np.full(len(recv), ph.kick_r), np.zeros(len(recv), dtype=np.int64),
                          np.ones(len(recv), dtype=np.int64))
            t_them = first.min(axis=1)
            land = B[np.arange(tx.size), np.minimum(t_them, T)]
            ok = (t_them <= T) & (D > 2.0)
            paths.append(B[ok])
            done.append(t_them[ok])
            value.append(self._their_eval(land[ok, 0], land[ok, 1]))
            owner.append(np.repeat([o[0] for o in recv], 3)[ok])

        # Include straight wing-breakaway angles (180 deg = straight toward our goal line -ph.hl)
        ga = math.degrees(math.atan2(-oy, gx - ox))
        ang_deg = np.unique(np.round(np.concatenate([
            ga + np.array([-70.0, -35.0, 0.0, 35.0, 70.0]),
            np.array([180.0, 160.0, -160.0])
        ]), 1))
        ang = np.radians(ang_deg)
        for L in (6.0, 12.0, 18.0):
            v_carry = 0.85 if L >= 12.0 else self.DRIBBLE_V
            ux, uy = np.cos(ang), np.sin(ang)
            travel = np.minimum(t * v_carry, L)
            B = np.stack([ox + ux[:, None] * travel, oy + uy[:, None] * travel], axis=2)
            ex, ey = ox + ux * L, oy + uy * L
            ok = (np.abs(ex) < ph.hl - 1.0) & (np.abs(ey) < ph.hw - 1.0)
            paths.append(B[ok])
            done.append(np.full(int(ok.sum()), int(math.ceil(L / v_carry))))
            # Boost threat value of deep wing breakaways so defenders prioritize cutting them off
            wing_threat = self._their_eval(ex[ok], ey[ok]) + np.where(ex[ok] < -10.0, 8.0, 0.0)
            value.append(wing_threat)
            owner.append(np.zeros(int(ok.sum()), dtype=np.int64))

        if math.hypot(gx - ox, oy) < 28.0:
            gy = np.array([-ph.gh + 1.0, 0.0, ph.gh - 1.0])
            dx, dy = gx - 2.0 - ox, gy - oy
            D = np.maximum(np.hypot(dx, dy), 1e-6)
            s0 = np.full(gy.size, ph.bmax)
            B = _kick_traj(ph, ox, oy, dx / D, dy / D, s0, np.ones(gy.size, dtype=np.int64), T)
            t_goal = _first_true(B[:, :, 0] <= gx)
            ok = t_goal <= T
            paths.append(B[ok])
            done.append(t_goal[ok])
            value.append(np.full(int(ok.sum()), 100.0))
            owner.append(np.full(int(ok.sum()), -1, dtype=np.int64))

        if not paths:
            return None
        B = np.concatenate(paths)
        if B.shape[0] == 0:
            return None
        return (B, np.concatenate(done).astype(np.int64), np.concatenate(value),
                np.concatenate(owner).astype(np.int64), recv)

    def _their_root(self, ctx, ox, oy):
        """The world mirrored to the opponents' point of view, as a _State,
        so the same pass/shot evaluators can be reused for their attack."""
        ph = self.ph
        field = [o for o in ctx["opps"] if not o[4]]
        mates = [(o[0], -o[1], -o[2], _norm_deg(o[3] + 180.0)) for o in field]
        ours = list(ctx["mates"]) + ([ctx["goalie"]] if ctx["goalie"] else [])
        n_field = len(ctx["mates"])
        lead = 0.6 if self.FIX_MIRROR_VEL else 0.0
        opps = [(m[0], -(m[1] + m[4] * lead), -(m[2] + m[5] * lead), _norm_deg(m[3] + 180.0),
                 i >= n_field) for i, m in enumerate(ours)]
        xs = sorted(m[1] for m in ours)
        line = xs[1] if len(xs) > 1 else (xs[0] if xs else 0.0)
        S = _State(ph, -ox, -oy, ctx["opp_first"], mates, opps, -line,
                   opp_vel=[(-m[4], -m[5]) for m in ours],
                   mate_vel={o[0]: (-o[5], -o[6]) for o in field}, lead=lead)
        gk = next((o for o in ctx["opps"] if o[4]), None)
        if gk is not None:
            S.own_gk = (-gk[1], -gk[2])
        S.theirs = True
        return S

    def _their_followups(self, ctx, ox, oy, B, t_done, E, owner):
        """Raise the value of their THEIR_K best actions by what they could do
        next (one more step of the chain search, discounted by THEIR_W), and
        add explicit follow-up shot paths for us to block."""
        ph = self.ph
        S = self._their_root(ctx, ox, oy)
        holder = ctx["opp_first"]
        T = B.shape[1] - 1
        E = E.copy()
        shots = []
        cand = [int(a) for a in np.argsort(-E, kind="stable")
                if owner[a] > 0 or (owner[a] == 0 and holder is not None)][:self.THEIR_K]
        for a in cand:
            t = int(min(t_done[a], T))
            lx, ly = float(B[a, t, 0]), float(B[a, t, 1])
            child = S.child(int(owner[a]) or holder, -lx, -ly, t)
            v = self._chain_value(child, 1, math.inf, drib=bool(owner[a] > 0), light=True,
                                  shoot=not self.THEIR_SHOT2)
            if np.isfinite(v):
                E[a] = max(E[a], self.THEIR_W * v)
            if not self.THEIR_SHOT2 or child.ox <= 20.0 or t >= T:
                continue
            _, pg, gy = self._eval_shots(child, ph.bmax, 1)
            k = int(np.argmax(pg))
            if pg[k] < self.SHOT_P:
                continue
            dx, dy = -ph.hl - lx, -float(gy[k]) - ly
            d = math.hypot(dx, dy) or 1.0
            leg = _kick_traj(ph, lx, ly, np.array([dx / d]), np.array([dy / d]),
                             np.array([ph.bmax]), np.array([1]), T - t)[0]
            path = np.concatenate([B[a, :t], leg])
            cross = int(_first_true((path[:, 0] <= -ph.hl)[None, :])[0])
            if cross <= T:
                shots.append((a, path, cross,
                              self.THEIR_W * (self.SHOT_V0 + self.SHOT_V1 * float(pg[k]))))
        return E, shots, len(cand)

    def _defend_point(self, agent, ctx):
        """This player's defensive target (the plan is shared via ctx)."""
        ph = self.ph
        wm = agent.wm
        me = wm.self.uniform_number
        self._support_target = None
        plan = ctx.get("defense")
        if plan is None:
            plan = ctx["defense"] = self._defense_plan(agent, ctx)
        if me not in plan:
            h = self._home(agent, ctx)
            return Vector2D(*self._goal_side(h.x(), h.y(), self._carrier(ctx)[0]))
        rx, ry = plan[me]
        rx = max(min(rx, ph.hl - 1.0), -ph.hl + 1.0)
        ry = max(min(ry, ph.hw - 1.0), -ph.hw + 1.0)
        return Vector2D(rx, ry)

    def _defense_plan(self, agent, ctx):
        """{uniform number: (x, y)} for every free (not chasing/blocking) player.

        See the module docstring. The opponents' value of a set of open
        actions is a soft-max (temperature 4) that also includes the value
        of the carrier just keeping the ball, and the assignment minimises
        that value plus 0.08 * weight * distance for each player's run.
        """
        ph = self.ph
        wm = agent.wm
        gx, gy = -ph.hl, 0.0
        if wm.kickable_opponent_existance:
            ox, oy = ctx["ball"][0], ctx["ball"][1]
            t_kick = 0
        else:
            ox, oy = ctx["opp_loc"]
            t_kick = min(ctx["opp_min"], HORIZON)

        busy = self._busy(ctx)
        players = [m for m in ctx["mates"] if m[0] not in busy]
        if self.MAN_MARK and players:
            return self._man_marks(ctx, players, ox, oy)
        stop = self._stopper(ctx, players) if self.STOPPER else None
        if stop is not None:
            players = [m for m in players if m[0] != stop[0]]
            if not players:
                return {stop[0]: stop[1]}
        acts = self._their_actions(ctx, ox, oy) if players else None
        if acts is None:
            return self._man_marks(ctx, players, ox, oy) if players else {}
        B, t_done, E, owner, recv = acts
        via = owner.copy()
        parent = np.full(E.size, -1, dtype=np.int64)
        if ox < self.THEIR_X and agent.budget.remaining_ms > self.THEIR_MIN_MS:
            t0 = time.perf_counter()
            E, shots, n2 = self._their_followups(ctx, ox, oy, B, t_done, E, owner)
            if shots:
                B = np.concatenate([B, np.stack([s[1] for s in shots])])
                t_done = np.concatenate([t_done, [s[2] for s in shots]]).astype(np.int64)
                E = np.concatenate([E, [s[3] for s in shots]])
                owner = np.concatenate([owner, np.full(len(shots), -1, dtype=np.int64)])
                via = np.concatenate([via, [owner[s[0]] for s in shots]]).astype(np.int64)
                parent = np.concatenate([parent, [s[0] for s in shots]]).astype(np.int64)
            note(agent, "their2", n=n2, s=len(shots),
                 ms=round(1000.0 * (time.perf_counter() - t0), 1))
        A = E.size

        best_by_recv = {}
        for a in range(A):
            u = int(via[a])
            if u > 0:
                best_by_recv[u] = max(best_by_recv.get(u, -1e9), float(E[a]))
        ranked = sorted(recv, key=lambda o: (-best_by_recv.get(o[0], -1e9), o[0]))[:3]

        # Avoid placing cover spot on top of the primary presser/blocker (fixes 0x801 collision)
        bplan = self._block_plan(ctx)
        bx_press, by_press = (bplan[1], bplan[2]) if bplan is not None else (ox, oy)

        spots, weights = [], []
        d_ball = math.hypot(ox - gx, oy - gy)
        if d_ball > 4.0:
            # Place sweeper/cover spot deeper than the presser (6.5m to 11.0m goal-side)
            step = min(max(0.42 * d_ball, 6.5), 11.0)
            cx_cov = ox + (gx - ox) / d_ball * step
            cy_cov = oy + (gy - oy) / d_ball * step * 0.85
            if math.hypot(cx_cov - bx_press, cy_cov - by_press) < 3.2:
                cx_cov = max(-ph.hl + 2.5, cx_cov - 3.5)
                cy_cov *= 0.75
            spots.append((cx_cov, cy_cov))
        else:
            spots.append((gx + 2.0, 0.0))
        weights.append(3.2)

        # If opponent carrier is wide on the wing, add a dedicated wing-cutoff spot ahead of them
        if abs(oy) > 12.0 and ox > -ph.hl + 8.0:
            spots.append((max(-ph.hl + 3.0, ox - 8.5), oy * 0.88))
            weights.append(2.6)

        for rank, o in enumerate(ranked):
            w = (2.2, 1.5, 1.1)[rank]
            fx, fy = o[1] + o[5] * 3.0, o[2] + o[6] * 3.0
            vx1, vy1 = gx - fx, gy - fy
            n1 = math.hypot(vx1, vy1) or 1.0
            vx2, vy2 = ox - fx, oy - fy
            n2 = math.hypot(vx2, vy2) or 1.0
            if ox > self.PRESS_X:
                wg, wb, gap = 0.65, 0.35, 2.0
            else:
                wg, wb, gap = 0.80, 0.20, (1.5 if n1 < 30.0 else 2.5)
            bx = wg * vx1 / n1 + wb * vx2 / n2
            by = wg * vy1 / n1 + wb * vy2 / n2
            nb = math.hypot(bx, by) or 1.0
            spots.append((fx + bx / nb * gap, fy + by / nb * gap))
            weights.append(w)
            if n2 > 6.0:
                k = max(0.5, 3.0 / n2)
                spots.append((ox + (fx - ox) * k, oy + (fy - oy) * k))
                weights.append(0.8 * w)
        n_mark = len(spots)
        spots.append((max(-ph.hl + 8.0, ox - 14.0), oy * 0.35))
        weights.append(1.0)
        n = len(players)
        for m in players:
            if len(spots) >= n:
                break
            spots.append((m[1], m[2]))
            weights.append(0.5)
        for r in range(1, len(spots)):
            if self.FIX_SPOTS and r < n_mark:
                spots[r] = (max(min(spots[r][0], ox - 1.0), -ph.hl + 2.0),
                            max(min(spots[r][1], ph.hw - 2.0), -ph.hw + 2.0))
            else:
                spots[r] = self._goal_side(spots[r][0], spots[r][1], ox)
            # Enforce minimum separation from the primary presser's target
            if math.hypot(spots[r][0] - bx_press, spots[r][1] - by_press) < 2.8:
                spots[r] = (max(-ph.hl + 2.0, spots[r][0] - 3.0), spots[r][1] * 0.8)
        R = len(spots)

        fixed = [m for m in ctx["mates"] if m[0] in busy]
        FP = [[m[1] + m[4] * 0.6, m[2] + m[5] * 0.6] for m in fixed]
        Fb = [m[3] for m in fixed]
        Fg = [0] * len(fixed)
        if stop is not None:
            FP.append([stop[1][0], stop[1][1]])
            Fb.append(math.degrees(math.atan2(oy - stop[1][1], ox - stop[1][0])))
            Fg.append(0)
        if ctx["goalie"]:
            g = ctx["goalie"]
            FP.append([g[1], g[2]])
            Fb.append(g[3])
            Fg.append(-1)
        SP = np.array(spots)
        Sb = np.degrees(np.arctan2(oy - SP[:, 1], ox - SP[:, 0]))
        P = np.vstack([np.array(FP).reshape(-1, 2), SP])
        first = _race(ph, B, P, np.concatenate([np.array(Fb, dtype=float), Sb]),
                      np.full(P.shape[0], ph.kick_r), np.array(Fg + [0] * R, dtype=np.int64),
                      np.ones(P.shape[0], dtype=np.int64))
        dribble = owner == 0
        margin = np.where(dribble, 2, 1)
        kills = first + margin[:, None] <= t_done[:, None]
        nf = len(FP)
        if len(fixed):
            kills[dribble, :len(fixed)] = False
        if ctx["goalie"]:
            kills[parent >= 0, nf - 1] = False
            if self.FIX_GK_BACK:
                tg = np.minimum(first[:, nf - 1], B.shape[1] - 1)
                behind = B[np.arange(B.shape[0]), tg, 0] < ctx["goalie"][1] - 0.5
                kills[behind, nf - 1] = False
        for a in np.nonzero(parent >= 0)[0]:
            kills[a] |= kills[parent[a]]
        killed_fixed = kills[:, :nf].any(axis=1) if nf else np.zeros(A, dtype=bool)
        order = [int(a) for a in np.argsort(-E, kind="stable") if not killed_fixed[a]]
        bit = {a: 1 << a for a in order}
        spot_mask = []
        for r in range(R):
            col = kills[:, nf + r]
            spot_mask.append(sum(bit[a] for a in order if col[a]))

        # Use REACH2 travel-time-aware reachability so t_kick=0 never zeroes out spot evaluation
        mask = [[0] * R for _ in range(n)]
        dcost = [[0.0] * R for _ in range(n)]
        for i, m in enumerate(players):
            mx, my = m[1] + m[4] * 0.6, m[2] + m[5] * 0.6
            for r in range(R):
                d = math.hypot(mx - spots[r][0], my - spots[r][1])
                # Penalize assigning a player who is currently ahead of the ball to a deep spot if another is already goal-side
                behind_pen = 1.5 if mx > ox + 2.0 and spots[r][0] < ox - 4.0 else 1.0
                dcost[i][r] = 0.08 * weights[r] * d * behind_pen
                late = max(int(np.searchsorted(ph.DASH, max(d - 0.5, 0.0))) - max(t_kick, 3), 0)
                col = first[:, nf + r] + late + margin <= t_done
                for a in np.nonzero(parent >= 0)[0]:
                    col[a] |= col[parent[a]]
                mask[i][r] = sum(bit[a] for a in order if col[a])

        e_hold = float(self._their_eval(ox, oy))
        Elist = [float(E[a]) for a in order]
        tau = 4.0
        memo = {}

        def their_value(killed):
            v = memo.get(killed)
            if v is None:
                top = e_hold
                for k, a in enumerate(order):
                    if not killed & bit[a]:
                        top = max(top, Elist[k])
                        break
                num, den = e_hold * math.exp((e_hold - top) / tau), math.exp((e_hold - top) / tau)
                for k, a in enumerate(order):
                    if killed & bit[a]:
                        continue
                    w = math.exp((Elist[k] - top) / tau)
                    if w < 1e-3:
                        break
                    num += w * Elist[k]
                    den += w
                v = memo[killed] = num / den
            return v

        need_cover = self.FORCE_COVER and d_ball < 45.0
        best, best_cost = None, None
        for perm in permutations(range(R), n):
            if need_cover and 0 not in perm:
                continue
            killed = 0
            cost = 0.0
            for i in range(n):
                killed |= mask[i][perm[i]]
                cost += dcost[i][perm[i]]
            cost += their_value(killed)
            if best_cost is None or cost < best_cost - 1e-9:
                best, best_cost = perm, cost
        plan = {players[i][0]: spots[best[i]] for i in range(n)}
        if stop is not None:
            plan[stop[0]] = stop[1]
        return plan

    def _stopper(self, ctx, players):
        """(player, spot) to stand goal-side of their deepest attacker (STOPPER, off)."""
        ph = self.ph
        field = [o for o in ctx["opps"] if not o[4] and o[0] != ctx["opp_first"]]
        if not field or not players:
            return None
        o = min(field, key=lambda o: (o[1], o[0]))
        if o[1] > 5.0:
            return None
        d = math.hypot(-ph.hl - o[1], o[2]) or 1.0
        sx = max(o[1] + (-ph.hl - o[1]) / d * self.STOP_GAP, -ph.hl + 1.5)
        sy = o[2] - o[2] / d * self.STOP_GAP
        m = min(players, key=lambda m: (math.hypot(m[1] - sx, m[2] - sy), m[0]))
        return m[0], (sx, sy)

    def _man_marks(self, ctx, players, ox, oy):
        """Plain man-marking (MAN_MARK, or fallback when there are no actions
        to model): a goal-side spot near each opponent, assigned by trying
        every permutation and minimising longest run + 0.3 * total run."""
        ph = self.ph
        gx = -ph.hl
        spots = []
        field = sorted((o for o in ctx["opps"] if not o[4] and o[0] != ctx["opp_first"]),
                       key=lambda o: o[0])
        deepest = min(field, key=lambda o: (o[1], o[0]))[0] if field and self.MARK_DEEP_GS else None
        for o in field:
            fx, fy = o[1] + 2.0 * o[5], o[2] + 2.0 * o[6]
            if ctx.get("midblock") and fx > self.MB_LINE:
                spots.append((self.MB_LINE - 3.0, max(min(fy, 15.0), -15.0)))
                continue
            if fx > ox + self.MARK_BEHIND:
                spots.append((max(-47.0, ox - 14.0), 0.3 * oy))
                continue
            wb, gap = (0.20, self.MARK_DEEP_GAP) if o[0] == deepest else (0.35, 1.8)
            if self.MARK_GS_ALL and o[0] != deepest and (fx < ox - 2.0 or fx < 0.0):
                wb = 0.25
            db = math.hypot(ox - fx, oy - fy) or 1.0
            dg = math.hypot(gx - fx, fy) or 1.0
            ux = wb * (ox - fx) / db + (1.0 - wb) * (gx - fx) / dg
            uy = wb * (oy - fy) / db + (1.0 - wb) * (-fy) / dg
            nu = math.hypot(ux, uy) or 1.0
            sx = min(fx + gap * ux / nu, ox - 1.0)
            if self.MARK_GS_ALL and (fx < ox - 2.0 or fx < -10.0):
                sx = min(sx, fx - 1.0)
            spots.append((max(min(sx, ph.hl - 1.0), -ph.hl + 1.0),
                          max(min(fy + gap * uy / nu, ph.hw - 1.0), -ph.hw + 1.0)))
        while len(spots) < len(players):
            k = len(spots)
            f = 0.3 if k % 2 == 0 else 0.6
            spots.append((ox + (gx - ox) * f, oy - oy * f))
        best, best_cost = None, None
        for perm in permutations(range(len(spots)), len(players)):
            ds = [math.hypot(players[i][1] - spots[perm[i]][0], players[i][2] - spots[perm[i]][1])
                  for i in range(len(players))]
            cost = max(ds) + 0.3 * sum(ds)
            if best_cost is None or cost < best_cost - 1e-9:
                best, best_cost = perm, cost
        return {players[i][0]: spots[best[i]] for i in range(len(players))}


# ==========================================================================
# SECTION: OFF-BALL POSITIONING MIXIN (agent/off_ball.py)
# ==========================================================================


class OffBallMixin:
    """The off_ball entry point and positioning helpers."""

    def off_ball(self, agent) -> None:
        """Framework entry point when this player cannot kick the ball. Any
        exception falls back to walking to the home position."""
        try:
            if self._stam_cycle_start(agent, False):
                return
            self._off_ball(agent)
        except Exception as e:
            note(agent, "off_ball_error", err=str(e)[:120])
            Action.go_to_point(agent, Tools.home_position(agent))

    def _off_ball(self, agent):
        """The off-ball rule list (see the module docstring)."""
        wm = agent.wm
        ph = self._phys(agent)
        ctx = self._context(agent)
        me = wm.self.uniform_number
        ball = wm.ball.position

        # 1. Automated Shot-Block Tackle
        if (self.IMP_TACKLE and not wm.kickable_opponent_existance
                and not wm.kickable_teammate_existance
                and wm.self.tackle_probability >= self.SHOTBLOCK_P):
            bvx, bvy = wm.ball.velocity.x(), wm.ball.velocity.y()
            if bvx < -0.5:
                path_x = ball.x() + bvx * ph.G
                cross = np.nonzero(path_x <= -ph.hl)[0]
                if cross.size:
                    tc = int(cross[0])
                    yc = ball.y() + bvy * ph.G[tc]
                    if abs(yc) < ph.gh + 1.5 and ctx["our_min"] >= tc:
                        want = (ball - Vector2D(-ph.hl, 0.0)).th().degree()
                        rel = _norm_deg(want - wm.self.body_direction)
                        dirs = np.arange(-90.0, 91.0, 10.0)
                        gain = (1.0 - np.abs(dirs) / 180.0) * np.cos(np.radians(rel - dirs))
                        k = int(np.argmax(gain))
                        note(agent, "shot_block", p=round(wm.self.tackle_probability, 2), t=tc)
                        Action.tackle(agent, float(dirs[k]), foul=False)
                        Action.intercept(agent)
                        return

        # 2. Full-Pitch & Wing Fair Tackle (Dispossesses FULL on wing breakaways & in midfield)
        opp_has_or_beats = (wm.kickable_opponent_existance
                            or (not wm.kickable_teammate_existance
                                and ctx["opp_min"] + 1 < ctx["our_min"]))
        tackle_thr = (self.TACKLE_DEF_P if (ball.x() < -18.0 or abs(ball.y()) > 18.0)
                      else self.TACKLE_OPEN_P)
        if (opp_has_or_beats and not wm.kickable_teammate_existance
                and wm.self.tackle_probability >= tackle_thr):
            # Aim tackle away from our goal (or upfield toward opponent goal if in midfield)
            aim_target = Vector2D(ph.hl, 0.0) if ball.x() > -15.0 else Vector2D(ball.x() + 15.0,
                                                                                  math.copysign(ph.hw, ball.y()))
            want = (aim_target - ball).th().degree()
            rel = _norm_deg(want - wm.self.body_direction)
            dirs = np.arange(-90.0, 91.0, 10.0)
            gain = (1.0 - np.abs(dirs) / 180.0) * np.cos(np.radians(rel - dirs))
            k = int(np.argmax(gain))
            if gain[k] >= 0.20 or ball.x() < -25.0:
                note(agent, "tackle", p=round(wm.self.tackle_probability, 2), dir=float(dirs[k]))
                Action.tackle(agent, float(dirs[k]), foul=False)
                Action.intercept(agent)
                return

        # 3. Deliberate Tactical Foul Tackle (Last-ditch outside box or midfield turnover)
        if (getattr(self, "FOULS", False) and wm.kickable_opponent_existance
                and not wm.kickable_teammate_existance
                and getattr(self, "_fouls", 0) < self.FOUL_MAX and int(wm.self.card) == 0
                and wm.self.foul_probability >= self.FOUL_P):
            d_goal = math.hypot(ball.x() + ph.hl, ball.y())
            why = None
            in_box = ball.x() < -ph.pen_x and abs(ball.y()) < ph.pen_hw
            if (self.LASTDITCH and self.LD_MIN <= d_goal <= self.LD_MAX and not in_box
                    and self._serious(ctx)):
                why = "last"
            elif (self.MID_FOUL and self.MID_X0 <= ball.x() <= self.MID_X1
                  and wm.self.tackle_probability < self.MID_TACKLE_MAX
                  and (not self.FOUL_COVER
                       or any(m[0] != me and m[1] < ball.x() - 3.0 for m in ctx["mates"]))):
                why = "mid"
            if why is not None:
                want = (ball - Vector2D(-ph.hl, 0.0)).th().degree()
                rel = _norm_deg(want - wm.self.body_direction)
                dirs = np.arange(-90.0, 91.0, 10.0)
                gain = (1.0 - np.abs(dirs) / 180.0) * np.cos(np.radians(rel - dirs))
                k = int(np.argmax(gain))
                self._fouls = getattr(self, "_fouls", 0) + 1
                note(agent, "foul", why=why, n=self._fouls, p=round(wm.self.foul_probability, 2))
                Action.tackle(agent, float(dirs[k]), foul=True)
                Action.intercept(agent)
                return

        # 4. Loose-Ball Chaser
        if not wm.kickable_teammate_existance and ctx["chaser"] == me:
            my_t = ctx["mate_t"].get(me, NEVER)
            if ctx["attack"] or my_t <= ctx["opp_min"] + 2:
                note(agent, "chase", t=my_t)
                Action.intercept(agent)
                Action.go_to_point(agent, ball, max_dash_power=100.0)
                return

        # 5. Primary Presser / Blocker Execution
        if not ctx["attack"] and me in self._busy(ctx):
            plan = self._block_plan(ctx)
            if plan is not None and plan[0] == me:
                power = 100.0 if wm.self.stamina > ph.st_floor + 200.0 else self._dash_power(agent, True, serious=True)
                engage = ((self.BLOCK_ENGAGE_HIGH if self.FIX_BLOCK else 12.0)
                          if ball.x() > self.PRESS_X
                          or (self.PRESS_ALL and ball.x() > self.PRESS_ALL_FROM)
                          else self.BLOCK_ENGAGE)
                carrier = next((o for o in ctx["opps"] if o[0] == ctx["opp_first"]), None)
                if (ctx["kickable_opp"] and carrier is not None
                        and math.hypot(carrier[5], carrier[6]) < self.HOLD_V):
                    engage = max(engage, self.HOLD_ENGAGE)
                d_ball = wm.self.position.dist(ball)
                if d_ball < engage:
                    aim = ball + wm.ball.velocity
                    rel = _norm_deg((aim - wm.self.position).th().degree() - wm.self.body_direction)
                    d_aim = wm.self.position.dist(aim)
                    straight = (d_ball < self.ENGAGE_D and abs(rel) <= self.ENGAGE_CONE
                                and d_aim * abs(math.sin(math.radians(rel))) < ph.kick_r - 0.2)
                    note(agent, "engage", t=ctx["mate_t"].get(me, NEVER), run=int(straight))
                    if straight:
                        ahead = wm.self.position + Vector2D.polar2vector(
                            d_aim * math.cos(math.radians(rel)) + 3.0, wm.self.body_direction)
                        Action.go_to_point(agent, ahead, max_dash_power=power, distance_threshold=0.3)
                    elif d_ball < 5.5 or wm.self.position.x() > ball.x() - 1.0:
                        # Sprint directly onto the predicted ball point to force a tackle/interception
                        Action.go_to_point(agent, aim, max_dash_power=power, distance_threshold=0.5)
                    else:
                        Action.go_to_point(agent, Vector2D(plan[1], plan[2]),
                                           max_dash_power=power, distance_threshold=0.5)
                    Action.intercept(agent)
                    return
                note(agent, "block", x=round(plan[1], 1), y=round(plan[2], 1))
                Action.go_to_point(agent, Vector2D(plan[1], plan[2]),
                                   max_dash_power=power, distance_threshold=0.5)
                Action.intercept(agent)
                return
            if plan is None and ctx["chaser"] == me:
                note(agent, "chase", t=ctx["mate_t"].get(me, NEVER))
                Action.intercept(agent)
                Action.go_to_point(agent, ball, max_dash_power=100.0)
                return

        # 6. Give-and-Go Continuation
        go = self._go
        if go is not None and (not ctx["attack"] or wm.cycle - go[0] > self.GO_CYCLES):
            self._go = go = None
        if go is not None and ctx["chaser"] != me:
            gx = go[1] if go[1] <= 0.0 else min(go[1], ctx["offside_x"] - 0.5)
            note(agent, "go", c=wm.cycle - go[0])
            Action.go_to_point(agent, Vector2D(gx, go[2]),
                               max_dash_power=self._dash_power(agent, True))
            Action.go_to_point(agent, self._home(agent, ctx))
            return

        # 7. Off-Ball Support (Attack) or Minimax Zonal/Lane Coverage (Defense)
        t_need, serious = None, False
        low = False
        if ctx["attack"]:
            target = self._support_point(agent, ctx)
            dist = wm.self.position.dist(target)
            urgent = (self._running == "run" and ball.x() > 10.0) or ctx.get("counter", False)
            label = self._running or "support"
        else:
            low = self.LOW_BLOCK and ball.x() > 8.0
            target = self._low_block_spot(ctx, me) if low else self._defend_point(agent, ctx)
            cx, cy = self._carrier(ctx)
            dist = wm.self.position.dist(target)
            serious = self._serious(ctx)
            # Any defender caught behind the ball or behind their target spot sprints at full power!
            out_of_pos = (wm.self.position.x() > target.x() + 1.5
                          or wm.self.position.x() > cx - 2.0)
            urgent = out_of_pos or dist > 5.0 or cx < 10.0
            if out_of_pos and cx < 15.0:
                serious = True
            v_threat = self.DEF_PACE_V if self.FIX_DEF_PACE else self.DRIBBLE_V
            t_need = (min(ctx["opp_min"], HORIZON)
                      + max(math.hypot(cx - target.x(), cy - target.y()) - 2.0, 0.0) / v_threat)
            label = "defend"
        far = wm.self.position.dist(ball) > 35.0 and not urgent
        power = self._dash_power(agent, urgent, dist, far, t_need, serious,
                                 attack=ctx["attack"])
        if low:
            power, label = 45.0, "low"
        if self.STAMINA_SAVE:
            power = self._save_power(agent, label, power, dist, serious, ball)
        if ctx["midblock"] and label == "defend" and not serious:
            power = min(power, self.MB_POWER)
        note(agent, label, s=int(serious), pw=int(power))
        Action.go_to_point(agent, target, max_dash_power=power)
        Action.go_to_point(agent, self._home(agent, ctx))

    def _home(self, agent, ctx):
        """Formation home position pulled towards the ball (COMPACT), onside."""
        ph = self.ph
        h = Tools.home_position(agent)
        bx = (ctx["our_loc"] if ctx["our_min"] <= ctx["opp_min"] else ctx["opp_loc"])[0]
        x = bx + self.COMPACT * (h.x() - bx)
        x = min(x, max(ctx["offside_x"], bx) - 0.5)
        x = max(min(x, ph.hl - 2.0), -ph.hl + 3.0)
        y = max(min(h.y(), ph.hw - 2.0), -ph.hw + 2.0)
        return Vector2D(x, y)

    def _busy(self, ctx):
        """Players taken by the ball this cycle: the chaser, or else the blocker."""
        ch = ctx["chaser"]
        if ch is not None and not ctx["kickable_mate"] and (
                ctx["attack"] or ctx["mate_t"].get(ch, NEVER) <= ctx["opp_min"] + 1):
            return {ch}
        plan = self._block_plan(ctx)
        return {plan[0] if plan is not None else ch}

    def _goal_side(self, x, y, ball_x):
        ph = self.ph
        x = max(min(x, ball_x - 3.5), -ph.hl + 2.0)
        return x, max(min(y, ph.hw - 3.0), -ph.hw + 3.0)

    def _carrier(self, ctx):
        if ctx["kickable_opp"]:
            return ctx["ball"][0], ctx["ball"][1]
        return ctx["opp_loc"]

    def _block_plan(self, ctx):
        """(player, x, y): who blocks the opponent carrier, and where.

        Predicts his dribble from where they win the ball towards our goal
        at DRIBBLE_V m/cycle, bent towards his current running direction,
        and picks the earliest point on that path some teammate can reach
        in time (preferring teammates already goal-side). Cached in ctx.
        """
        if "block" in ctx:
            return ctx["block"]
        ph = self.ph
        plan = None
        mates = ctx["mates"]
        if ctx["opp_min"] < NEVER and mates:
            lx, ly = ctx["opp_loc"]
            gx = -ph.hl
            d = math.hypot(gx - lx, ly)
            if d > 1.0:
                ux, uy = (gx - lx) / d, -ly / d
                first = next((o for o in ctx["opps"] if o[0] == ctx["opp_first"]), None)
                if first is not None and math.hypot(first[5], first[6]) > 0.25:
                    sp = math.hypot(first[5], first[6])
                    # Weight actual sprint direction heavily (0.75) to cut off wing breakaways
                    w_vel = 0.75 if sp > 0.35 else 0.50
                    ux = (1.0 - w_vel) * ux + w_vel * first[5] / sp
                    uy = (1.0 - w_vel) * uy + w_vel * first[6] / sp
                    n = math.hypot(ux, uy) or 1.0
                    ux, uy = ux / n, uy / n
                k = np.arange(1, 41)
                k = k[self.DRIBBLE_V * k <= d - 1.0]
                if k.size:
                    fx = np.clip(lx + ux * self.DRIBBLE_V * k, -ph.hl + 1.5, ph.hl - 1.5)
                    fy = np.clip(ly + uy * self.DRIBBLE_V * k, -ph.hw + 1.5, ph.hw - 1.5)
                    MX = np.array([m[1] + m[4] * 0.6 for m in mates])
                    MY = np.array([m[2] + m[5] * 0.6 for m in mates])
                    dist = np.hypot(fx[:, None] - MX, fy[:, None] - MY)
                    steps = np.searchsorted(ph.DASH, np.maximum(dist - ph.kick_r, 0.0))
                    ok = steps <= (ctx["opp_min"] + k)[:, None]
                    if self.FIX_BLOCK:
                        gs = np.array([m[1] <= lx + 1.0 for m in mates])
                        if (ok & gs[None, :]).any():
                            ok = ok & gs[None, :]
                    rows = np.nonzero(ok.any(axis=1))[0]
                    if rows.size:
                        r = int(rows[0])
                        cand = [(int(steps[r, j]), mates[j][0]) for j in range(len(mates)) if ok[r, j]]
                        who = min(cand)[1]
                        sy = float(fy[r])
                        plan = (who, float(fx[r]), sy - math.copysign(min(abs(sy), self.SHADE), sy))
        ctx["block"] = plan
        return plan

    def _dash_power(self, agent, urgent, dist=None, far=False, t_need=None, serious=False,
                    attack=False):
        """Dash power for a positioning run.

        Full power when the situation is serious or late in a half; the
        stamina-neutral FREE_POWER when stamina is down to the floor plus
        RESERVE; otherwise scaled by stamina, urgency and distance, and
        paced so the player arrives just in time (t_need) rather than early.
        """
        ph = self.ph
        c = agent.wm.cycle
        left = (HALF_CYCLES if c < HALF_CYCLES else MATCH_CYCLES) - c
        st = agent.wm.self.stamina
        if (self.STAMINA_MODE == 1 or (self.STAMINA_MODE == 2 and attack)) and left > LATE_CYCLES:
            return self._dash_power_h(agent, urgent, dist, far, t_need, serious)
        ratio = st / ph.stamina_max
        spare = st - ph.st_floor - RESERVE
        if serious:
            power = 100.0
        elif spare <= 0.0:
            power = FREE_POWER
        elif left <= LATE_CYCLES:
            power = 100.0
        elif urgent:
            power = 100.0 if ratio > 0.40 else 80.0 if ratio > 0.32 else 55.0
        else:
            power = 100.0 if ratio > 0.60 else 75.0 if ratio > 0.42 else FREE_POWER
        hurry = self.FIX_DEF_PACE and (urgent or serious)
        if dist is not None and not hurry:
            power = min(power, 45.0 if dist < 2.0 else 75.0 if dist < 5.0 else 100.0)
        if t_need is not None and dist is not None and not serious and not hurry:
            power = min(power, max(100.0 * dist / max(t_need - PACE_SLACK, 1.0), FREE_POWER))
        if far and not serious and not hurry:
            power = min(power, 65.0)
        return power

    def _save_power(self, agent, label, power, dist, serious, ball):
        """Extra power caps per run type (STAMINA_SAVE, off)."""
        wm = agent.wm
        c = wm.cycle
        if (HALF_CYCLES if c < HALF_CYCLES else MATCH_CYCLES) - c <= LATE_CYCLES:
            return power
        st = wm.self.stamina
        if label == "rest" and dist < 12.0:
            return min(power, 60.0)
        if label in ("diag", "support"):
            return min(power, 75.0)
        if label == "run" and st < 5000.0:
            return min(power, 75.0)
        if self.STAMINA_SAVE >= 2 and label == "defend" and not serious and ball.x() > 10.0:
            return min(power, 70.0)
        return power

    def _dash_power_h(self, agent, urgent, dist, far, t_need, serious):
        """Hysteresis recovery mode (STAMINA_MODE 1/2, off): below 50% stamina
        walk until back above 70%, except in serious situations."""
        wm = agent.wm
        ph = self.ph
        st = wm.self.stamina
        if st < 0.5 * ph.stamina_max:
            self._recover = True
        elif st > 0.7 * ph.stamina_max:
            self._recover = False
        inc = FREE_POWER * max(float(wm.self.recovery), 0.1)
        line = wm.our_defense_line_x
        if serious or (wm.self.position.x() < line + 0.5 and wm.ball.position.x() < line + 20.0):
            return 100.0
        if self._recover:
            power = FREE_POWER if urgent else max(inc - 25.0, 0.0)
        elif urgent:
            power = 100.0
        elif wm.kickable_teammate_existance and wm.ball.dist_from_self < 20.0:
            power = min(1.1 * inc, 100.0)
        else:
            power = min(1.7 * inc, 100.0)
        if dist is not None:
            power = min(power, 40.0 if dist < 3.0 else 70.0 if dist < 6.0 else 100.0)
        if t_need is not None and dist is not None:
            power = min(power, max(100.0 * dist / max(t_need - PACE_SLACK, 1.0), FREE_POWER))
        if far:
            power = min(power, 60.0)
        return power

    def _serious(self, ctx):
        """True when the opponents will win the ball somewhere dangerous: past
        x = -10, or within 36 m of our goal with at most one of us behind the
        ball (22 m: at most one covering the line to goal). Defenders then
        run at full power."""
        if "serious" in ctx:
            return ctx["serious"]
        ph = self.ph
        out = False
        if not ctx["attack"] and ctx["opp_min"] < NEVER and ctx["opp_min"] <= ctx["our_min"]:
            cx, cy = self._carrier(ctx)
            d = math.hypot(cx + ph.hl, cy)
            behind = [m for m in ctx["mates"] if m[1] < cx - 1.5]
            close = sum(1 for m in behind
                        if abs(m[2] - cy * (m[1] + ph.hl) / max(cx + ph.hl, 1.0)) < 12.0)
            out = (d < 36.0 and len(behind) <= 1) or (d < 22.0 and close <= 1) or cx < -10.0
        ctx["serious"] = out
        return out

    def _support_point(self, agent, ctx):
        """Where to stand in attack.

        Special roles first (rest defence, far-post / cut-back finish), then
        either an outlet near a pressed carrier, a forward or diagonal run,
        or the home area. Candidate points are scored by how open they are
        for a pass (race again), the value of receiving there, distance to
        home/self and crowding by teammates; the previous target gets a
        small bonus so players don't dither.
        """
        ph = self.ph
        wm = agent.wm
        me = wm.self.uniform_number
        home = self._home(agent, ctx)
        hx, hy = home.x(), home.y()
        if wm.kickable_teammate_existance:
            ox, oy = ctx["ball"][0], ctx["ball"][1]
        else:
            ox, oy = ctx["our_loc"]
        rest = (self._shell_spot(ctx, me, ox) if self.LOW_BLOCK
                else self._rest_spot(ctx, me, ox) if self.REST_DEFENSE else None)
        if rest is not None:
            self._running = "rest"
            self._support_target = None
            return rest
        fin = self._finish_spot(ctx, me, ox, oy) if self.FINISH else None
        if fin is not None:
            self._running = "run"
            self._support_target = None
            return fin
        outlet = wm.kickable_teammate_existance and self._is_outlet(ctx, me, ox, oy)
        run = None if outlet else self._run_spot(ctx, me, ox, oy)
        self._running = run[2] if run else None
        if outlet:
            cx, cy, rings = ox, oy, ((5.0, 12), (8.0, 12), (11.0, 12))
            pts = []
        elif run and run[2] == "run":
            hx, hy = run[0], run[1]
            cx, cy, rings = hx, hy, ((3.0, 8), (6.0, 8))
            pts = [(hx, hy)]
        elif run:
            hx, hy = run[0], run[1]
            side = 1.0 if hy >= oy else -1.0
            cx, cy, rings = hx, hy, ()
            pts = [(ox + ddx, oy + side * ddy) for ddx in (5.0, 8.0, 12.0)
                   for ddy in (3.0, 6.0, 9.0, 12.0)]
        else:
            cx, cy, rings = hx, hy, ((4.0, 8), (8.0, 8))
            pts = [(hx, hy)]
        for r, n in rings:
            for i in range(n):
                a = 2.0 * math.pi * i / n
                pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        if self._support_target is not None:
            pts.append(self._support_target)
        pts = np.array(pts)
        line = max(ctx["offside_x"], ox) - 0.7
        px = np.where((pts[:, 0] > line) & (pts[:, 0] > 0.0), line, pts[:, 0])
        px = np.clip(px, -ph.hl + 2.0, ph.hl - 2.0)
        py = np.clip(pts[:, 1], -ph.hw + 2.0, ph.hw - 2.0)

        dx, dy = px - ox, py - oy
        D = np.maximum(np.hypot(dx, dy), 1e-6)
        s0 = np.minimum(ph.bmax, D * (1.0 - ph.decay) + 1.5)
        delay = np.ones(px.size, dtype=np.int64)
        T = 25
        B = _kick_traj(ph, ox, oy, dx / D, dy / D, s0, delay, T)
        opps = ctx["opps"]
        if opps:
            OP = np.array([[o[1], o[2]] for o in opps])
            ob = np.array([o[3] for o in opps])
            og = np.array([1 if o[4] else 0 for o in opps], dtype=np.int64)
            first = _race(ph, B, OP, ob, np.full(len(opps), ph.kick_r + OPP_MARGIN), og,
                          np.ones(len(opps), dtype=np.int64))
            t_th = first.min(axis=1)
            sp = np.hypot(px[:, None] - OP[:, 0], py[:, None] - OP[:, 1]).min(axis=1)
        else:
            t_th = np.full(px.size, NEVER)
            sp = np.full(px.size, 20.0)
        arrive = np.searchsorted(ph.G, D / s0) + delay
        margin = np.clip(t_th - arrive, -6, 6).astype(float)
        openness = np.where(D < 3.0, 0.3, _sig((margin - 0.5) / 0.7))

        crowd = np.zeros(px.size)
        for m in ctx["mates"]:
            if m[0] == me:
                continue
            d = np.hypot(px - m[1], py - m[2])
            crowd += np.clip(6.0 - d, 0.0, None) / 6.0 * 0.012

        mx, my = wm.self.position.x(), wm.self.position.y()
        if outlet:
            sep = np.full(px.size, 90.0)
            if opps:
                aw = np.degrees(np.arctan2(dy, dx))
                ao = np.degrees(np.arctan2(OP[:, 1] - oy, OP[:, 0] - ox))
                do = np.hypot(OP[:, 0] - ox, OP[:, 1] - oy)
                gap = np.abs(_norm_deg(aw[:, None] - ao[None, :]))
                gap = np.where(do[None, :] < D[:, None] + 2.0, gap, 90.0)
                sep = gap.min(axis=1)
            score = (openness * (ph.v_keep(px, py, sp) + 0.02)
                     + 0.01 * np.clip(sep / 30.0, 0.0, 1.0)
                     - 0.0025 * np.hypot(px - mx, py - my)
                     - crowd)
        else:
            score = (openness * (ph.v_keep(px, py, sp) + 0.02)
                     - 0.0025 * np.hypot(px - hx, py - hy)
                     - 0.0012 * np.hypot(px - mx, py - my)
                     - crowd)
        i = int(np.argmax(score))
        if self._support_target is not None:
            j = px.size - 1
            if score[j] + 0.004 >= score[i]:
                i = j
        self._support_target = (float(px[i]), float(py[i]))
        return Vector2D(float(px[i]), float(py[i]))

    def _finish_spot(self, ctx, me, ox, oy):
        """Near their box, the two most advanced players take the far post
        and the cut-back spot (whichever pairing means less running)."""
        if ox < self.FINISH_X:
            return None
        ph = self.ph
        rest = [m for m in ctx["mates"] if m[0] != ctx["chaser"]]
        anchor = (min(rest, key=lambda m: (m[1], m[0]))[0]
                  if self.REST_DEFENSE and len(rest) > 2 else None)
        cand = sorted((m for m in rest if m[0] != anchor), key=lambda m: (-m[1], m[0]))[:2]
        if len(cand) < 2 or me not in [m[0] for m in cand]:
            return None
        line = max(ctx["offside_x"], ox) - 0.7
        s = 1.0 if oy >= 0.0 else -1.0
        far = (min(line, 46.5), -s * 5.5)
        cut = (min(line, max(ox - 4.5, 36.5)), -s * 1.0)
        a, b = cand
        d_ab = math.hypot(a[1] - far[0], a[2] - far[1]) + math.hypot(b[1] - cut[0], b[2] - cut[1])
        d_ba = math.hypot(a[1] - cut[0], a[2] - cut[1]) + math.hypot(b[1] - far[0], b[2] - far[1])
        spot = ({a[0]: far, b[0]: cut} if d_ab <= d_ba else {a[0]: cut, b[0]: far})[me]
        return Vector2D(min(spot[0], ph.hl - 2.0), spot[1])

    def _shell_spot(self, ctx, me, ox):
        if ox >= 25.0:
            return None
        rest = sorted((m for m in ctx["mates"] if m[0] != ctx["chaser"]),
                      key=lambda m: (m[1], m[0]))[:2]
        ids = [m[0] for m in rest]
        if me not in ids:
            return None
        lo = -self.ph.hl + 3.0
        if ids.index(me) == 0:
            return Vector2D(max(min(ox - 14.0, -5.0), lo), -6.0)
        return Vector2D(max(min(ox - 10.0, 0.0), lo), 6.0)

    def _low_block_spot(self, ctx, me):
        busy = self._busy(ctx)
        ids = [m[0] for m in sorted((m for m in ctx["mates"] if m[0] not in busy),
                                    key=lambda m: (m[2], m[0]))]
        n = len(ids)
        k = ids.index(me) if me in ids else 0
        ys = (0.0,) if n <= 1 else tuple(-9.0 + 18.0 * i / (n - 1) for i in range(n))
        y = ys[min(k, len(ys) - 1)] + 0.3 * ctx["ball"][1]
        return Vector2D(-12.0, max(min(y, self.ph.hw - 2.0), -self.ph.hw + 2.0))

    def _rest_spot(self, ctx, me, ox):
        """Rest defence: the deepest non-chaser stays behind the play."""
        if ox <= -10.0:
            return None
        rest = [m for m in ctx["mates"] if m[0] != ctx["chaser"]]
        if not rest or min(rest, key=lambda m: (m[1], m[0]))[0] != me:
            return None
        fx, fy = min(((o[1], o[2]) for o in ctx["opps"] if not o[4]), default=(0.0, 0.0))
        x = max(min(ox - 15.0, fx - 3.5, 2.0), -self.ph.hl + 4.0)
        return Vector2D(x, max(min(fy * 0.7, 10.0), -10.0))

    def _run_spot(self, ctx, me, ox, oy):
        """Forward runs for the two most advanced players: into the box
        ("run") high up the pitch, otherwise a diagonal ahead of the ball."""
        if ox < self.DIAG_X0:
            return None
        ph = self.ph
        bx, by = ctx["ball"][0], ctx["ball"][1]
        cand = [m for m in ctx["mates"]
                if m[0] != ctx["chaser"] and math.hypot(m[1] - bx, m[2] - by) > ph.kick_r + 0.3]
        run = sorted(cand, key=lambda m: (-m[1], m[0]))[:2]
        ids = [m[0] for m in run]
        if me not in ids:
            return None
        if len(run) == 2:
            hi = max(run, key=lambda m: (m[2], m[0]))[0]
            side = 1.0 if me == hi else -1.0
        else:
            side = 1.0 if by < 0.0 else -1.0
        if ox > self.RUN_SPLIT_X or (ox > 0.0 and ids.index(me) == 0):
            x = min(ctx["offside_x"] - 1.2, max(ox + 10.0, 36.0), ph.hl - 7.0)
            return x, side * 8.0, "run"
        x = min(ox + 8.0, max(ctx["offside_x"], ox) - 1.0)
        y = max(min(oy + side * 8.0, ph.hw - 3.0), -ph.hw + 3.0)
        return x, y, "diag"

    def _is_outlet(self, ctx, me, ox, oy):
        """Whether this player is one of the two nearest outlets for a pressed carrier."""
        if not any(math.hypot(o[1] - ox, o[2] - oy) < self.OUTLET_D for o in ctx["opps"]):
            return False
        near = sorted((math.hypot(m[1] - ox, m[2] - oy), m[0]) for m in ctx["mates"])
        near = [u for d, u in near if d > self.ph.kick_r + 0.3][:2]
        return me in near


# ==========================================================================
# SECTION: AGENT PLAYERS (agent/player.py)
# ==========================================================================


class Agent1(StarterAgent, AgentConfig, OnBallMixin, EvaluatorMixin, SearchMixin, OffBallMixin, DefenseMixin):
    """Search-based 5-a-side player.

    The framework calls `on_ball` (OnBallMixin) when this player can kick
    the ball and `off_ball` (OffBallMixin) otherwise, only in normal play;
    set pieces and the goalkeeper are run by the framework. Players never
    communicate: they coordinate because each runs the same deterministic
    computation on the same (exact) world model.
    """

    def __init__(self):
        super().__init__()
        self.ph = None                # _Phys tables, built on first use
        self._ctx = None              # per-cycle shared context (OnBallMixin._context)
        self._commit = None           # two-kick pass/shot to continue next cycle
        self._support_target = None   # last support spot (hysteresis)
        self._prev_chaser = None      # last cycle's chaser (stickiness)
        self._running = None          # current attacking run type
        self._vmax_cache = {}         # max one-kick speed per direction, this cycle
        self._drib_wins = 0           # debug counter: dribbles chosen in the chain search
        self._recover = False         # hysteresis recovery mode state
        self._opp_poss = 0            # cycles the opponents have had the ball
        self._opp_poss_cycle = -1
        self._go = None               # active give-and-go run (cycle, x, y)
        self._stam = None             # StaminaPredictor
        self._stam_cycle = -1
        self._budget = {}             # teammates' sprint budgets (full dashes left)
        self._sig_done = set()        # stamina signalling bookkeeping (off)
        self._sig_pending = {}
        self._floor_roles = (-1, set())  # teammates expected to sprint this cycle


class Agent2:
    """Secondary agent stub."""
    pass


__all__ = ['Agent1', 'Agent2', 'AgentConfig', 'StaminaPredictor', '_Phys', '_State', '_KickResult']
