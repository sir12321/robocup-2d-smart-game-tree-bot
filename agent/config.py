"""Tuning constants and feature flags for Agent1.

Units: distances in metres, time in server cycles (10 cycles = 1 s), speeds in
metres per cycle, stamina in server stamina units (max 8000). Coordinates are
always from our point of view: our goal at x = -52.5, attacking towards +x.

Module-level constants are shared by the helper classes (physics, search
state, stamina model). `AgentConfig` holds the per-agent knobs and is mixed
into `Agent1`, so they are read as `self.NAME`.

Where the numbers come from: action and evaluation constants were tuned by
running matches and calibrated against our match logs; the logistic fits
(PASS_FIT, CARRY_FIT, GK_A/GK_P/GK_N) and the stamina GAP tables were fitted
to those logs. Many boolean flags (FIX_*, IMP_*, ...) switch on a fix or
improvement that was A/B-tested against the version without it. Names marked
"unused" are leftovers from experiments: nothing reads them.
"""
import numpy as np

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
