"""Server physics and the race model, the foundation of every decision.

The central question the agent asks is "for this ball trajectory, at which
cycle can each player first control the ball?" `_race` answers it for many
trajectories and players at once with numpy. Pass, dribble and shot
evaluation, the choice of chaser and blocker, and the defensive assignment
are all built on the answer.
"""
import math
import numpy as np

from .config import HORIZON, NEVER, OPP_MARGIN, INERTIA, _sig


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