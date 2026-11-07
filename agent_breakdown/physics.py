"""Module header, stamina recovery tables, server physics, the race model, search states.

Part view of agent.py, written by `agent_breakdown/parts.py split`.
Edit the code between the begin/end markers, then run `agent_breakdown/parts.py build`
to write it back into agent.py. Text outside the markers is ignored."""

# ==== begin <module> ====

import math
import time
from itertools import permutations

import numpy as np
from pyrusgeom.angle_deg import AngleDeg
from pyrusgeom.geom_2d import Vector2D
from src.student.interface import StarterAgent, Action, Tools, note


HORIZON = 40
MATCH_CYCLES = 1500
HALF_CYCLES = 750
LATE_CYCLES = 100
RESERVE = 600.0
FREE_POWER = 45.0
PACE_SLACK = 3.0
SPRINT_NET = 55.0
NEVER = 999
GAMMA = 0.985
OPP_MARGIN = 0.3
INERTIA = 5.0
OPP_PURSUIT = 0.8
OPP_UNC = 0.2
OPP_UNC_MAX = 2.0
GK_REACH = 0.5

# ==== end <module> ====

# ==== begin _sig ====
def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))
# ==== end _sig ====

# ==== begin _norm_deg ====


def _norm_deg(a):
    return (a + 180.0) % 360.0 - 180.0
# ==== end _norm_deg ====

# ==== begin _Phys ====


class _Phys:

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
        self.kick_rand = float(getattr(sp, "kick_rand", 0.0)) or 0.1
        self.kick_power_rate = float(getattr(sp, "kick_power_rate", 0.0)) or 0.027
        self.kick_max = float(getattr(sp, "ball_accel_max", 0.0)) or 2.7
        self.tackle_dist = float(getattr(sp, "tackle_dist", 0.0)) or 2.0
        self.tackle_width = float(getattr(sp, "tackle_width", 0.0)) or 1.25
        self.tackle_exp = float(getattr(sp, "tackle_exponent", 0.0)) or 6.0
        self._tk = {}

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

    def tackle_reach(self, p):
        if p not in self._tk:
            q = max(1.0 - p, 1e-6) ** (1.0 / self.tackle_exp)
            a = np.radians(15.0 * np.arange(13))
            r = q / ((np.abs(np.cos(a)) / self.tackle_dist) ** self.tackle_exp
                     + (np.abs(np.sin(a)) / self.tackle_width) ** self.tackle_exp) ** (1.0 / self.tackle_exp)
            self._tk[p] = np.where(a <= math.pi / 2.0 + 1e-9, r, 0.0)
        return self._tk[p]

    def threat(self, x, y):
        wide = 1.0 - 0.4 * np.clip((np.abs(y) - 10.0) / 15.0, 0.0, 1.0)
        return 0.05 + 0.30 * wide / (1.0 + np.exp(-(x - 22.0) / 9.0))

    def danger(self, x, y):
        return self.threat(-x, y)

    def v_keep(self, x, y, space):
        return self.threat(x, y) + 0.015 * np.clip(space, 0.0, 8.0) / 8.0

    def v_lose(self, x, y):
        return -(self.danger(x, y) + 0.03)

    def v_out(self, x, y):
        return np.where(np.asarray(x) > 0.0, self.v_lose(x, y), -(0.35 * self.danger(x, y) + 0.015))
# ==== end _Phys ====

# ==== begin _kick_traj ====


def _kick_traj(ph, ox, oy, ux, uy, s0, delay, T):
    t = np.arange(T + 1)[None, :]
    k = np.clip(t - delay[:, None], 0, ph.G.size - 1)
    g = ph.G[k] * s0[:, None]
    ox = np.broadcast_to(ox, s0.shape)[:, None]
    oy = np.broadcast_to(oy, s0.shape)[:, None]
    return np.stack([ox + ux[:, None] * g, oy + uy[:, None] * g], axis=2)
# ==== end _kick_traj ====

# ==== begin _race ====


def _race(ph, B, P, body, ctrl, gk, react, vel=None, omni=None, tired=None, tackle=None):
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
    cr = c
    if tackle is not None:
        tk_mask, tk = tackle
        c = c - np.where(tk_mask, OPP_MARGIN, 0.0)
        cr = c + np.where(tk_mask, max(tk[0] - ph.kick_r, 0.0), 0.0)
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
        ok = d <= cr + ph.DASH[avail]
    else:
        ok = d <= cr + ph.DASH_TIRED[np.asarray(tired, dtype=np.int64)[None, None, :], avail]
    if omni is not None and np.any(omni):
        k = np.clip(np.rint(diff / 15.0).astype(np.int64), 0, 12)
        now = np.clip(t - react[None, None, :], 0, ph.DASH.size - 1)
        if tackle is None:
            reach = ph.DASH_DIR[k, now]
        else:
            tab = np.stack([ph.DASH_DIR, ph.DASH_DIR + np.maximum(tk - ph.kick_r, 0.0)[:, None]])
            reach = tab[tk_mask.astype(np.int64)[None, None, :], k, now]
        ok |= omni[None, None, :] & (d <= c + reach)
    hit = ok.any(axis=1)
    return np.where(hit, ok.argmax(axis=1), NEVER)
# ==== end _race ====

# ==== begin _first_true ====


def _first_true(mask):
    hit = mask.any(axis=1)
    return np.where(hit, mask.argmax(axis=1), NEVER)
# ==== end _first_true ====

# ==== begin _State ====


class _State:

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
        self.kick_cond = None
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
        if self.nop == 0:
            return np.full(np.shape(x), 20.0)
        d = np.hypot(np.asarray(x)[..., None] - self.op_pos[:, 0],
                     np.asarray(y)[..., None] - self.op_pos[:, 1])
        return d.min(axis=-1)

    def child(self, receiver_unum, lx, ly, t, body=None):
        ph = self.ph
        drift = (1.0 - ph.pdecay ** t) / (1.0 - ph.pdecay)
        reach = float(ph.DASH[min(max(t - 1, 0), ph.DASH.size - 1)])
        opps, unc = [], []
        coast = drift - self.lead
        for k, (u, x, y, b, g) in enumerate(self.opps):
            x, y = x + self.ov[k, 0] * coast, y + self.ov[k, 1] * coast
            r = reach * (GK_REACH if g else 1.0)
            dx, dy = lx - x, ly - y
            d = math.hypot(dx, dy)
            step = min(max(d - 1.0, 0.0), OPP_PURSUIT * r)
            if d > 1e-6:
                x, y = x + dx / d * step, y + dy / d * step
                b = math.degrees(math.atan2(dy, dx))
            opps.append((u, x, y, b, g))
            unc.append(min(OPP_UNC * r, OPP_UNC_MAX))
        mates, mate_vel = [], {}
        keep = ph.pdecay ** t
        cap = max(self.offside_x, lx) - 0.5
        for m in self.mates:
            if m[0] == receiver_unum:
                mates.append((m[0], lx, ly, m[3] if body is None else body))
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
                   mate_vel=mate_vel, opp_unc=unc, tired_map=self.tired_map)
        s.own_gk = self.own_gk
        s.adv = self.adv
        return s
# ==== end _State ====

# ==== begin _KickResult ====


class _KickResult:
    __slots__ = ("value", "p", "t_us", "t_th", "who", "lx", "ly", "succ", "v_fail",
                 "vs", "tx", "ty", "s0", "delay", "t_out")
# ==== end _KickResult ====

