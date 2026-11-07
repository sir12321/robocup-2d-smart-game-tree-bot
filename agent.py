
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

def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def _norm_deg(a):
    return (a + 180.0) % 360.0 - 180.0


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

US_GAP_GLOBAL = None
US_GAP_TABLE = {
    (0, 0, 0): (0.489, -0.000, 9.238, 24.130, 0.001, 0.007),
    (0, 0, 1): (0.612, 0.001, 54.153, 24.954, 0.000, 0.011),
    (3, 0, 0): (1.588, 0.003, 39.307, 77.636, 83.611, -0.004),
    (3, 1, 1): (1.059, 18.894, -0.000, 12.061, 0.321, -23.924),
    'taker': (0.174, 32.342, 36.515, 39.887, 41.534, 63.557),
}


class StaminaPredictor:
    def __init__(self, sp):
        self.last_need = {}
        self.gap_global, self.gap_table = GAP_GLOBAL_V3, GAP_TABLE_V3
        self.gk_global, self.gk_table = GAP_GLOBAL_V2, GAP_TABLE_V2
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
        self.eff_inc = g("effort_inc", 0.01)
        self.eff_dec_thr = g("effort_dec_thr", 0.3) * self.st_max
        self.eff_inc_thr = g("effort_inc_thr", 0.6) * self.st_max
        self.rec_dec_thr = g("recover_dec_thr", 0.3) * self.st_max
        half = g("half_time", 75.0)
        self.half_cycles = half * 10.0 if half < 1000 else half
        self.touch_ball = g("player_size", 0.3) + g("ball_size", 0.085) + 0.03
        self.touch_player = 2.0 * g("player_size", 0.3) + 0.03
        self.half_x = g("pitch_half_length", 52.5)
        self.half_y = g("pitch_half_width", 34.0)
        self.state = {}
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
        if cycle == self.last:
            return
        self.last_need = {}
        half = self.last is not None and self.last < self.half_cycles <= cycle
        if half:
            self.state = {k: self._fresh() for k in self.state}
            self.prev = {}
        gap = 0 if self.last is None else cycle - self.last - 1
        goal = self.score is not None and score != self.score
        out = self.ball is not None and (abs(self.ball[0]) > self.half_x or abs(self.ball[1]) > self.half_y)
        if self.ball is not None:
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
                if k[0] == "us" and not o[5] and o0:
                    dx = o[0] - o0[0]
                    self._gap_us(s, gap, goal, out, max(-dx, 0.0), max(dx, 0.0), abs(o[1] - o0[1]), disp,
                                 k == taker)
                else:
                    self._gap(s, gap, goal, disp, out, o[5])
                o0 = None
            if o0 is not None:
                need = self._need(o0, o, s[1], self._contact(k, obs, ball))
                self.last_need[k] = need
                if need > 0.0:
                    low = need + self.dash_threshold()
                    if s[0] < low:
                        s[0] = min(low, self.st_max)
                self._step(s, need)
            self.prev[k] = o
        self.last, self.score, self.ball = cycle, score, ball


    def estimate(self, key):
        s = self.state.get(key)
        return s[0] if s else self.st_max

    def dash_threshold(self):
        return max(self.eff_dec_thr, self.rec_dec_thr)

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
        if contact and math.hypot(o1[2] - o0[2] * self.decay, o1[3] - o0[3] * self.decay) < 0.004 * self.decay:
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
        st, eff, rec, cap = s
        st = max(st - min(need, st + self.extra), 0.0)
        if st >= self.eff_inc_thr and eff < 1.0:
            eff = min(eff + self.eff_inc, 1.0)
        inc = min(self.inc_max * rec, self.st_max - st, cap)
        s[:] = [st + inc, eff, rec, max(cap - inc, 0.0)]

    def _gap(self, s, gap, goal, disp, out, goalie=False):
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


def _kick_traj(ph, ox, oy, ux, uy, s0, delay, T):
    t = np.arange(T + 1)[None, :]
    k = np.clip(t - delay[:, None], 0, ph.G.size - 1)
    g = ph.G[k] * s0[:, None]
    ox = np.broadcast_to(ox, s0.shape)[:, None]
    oy = np.broadcast_to(oy, s0.shape)[:, None]
    return np.stack([ox + ux[:, None] * g, oy + uy[:, None] * g], axis=2)


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


def _first_true(mask):
    hit = mask.any(axis=1)
    return np.where(hit, mask.argmax(axis=1), NEVER)


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


class _KickResult:
    __slots__ = ("value", "p", "t_us", "t_th", "who", "lx", "ly", "succ", "v_fail",
                 "vs", "tx", "ty", "s0", "delay", "t_out")


class Agent1(StarterAgent):

    PASS_M0 = 0.8
    PASS_S = 0.6
    FIT_M0 = -1.9
    FIT_S = 2.8
    RINGS = ((3.0, 8), (6.0, 8), (9.0, 12))
    END_SPEEDS = (1.0, 1.4, 1.8)
    CHILD_RINGS = ((3.0, 8), (6.0, 8))
    CHILD_SPEEDS = (1.0, 1.8)
    MULTI_DELAY = 1
    KICK_LATE = 2
    P_PASS = 0.8
    P_DRIB = 0.9
    P_ATT = 0.65
    P_DRIB_ATT = 0.75
    LOSS_ATT = 30.0
    BACK_W = 1.0
    SLOT_W = 15.0
    DRIB_K = 6
    CHANNEL_V = 1.7
    CONE = 35.0
    BACK_X = -10.0
    DIAG_X0 = -15.0
    ENGAGE_D = 3.5
    ENGAGE_CONE = 25.0
    HOLD_ENGAGE = 15.0
    HOLD_V = 0.08
    CARRY_IN_P = 0.8
    CARRY_IN_D = 3.0
    V_BACKPASS = -0.85
    SHADE = 0.6
    SHOT_P = 0.7
    CHILD_K = 3
    BEAM_W = 10
    BEAM_D = 3
    BEAM_DIV = 3
    CHILD_ADV = 0.35
    BEAM_CARRIES = 2
    HOP_COST = 0.5
    SHOT_V0_US = 82.0
    BOX_SHOT_P = 0.7
    MARK_BEHIND = 8.0
    SHOT_KICK_LATE = 2
    FOUL_P = 0.8
    TOUCH_ST = 3000.0
    TOUCH_N = 2
    TOUCH_PRESS = 2.0
    TOUCH_P = 0.8
    CHAIN_MS = 300.0
    ONBALL_CAP_MS = 400.0
    ONBALL_SHARE = 10.0
    CHILD_DRIB_K = 2
    MID_LOSS = 18.0
    SHOT_V1 = 40.0
    SHOT_P_PRESSED = 0.3
    RUN_SPLIT_X = 20.0
    CLEAR_X = -30.0
    PRESS_D = 1.8
    HOLD_BAN_D = 2.2
    OUTLET_D = 4.0
    DRIBBLE_V = 0.73
    COMPACT = 0.7
    KB_GAP = 9.0
    KB_MIN_R = 7.0
    KB_MAX_R = 34.0
    ROLE_DEF = False
    ROLE_H = 5
    ROLE_K = 6
    ROLE_RUN = 0.05
    GK_M = 1.0
    GK_S = 0.5
    KICK_SCEN = False
    KICK_CONF = 0.6
    KS_BAND = 4
    KS_CHILD = False
    KN_RATE = 1.25
    CTRL_PHYS = False
    V_CTRL = 2.0
    TACKLE_REACH = False
    TACKLE_P = 0.8
    NOISE_SHOT = False
    SHOT_CONF = 0.71
    CENTER_GUARD = False
    GUARD_GAP = 2.0
    GUARD_GAP_ATT = 3.0
    GUARD_WG = 0.7
    GUARD_STICK = 4.0
    GUARD_URGENT = 2.5
    GUARD_REST = True
    TACKLE_ZONE = True
    TZ_X = 1.0
    FOUL_MAX_X = 20.0
    TZ_P_BOX = 0.6
    BACKPASS_X = 30.0
    SP_T = 30
    SP_BALL_D = 35.0
    SP_R = 25.0
    TZ_Y = 0.0
    CIRC_W = 0.5
    SELF_GOAL_W = 0.0
    CIRC_K = 1.0
    DEF_ROLES = False
    MARK_X = -5.0
    MARK_VIEW = 0.45
    COVER_FRAC = 0.35
    COVER_NEAR = 3.0
    COVER_FAR = 5.0
    JOB_STICK = 3.0
    SHAPE_BACK = 14.0
    SHAPE_SIDE_D = 6.0
    SHAPE_W = 7.0
    SHAPE_GOAL = 11.0
    REST_AHEAD = 10.0
    REST_BEHIND = 10.0
    REST_SPACE = 25.0
    LANE_GAP = 2.0
    BACK_STAY = 10.0
    JOCKEY_GAP = 1.2
    JOCKEY_GAP_BOX = 0.8
    FRONT_CLOSE = 4.0
    FRONT_STICK = 2
    HARRY_D = 6.0
    FRONT_STEP = 6.0
    MAN_LEAD = 4.0
    FRONT_AHEAD = 8.0
    CONTEST_T = 12
    CONTEST_D = 3
    STAM_MARGIN = 600.0
    OUTLET = False
    OUTLET_AHEAD = 15.0
    OUTLET_MIN = 8.0
    OUTLET_ATT = 2
    OUTLET_W = 16.0
    TURN_D = 2.2
    TURN_V = 0.5
    COUNTER = False
    COUNTER_T = 40
    COUNTER_DEF = 8
    CTR_RUN = 25.0
    CTR_LANE = 10.0
    FWD_CTR = 0.4
    BACK_BIAS_CTR = 1.0
    HOLD_CTR = 3.0
    OUT_Q = 0.5
    BACK_BIAS = 0.3
    TZ_P = 0.8
    RECOVER = True
    REC_ON = 0.5
    REC_OFF = 0.7
    REC_GAIN = 25.0
    DF_LINE_D = 20.0
    REC_SAFE_R = 25.0
    REC_SIGNAL = True
    REC_SIG_TOL = 0.5
    REC_SIG_N = 2
    REC_SIG_OFF = 15.0
    REC_SIG_GAP = 10
    KICK2_REAL = False
    KICK2_MID = 0.78
    KICK2_LOW = 0.42
    KICK2_ANG = 7.0
    REC_P_W = 25
    REC_P_WOFF = 40
    REC_P_MON = 0.0
    REC_P_MOFF = -300.0
    BLOCK_FRONT = False
    def __init__(self):
        super().__init__()
        self.ph = None
        self._ctx = None
        self._commit = None
        self._support_target = None
        self._prev_chaser = None
        self._running = None
        self._vmax_cache = {}
        self._stam = None
        self._stam_cycle = -1
        self._budget = {}
        self._floor_roles = (-1, set())
        self._guard_prev = None
        self._front_prev = None
        self._front_on = None
        self._job_prev = {}
        self._roles_c = -10
        self._outlet_y = None
        self._def_c0 = None
        self._counter_c = None
        self._ctx_c = -10
        self._counter_now = False
        self._budget_low = {}
        self._last_att_c = -100
        self._sp_c = -100
        self._recover = False
        self._rec_seen = {}


    def _phys(self, agent):
        if self.ph is None:
            self.ph = _Phys(agent.server_params)
        return self.ph

    def _allow_ms(self, agent, on_ball):
        rem = agent.budget.remaining_ms
        if rem == float("inf"):
            rem = 60000.0
        left = max(60, MATCH_CYCLES - agent.wm.cycle)
        per = rem / left
        cap = self.ONBALL_CAP_MS if on_ball else 25.0
        allow = min(per * (self.ONBALL_SHARE if on_ball else 0.7), cap,
                    agent.budget.decision_remaining_ms - 300.0)
        return max(allow, 1.0)

    def _context(self, agent):
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
            best = min((mate_t[i] - (0.5 if mates[i][0] == self._prev_chaser else 0.0),
                        mates[i][0]) for i in range(nm))
            chaser = best[1]
            best_t = int(mate_t[[m[0] for m in mates].index(chaser)])
            if gk_t < best_t:
                gi = min(int(gk_t), T)
                behind = B[0, gi, 0] < goalie[1] - 0.5
                if B[0, gi, 0] < -ph.pen_x and abs(B[0, gi, 1]) < ph.pen_hw and not behind:
                    chaser = None
        self._prev_chaser = chaser
        our_min = min(int(mate_t.min()) if nm else NEVER, int(gk_t))

        oi = min(opp_min, T)
        opp_first = None
        if opp_t.size and opp_min < NEVER:
            opp_first = opps[int(opp_t.argmin())][0]
        ui = min(our_min, T)

        self._ctx = {
            "key": key, "mates": mates, "goalie": goalie, "opps": opps,
            "mate_t": {mates[i][0]: int(mate_t[i]) for i in range(nm)},
            "ball": (bx, by, bvx, bvy), "chaser": chaser,
            "our_min": our_min, "opp_min": opp_min, "opp_first": opp_first,
            "our_loc": (float(B[0, ui, 0]), float(B[0, ui, 1])),
            "opp_loc": (float(B[0, oi, 0]), float(B[0, oi, 1])),
            "attack": (bool(wm.kickable_teammate_existance)
                       or (not wm.kickable_opponent_existance and nm > 0
                           and int(mate_t.min()) + (2 if bx > -25.0 else 5) <= opp_min)),
            "offside_x": wm.offside_line_x,
            "kickable_mate": bool(wm.kickable_teammate_existance),
            "kickable_opp": bool(wm.kickable_opponent_existance),
        }
        gd = None
        self._ctx["guard"], self._ctx["guard_spot"], self._ctx["guard_target"] = gd or (None, None, None)
        self._ctx["counter"] = self._counter_state(wm, self._ctx["attack"])
        return self._ctx

    def _root_state(self, agent, ctx):
        wm = agent.wm
        mates = [(m[0], m[1], m[2], m[3]) for m in ctx["mates"]]
        opps = [(o[0], o[1] + o[5] * 0.6, o[2] + o[6] * 0.6, o[3], o[4])
                for o in ctx["opps"]]
        bx, by = ctx["ball"][0], ctx["ball"][1]
        S = _State(self.ph, bx, by, wm.self.uniform_number, mates, opps,
                   ctx["offside_x"], opp_vel=[(o[5], o[6]) for o in ctx["opps"]],
                   mate_vel={m[0]: (m[4], m[5]) for m in ctx["mates"]}, lead=0.6,
                   tired_map=self._budget_low or self._budget)
        S.adv = self.CHILD_ADV
        kr = float(getattr(wm.self, "kick_rate", 0.0) or 0.0)
        if kr > 0.0:
            bvx, bvy = ctx["ball"][2], ctx["ball"][3]
            rate = (1.5 - kr / self.ph.kick_power_rate
                    + 0.5 + 0.5 * math.hypot(bvx, bvy) / (self.ph.bmax * self.ph.decay))
            S.kick_cond = (rate, kr, bvx, bvy)
        g = ctx.get("goalie")
        if g:
            S.own_gk = (g[1], g[2])
        return S

    def _risk(self, agent):
        wm = agent.wm
        lead = wm.our_team_score - wm.their_team_score
        late = wm.cycle > 1150
        if late and lead > 0:
            return self.PASS_M0 + 0.4, 1.0
        if late and lead < 0:
            return self.PASS_M0 - 0.2, 1.15
        return self.PASS_M0, 1.0


    def _pass_targets(self, S, rings=None, speeds=None, space=True):
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
        th = np.array([(x + dx, y + dy) for (_, x, y, _) in S.recv if x > -15.0
                       for dx in (8.0, 14.0) for dy in (-4.0, 0.0, 4.0)]).reshape(-1, 2)
        Dt = np.hypot(th[:, 0] - S.ox, th[:, 1] - S.oy)
        ok = ((np.abs(th[:, 0]) < ph.hl - 1.0) & (np.abs(th[:, 1]) < ph.hw - 1.0)
              & (Dt > 3.0) & (Dt < 40.0))
        for e in (0.3, 0.65):
            TX.append(th[ok, 0])
            TY.append(th[ok, 1])
            S0.append(np.minimum(ph.bmax, Dt[ok] * (1.0 - ph.decay) + e))
        return np.concatenate(TX), np.concatenate(TY), np.concatenate(S0)

    def _eval_kicks(self, S, tx, ty, s0, delay, m0=None, T=HORIZON, k_multi=1.0):
        r = self._eval_kicks1(S, tx, ty, s0, delay, m0=m0, T=T, k_multi=k_multi)
        return r

    def _eval_shots(self, S, s0, delay, n=13, T=HORIZON):
        ph = S.ph
        gy = np.linspace(-ph.gh + 0.6, ph.gh - 0.6, n)
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
        t_th = first.min(axis=1) if first.shape[1] else np.full(B.shape[0], NEVER)
        yc = S.oy + (ph.hl - S.ox) * dy / np.maximum(dx, 1e-6)
        dist = np.hypot(ph.hl - S.ox, yc - S.oy)
        e = 0.03 * dist + 0.3
        lo = np.maximum(yc - e, -ph.gh + 0.15)
        hi = np.minimum(yc + e, ph.gh - 0.15)
        p_on = np.clip((hi - lo) / (2.0 * e), 0.0, 1.0)
        rel = np.clip(t_th - t_goal, -6, 6).astype(float)
        p_beat = np.where(t_goal > T, 0.0, _sig((rel + 1.0) / 0.45))
        if gk.size:
            g = gk[:1]
            t_gk = _race(ph, B, S.op_pos[g], S.op_body[g], np.full(1, ph.catch_r), S.op_gk[g],
                         np.ones(1, dtype=np.int64), vel=S.ov[g], omni=np.ones(1, dtype=bool))[:, 0]
            p_beat = p_beat * np.where(t_goal > T, 0.0, _sig((t_gk - t_goal + self.GK_M) / self.GK_S))
        pg = p_on * p_beat
        return pg - (1.0 - pg) * 0.02, pg, gy

    def _deep_value(self, S, res, i, depth, deadline):
        best = float(res.vs[i])
        if depth <= 1 or time.perf_counter() > deadline:
            return best
        recv = S.recv[int(res.who[i])][0]
        child = S.child(recv, float(res.lx[i]), float(res.ly[i]), int(res.t_us[i]))
        if child.ox > 10.0:
            sv, _, _ = self._eval_shots(child, self.ph.bmax, 1)
            best = max(best, 0.92 * float(sv.max()))
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
            best = max(best, 0.88 * float(cv.max()))
        return best

    def _kick_delay(self, agent, S, tx, ty, s0):
        ang = np.degrees(np.arctan2(ty - S.oy, tx - S.ox))
        vmax = np.array([self._vmax(agent, a) for a in ang])
        return np.where(vmax >= s0 * 0.98, 0, self.MULTI_DELAY).astype(np.int64)

    def _shot1_speeds(self, agent, S, n=13):
        ph = self.ph
        gy = np.linspace(-ph.gh + 0.6, ph.gh - 0.6, n)
        return np.array([min(self._vmax(agent, math.degrees(math.atan2(y - S.oy, ph.hl + 2.0 - S.ox))),
                             ph.bmax) for y in gy])

    def _vmax(self, agent, angle_deg):
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


    def _stam_cycle_start(self, agent):
        wm = agent.wm
        if wm.self.is_goalie:
            return
        ph = self._phys(agent)
        if wm.cycle != self._stam_cycle:
            self._stam_cycle = wm.cycle
            if self._stam is None:
                self._stam = StaminaPredictor(agent.server_params)
                self._stam.half_cycles = float(HALF_CYCLES)
            P = self._stam
            prev = P.last
            P.update(agent)
            if self._floor_roles[0] == prev and wm.cycle == (prev or -9) + 1:
                for u in self._floor_roles[1]:
                    need = P.last_need.get(("us", u))
                    if (need is not None and abs(need - 45.0) <= 0.3 and ("us", u) in P.state
                            and P.state[("us", u)][0] > ph.st_floor + 1.0):
                        P.state[("us", u)][0] = ph.st_floor
                        note(agent, "floor_hit", u=u)
            self._budget = {k[1]: int(min(max((P.estimate(k) - ph.st_floor) / SPRINT_NET, 0.0), ph.J))
                            for k in P.state if k[0] == "us"}
            self._budget_low = {k[1]: int(min(max((P.estimate(k) - self.STAM_MARGIN - ph.st_floor)
                                                  / SPRINT_NET, 0.0), ph.J))
                                for k in P.state if k[0] == "us"}
            self._budget_low[wm.self.uniform_number] = int(min(max(
                (float(wm.self.stamina) - ph.st_floor) / SPRINT_NET, 0.0), ph.J))
            st = float(wm.self.stamina)
            cap = float(getattr(wm.self, "stamina_capacity", 1.0) or 0.0)
            if cap < 0.01:
                self._recover = False
            elif st < self.REC_ON * ph.stamina_max:
                self._recover = True
            elif st > self.REC_OFF * ph.stamina_max:
                self._recover = False
            if wm.cycle % 25 == 0:
                note(agent, "stam", rec=int(self._recover), cap=round(cap),
                     **{f"u{k[1]}": round(P.estimate(k)) for k in sorted(P.state) if k[0] == "us"})
            if self.RECOVER and self.REC_SIGNAL:
                self._rec_signal(agent, P)
        if self._floor_roles[0] != wm.cycle:
            self._floor_roles = (wm.cycle, self._flat_out(self._context(agent)))

    def _flat_out(self, ctx):
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
        return np.array([self._budget.get(m[0], ph.J) for m in mates]
                        + ([ph.J] if goalie else []) + [ph.J] * len(opps), dtype=np.int64)


    def on_ball(self, agent) -> None:
        try:
            self._stam_cycle_start(agent)
            self._on_ball(agent)
        except Exception as e:
            note(agent, "on_ball_error", err=str(e)[:120])
            Action.clear(agent)

    def _on_ball(self, agent):
        wm = agent.wm
        ph = self._phys(agent)
        if wm.self.is_goalie:
            Action.clear(agent)
            return
        start = time.perf_counter()
        deadline = start + self._allow_ms(agent, True) / 1000.0
        ctx = self._context(agent)
        self._counter_now = bool(ctx.get("counter"))
        if self._counter_now:
            note(agent, "ctr")
        S = self._root_state(agent, ctx)
        m0, shot_w = self._risk(agent)

        c = self._commit
        if c is not None and wm.cycle - c["cycle"] == 1 and self._pressure(S)[0] < 1.5:
            ang = math.degrees(math.atan2(c["ty"] - S.oy, c["tx"] - S.ox))
            if not self._cone_mask(S, [ang])[0]:
                note(agent, "abort_" + c["kind"])
                c = None
        if c is not None and c["kind"] == "pass" and wm.cycle - c["cycle"] == 1:
            D = math.hypot(c["tx"] - S.ox, c["ty"] - S.oy)
            n_left = max(c["arrive"] - 1, 1)
            s0 = min(ph.bmax, D * (1.0 - ph.decay) / (1.0 - ph.decay ** n_left))
            r = self._eval_kicks(S, np.array([c["tx"]]), np.array([c["ty"]]),
                                 np.array([s0]), np.zeros(1, dtype=np.int64))
            if not r.succ[0] or r.p[0] < 0.5:
                note(agent, "abort_lane", p=round(float(r.p[0]), 2))
                c = None
        if (c is not None and wm.cycle - c["cycle"] == 1 and c["age"] < 2
                and wm.last_kicker_uniform_number == wm.self.uniform_number):
            c["cycle"], c["age"] = wm.cycle, c["age"] + 1
            note(agent, "continue_" + c["kind"])
            Action.kick(agent, Vector2D(c["tx"], c["ty"]), c["arrive"],
                        accept_late=self.SHOT_KICK_LATE if c["kind"] == "shot" else self.KICK_LATE)
            if c["kind"] == "shot":
                Action.kick_one_step(agent, Vector2D(c["tx"], c["ty"]))
            Action.hold(agent)
            return
        self._commit = None

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
        if math.hypot(ph.hl - bx, by) < 32.0:
            best_shot = None
            v1 = self._vmax(agent, math.degrees(math.atan2(-by, ph.hl - bx)))
            v1s = self._shot1_speeds(agent, S)
            if v1s.max() >= 1.8:
                sv, pg, gy = self._eval_shots(S, v1s, 0)
                i = int(np.argmax(sv))
                best_shot = (float(sv[i]) * shot_w, "shot1", (float(gy[i]), float(pg[i])))
            sv, pg, gy = self._eval_shots(S, ph.bmax, 1 if v1 > 1.2 else 2)
            i = int(np.argmax(sv))
            cand = (float(sv[i]) * shot_w, "shot", (float(gy[i]), float(pg[i])))
            if best_shot is None or cand[0] > best_shot[0]:
                best_shot = cand
            if not heavy or best_shot[2][1] >= 0.15:
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
            drib = [o for o, c in zip(drib, clear) if c and o[2][2] >= 0.5]
        options.extend(drib)

        if (d_opp >= self.HOLD_BAN_D and t_opp > 2) or self._by_their_line(S):
            options.append(self._eval_hold(agent, S))
        if S.ox < self.CLEAR_X:
            options.append(self._eval_clear(S))

        if res is not None:
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
            options.extend(last or [(-1.0, "clear", None)])

        tops = {}
        for o in options:
            tops[o[1]] = max(tops.get(o[1], -9.0), round(o[0], 3))
        note(agent, "fb", d=round(d_opp, 1), last=int(cornered), **tops)
        best = max(options, key=lambda o: o[0])
        self._execute(agent, S, best, res)

    def _pressure(self, S):
        if S.nop == 0:
            return 99.0, NEVER
        d = float(np.hypot(S.op_pos[:, 0] - S.ox, S.op_pos[:, 1] - S.oy).min())
        T = 4
        B = np.tile(np.array([S.ox, S.oy]), (1, T + 1, 1))
        first = _race(S.ph, B, S.op_pos, S.op_body, np.full(S.nop, S.ph.kick_r + 0.5),
                      S.op_gk, np.zeros(S.nop, dtype=np.int64))
        return d, int(first.min())

    def _multi_keep(self, S):
        d = self._pressure(S)[0]
        return (0.32 if d < 1.8 else 0.48 if d < 2.2 else 0.72 if d < 2.6
                else 0.87 if d < 3.5 else 0.94 if d < 4.0 else 1.0)

    @staticmethod
    def _live(r):
        return r.succ & (r.t_th > r.t_us)

    def _channel_kick(self, agent, S):
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
        angs = np.asarray(angs, dtype=float)
        if S.nop == 0:
            return np.ones(angs.size, dtype=bool)
        d = np.hypot(S.op_pos[:, 0] - S.ox, S.op_pos[:, 1] - S.oy)
        a = np.degrees(np.arctan2(S.op_pos[:, 1] - S.oy, S.op_pos[:, 0] - S.ox))
        off = np.abs(_norm_deg(angs[:, None] - a[None, :]))
        ok = ((off > self.CONE) | (d[None, :] >= 3.0)).all(axis=1)
        wide = ok & ((off > 90.0) | (d[None, :] >= S.ph.kick_r + 0.3)).all(axis=1)
        return wide if wide.any() else ok

    def _bounce_pass(self, agent, S):
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


    def _eval_kicks1(self, S, tx, ty, s0, delay, m0=None, T=HORIZON, k_multi=1.0):
        ph = S.ph
        m0 = self.PASS_M0 if m0 is None else m0
        C = tx.size
        dx, dy = tx - S.ox, ty - S.oy
        D = np.maximum(np.hypot(dx, dy), 1e-6)
        ux, uy = dx / D, dy / D
        B = _kick_traj(ph, S.ox, S.oy, ux, uy, s0, delay, T)
        tackle = None
        first = _race(ph, B, S.P_race, S.body, S.ctrl, S.gk, S.react, vel=S.vel, omni=S.omni,
                      tired=S.tired, tackle=tackle)
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
        margin = np.clip(t_th - t_us, -6, 6).astype(float)
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
        r.t_out = out
        return r

    def _field_eval(self, x, y):
        slot = self.SLOT_W * _sig((x - 31.0) / 2.0) * _sig((12.0 - np.abs(y)) / 2.0)
        return x + 100.0 * (self.ph.threat(x, y) - 0.05) + slot

    def _chain_scores(self, S, r, pressed):
        att = (S.ox > 15.0) | (r.lx > 22.0)
        safe = r.succ & (r.p >= np.where(att, self.P_ATT, self.P_PASS))
        score = (self._field_eval(r.lx, r.ly) - 0.05 * r.t_us
                 - np.where(att, self.LOSS_ATT, self.MID_LOSS) * (1.0 - r.p))
        if S.ox > self.BACK_X and not pressed:
            score = score - self.BACK_W * np.maximum(0.0, S.ox - 4.0 - r.lx)
        if self.BACK_BIAS:
            bias = self.BACK_BIAS_CTR if self._counter_now else self.BACK_BIAS
            score = score - bias * np.maximum(0.0, S.ox - r.lx)
        if self._counter_now:
            score = score + self.FWD_CTR * np.maximum(0.0, r.lx - S.ox)
        return safe, np.where(safe, score, -np.inf)

    def _step_cost(self, S, r):
        att = (S.ox > 15.0) | (r.lx > 22.0)
        return 0.05 * r.t_us + np.where(att, self.LOSS_ATT, self.MID_LOSS) * (1.0 - r.p)

    def _on_ball_chain(self, agent, S, deadline, shot_w):
        ph = self.ph
        opts = []

        if math.hypot(ph.hl - S.ox, S.oy) < 30.0:
            v1 = self._vmax(agent, math.degrees(math.atan2(-S.oy, ph.hl - S.ox)))
            v1s = self._shot1_speeds(agent, S)
            if v1s.max() >= 1.8:
                sv, pg, gy = self._eval_shots(S, v1s, 0)
                kind = "shot1"
                d_opp, t_opp = self._pressure(S)
                if pg.max() * shot_w < self.SHOT_P and d_opp >= self.HOLD_BAN_D and t_opp > 2:
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
            if pg[i] * shot_w >= p_need:
                carry = self._carry_in(agent, S, float(pg[i]))
                if carry is not None:
                    note(agent, "carry_in", pg=round(float(pg[i]), 2), pg2=round(carry[2], 2))
                    self._execute(agent, S, (carry[2], "dribble", (carry[0], carry[1], carry[3], carry[4])),
                                  None)
                    return True
                self._execute(agent, S, (float(sv[i]), kind, (float(gy[i]), float(pg[i]))), None)
                return True

        t0 = time.perf_counter()
        pressed = self._pressure(S)[0] < self.PRESS_D
        tx, ty, s0 = self._pass_targets(S)
        res = None
        if tx.size:
            delay = self._kick_delay(agent, S, tx, ty, s0)
            res = self._eval_kicks(S, tx, ty, s0, delay, k_multi=self._multi_keep(S))
            safe, base = self._chain_scores(S, res, pressed)
            score = base.copy()
        drib = self._drib_options(S, pressed)
        cost = self._step_cost(S, res) if res is not None else None
        dcost = [float(self._field_eval(d[1], d[2])) - d[0] for d in drib]
        level = self._beam_chain(S, res, safe if res is not None else None,
                                 base if res is not None else None, cost,
                                 score if res is not None else None, drib, dcost, deadline)
        if res is not None:
            j = int(np.argmax(score))
            if np.isfinite(score[j]):
                opts.append((float(score[j]), "pass", j))
        for d in drib:
            opts.append((d[0], "dribble", (d[1], d[2], d[5], d[6])))

        T = 4
        B = np.tile(np.array([S.ox, S.oy]), (1, T + 1, 1))
        first = _race(ph, B, S.op_pos, S.op_body, np.full(S.nop, ph.kick_r + 0.5),
                      S.op_gk, np.zeros(S.nop, dtype=np.int64))
        t_th = float(first.min()) if S.nop else NEVER
        if t_th > T:
            opts.append((float(self._field_eval(S.ox, S.oy)) - 1.5
                         - (self.HOLD_CTR if self._counter_now else 0.0), "hold", None))

        if not opts:
            return False
        best = max(opts, key=lambda o: o[0])
        note(agent, "chain", k=best[1], lv=level,
             ms=round(1000.0 * (time.perf_counter() - t0), 1),
             cut=int(time.perf_counter() > deadline))
        self._execute(agent, S, best, res)
        return True

    def _beam_chain(self, S, res, safe, base, cost, score, drib, dcost, deadline):
        nodes = []
        if res is not None:
            for i in np.flatnonzero(safe):
                nodes.append((float(base[i]), float(cost[i]), int(i),
                              lambda i=i: self._child(S, res, i), self.BEAM_CARRIES,
                              (-1, int(res.who[i]))))
        for n_d, d in enumerate(drib[:self.DRIB_K]):
            nodes.append((d[0], dcost[n_d], -1 - n_d,
                          lambda d=d: S.child(S.kicker, d[1], d[2], d[3], body=d[4]),
                          self.BEAM_CARRIES - 1, (-1, -1 - n_d)))
        def raise_root(root, v):
            if root >= 0:
                score[root] = max(score[root], v)
            else:
                drib[-1 - root][0] = max(drib[-1 - root][0], v)

        return self._beam_run(nodes, self._beam_expand, raise_root, self.BEAM_W, self.BEAM_D,
                              deadline)

    def _beam_run(self, nodes, expand, raise_root, width, depth, deadline):
        beam = self._beam_keep(nodes, width)
        layer = 0
        while beam and layer < depth and time.perf_counter() < deadline:
            layer += 1
            nxt = []
            for b, (_, g, root, make, may_drib, _) in enumerate(beam):
                if time.perf_counter() > deadline:
                    break
                out = expand(make(), g, may_drib, b)
                for v, g2, make2, may2, key in out:
                    raise_root(root, v)
                    if make2 is not None:
                        nxt.append((v, g2, root, make2, may2, key))
            beam = self._beam_keep(nxt, width)
        return layer

    def _beam_keep(self, nodes, width):
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

    def _beam_expand(self, C, g, may_drib, b):
        ph = self.ph
        out = []
        if C.ox > 20.0:
            _, pg, _ = self._eval_shots(C, ph.bmax, 1)
            if pg.max() >= self.SHOT_P:
                out.append((self.SHOT_V0_US + self.SHOT_V1 * float(pg.max()) - g, 0.0, None,
                            False, None))
        pressed = self._pressure(C)[0] < self.PRESS_D
        tx, ty, s0 = self._pass_targets(C, rings=self.CHILD_RINGS, speeds=self.CHILD_SPEEDS)
        if tx.size:
            r = self._eval_kicks(C, tx, ty, s0, np.ones(tx.size, dtype=np.int64))
            safe, sc = self._chain_scores(C, r, pressed)
            if safe.any():
                step = self._step_cost(C, r)
                for j in np.flatnonzero(safe):
                    out.append((float(sc[j]) - self.HOP_COST - g,
                                g + self.HOP_COST + float(step[j]),
                                lambda j=j: self._child(C, r, j), self.BEAM_CARRIES,
                                (b, C.recv[int(r.who[j])][0])))
        if may_drib > 0:
            for d in self._drib_options(C, pressed)[:self.CHILD_DRIB_K]:
                step = float(self._field_eval(d[1], d[2])) - d[0]
                out.append((d[0] - 1.0 - g, g + 1.0 + step,
                            lambda d=d: C.child(C.kicker, d[1], d[2], d[3], body=d[4]),
                            may_drib - 1, (b, -1)))
        return out

    def _carry_in(self, agent, S, pg):
        ph = self.ph
        if pg >= self.CARRY_IN_P or S.ox > ph.hl - 12.0 or S.nop == 0:
            return None
        if self._pressure(S)[0] < self.CARRY_IN_D:
            return None
        ga = math.atan2(-S.oy, ph.hl - S.ox)
        cands = [c for c in self._dribs(S)
                 if c[0] >= 0.9 and abs(_norm_deg(math.degrees(math.atan2(c[2] - S.oy, c[1] - S.ox) - ga)))
                 <= 40.0]
        cands.sort(key=lambda c: math.hypot(ph.hl - c[1], c[2]))
        best = None
        for p, ex, ey, K, _, _, exe in cands[:6]:
            _, pg2, _ = self._eval_shots(S.child(S.kicker, ex, ey, K), ph.bmax, 1)
            v = p * float(pg2.max())
            if v > pg + 0.05 and (best is None or v > best[2]):
                best = (ex, ey, v, p, exe)
        return best

    def _child(self, S, res, i):
        recv = S.recv[int(res.who[i])][0]
        return S.child(recv, float(res.lx[i]), float(res.ly[i]), int(res.t_us[i]))

    def _drib_options(self, S, pressed):
        attack = S.ox > 0.0
        p_min = self.P_DRIB_ATT if attack else self.P_DRIB
        out = []
        for p, ex, ey, K, _, _, exe in self._dribs(S):
            if p < p_min:
                continue
            score = float(self._field_eval(ex, ey)) - 0.05 * K
            if self.CIRC_W and not self._counter_now:
                gx = S.ph.hl - ex
                score -= self.CIRC_W * (math.hypot(gx, self.CIRC_K * ey) - gx)
            if self._counter_now:
                score += self.FWD_CTR * max(0.0, ex - S.ox) - self.BACK_BIAS_CTR * max(0.0, S.ox - ex)
            if attack:
                score -= self.LOSS_ATT * (1.0 - p)
            if S.ox > self.BACK_X and not pressed:
                score -= self.BACK_W * max(0.0, S.ox - 4.0 - ex)
            out.append([score, ex, ey, K, math.degrees(math.atan2(ey - S.oy, ex - S.ox)), p, exe])
        out.sort(key=lambda d: -d[0])
        return out

    def _eval_dribbles(self, agent, S):
        ph = S.ph
        out = []
        for p, ex, ey, K, lx, ly, exe in self._dribs(S):
            value = (p * GAMMA ** K * float(ph.v_keep(ex, ey, S.space(ex, ey)))
                     + (1.0 - p) * float(ph.v_lose(lx, ly)))
            out.append((value, "dribble", (ex, ey, p, exe, K, lx, ly)))
        return out

    def _dribs(self, S):
        ph = S.ph
        me = next((m for m in S.mates if m[0] == S.kicker), None)
        if me is None:
            return []
        px, py, body = me[1], me[2], me[3]
        pvx, pvy = S.mv.get(S.kicker, (0.0, 0.0))
        if S.kick_cond is not None:
            _, kr, bvx, bvy = S.kick_cond
        else:
            kr, bvx, bvy = 0.85 * ph.kick_power_rate, 0.0, 0.0
        acc_max = min(ph.kick_max, 100.0 * kr)
        vmax = 1.05
        acc = 0.6
        light = S.kick_cond is None
        sprint = ph.J if S.tired_map is None else S.tired_map.get(S.kicker, ph.J)

        def run(ang, n_turn, n_dash):
            x, y, vx, vy = px, py, pvx, pvy
            ux, uy = math.cos(math.radians(ang)), math.sin(math.radians(ang))
            out = []
            for k in range(1 + n_turn + n_dash):
                if k > n_turn:
                    a = acc if k - n_turn <= sprint else 0.45 * acc
                    vx, vy = vx + a * ux, vy + a * uy
                    v = math.hypot(vx, vy)
                    if v > vmax:
                        vx, vy = vx * vmax / v, vy * vmax / v
                x, y = x + vx, y + vy
                vx, vy = vx * ph.pdecay, vy * ph.pdecay
                out.append((x, y))
            return out

        def kickable(tx, ty, n):
            g = ph.G[n]
            fvx, fvy = (tx - S.ox) / g, (ty - S.oy) / g
            if math.hypot(fvx, fvy) > ph.bmax or math.hypot(fvx - bvx, fvy - bvy) > acc_max:
                return None
            return fvx, fvy

        cands = []
        sp0 = math.hypot(pvx, pvy) * ph.pdecay
        for a in range(0, 16, 2 if light else 1):
            rel = 22.5 * a
            ang = _norm_deg(body + rel)
            if px < 16.0 and abs(ang) > 100.0:
                continue
            if px < -36.0 and abs(py) < 20.0 and abs(ang) > 45.0:
                continue
            n_turn, diff, spd = 0, abs(_norm_deg(rel)), sp0
            while diff > 10.0:
                diff -= 180.0 / (1.0 + INERTIA * spd)
                spd *= ph.pdecay
                n_turn += 1
            if n_turn >= 3:
                continue
            ux, uy = math.cos(math.radians(ang)), math.sin(math.radians(ang))
            pos = run(ang, n_turn, 4)
            for n_dash in (4, 3, 2):
                n = 1 + n_turn + n_dash
                tx, ty = pos[n - 1][0] + 0.525 * ux, pos[n - 1][1] + 0.525 * uy
                if abs(tx) > ph.hl - 1.0 or abs(ty) > ph.hw - 1.0:
                    continue
                fv = kickable(tx, ty, n)
                if fv is None:
                    continue
                if math.hypot(S.ox + fv[0] - pos[0][0], S.oy + fv[1] - pos[0][1]) < 0.485:
                    continue
                cands.append(("sdrib", tx, ty, n, tx, ty, fv[0], fv[1], ang))
        max_turn = 180.0 / (1.0 + INERTIA * sp0)
        top = (6 if S.ox < -20.0 else 7 if S.ox < 0.0 else 13 if S.ox < 10.0
               else 15 if S.ox < 20.0 else 20)
        top = min(top, int(sprint))
        if top >= 5:
            for a in range(0, 60, 3 if light else 1):
                rel = _norm_deg(6.0 * a)
                if abs(rel) > max_turn:
                    continue
                n_turn = 0 if a == 0 else 1
                ang = _norm_deg(body + rel)
                if S.ox < ph.pen_x + 5.0 and abs(ang) > 85.0:
                    continue
                pos = run(ang, n_turn, top)
                n_dash = top
                while n_dash >= 5:
                    n = 1 + n_turn + n_dash
                    rx, ry = pos[n - 1]
                    ok = (rx < ph.hl - 2.5 and abs(ry) < ph.hw - 1.0
                          and math.hypot(rx + ph.hl, ry) >= 18.0)
                    fv = kickable(rx, ry, n) if ok else None
                    if fv is not None:
                        cands.append(("selfpass", rx, ry, n, rx, ry, fv[0], fv[1], ang))
                    n_dash -= 2 if n_dash > 10 else 1
        if not cands:
            return []

        C = len(cands)
        n_arr = np.array([c[3] for c in cands], dtype=np.int64)
        T = int(n_arr.max()) + 4
        t = np.arange(T + 1)
        fvx = np.array([c[6] for c in cands])
        fvy = np.array([c[7] for c in cands])
        k = np.minimum(t[None, :], n_arr[:, None])
        B = np.stack([S.ox + fvx[:, None] * ph.G[k], S.oy + fvy[:, None] * ph.G[k]], axis=2)
        out = []
        if S.nop:
            first = _race(ph, B, S.op_pos, S.op_body, S.op_ctrl, S.op_gk,
                          np.zeros(S.nop, dtype=np.int64), vel=S.ov)
            tx = np.array([c[1] for c in cands])
            ty = np.array([c[2] for c in cands])
            mang = np.arctan2(ty - S.oy, tx - S.ox)
            relx = ((S.op_pos[None, :, 0] - S.ox) * np.cos(mang)[:, None]
                    + (S.op_pos[None, :, 1] - S.oy) * np.sin(mang)[:, None])
            short = np.array([c[0] == "sdrib" for c in cands])
            bonus = np.where(short[:, None],
                             (tx < 30.0)[:, None].astype(int) + (tx < 0.0)[:, None].astype(int),
                             np.where(relx > 0.8, 1 + (tx < 27.0)[:, None].astype(int), 0))
            margin = np.where(relx < -4.0, NEVER, first - bonus - n_arr[:, None]).min(axis=1)
        else:
            margin = np.full(C, NEVER)
        for i, c in enumerate(cands):
            if margin[i] < 1:
                continue
            p = 0.97 if margin[i] >= NEVER // 2 else min(0.97, 0.88 + 0.03 * float(margin[i]))
            out.append((p, c[4], c[5], c[3], c[1], c[2], (c[1], c[2], c[3], c[0])))
        return out

    def _eval_hold(self, agent, S):
        ph = S.ph
        d, _ = self._pressure(S)
        p = 0.0 if d < 1.0 else 0.45 if d < 2.0 else 0.8 if d < 3.0 else 0.95 if d < 5.0 else 0.98
        lose = float(ph.v_lose(S.ox, S.oy))
        if self._by_their_line(S):
            lose = self.OUT_Q * float(ph.v_keep(S.ox, S.oy, 0.0)) + (1.0 - self.OUT_Q) * lose
        v = (p * (float(ph.v_keep(S.ox, S.oy, S.space(S.ox, S.oy))) - 0.012) + (1.0 - p) * lose)
        return (v, "hold", None)

    def _eval_clear(self, S):
        ph = S.ph
        lx = min(ph.hl - 5.0, S.ox + 30.0)
        ly = math.copysign(min(ph.hw - 3.0, abs(S.oy) + 12.0), S.oy if S.oy else 1.0)
        v = 0.35 * float(ph.v_keep(lx, ly, 4.0)) + 0.65 * float(ph.v_out(lx, ly))
        return (v, "clear", None)

    def _execute(self, agent, S, best, res):
        if self._touch_out(agent, S, best, res):
            return
        self._execute_base(agent, S, best, res)

    def _touch_out(self, agent, S, best, res):
        ph = self.ph
        if S.ox >= 0.0 or self._pressure(S)[0] > self.TOUCH_PRESS:
            return False
        value, kind, payload = best
        if kind in ("shot", "shot1", "clear"):
            return False
        if kind == "pass":
            p = float(res.p[payload])
        elif kind in ("pass1", "channel"):
            p = float(getattr(payload[0], "p")[payload[1]])
        elif kind == "dribble":
            p = float(payload[2])
        else:
            p = 0.5
        if p >= self.TOUCH_P:
            return False
        me = agent.wm.self.uniform_number
        st = {m[0]: (agent.wm.self.stamina if m[0] == me else
                     (self._stam.estimate(("us", m[0])) if self._stam is not None else ph.stamina_max))
              for m in self._context(agent)["mates"]}
        tired = sum(1 for v in st.values() if v < self.TOUCH_ST)
        if tired < self.TOUCH_N:
            return False
        side = 1.0 if S.oy >= 0.0 else -1.0
        tx = min(S.ox + 10.0, ph.hl - 2.0)
        note(agent, "touch_out", k=kind, p=round(p, 2), tired=tired, x=round(S.ox, 1))
        Action.kick_one_step(agent, Vector2D(tx, side * (ph.hw + 5.0)))
        return True

    def _execute_base(self, agent, S, best, res):
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
            Action.kick(agent, Vector2D(tx, ty), arrive, accept_late=self.KICK_LATE)
            Action.hold(agent)
        elif kind == "pass1":
            r, i = payload
            note(agent, "pass", v=round(value, 3), p=round(float(r.p[i]), 2),
                 t=int(r.t_us[i]), m=int(min(r.t_th[i], 99)) - int(r.t_us[i]),
                 d=0, s0=round(float(r.s0[i]), 2), one=1)
            Action.kick_one_step(agent, Vector2D(float(r.tx[i]), float(r.ty[i])))
        elif kind == "channel":
            r, i = payload
            note(agent, "channel", v=round(value, 3), s0=round(float(r.s0[i]), 2))
            Action.kick_one_step(agent, Vector2D(float(r.tx[i]), float(r.ty[i])))
        elif kind in ("shot", "shot1"):
            gy, pg = payload
            gy = S.oy + (gy - S.oy) * (ph.hl - S.ox) / max(ph.hl + 2.0 - S.ox, 1.0)
            target = Vector2D(ph.hl, gy)
            note(agent, kind, v=round(value, 3), pg=round(pg, 2))
            if kind == "shot1":
                Action.kick_one_step(agent, target)
            else:
                D = math.hypot(ph.hl - S.ox, gy - S.oy)
                arrive = 2
                while arrive < 60 and D * (1.0 - ph.decay) / (1.0 - ph.decay ** arrive) >= ph.bmax:
                    arrive += 1
                self._commit = {"cycle": wm.cycle, "age": 0, "kind": "shot",
                                "tx": ph.hl, "ty": gy, "arrive": arrive}
                Action.kick(agent, target, arrive, accept_late=self.SHOT_KICK_LATE)
                Action.kick_one_step(agent, target)
        elif kind == "dribble":
            tx, ty, arrive, sub = payload[3]
            note(agent, sub, v=round(value, 3), p=round(float(payload[2]), 2), n=int(arrive))
            Action.kick(agent, Vector2D(tx, ty), int(arrive), accept_late=1)
            Action.dribble(agent, Vector2D(tx, ty))
        elif kind == "hold":
            note(agent, "hold", v=round(value, 3))
            Action.hold(agent)
        else:
            note(agent, "clear", v=round(value, 3))
        if kind != "clear" and S.ox > self.CLEAR_X:
            Action.hold(agent)
        Action.clear(agent)

    def _by_their_line(self, S):
        ph = S.ph
        return S.ox > ph.pen_x and (S.ox > ph.hl - 8.0 or abs(S.oy) > ph.hw - 6.0)


    def off_ball(self, agent) -> None:
        try:
            self._stam_cycle_start(agent)
            self._off_ball(agent)
        except Exception as e:
            note(agent, "off_ball_error", err=str(e)[:120])
            Action.go_to_point(agent, Tools.home_position(agent))

    def _off_ball(self, agent):
        wm = agent.wm
        ph = self._phys(agent)
        ctx = self._context(agent)
        me = wm.self.uniform_number
        ball = wm.ball.position

        if (self.TACKLE_ZONE and wm.kickable_opponent_existance and not wm.kickable_teammate_existance
                and not (abs(ball.x()) > ph.hl - self.TZ_X and abs(ball.y()) < ph.pen_hw + self.TZ_Y)):
            mp = wm.self.position
            own_box = mp.x() < -ph.pen_x + 0.5 and abs(mp.y()) < ph.pen_hw + 0.5
            fair = wm.self.tackle_probability >= (self.TZ_P_BOX if own_box else self.TZ_P)
            foul = (wm.self.foul_probability >= self.FOUL_P and int(wm.self.card) == 0
                    and not own_box and ball.x() <= self.FOUL_MAX_X)
            if foul or fair:
                want = (ball - Vector2D(-ph.hl, 0.0)).th().degree()
                rel = _norm_deg(want - wm.self.body_direction)
                dirs = np.arange(-90.0, 91.0, 10.0)
                gain = (1.0 - np.abs(dirs) / 180.0) * np.cos(np.radians(rel - dirs))
                bv = wm.ball.velocity
                ra = np.radians(wm.self.body_direction + dirs)
                pw = ph.kick_max * (1.0 - np.abs(dirs) / 180.0)
                vx, vy = bv.x() + pw * np.cos(ra), bv.y() + pw * np.sin(ra)
                sp = np.maximum(np.hypot(vx, vy), 1e-6)
                run = np.minimum(sp / (1.0 - ph.decay), 20.0)
                inside = ((np.abs(ball.x() + vx / sp * run) < ph.hl)
                          & (np.abs(ball.y() + vy / sp * run) < ph.hw))
                safe = np.ones(dirs.size, dtype=bool)
                if ball.x() < -ph.hl + self.BACKPASS_X:
                    gx, gy = -ph.hl - ball.x(), -ball.y()
                    gd = math.hypot(gx, gy) or 1.0
                    safe = (vx * gx + vy * gy) / (sp * gd) < 0.3
                if (inside & safe).any():
                    gain = np.where(inside & safe, gain, -np.inf)
                elif safe.any():
                    gain = np.where(safe, gain, -np.inf)
                elif inside.any():
                    gain = np.where(inside, gain, -np.inf)
                k = int(np.argmax(gain))
                note(agent, "zone_tackle", foul=int(foul), p=round(wm.self.tackle_probability, 2))
                Action.tackle(agent, float(dirs[k]), foul=foul)
                Action.intercept(agent)
                return

        if not wm.kickable_teammate_existance and ctx["chaser"] == me:
            my_t = ctx["mate_t"].get(me, NEVER)
            if ctx["attack"] or my_t <= ctx["opp_min"] + 1 or self._contest(ctx):
                won = ctx["attack"] or my_t <= ctx["opp_min"] + 1
                note(agent, "chase" if won else "contest", t=my_t)
                Action.intercept(agent)
                Action.go_to_point(agent, ball, max_dash_power=100.0)
                return
        if not ctx["attack"] and me in self._busy(ctx):
            plan = self._block_plan(ctx)
            if plan is not None and plan[0] == me:
                power = self._dash_power(agent, True, serious=self._serious(ctx))
                engage = 12.0
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
                                and d_aim * abs(math.sin(math.radians(rel))) < ph.kick_r - 0.25)
                    hold_run = not straight and engage == self.HOLD_ENGAGE and d_ball >= self.ENGAGE_D
                    note(agent, "engage", t=ctx["mate_t"].get(me, NEVER),
                         run=2 if hold_run else int(straight))
                    if straight:
                        ahead = wm.self.position + Vector2D.polar2vector(
                            d_aim * math.cos(math.radians(rel)) + 3.0, wm.self.body_direction)
                        Action.go_to_point(agent, ahead,
                                           max_dash_power=power, distance_threshold=0.5)
                    elif hold_run:
                        Action.go_to_point(agent, ball, max_dash_power=power,
                                           distance_threshold=1.0)
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
                return

        t_need, serious = None, False
        if ctx["attack"]:
            target = self._support_point(agent, ctx)
            dist = wm.self.position.dist(target)
            urgent = (self._running == "run" and ball.x() > 15.0) or self._running == "counter"
            label = self._running or "support"
        else:
            target = self._defend_point(agent, ctx)
            cx, cy = self._carrier(ctx)
            dist = wm.self.position.dist(target)
            serious = self._serious(ctx)
            bv = wm.ball.velocity
            deep = ball.x() > 0.0 and not (bv.x() < -1.0 and bv.r() > 1.5)
            urgent = not deep and ((cx < 15.0 and wm.self.position.x() > cx - 4.0)
                                   or (ball.x() < -10.0 and dist > 4.0) or dist > 12.0)
            t_need = (min(ctx["opp_min"], HORIZON)
                      + max(math.hypot(cx - target.x(), cy - target.y()) - 2.0, 0.0) / self.DRIBBLE_V)
            label = "defend"
        if me == ctx.get("guard") and (not ctx["attack"] or self._running == "guard"):
            urgent = urgent or dist > self.GUARD_URGENT
            label = "guard"
        far = wm.self.position.dist(ball) > 30.0 and not (ctx["attack"] and self._running == "counter")
        power = self._dash_power(agent, urgent, dist, far, t_need, serious)
        capped = False
        if self.RECOVER:
            rec = self._recover_cap(agent, ctx, serious)
            if rec is not None and rec < power:
                power, capped = rec, True
                label += "_rec"
        note(agent, label, s=int(serious), pw=int(power))
        Action.go_to_point(agent, target, max_dash_power=power)
        Action.go_to_point(agent, self._home(agent, ctx), max_dash_power=power if capped else 100.0)

    def _home(self, agent, ctx):
        ph = self.ph
        h = Tools.home_position(agent)
        bx = (ctx["our_loc"] if ctx["our_min"] <= ctx["opp_min"] else ctx["opp_loc"])[0]
        x = bx + self.COMPACT * (h.x() - bx)
        x = min(x, max(ctx["offside_x"], bx) - 0.5)
        x = max(min(x, ph.hl - 2.0), -ph.hl + 3.0)
        y = max(min(h.y(), ph.hw - 2.0), -ph.hw + 2.0)
        return Vector2D(x, y)

    def _busy(self, ctx):
        ch = ctx["chaser"]
        if ch is not None and not ctx["kickable_mate"] and (
                ctx["attack"] or ctx["mate_t"].get(ch, NEVER) <= ctx["opp_min"] + 1):
            return {ch}
        plan = self._block_plan(ctx)
        return {plan[0] if plan is not None else ch}

    def _goal_side(self, x, y, ball_x):
        ph = self.ph
        x = max(min(x, ball_x - 4.0), -ph.hl + 2.0)
        return x, max(min(y, 14.0), -14.0)

    def _carrier(self, ctx):
        if ctx["kickable_opp"]:
            return ctx["ball"][0], ctx["ball"][1]
        return ctx["opp_loc"]

    def _block_plan(self, ctx):
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
                    ux, uy = 0.5 * ux + 0.5 * first[5] / sp, 0.5 * uy + 0.5 * first[6] / sp
                    n = math.hypot(ux, uy) or 1.0
                    ux, uy = ux / n, uy / n
                k = np.arange(1, 41)
                k = k[self.DRIBBLE_V * k <= d - 1.0]
                if k.size:
                    fx = lx + ux * self.DRIBBLE_V * k
                    fy = ly + uy * self.DRIBBLE_V * k
                    MX = np.array([m[1] + m[4] * 0.6 for m in mates])
                    MY = np.array([m[2] + m[5] * 0.6 for m in mates])
                    dist = np.hypot(fx[:, None] - MX, fy[:, None] - MY)
                    steps = np.searchsorted(ph.DASH, np.maximum(dist - ph.kick_r, 0.0))
                    ok = steps <= (ctx["opp_min"] + k)[:, None]
                    if ctx.get("guard") is not None:
                        free = np.array([m[0] != ctx["guard"] for m in mates])
                        if (ok & free[None, :]).any():
                            ok = ok & free[None, :]
                    rows = np.nonzero(ok.any(axis=1))[0]
                    if rows.size:
                        r = int(rows[0])
                        cand = [(int(steps[r, j]), mates[j][0]) for j in range(len(mates)) if ok[r, j]]
                        who = min(cand)[1]
                        sy = float(fy[r])
                        plan = (who, float(fx[r]), sy - math.copysign(min(abs(sy), self.SHADE), sy))
        ctx["block"] = plan
        return plan

    def _dash_power(self, agent, urgent, dist=None, far=False, t_need=None, serious=False):
        ph = self.ph
        c = agent.wm.cycle
        left = (HALF_CYCLES if c < HALF_CYCLES else MATCH_CYCLES) - c
        st = agent.wm.self.stamina
        ratio = st / ph.stamina_max
        spare = st - ph.st_floor - RESERVE
        if serious:
            power = 100.0
        elif spare <= 0.0:
            power = FREE_POWER
        elif left <= LATE_CYCLES:
            power = 100.0
        elif urgent:
            power = 100.0 if ratio > 0.45 else 75.0 if ratio > 0.35 else 50.0
        else:
            power = 100.0 if ratio > 0.65 else 70.0 if ratio > 0.45 else FREE_POWER
        if dist is not None:
            power = min(power, 40.0 if dist < 3.0 else 70.0 if dist < 6.0 else 100.0)
        if t_need is not None and dist is not None and not serious:
            power = min(power, max(100.0 * dist / max(t_need - PACE_SLACK, 1.0), FREE_POWER))
        if far:
            power = min(power, 60.0)
        return power

    def _recover_cap(self, agent, ctx, serious):
        if not self._recover:
            return None
        wm = agent.wm
        c = wm.cycle
        if serious or (HALF_CYCLES if c < HALF_CYCLES else MATCH_CYCLES) - c <= LATE_CYCLES:
            return None
        me = wm.self.uniform_number
        mx = wm.self.position.x()
        bx, by = ctx["ball"][0], ctx["ball"][1]
        if math.hypot(bx + self.ph.hl, by) < self.REC_SAFE_R:
            return None
        line = min([m[1] for m in ctx["mates"] if m[0] != me] + [bx])
        if mx < line and bx < line + self.DF_LINE_D:
            return None
        inc = ((self._stam.inc_max if self._stam is not None else 45.0)
               * float(getattr(wm.self, "recovery", 1.0) or 1.0))
        return max(inc - self.REC_GAIN, 0.0)

    def _rec_signal(self, agent, P):
        wm = agent.wm
        ph = self.ph
        c = wm.cycle
        me = wm.self.uniform_number
        ctx = self._context(agent)
        bx, by = ctx["ball"][0], ctx["ball"][1]
        busy = set(self._busy(ctx)) if not ctx["attack"] else set()
        exempt = (self._serious(ctx)
                  or (HALF_CYCLES if c < HALF_CYCLES else MATCH_CYCLES) - c <= LATE_CYCLES
                  or math.hypot(bx + ph.hl, by) < self.REC_SAFE_R)
        for m in ctx["mates"]:
            v = m[0]
            key = ("us", v)
            need, s = P.last_need.get(key), P.state.get(key)
            if v == me or s is None:
                continue
            rec_p = P.inc_max * s[2] - self.REC_GAIN
            hits, off_c = self._rec_seen.get(v, (0, None))
            if need is None:
                pass
            elif abs(need - rec_p) <= self.REC_SIG_TOL:
                hits += 1
                if hits >= self.REC_SIG_N:
                    hi = self.REC_OFF * ph.stamina_max
                    if off_c is not None and c - off_c <= self.REC_SIG_GAP:
                        hi = min(hi, self.REC_ON * ph.stamina_max + P.inc_max * (c - off_c))
                    off_c = None
                    if s[0] > hi:
                        note(agent, "rec_sig", u=v, est=round(s[0]), to=round(hi))
                        s[0] = hi
            elif need > 0.5:
                hits = 0
                line = min([o[1] for o in ctx["mates"] if o[0] != v] + [bx])
                deep = m[1] < line and bx < line + self.DF_LINE_D
                if (need >= rec_p + self.REC_SIG_OFF and not exempt and not deep
                        and v != ctx["chaser"] and v not in busy
                        and math.hypot(m[1] - bx, m[2] - by) > 2.0):
                    off_c = c
            self._rec_seen[v] = (hits, off_c)

    def _serious(self, ctx):
        if "serious" in ctx:
            return ctx["serious"]
        ph = self.ph
        out = False
        if not ctx["attack"] and ctx["opp_min"] < NEVER and ctx["opp_min"] <= ctx["our_min"]:
            cx, cy = self._carrier(ctx)
            d = math.hypot(cx + ph.hl, cy)
            behind = [m for m in ctx["mates"] if m[1] < cx - 1.0]
            close = sum(1 for m in behind
                        if abs(m[2] - cy * (m[1] + ph.hl) / max(cx + ph.hl, 1.0)) < 12.0)
            out = (d < 30.0 and not behind) or (d < 15.0 and close <= 1)
        ctx["serious"] = out
        return out

    def _support_point(self, agent, ctx):
        ph = self.ph
        wm = agent.wm
        me = wm.self.uniform_number
        home = self._home(agent, ctx)
        hx, hy = home.x(), home.y()
        if wm.kickable_teammate_existance:
            ox, oy = ctx["ball"][0], ctx["ball"][1]
        else:
            ox, oy = ctx["our_loc"]
        if self.GUARD_REST and ctx.get("guard") == me:
            self._running = "guard"
            self._support_target = None
            return Vector2D(*ctx["guard_spot"])
        outlet = wm.kickable_teammate_existance and self._is_outlet(ctx, me, ox, oy)
        run = None if outlet else self._run_spot(ctx, me, ox, oy)
        self._running = run[2] if run else None
        if outlet:
            cx, cy, rings = ox, oy, ((5.0, 12), (8.0, 12), (11.0, 12))
            pts = []
        elif run and run[2] in ("run", "counter"):
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

    def _run_spot(self, ctx, me, ox, oy):
        counter = ctx.get("counter")
        if ox < self.DIAG_X0 and not counter:
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
        if counter and ox <= self.RUN_SPLIT_X:
            line = max(ctx["offside_x"], ox) - 1.0
            x = min(line, ox + self.CTR_RUN, ph.hl - 8.0)
            return x, side * self.CTR_LANE, "counter"
        if ox > self.RUN_SPLIT_X or (ox > 0.0 and ids.index(me) == 0):
            x = min(ctx["offside_x"] - 1.5, max(ox + 10.0, 36.0), ph.hl - 8.0)
            return x, side * 8.5, "run"
        x = min(ox + 8.0, max(ctx["offside_x"], ox) - 1.0)
        y = max(min(oy + side * 8.0, ph.hw - 3.0), -ph.hw + 3.0)
        return x, y, "diag"

    def _is_outlet(self, ctx, me, ox, oy):
        if not any(math.hypot(o[1] - ox, o[2] - oy) < self.OUTLET_D for o in ctx["opps"]):
            return False
        near = sorted((math.hypot(m[1] - ox, m[2] - oy), m[0]) for m in ctx["mates"])
        near = [u for d, u in near if d > self.ph.kick_r + 0.3][:1 if ctx.get("counter") else 2]
        return me in near

    def _counter_state(self, wm, attack):
        cyc = wm.cycle
        if cyc - self._ctx_c > 2:
            self._def_c0, self._counter_c = None, None
            self._sp_c = cyc
        self._ctx_c = cyc
        if attack:
            self._last_att_c = cyc
            if self._def_c0 is not None:
                spell = cyc - self._def_c0
                if spell >= self.COUNTER_DEF:
                    self._counter_c = cyc
                elif spell >= 3:
                    self._counter_c = None
                self._def_c0 = None
        else:
            if self._def_c0 is None:
                self._def_c0 = cyc
            if cyc - self._def_c0 >= 3:
                self._counter_c = None
        return bool(self.COUNTER and attack and self._counter_c is not None and int(wm.game_mode_type) == 2
                    and cyc - self._counter_c <= self.COUNTER_T)


    def _defend_point(self, agent, ctx):
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
        if agent.wm.kickable_opponent_existance:
            ox, oy = ctx["ball"][0], ctx["ball"][1]
            t_kick = 0
        else:
            ox, oy = ctx["opp_loc"]
            t_kick = min(ctx["opp_min"], HORIZON)

        busy = self._busy(ctx)
        players = [m for m in ctx["mates"] if m[0] not in busy]
        guard = ctx.get("guard")
        if guard is not None and any(m[0] == guard for m in players):
            rest = [m for m in players if m[0] != guard]
            if rest:
                plan = self._man_marks(ctx, rest, ox, oy, skip=ctx["guard_target"])
            else:
                plan = {}
            plan[guard] = ctx["guard_spot"]
            return plan
        if not players:
            return {}
        if self._set_piece(ctx):
            return self._sp_covers(ctx, players, ox, oy)
        return self._man_marks(ctx, players, ox, oy)

    def _set_piece(self, ctx):
        return (ctx["key"][0] - self._sp_c <= self.SP_T and not ctx["attack"]
                and math.hypot(ctx["ball"][0] + self.ph.hl, ctx["ball"][1]) < self.SP_BALL_D)

    def _sp_covers(self, ctx, players, ox, oy):
        gx = -self.ph.hl
        lead = self.MAN_LEAD
        men = [(o[0], o[1] + lead * o[5], o[2] + lead * o[6]) for o in ctx["opps"]
               if not o[4] and o[0] != ctx["opp_first"] and math.hypot(o[1] - gx, o[2]) < self.SP_R]
        men.sort(key=lambda f: (-self._goal_view(f[1], f[2]), f[0]))
        spots = [self._lane_spot(fx, fy, ox, oy, "cover") for _, fx, fy in men[:len(players)]]
        best, best_cost = (), None
        for who in permutations(range(len(players)), len(spots)):
            ds = [math.hypot(players[i][1] + 0.6 * players[i][4] - spots[k][0],
                             players[i][2] + 0.6 * players[i][5] - spots[k][1])
                  for k, i in enumerate(who)]
            cost = max(ds) + 0.3 * sum(ds) if ds else 0.0
            if best_cost is None or cost < best_cost - 1e-9:
                best, best_cost = who, cost
        plan = {players[i][0]: spots[k] for k, i in enumerate(best)}
        for n, m in enumerate(m for i, m in enumerate(players) if i not in best):
            f = 0.3 if n % 2 == 0 else 0.6
            plan[m[0]] = (ox + (gx - ox) * f, oy - oy * f)
        return plan

    def _man_marks(self, ctx, players, ox, oy, skip=None):
        ph = self.ph
        gx = -ph.hl
        spots = []
        field = sorted((o for o in ctx["opps"] if not o[4] and o[0] != ctx["opp_first"]
                        and o[0] != skip), key=lambda o: o[0])
        for o in field:
            fx, fy = o[1] + 2.0 * o[5], o[2] + 2.0 * o[6]
            if fx > ox + self.MARK_BEHIND:
                spots.append((max(-47.0, ox - 14.0), 0.3 * oy))
                continue
            wb, gap = 0.75, 1.5
            db = math.hypot(ox - fx, oy - fy) or 1.0
            dg = math.hypot(gx - fx, fy) or 1.0
            ux = wb * (ox - fx) / db + (1.0 - wb) * (gx - fx) / dg
            uy = wb * (oy - fy) / db + (1.0 - wb) * (-fy) / dg
            nu = math.hypot(ux, uy) or 1.0
            sx = min(fx + gap * ux / nu, ox + 1.0)
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


    def _contest(self, ctx):
        ch = ctx["chaser"]
        return (ch is not None and not ctx["kickable_mate"] and not ctx["kickable_opp"]
                and ctx["key"][0] - self._last_att_c <= self.CONTEST_T
                and ctx["mate_t"].get(ch, NEVER) <= ctx["opp_min"] + self.CONTEST_D)


    def _goal_view(self, x, y):
        ph = self.ph
        dx = abs(-ph.hl - x)
        return math.atan2(2.0 * ph.gh * dx, dx * dx + y * y - ph.gh * ph.gh)

    def _lane_spot(self, fx, fy, ox, oy, kind):
        d = math.hypot(ox - fx, oy - fy)
        if d < 1e-6:
            return fx, fy
        if kind == "rest":
            gx = -self.ph.hl
            dg = math.hypot(gx - fx, fy) or 1.0
            w = self.GUARD_WG
            vx = w * (gx - fx) / dg + (1.0 - w) * (ox - fx) / d
            vy = w * (-fy) / dg + (1.0 - w) * (oy - fy) / d
            n = math.hypot(vx, vy) or 1.0
            return fx + self.GUARD_GAP * vx / n, fy + self.GUARD_GAP * vy / n
        if kind == "mark":
            k = min(self.LANE_GAP, 0.5 * d)
        else:
            k = min(max(self.COVER_FRAC * d, self.COVER_NEAR), self.COVER_FAR, 0.8 * d)
        return fx + (ox - fx) / d * k, fy + (oy - fy) / d * k
