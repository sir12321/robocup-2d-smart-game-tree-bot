"""Stamina estimation for every player on the pitch.

The world model gives a player only his own stamina. To know who can still
sprint, every agent replays the server's stamina rules for all 10 players:
each cycle it infers the dash power that explains a player's observed change
of velocity, charges it, and adds the recovery. Stoppages (cycles we never
see) are bridged with recovery tables fitted to our match logs (see the GAP
tables in agent/config.py).
"""
import math

from .config import (
    GAP_GLOBAL_V2,
    GAP_TABLE_V2,
    GAP_GLOBAL_V3,
    GAP_TABLE_V3,
    GAP_GLOBAL,
    GAP_TABLE,
    US_GAP_GLOBAL,
    US_GAP_TABLE,
    GAP_SD,
    DRIFT_SD2,
    SURE_MARGIN,
)


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