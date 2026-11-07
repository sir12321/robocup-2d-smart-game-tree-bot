"""Stamina estimates for every player, sprint budgets, recovery mode and its signal.

Part view of agent.py, written by `agent_breakdown/parts.py split`.
Edit the code between the begin/end markers, then run `agent_breakdown/parts.py build`
to write it back into agent.py. Text outside the markers is ignored."""

# ==== begin StaminaPredictor ====


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
# ==== end StaminaPredictor ====

class Agent1:  # part view only: these are methods of agent.py's Agent1
    # ==== begin Agent1._stam_cycle_start ====


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
    # ==== end Agent1._stam_cycle_start ====

    # ==== begin Agent1._flat_out ====

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
    # ==== end Agent1._flat_out ====

    # ==== begin Agent1._race_budgets ====

    def _race_budgets(self, ph, mates, goalie, opps):
        return np.array([self._budget.get(m[0], ph.J) for m in mates]
                        + ([ph.J] if goalie else []) + [ph.J] * len(opps), dtype=np.int64)
    # ==== end Agent1._race_budgets ====

    # ==== begin Agent1._recover_cap ====

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
    # ==== end Agent1._recover_cap ====

    # ==== begin Agent1._rec_signal ====

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
    # ==== end Agent1._rec_signal ====

