"""off_ball entry point: zone tackle, chase/contest, blocker engage, dispatch; home slot, dash power.

Part view of agent.py, written by `agent_breakdown/parts.py split`.
Edit the code between the begin/end markers, then run `agent_breakdown/parts.py build`
to write it back into agent.py. Text outside the markers is ignored."""

class Agent1:  # part view only: these are methods of agent.py's Agent1
    # ==== begin Agent1.off_ball ====


    def off_ball(self, agent) -> None:
        try:
            self._stam_cycle_start(agent)
            self._off_ball(agent)
        except Exception as e:
            note(agent, "off_ball_error", err=str(e)[:120])
            Action.go_to_point(agent, Tools.home_position(agent))
    # ==== end Agent1.off_ball ====

    # ==== begin Agent1._off_ball ====

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
    # ==== end Agent1._off_ball ====

    # ==== begin Agent1._home ====

    def _home(self, agent, ctx):
        ph = self.ph
        h = Tools.home_position(agent)
        bx = (ctx["our_loc"] if ctx["our_min"] <= ctx["opp_min"] else ctx["opp_loc"])[0]
        x = bx + self.COMPACT * (h.x() - bx)
        x = min(x, max(ctx["offside_x"], bx) - 0.5)
        x = max(min(x, ph.hl - 2.0), -ph.hl + 3.0)
        y = max(min(h.y(), ph.hw - 2.0), -ph.hw + 2.0)
        return Vector2D(x, y)
    # ==== end Agent1._home ====

    # ==== begin Agent1._busy ====

    def _busy(self, ctx):
        ch = ctx["chaser"]
        if ch is not None and not ctx["kickable_mate"] and (
                ctx["attack"] or ctx["mate_t"].get(ch, NEVER) <= ctx["opp_min"] + 1):
            return {ch}
        plan = self._block_plan(ctx)
        return {plan[0] if plan is not None else ch}
    # ==== end Agent1._busy ====

    # ==== begin Agent1._dash_power ====

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
    # ==== end Agent1._dash_power ====

    # ==== begin Agent1._contest ====


    def _contest(self, ctx):
        ch = ctx["chaser"]
        return (ch is not None and not ctx["kickable_mate"] and not ctx["kickable_opp"]
                and ctx["key"][0] - self._last_att_c <= self.CONTEST_T
                and ctx["mate_t"].get(ch, NEVER) <= ctx["opp_min"] + self.CONTEST_D)
    # ==== end Agent1._contest ====

