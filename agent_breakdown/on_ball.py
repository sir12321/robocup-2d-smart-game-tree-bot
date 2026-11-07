"""on_ball entry point: committed kicks, fallback evaluator, pressure, hold, clear, execution.

Part view of agent.py, written by `agent_breakdown/parts.py split`.
Edit the code between the begin/end markers, then run `agent_breakdown/parts.py build`
to write it back into agent.py. Text outside the markers is ignored."""

class Agent1:  # part view only: these are methods of agent.py's Agent1
    # ==== begin Agent1.on_ball ====


    def on_ball(self, agent) -> None:
        try:
            self._stam_cycle_start(agent)
            self._on_ball(agent)
        except Exception as e:
            note(agent, "on_ball_error", err=str(e)[:120])
            Action.clear(agent)
    # ==== end Agent1.on_ball ====

    # ==== begin Agent1._on_ball ====

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
    # ==== end Agent1._on_ball ====

    # ==== begin Agent1._pressure ====

    def _pressure(self, S):
        if S.nop == 0:
            return 99.0, NEVER
        d = float(np.hypot(S.op_pos[:, 0] - S.ox, S.op_pos[:, 1] - S.oy).min())
        T = 4
        B = np.tile(np.array([S.ox, S.oy]), (1, T + 1, 1))
        first = _race(S.ph, B, S.op_pos, S.op_body, np.full(S.nop, S.ph.kick_r + 0.5),
                      S.op_gk, np.zeros(S.nop, dtype=np.int64))
        return d, int(first.min())
    # ==== end Agent1._pressure ====

    # ==== begin Agent1._multi_keep ====

    def _multi_keep(self, S):
        d = self._pressure(S)[0]
        return (0.32 if d < 1.8 else 0.48 if d < 2.2 else 0.72 if d < 2.6
                else 0.87 if d < 3.5 else 0.94 if d < 4.0 else 1.0)
    # ==== end Agent1._multi_keep ====

    # ==== begin Agent1._eval_hold ====

    def _eval_hold(self, agent, S):
        ph = S.ph
        d, _ = self._pressure(S)
        p = 0.0 if d < 1.0 else 0.45 if d < 2.0 else 0.8 if d < 3.0 else 0.95 if d < 5.0 else 0.98
        lose = float(ph.v_lose(S.ox, S.oy))
        if self._by_their_line(S):
            lose = self.OUT_Q * float(ph.v_keep(S.ox, S.oy, 0.0)) + (1.0 - self.OUT_Q) * lose
        v = (p * (float(ph.v_keep(S.ox, S.oy, S.space(S.ox, S.oy))) - 0.012) + (1.0 - p) * lose)
        return (v, "hold", None)
    # ==== end Agent1._eval_hold ====

    # ==== begin Agent1._eval_clear ====

    def _eval_clear(self, S):
        ph = S.ph
        lx = min(ph.hl - 5.0, S.ox + 30.0)
        ly = math.copysign(min(ph.hw - 3.0, abs(S.oy) + 12.0), S.oy if S.oy else 1.0)
        v = 0.35 * float(ph.v_keep(lx, ly, 4.0)) + 0.65 * float(ph.v_out(lx, ly))
        return (v, "clear", None)
    # ==== end Agent1._eval_clear ====

    # ==== begin Agent1._execute ====

    def _execute(self, agent, S, best, res):
        if self._touch_out(agent, S, best, res):
            return
        self._execute_base(agent, S, best, res)
    # ==== end Agent1._execute ====

    # ==== begin Agent1._touch_out ====

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
    # ==== end Agent1._touch_out ====

    # ==== begin Agent1._execute_base ====

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
    # ==== end Agent1._execute_base ====

    # ==== begin Agent1._by_their_line ====

    def _by_their_line(self, S):
        ph = S.ph
        return S.ox > ph.pen_x and (S.ox > ph.hl - 8.0 or abs(S.oy) > ph.hw - 6.0)
    # ==== end Agent1._by_their_line ====

