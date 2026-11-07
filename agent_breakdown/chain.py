"""Game-tree beam search over pass/dribble chains and its scoring.

Part view of agent.py, written by `agent_breakdown/parts.py split`.
Edit the code between the begin/end markers, then run `agent_breakdown/parts.py build`
to write it back into agent.py. Text outside the markers is ignored."""

class Agent1:  # part view only: these are methods of agent.py's Agent1
    # ==== begin Agent1._field_eval ====

    def _field_eval(self, x, y):
        slot = self.SLOT_W * _sig((x - 31.0) / 2.0) * _sig((12.0 - np.abs(y)) / 2.0)
        return x + 100.0 * (self.ph.threat(x, y) - 0.05) + slot
    # ==== end Agent1._field_eval ====

    # ==== begin Agent1._chain_scores ====

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
    # ==== end Agent1._chain_scores ====

    # ==== begin Agent1._step_cost ====

    def _step_cost(self, S, r):
        att = (S.ox > 15.0) | (r.lx > 22.0)
        return 0.05 * r.t_us + np.where(att, self.LOSS_ATT, self.MID_LOSS) * (1.0 - r.p)
    # ==== end Agent1._step_cost ====

    # ==== begin Agent1._on_ball_chain ====

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
    # ==== end Agent1._on_ball_chain ====

    # ==== begin Agent1._beam_chain ====

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
    # ==== end Agent1._beam_chain ====

    # ==== begin Agent1._beam_run ====

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
    # ==== end Agent1._beam_run ====

    # ==== begin Agent1._beam_keep ====

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
    # ==== end Agent1._beam_keep ====

    # ==== begin Agent1._beam_expand ====

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
    # ==== end Agent1._beam_expand ====

    # ==== begin Agent1._child ====

    def _child(self, S, res, i):
        recv = S.recv[int(res.who[i])][0]
        return S.child(recv, float(res.lx[i]), float(res.ly[i]), int(res.t_us[i]))
    # ==== end Agent1._child ====

