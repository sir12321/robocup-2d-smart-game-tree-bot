"""Shot evaluation (P(goal) incl. the keeper race) and the carry-in before a shot.

Part view of agent.py, written by `agent_breakdown/parts.py split`.
Edit the code between the begin/end markers, then run `agent_breakdown/parts.py build`
to write it back into agent.py. Text outside the markers is ignored."""

class Agent1:  # part view only: these are methods of agent.py's Agent1
    # ==== begin Agent1._eval_shots ====

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
    # ==== end Agent1._eval_shots ====

    # ==== begin Agent1._shot1_speeds ====

    def _shot1_speeds(self, agent, S, n=13):
        ph = self.ph
        gy = np.linspace(-ph.gh + 0.6, ph.gh - 0.6, n)
        return np.array([min(self._vmax(agent, math.degrees(math.atan2(y - S.oy, ph.hl + 2.0 - S.ox))),
                             ph.bmax) for y in gy])
    # ==== end Agent1._shot1_speeds ====

    # ==== begin Agent1._carry_in ====

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
    # ==== end Agent1._carry_in ====

