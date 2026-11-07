"""Pass targets, kick evaluation (race), kick timing, look-ahead, first-time and special kicks.

Part view of agent.py, written by `agent_breakdown/parts.py split`.
Edit the code between the begin/end markers, then run `agent_breakdown/parts.py build`
to write it back into agent.py. Text outside the markers is ignored."""

class Agent1:  # part view only: these are methods of agent.py's Agent1
    # ==== begin Agent1._pass_targets ====


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
    # ==== end Agent1._pass_targets ====

    # ==== begin Agent1._eval_kicks ====

    def _eval_kicks(self, S, tx, ty, s0, delay, m0=None, T=HORIZON, k_multi=1.0):
        r = self._eval_kicks1(S, tx, ty, s0, delay, m0=m0, T=T, k_multi=k_multi)
        return r
    # ==== end Agent1._eval_kicks ====

    # ==== begin Agent1._deep_value ====

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
    # ==== end Agent1._deep_value ====

    # ==== begin Agent1._kick_delay ====

    def _kick_delay(self, agent, S, tx, ty, s0):
        ang = np.degrees(np.arctan2(ty - S.oy, tx - S.ox))
        vmax = np.array([self._vmax(agent, a) for a in ang])
        return np.where(vmax >= s0 * 0.98, 0, self.MULTI_DELAY).astype(np.int64)
    # ==== end Agent1._kick_delay ====

    # ==== begin Agent1._vmax ====

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
    # ==== end Agent1._vmax ====

    # ==== begin Agent1._live ====

    @staticmethod
    def _live(r):
        return r.succ & (r.t_th > r.t_us)
    # ==== end Agent1._live ====

    # ==== begin Agent1._channel_kick ====

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
    # ==== end Agent1._channel_kick ====

    # ==== begin Agent1._cone_mask ====

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
    # ==== end Agent1._cone_mask ====

    # ==== begin Agent1._bounce_pass ====

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
    # ==== end Agent1._bounce_pass ====

    # ==== begin Agent1._eval_kicks1 ====


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
    # ==== end Agent1._eval_kicks1 ====

