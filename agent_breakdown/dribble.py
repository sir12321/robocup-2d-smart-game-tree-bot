"""Dribbles and self-passes simulated with player physics, their scores and values.

Part view of agent.py, written by `agent_breakdown/parts.py split`.
Edit the code between the begin/end markers, then run `agent_breakdown/parts.py build`
to write it back into agent.py. Text outside the markers is ignored."""

class Agent1:  # part view only: these are methods of agent.py's Agent1
    # ==== begin Agent1._drib_options ====

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
    # ==== end Agent1._drib_options ====

    # ==== begin Agent1._eval_dribbles ====

    def _eval_dribbles(self, agent, S):
        ph = S.ph
        out = []
        for p, ex, ey, K, lx, ly, exe in self._dribs(S):
            value = (p * GAMMA ** K * float(ph.v_keep(ex, ey, S.space(ex, ey)))
                     + (1.0 - p) * float(ph.v_lose(lx, ly)))
            out.append((value, "dribble", (ex, ey, p, exe, K, lx, ly)))
        return out
    # ==== end Agent1._eval_dribbles ====

    # ==== begin Agent1._dribs ====

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
    # ==== end Agent1._dribs ====

