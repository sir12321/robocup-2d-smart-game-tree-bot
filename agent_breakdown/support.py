"""Attacking movement without the ball: support spots, runs/diagonals, outlets.

Part view of agent.py, written by `agent_breakdown/parts.py split`.
Edit the code between the begin/end markers, then run `agent_breakdown/parts.py build`
to write it back into agent.py. Text outside the markers is ignored."""

class Agent1:  # part view only: these are methods of agent.py's Agent1
    # ==== begin Agent1._support_point ====

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
    # ==== end Agent1._support_point ====

    # ==== begin Agent1._run_spot ====

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
    # ==== end Agent1._run_spot ====

    # ==== begin Agent1._is_outlet ====

    def _is_outlet(self, ctx, me, ox, oy):
        if not any(math.hypot(o[1] - ox, o[2] - oy) < self.OUTLET_D for o in ctx["opps"]):
            return False
        near = sorted((math.hypot(m[1] - ox, m[2] - oy), m[0]) for m in ctx["mates"])
        near = [u for d, u in near if d > self.ph.kick_r + 0.3][:1 if ctx.get("counter") else 2]
        return me in near
    # ==== end Agent1._is_outlet ====

