"""Defending: goal-side recovery, blocker plan, man-marking with min-max assignment, set-piece covers.

Part view of agent.py, written by `agent_breakdown/parts.py split`.
Edit the code between the begin/end markers, then run `agent_breakdown/parts.py build`
to write it back into agent.py. Text outside the markers is ignored."""

class Agent1:  # part view only: these are methods of agent.py's Agent1
    # ==== begin Agent1._goal_side ====

    def _goal_side(self, x, y, ball_x):
        ph = self.ph
        x = max(min(x, ball_x - 4.0), -ph.hl + 2.0)
        return x, max(min(y, 14.0), -14.0)
    # ==== end Agent1._goal_side ====

    # ==== begin Agent1._carrier ====

    def _carrier(self, ctx):
        if ctx["kickable_opp"]:
            return ctx["ball"][0], ctx["ball"][1]
        return ctx["opp_loc"]
    # ==== end Agent1._carrier ====

    # ==== begin Agent1._block_plan ====

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
    # ==== end Agent1._block_plan ====

    # ==== begin Agent1._serious ====

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
    # ==== end Agent1._serious ====

    # ==== begin Agent1._defend_point ====


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
    # ==== end Agent1._defend_point ====

    # ==== begin Agent1._defense_plan ====

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
    # ==== end Agent1._defense_plan ====

    # ==== begin Agent1._set_piece ====

    def _set_piece(self, ctx):
        return (ctx["key"][0] - self._sp_c <= self.SP_T and not ctx["attack"]
                and math.hypot(ctx["ball"][0] + self.ph.hl, ctx["ball"][1]) < self.SP_BALL_D)
    # ==== end Agent1._set_piece ====

    # ==== begin Agent1._sp_covers ====

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
    # ==== end Agent1._sp_covers ====

    # ==== begin Agent1._man_marks ====

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
    # ==== end Agent1._man_marks ====

    # ==== begin Agent1._goal_view ====


    def _goal_view(self, x, y):
        ph = self.ph
        dx = abs(-ph.hl - x)
        return math.atan2(2.0 * ph.gh * dx, dx * dx + y * y - ph.gh * ph.gh)
    # ==== end Agent1._goal_view ====

    # ==== begin Agent1._lane_spot ====

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
    # ==== end Agent1._lane_spot ====

