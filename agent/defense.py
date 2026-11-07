"""Defensive positioning: a minimax assignment of defenders to spots.

When the opponents have (or will win) the ball, we enumerate what their
carrier could do next: passes to each of his teammates, dribbles in several
directions and lengths, and shots. Each action is valued from their point
of view, with a one-step follow-up look-ahead for the best ones when the
ball is not too far upfield and CPU time allows. We then pick
candidate spots for our free defenders (a cover spot goal-side of the ball,
goal-side marks on the most dangerous receivers, a wing cut-off, ...) and
race each spot against each action's ball path: a defender at a spot
"kills" an action if he gets to its path in time. Every assignment of
players to spots is tried (`permutations`) and we keep the one that
minimises the opponents' soft-max value over the actions still open, plus a
small cost for running.
"""
import math
import time
from itertools import permutations
import numpy as np
from pyrusgeom.geom_2d import Vector2D
from src.student.interface import note

from .config import HORIZON, NEVER, _norm_deg
from .physics import _kick_traj, _race, _first_true
from .state import _State


class DefenseMixin:
    """Where our free defenders stand when the opponents have the ball."""

    def _their_eval(self, x, y):
        """Value to the opponents of having the ball at (x, y)."""
        return self._field_eval(-x, y)

    def _their_actions(self, ctx, ox, oy, T=30):
        """The opponent carrier's candidate actions from (ox, oy).

        Returns (paths B, cycle each action completes, their value, owner,
        receivers), where owner is the receiving opponent's number for
        passes, 0 for dribbles and -1 for shots; None if there are none.
        """
        ph = self.ph
        gx = -ph.hl
        holder = ctx["opp_first"]
        recv = [o for o in ctx["opps"] if not o[4] and o[0] != holder]
        paths, done, value, owner = [], [], [], []
        t = np.arange(T + 1)[None, :]

        if recv:
            pts = []
            for o in recv:
                fx, fy = o[1] + o[5] * 3.0, o[2] + o[6] * 3.0
                d = math.hypot(gx - fx, fy) or 1.0
                pts.append((fx, fy))
                pts.append((fx + (gx - fx) / d * 7.0, fy - fy / d * 7.0))
                # Also model straight wing/channel through-balls ahead of receivers
                pts.append((max(fx - 9.0, -ph.hl + 2.0), fy))
            P = np.array(pts)
            tx = np.clip(P[:, 0], -ph.hl + 1.0, ph.hl - 1.0)
            ty = np.clip(P[:, 1], -ph.hw + 1.0, ph.hw - 1.0)
            dx, dy = tx - ox, ty - oy
            D = np.maximum(np.hypot(dx, dy), 1e-6)
            s0 = np.minimum(ph.bmax, D * (1.0 - ph.decay) + 1.6)
            B = _kick_traj(ph, ox, oy, dx / D, dy / D, s0, np.ones(tx.size, dtype=np.int64), T)
            OP = np.array([[o[1] + o[5] * 0.6, o[2] + o[6] * 0.6] for o in recv])
            first = _race(ph, B, OP, np.array([o[3] for o in recv]),
                          np.full(len(recv), ph.kick_r), np.zeros(len(recv), dtype=np.int64),
                          np.ones(len(recv), dtype=np.int64))
            t_them = first.min(axis=1)
            land = B[np.arange(tx.size), np.minimum(t_them, T)]
            ok = (t_them <= T) & (D > 2.0)
            paths.append(B[ok])
            done.append(t_them[ok])
            value.append(self._their_eval(land[ok, 0], land[ok, 1]))
            owner.append(np.repeat([o[0] for o in recv], 3)[ok])

        # Include straight wing-breakaway angles (180 deg = straight toward our goal line -ph.hl)
        ga = math.degrees(math.atan2(-oy, gx - ox))
        ang_deg = np.unique(np.round(np.concatenate([
            ga + np.array([-70.0, -35.0, 0.0, 35.0, 70.0]),
            np.array([180.0, 160.0, -160.0])
        ]), 1))
        ang = np.radians(ang_deg)
        for L in (6.0, 12.0, 18.0):
            v_carry = 0.85 if L >= 12.0 else self.DRIBBLE_V
            ux, uy = np.cos(ang), np.sin(ang)
            travel = np.minimum(t * v_carry, L)
            B = np.stack([ox + ux[:, None] * travel, oy + uy[:, None] * travel], axis=2)
            ex, ey = ox + ux * L, oy + uy * L
            ok = (np.abs(ex) < ph.hl - 1.0) & (np.abs(ey) < ph.hw - 1.0)
            paths.append(B[ok])
            done.append(np.full(int(ok.sum()), int(math.ceil(L / v_carry))))
            # Boost threat value of deep wing breakaways so defenders prioritize cutting them off
            wing_threat = self._their_eval(ex[ok], ey[ok]) + np.where(ex[ok] < -10.0, 8.0, 0.0)
            value.append(wing_threat)
            owner.append(np.zeros(int(ok.sum()), dtype=np.int64))

        if math.hypot(gx - ox, oy) < 28.0:
            gy = np.array([-ph.gh + 1.0, 0.0, ph.gh - 1.0])
            dx, dy = gx - 2.0 - ox, gy - oy
            D = np.maximum(np.hypot(dx, dy), 1e-6)
            s0 = np.full(gy.size, ph.bmax)
            B = _kick_traj(ph, ox, oy, dx / D, dy / D, s0, np.ones(gy.size, dtype=np.int64), T)
            t_goal = _first_true(B[:, :, 0] <= gx)
            ok = t_goal <= T
            paths.append(B[ok])
            done.append(t_goal[ok])
            value.append(np.full(int(ok.sum()), 100.0))
            owner.append(np.full(int(ok.sum()), -1, dtype=np.int64))

        if not paths:
            return None
        B = np.concatenate(paths)
        if B.shape[0] == 0:
            return None
        return (B, np.concatenate(done).astype(np.int64), np.concatenate(value),
                np.concatenate(owner).astype(np.int64), recv)

    def _their_root(self, ctx, ox, oy):
        """The world mirrored to the opponents' point of view, as a _State,
        so the same pass/shot evaluators can be reused for their attack."""
        ph = self.ph
        field = [o for o in ctx["opps"] if not o[4]]
        mates = [(o[0], -o[1], -o[2], _norm_deg(o[3] + 180.0)) for o in field]
        ours = list(ctx["mates"]) + ([ctx["goalie"]] if ctx["goalie"] else [])
        n_field = len(ctx["mates"])
        lead = 0.6 if self.FIX_MIRROR_VEL else 0.0
        opps = [(m[0], -(m[1] + m[4] * lead), -(m[2] + m[5] * lead), _norm_deg(m[3] + 180.0),
                 i >= n_field) for i, m in enumerate(ours)]
        xs = sorted(m[1] for m in ours)
        line = xs[1] if len(xs) > 1 else (xs[0] if xs else 0.0)
        S = _State(ph, -ox, -oy, ctx["opp_first"], mates, opps, -line,
                   opp_vel=[(-m[4], -m[5]) for m in ours],
                   mate_vel={o[0]: (-o[5], -o[6]) for o in field}, lead=lead)
        gk = next((o for o in ctx["opps"] if o[4]), None)
        if gk is not None:
            S.own_gk = (-gk[1], -gk[2])
        S.theirs = True
        return S

    def _their_followups(self, ctx, ox, oy, B, t_done, E, owner):
        """Raise the value of their THEIR_K best actions by what they could do
        next (one more step of the chain search, discounted by THEIR_W), and
        add explicit follow-up shot paths for us to block."""
        ph = self.ph
        S = self._their_root(ctx, ox, oy)
        holder = ctx["opp_first"]
        T = B.shape[1] - 1
        E = E.copy()
        shots = []
        cand = [int(a) for a in np.argsort(-E, kind="stable")
                if owner[a] > 0 or (owner[a] == 0 and holder is not None)][:self.THEIR_K]
        for a in cand:
            t = int(min(t_done[a], T))
            lx, ly = float(B[a, t, 0]), float(B[a, t, 1])
            child = S.child(int(owner[a]) or holder, -lx, -ly, t)
            v = self._chain_value(child, 1, math.inf, drib=bool(owner[a] > 0), light=True,
                                  shoot=not self.THEIR_SHOT2)
            if np.isfinite(v):
                E[a] = max(E[a], self.THEIR_W * v)
            if not self.THEIR_SHOT2 or child.ox <= 20.0 or t >= T:
                continue
            _, pg, gy = self._eval_shots(child, ph.bmax, 1)
            k = int(np.argmax(pg))
            if pg[k] < self.SHOT_P:
                continue
            dx, dy = -ph.hl - lx, -float(gy[k]) - ly
            d = math.hypot(dx, dy) or 1.0
            leg = _kick_traj(ph, lx, ly, np.array([dx / d]), np.array([dy / d]),
                             np.array([ph.bmax]), np.array([1]), T - t)[0]
            path = np.concatenate([B[a, :t], leg])
            cross = int(_first_true((path[:, 0] <= -ph.hl)[None, :])[0])
            if cross <= T:
                shots.append((a, path, cross,
                              self.THEIR_W * (self.SHOT_V0 + self.SHOT_V1 * float(pg[k]))))
        return E, shots, len(cand)

    def _defend_point(self, agent, ctx):
        """This player's defensive target (the plan is shared via ctx)."""
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

    def _defense_plan(self, agent, ctx):
        """{uniform number: (x, y)} for every free (not chasing/blocking) player.

        See the module docstring. The opponents' value of a set of open
        actions is a soft-max (temperature 4) that also includes the value
        of the carrier just keeping the ball, and the assignment minimises
        that value plus 0.08 * weight * distance for each player's run.
        """
        ph = self.ph
        wm = agent.wm
        gx, gy = -ph.hl, 0.0
        if wm.kickable_opponent_existance:
            ox, oy = ctx["ball"][0], ctx["ball"][1]
            t_kick = 0
        else:
            ox, oy = ctx["opp_loc"]
            t_kick = min(ctx["opp_min"], HORIZON)

        busy = self._busy(ctx)
        players = [m for m in ctx["mates"] if m[0] not in busy]
        if self.MAN_MARK and players:
            return self._man_marks(ctx, players, ox, oy)
        stop = self._stopper(ctx, players) if self.STOPPER else None
        if stop is not None:
            players = [m for m in players if m[0] != stop[0]]
            if not players:
                return {stop[0]: stop[1]}
        acts = self._their_actions(ctx, ox, oy) if players else None
        if acts is None:
            return self._man_marks(ctx, players, ox, oy) if players else {}
        B, t_done, E, owner, recv = acts
        via = owner.copy()
        parent = np.full(E.size, -1, dtype=np.int64)
        if ox < self.THEIR_X and agent.budget.remaining_ms > self.THEIR_MIN_MS:
            t0 = time.perf_counter()
            E, shots, n2 = self._their_followups(ctx, ox, oy, B, t_done, E, owner)
            if shots:
                B = np.concatenate([B, np.stack([s[1] for s in shots])])
                t_done = np.concatenate([t_done, [s[2] for s in shots]]).astype(np.int64)
                E = np.concatenate([E, [s[3] for s in shots]])
                owner = np.concatenate([owner, np.full(len(shots), -1, dtype=np.int64)])
                via = np.concatenate([via, [owner[s[0]] for s in shots]]).astype(np.int64)
                parent = np.concatenate([parent, [s[0] for s in shots]]).astype(np.int64)
            note(agent, "their2", n=n2, s=len(shots),
                 ms=round(1000.0 * (time.perf_counter() - t0), 1))
        A = E.size

        best_by_recv = {}
        for a in range(A):
            u = int(via[a])
            if u > 0:
                best_by_recv[u] = max(best_by_recv.get(u, -1e9), float(E[a]))
        ranked = sorted(recv, key=lambda o: (-best_by_recv.get(o[0], -1e9), o[0]))[:3]

        # Avoid placing cover spot on top of the primary presser/blocker (fixes 0x801 collision)
        bplan = self._block_plan(ctx)
        bx_press, by_press = (bplan[1], bplan[2]) if bplan is not None else (ox, oy)

        spots, weights = [], []
        d_ball = math.hypot(ox - gx, oy - gy)
        if d_ball > 4.0:
            # Place sweeper/cover spot deeper than the presser (6.5m to 11.0m goal-side)
            step = min(max(0.42 * d_ball, 6.5), 11.0)
            cx_cov = ox + (gx - ox) / d_ball * step
            cy_cov = oy + (gy - oy) / d_ball * step * 0.85
            if math.hypot(cx_cov - bx_press, cy_cov - by_press) < 3.2:
                cx_cov = max(-ph.hl + 2.5, cx_cov - 3.5)
                cy_cov *= 0.75
            spots.append((cx_cov, cy_cov))
        else:
            spots.append((gx + 2.0, 0.0))
        weights.append(3.2)

        # If opponent carrier is wide on the wing, add a dedicated wing-cutoff spot ahead of them
        if abs(oy) > 12.0 and ox > -ph.hl + 8.0:
            spots.append((max(-ph.hl + 3.0, ox - 8.5), oy * 0.88))
            weights.append(2.6)

        for rank, o in enumerate(ranked):
            w = (2.2, 1.5, 1.1)[rank]
            fx, fy = o[1] + o[5] * 3.0, o[2] + o[6] * 3.0
            vx1, vy1 = gx - fx, gy - fy
            n1 = math.hypot(vx1, vy1) or 1.0
            vx2, vy2 = ox - fx, oy - fy
            n2 = math.hypot(vx2, vy2) or 1.0
            if ox > self.PRESS_X:
                wg, wb, gap = 0.65, 0.35, 2.0
            else:
                wg, wb, gap = 0.80, 0.20, (1.5 if n1 < 30.0 else 2.5)
            bx = wg * vx1 / n1 + wb * vx2 / n2
            by = wg * vy1 / n1 + wb * vy2 / n2
            nb = math.hypot(bx, by) or 1.0
            spots.append((fx + bx / nb * gap, fy + by / nb * gap))
            weights.append(w)
            if n2 > 6.0:
                k = max(0.5, 3.0 / n2)
                spots.append((ox + (fx - ox) * k, oy + (fy - oy) * k))
                weights.append(0.8 * w)
        n_mark = len(spots)
        spots.append((max(-ph.hl + 8.0, ox - 14.0), oy * 0.35))
        weights.append(1.0)
        n = len(players)
        for m in players:
            if len(spots) >= n:
                break
            spots.append((m[1], m[2]))
            weights.append(0.5)
        for r in range(1, len(spots)):
            if self.FIX_SPOTS and r < n_mark:
                spots[r] = (max(min(spots[r][0], ox - 1.0), -ph.hl + 2.0),
                            max(min(spots[r][1], ph.hw - 2.0), -ph.hw + 2.0))
            else:
                spots[r] = self._goal_side(spots[r][0], spots[r][1], ox)
            # Enforce minimum separation from the primary presser's target
            if math.hypot(spots[r][0] - bx_press, spots[r][1] - by_press) < 2.8:
                spots[r] = (max(-ph.hl + 2.0, spots[r][0] - 3.0), spots[r][1] * 0.8)
        R = len(spots)

        fixed = [m for m in ctx["mates"] if m[0] in busy]
        FP = [[m[1] + m[4] * 0.6, m[2] + m[5] * 0.6] for m in fixed]
        Fb = [m[3] for m in fixed]
        Fg = [0] * len(fixed)
        if stop is not None:
            FP.append([stop[1][0], stop[1][1]])
            Fb.append(math.degrees(math.atan2(oy - stop[1][1], ox - stop[1][0])))
            Fg.append(0)
        if ctx["goalie"]:
            g = ctx["goalie"]
            FP.append([g[1], g[2]])
            Fb.append(g[3])
            Fg.append(-1)
        SP = np.array(spots)
        Sb = np.degrees(np.arctan2(oy - SP[:, 1], ox - SP[:, 0]))
        P = np.vstack([np.array(FP).reshape(-1, 2), SP])
        first = _race(ph, B, P, np.concatenate([np.array(Fb, dtype=float), Sb]),
                      np.full(P.shape[0], ph.kick_r), np.array(Fg + [0] * R, dtype=np.int64),
                      np.ones(P.shape[0], dtype=np.int64))
        dribble = owner == 0
        margin = np.where(dribble, 2, 1)
        kills = first + margin[:, None] <= t_done[:, None]
        nf = len(FP)
        if len(fixed):
            kills[dribble, :len(fixed)] = False
        if ctx["goalie"]:
            kills[parent >= 0, nf - 1] = False
            if self.FIX_GK_BACK:
                tg = np.minimum(first[:, nf - 1], B.shape[1] - 1)
                behind = B[np.arange(B.shape[0]), tg, 0] < ctx["goalie"][1] - 0.5
                kills[behind, nf - 1] = False
        for a in np.nonzero(parent >= 0)[0]:
            kills[a] |= kills[parent[a]]
        killed_fixed = kills[:, :nf].any(axis=1) if nf else np.zeros(A, dtype=bool)
        order = [int(a) for a in np.argsort(-E, kind="stable") if not killed_fixed[a]]
        bit = {a: 1 << a for a in order}
        spot_mask = []
        for r in range(R):
            col = kills[:, nf + r]
            spot_mask.append(sum(bit[a] for a in order if col[a]))

        # Use REACH2 travel-time-aware reachability so t_kick=0 never zeroes out spot evaluation
        mask = [[0] * R for _ in range(n)]
        dcost = [[0.0] * R for _ in range(n)]
        for i, m in enumerate(players):
            mx, my = m[1] + m[4] * 0.6, m[2] + m[5] * 0.6
            for r in range(R):
                d = math.hypot(mx - spots[r][0], my - spots[r][1])
                # Penalize assigning a player who is currently ahead of the ball to a deep spot if another is already goal-side
                behind_pen = 1.5 if mx > ox + 2.0 and spots[r][0] < ox - 4.0 else 1.0
                dcost[i][r] = 0.08 * weights[r] * d * behind_pen
                late = max(int(np.searchsorted(ph.DASH, max(d - 0.5, 0.0))) - max(t_kick, 3), 0)
                col = first[:, nf + r] + late + margin <= t_done
                for a in np.nonzero(parent >= 0)[0]:
                    col[a] |= col[parent[a]]
                mask[i][r] = sum(bit[a] for a in order if col[a])

        e_hold = float(self._their_eval(ox, oy))
        Elist = [float(E[a]) for a in order]
        tau = 4.0
        memo = {}

        def their_value(killed):
            v = memo.get(killed)
            if v is None:
                top = e_hold
                for k, a in enumerate(order):
                    if not killed & bit[a]:
                        top = max(top, Elist[k])
                        break
                num, den = e_hold * math.exp((e_hold - top) / tau), math.exp((e_hold - top) / tau)
                for k, a in enumerate(order):
                    if killed & bit[a]:
                        continue
                    w = math.exp((Elist[k] - top) / tau)
                    if w < 1e-3:
                        break
                    num += w * Elist[k]
                    den += w
                v = memo[killed] = num / den
            return v

        need_cover = self.FORCE_COVER and d_ball < 45.0
        best, best_cost = None, None
        for perm in permutations(range(R), n):
            if need_cover and 0 not in perm:
                continue
            killed = 0
            cost = 0.0
            for i in range(n):
                killed |= mask[i][perm[i]]
                cost += dcost[i][perm[i]]
            cost += their_value(killed)
            if best_cost is None or cost < best_cost - 1e-9:
                best, best_cost = perm, cost
        plan = {players[i][0]: spots[best[i]] for i in range(n)}
        if stop is not None:
            plan[stop[0]] = stop[1]
        return plan

    def _stopper(self, ctx, players):
        """(player, spot) to stand goal-side of their deepest attacker (STOPPER, off)."""
        ph = self.ph
        field = [o for o in ctx["opps"] if not o[4] and o[0] != ctx["opp_first"]]
        if not field or not players:
            return None
        o = min(field, key=lambda o: (o[1], o[0]))
        if o[1] > 5.0:
            return None
        d = math.hypot(-ph.hl - o[1], o[2]) or 1.0
        sx = max(o[1] + (-ph.hl - o[1]) / d * self.STOP_GAP, -ph.hl + 1.5)
        sy = o[2] - o[2] / d * self.STOP_GAP
        m = min(players, key=lambda m: (math.hypot(m[1] - sx, m[2] - sy), m[0]))
        return m[0], (sx, sy)

    def _man_marks(self, ctx, players, ox, oy):
        """Plain man-marking (MAN_MARK, or fallback when there are no actions
        to model): a goal-side spot near each opponent, assigned by trying
        every permutation and minimising longest run + 0.3 * total run."""
        ph = self.ph
        gx = -ph.hl
        spots = []
        field = sorted((o for o in ctx["opps"] if not o[4] and o[0] != ctx["opp_first"]),
                       key=lambda o: o[0])
        deepest = min(field, key=lambda o: (o[1], o[0]))[0] if field and self.MARK_DEEP_GS else None
        for o in field:
            fx, fy = o[1] + 2.0 * o[5], o[2] + 2.0 * o[6]
            if ctx.get("midblock") and fx > self.MB_LINE:
                spots.append((self.MB_LINE - 3.0, max(min(fy, 15.0), -15.0)))
                continue
            if fx > ox + self.MARK_BEHIND:
                spots.append((max(-47.0, ox - 14.0), 0.3 * oy))
                continue
            wb, gap = (0.20, self.MARK_DEEP_GAP) if o[0] == deepest else (0.35, 1.8)
            if self.MARK_GS_ALL and o[0] != deepest and (fx < ox - 2.0 or fx < 0.0):
                wb = 0.25
            db = math.hypot(ox - fx, oy - fy) or 1.0
            dg = math.hypot(gx - fx, fy) or 1.0
            ux = wb * (ox - fx) / db + (1.0 - wb) * (gx - fx) / dg
            uy = wb * (oy - fy) / db + (1.0 - wb) * (-fy) / dg
            nu = math.hypot(ux, uy) or 1.0
            sx = min(fx + gap * ux / nu, ox - 1.0)
            if self.MARK_GS_ALL and (fx < ox - 2.0 or fx < -10.0):
                sx = min(sx, fx - 1.0)
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