"""Decisions for players without the ball (`off_ball`).

Each cycle a player takes the first rule that applies:
  1. shot-block tackle if the ball is heading into our goal;
  2. fair tackle if the opponents have, or will win, the ball;
  3. deliberate foul (last-ditch, or in midfield with cover behind);
  4. chase the ball if this player is the chaser;
  5. block: the teammate who can reach the opponent carrier's predicted
     dribble path first runs there and engages;
  6. continue a give-and-go run;
  7. position: in attack, the best open support spot (or a forward run,
     rest-defence spot, far-post / cut-back spot); in defence, the spot
     from DefenseMixin's assignment. Dash power is chosen to save stamina
     unless the situation is urgent.
"""
import math
import numpy as np
from pyrusgeom.geom_2d import Vector2D
from src.student.interface import Action, Tools, note

from .config import (
    HORIZON,
    MATCH_CYCLES,
    HALF_CYCLES,
    LATE_CYCLES,
    RESERVE,
    FREE_POWER,
    PACE_SLACK,
    NEVER,
    OPP_MARGIN,
    ENGAGE_CONE,
    _norm_deg,
    _sig,
)
from .physics import _Phys, _kick_traj, _race


class OffBallMixin:
    """The off_ball entry point and positioning helpers."""

    def off_ball(self, agent) -> None:
        """Framework entry point when this player cannot kick the ball. Any
        exception falls back to walking to the home position."""
        try:
            if self._stam_cycle_start(agent, False):
                return
            self._off_ball(agent)
        except Exception as e:
            note(agent, "off_ball_error", err=str(e)[:120])
            Action.go_to_point(agent, Tools.home_position(agent))

    def _off_ball(self, agent):
        """The off-ball rule list (see the module docstring)."""
        wm = agent.wm
        ph = self._phys(agent)
        ctx = self._context(agent)
        me = wm.self.uniform_number
        ball = wm.ball.position

        # 1. Automated Shot-Block Tackle
        if (self.IMP_TACKLE and not wm.kickable_opponent_existance
                and not wm.kickable_teammate_existance
                and wm.self.tackle_probability >= self.SHOTBLOCK_P):
            bvx, bvy = wm.ball.velocity.x(), wm.ball.velocity.y()
            if bvx < -0.5:
                path_x = ball.x() + bvx * ph.G
                cross = np.nonzero(path_x <= -ph.hl)[0]
                if cross.size:
                    tc = int(cross[0])
                    yc = ball.y() + bvy * ph.G[tc]
                    if abs(yc) < ph.gh + 1.5 and ctx["our_min"] >= tc:
                        want = (ball - Vector2D(-ph.hl, 0.0)).th().degree()
                        rel = _norm_deg(want - wm.self.body_direction)
                        dirs = np.arange(-90.0, 91.0, 10.0)
                        gain = (1.0 - np.abs(dirs) / 180.0) * np.cos(np.radians(rel - dirs))
                        k = int(np.argmax(gain))
                        note(agent, "shot_block", p=round(wm.self.tackle_probability, 2), t=tc)
                        Action.tackle(agent, float(dirs[k]), foul=False)
                        Action.intercept(agent)
                        return

        # 2. Full-Pitch & Wing Fair Tackle (Dispossesses FULL on wing breakaways & in midfield)
        opp_has_or_beats = (wm.kickable_opponent_existance
                            or (not wm.kickable_teammate_existance
                                and ctx["opp_min"] + 1 < ctx["our_min"]))
        tackle_thr = (self.TACKLE_DEF_P if (ball.x() < -18.0 or abs(ball.y()) > 18.0)
                      else self.TACKLE_OPEN_P)
        if (opp_has_or_beats and not wm.kickable_teammate_existance
                and wm.self.tackle_probability >= tackle_thr):
            # Aim tackle away from our goal (or upfield toward opponent goal if in midfield)
            aim_target = Vector2D(ph.hl, 0.0) if ball.x() > -15.0 else Vector2D(ball.x() + 15.0,
                                                                                  math.copysign(ph.hw, ball.y()))
            want = (aim_target - ball).th().degree()
            rel = _norm_deg(want - wm.self.body_direction)
            dirs = np.arange(-90.0, 91.0, 10.0)
            gain = (1.0 - np.abs(dirs) / 180.0) * np.cos(np.radians(rel - dirs))
            k = int(np.argmax(gain))
            if gain[k] >= 0.20 or ball.x() < -25.0:
                note(agent, "tackle", p=round(wm.self.tackle_probability, 2), dir=float(dirs[k]))
                Action.tackle(agent, float(dirs[k]), foul=False)
                Action.intercept(agent)
                return

        # 3. Deliberate Tactical Foul Tackle (Last-ditch outside box or midfield turnover)
        if (getattr(self, "FOULS", False) and wm.kickable_opponent_existance
                and not wm.kickable_teammate_existance
                and getattr(self, "_fouls", 0) < self.FOUL_MAX and int(wm.self.card) == 0
                and wm.self.foul_probability >= self.FOUL_P):
            d_goal = math.hypot(ball.x() + ph.hl, ball.y())
            why = None
            in_box = ball.x() < -ph.pen_x and abs(ball.y()) < ph.pen_hw
            if (self.LASTDITCH and self.LD_MIN <= d_goal <= self.LD_MAX and not in_box
                    and self._serious(ctx)):
                why = "last"
            elif (self.MID_FOUL and self.MID_X0 <= ball.x() <= self.MID_X1
                  and wm.self.tackle_probability < self.MID_TACKLE_MAX
                  and (not self.FOUL_COVER
                       or any(m[0] != me and m[1] < ball.x() - 3.0 for m in ctx["mates"]))):
                why = "mid"
            if why is not None:
                want = (ball - Vector2D(-ph.hl, 0.0)).th().degree()
                rel = _norm_deg(want - wm.self.body_direction)
                dirs = np.arange(-90.0, 91.0, 10.0)
                gain = (1.0 - np.abs(dirs) / 180.0) * np.cos(np.radians(rel - dirs))
                k = int(np.argmax(gain))
                self._fouls = getattr(self, "_fouls", 0) + 1
                note(agent, "foul", why=why, n=self._fouls, p=round(wm.self.foul_probability, 2))
                Action.tackle(agent, float(dirs[k]), foul=True)
                Action.intercept(agent)
                return

        # 4. Loose-Ball Chaser
        if not wm.kickable_teammate_existance and ctx["chaser"] == me:
            my_t = ctx["mate_t"].get(me, NEVER)
            if ctx["attack"] or my_t <= ctx["opp_min"] + 2:
                note(agent, "chase", t=my_t)
                Action.intercept(agent)
                Action.go_to_point(agent, ball, max_dash_power=100.0)
                return

        # 5. Primary Presser / Blocker Execution
        if not ctx["attack"] and me in self._busy(ctx):
            plan = self._block_plan(ctx)
            if plan is not None and plan[0] == me:
                power = 100.0 if wm.self.stamina > ph.st_floor + 200.0 else self._dash_power(agent, True, serious=True)
                engage = ((self.BLOCK_ENGAGE_HIGH if self.FIX_BLOCK else 12.0)
                          if ball.x() > self.PRESS_X
                          or (self.PRESS_ALL and ball.x() > self.PRESS_ALL_FROM)
                          else self.BLOCK_ENGAGE)
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
                                and d_aim * abs(math.sin(math.radians(rel))) < ph.kick_r - 0.2)
                    note(agent, "engage", t=ctx["mate_t"].get(me, NEVER), run=int(straight))
                    if straight:
                        ahead = wm.self.position + Vector2D.polar2vector(
                            d_aim * math.cos(math.radians(rel)) + 3.0, wm.self.body_direction)
                        Action.go_to_point(agent, ahead, max_dash_power=power, distance_threshold=0.3)
                    elif d_ball < 5.5 or wm.self.position.x() > ball.x() - 1.0:
                        # Sprint directly onto the predicted ball point to force a tackle/interception
                        Action.go_to_point(agent, aim, max_dash_power=power, distance_threshold=0.5)
                    else:
                        Action.go_to_point(agent, Vector2D(plan[1], plan[2]),
                                           max_dash_power=power, distance_threshold=0.5)
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
                Action.go_to_point(agent, ball, max_dash_power=100.0)
                return

        # 6. Give-and-Go Continuation
        go = self._go
        if go is not None and (not ctx["attack"] or wm.cycle - go[0] > self.GO_CYCLES):
            self._go = go = None
        if go is not None and ctx["chaser"] != me:
            gx = go[1] if go[1] <= 0.0 else min(go[1], ctx["offside_x"] - 0.5)
            note(agent, "go", c=wm.cycle - go[0])
            Action.go_to_point(agent, Vector2D(gx, go[2]),
                               max_dash_power=self._dash_power(agent, True))
            Action.go_to_point(agent, self._home(agent, ctx))
            return

        # 7. Off-Ball Support (Attack) or Minimax Zonal/Lane Coverage (Defense)
        t_need, serious = None, False
        low = False
        if ctx["attack"]:
            target = self._support_point(agent, ctx)
            dist = wm.self.position.dist(target)
            urgent = (self._running == "run" and ball.x() > 10.0) or ctx.get("counter", False)
            label = self._running or "support"
        else:
            low = self.LOW_BLOCK and ball.x() > 8.0
            target = self._low_block_spot(ctx, me) if low else self._defend_point(agent, ctx)
            cx, cy = self._carrier(ctx)
            dist = wm.self.position.dist(target)
            serious = self._serious(ctx)
            # Any defender caught behind the ball or behind their target spot sprints at full power!
            out_of_pos = (wm.self.position.x() > target.x() + 1.5
                          or wm.self.position.x() > cx - 2.0)
            urgent = out_of_pos or dist > 5.0 or cx < 10.0
            if out_of_pos and cx < 15.0:
                serious = True
            v_threat = self.DEF_PACE_V if self.FIX_DEF_PACE else self.DRIBBLE_V
            t_need = (min(ctx["opp_min"], HORIZON)
                      + max(math.hypot(cx - target.x(), cy - target.y()) - 2.0, 0.0) / v_threat)
            label = "defend"
        far = wm.self.position.dist(ball) > 35.0 and not urgent
        power = self._dash_power(agent, urgent, dist, far, t_need, serious,
                                 attack=ctx["attack"])
        if low:
            power, label = 45.0, "low"
        if self.STAMINA_SAVE:
            power = self._save_power(agent, label, power, dist, serious, ball)
        if ctx["midblock"] and label == "defend" and not serious:
            power = min(power, self.MB_POWER)
        note(agent, label, s=int(serious), pw=int(power))
        Action.go_to_point(agent, target, max_dash_power=power)
        Action.go_to_point(agent, self._home(agent, ctx))

    def _home(self, agent, ctx):
        """Formation home position pulled towards the ball (COMPACT), onside."""
        ph = self.ph
        h = Tools.home_position(agent)
        bx = (ctx["our_loc"] if ctx["our_min"] <= ctx["opp_min"] else ctx["opp_loc"])[0]
        x = bx + self.COMPACT * (h.x() - bx)
        x = min(x, max(ctx["offside_x"], bx) - 0.5)
        x = max(min(x, ph.hl - 2.0), -ph.hl + 3.0)
        y = max(min(h.y(), ph.hw - 2.0), -ph.hw + 2.0)
        return Vector2D(x, y)

    def _busy(self, ctx):
        """Players taken by the ball this cycle: the chaser, or else the blocker."""
        ch = ctx["chaser"]
        if ch is not None and not ctx["kickable_mate"] and (
                ctx["attack"] or ctx["mate_t"].get(ch, NEVER) <= ctx["opp_min"] + 1):
            return {ch}
        plan = self._block_plan(ctx)
        return {plan[0] if plan is not None else ch}

    def _goal_side(self, x, y, ball_x):
        ph = self.ph
        x = max(min(x, ball_x - 3.5), -ph.hl + 2.0)
        return x, max(min(y, ph.hw - 3.0), -ph.hw + 3.0)

    def _carrier(self, ctx):
        if ctx["kickable_opp"]:
            return ctx["ball"][0], ctx["ball"][1]
        return ctx["opp_loc"]

    def _block_plan(self, ctx):
        """(player, x, y): who blocks the opponent carrier, and where.

        Predicts his dribble from where they win the ball towards our goal
        at DRIBBLE_V m/cycle, bent towards his current running direction,
        and picks the earliest point on that path some teammate can reach
        in time (preferring teammates already goal-side). Cached in ctx.
        """
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
                    # Weight actual sprint direction heavily (0.75) to cut off wing breakaways
                    w_vel = 0.75 if sp > 0.35 else 0.50
                    ux = (1.0 - w_vel) * ux + w_vel * first[5] / sp
                    uy = (1.0 - w_vel) * uy + w_vel * first[6] / sp
                    n = math.hypot(ux, uy) or 1.0
                    ux, uy = ux / n, uy / n
                k = np.arange(1, 41)
                k = k[self.DRIBBLE_V * k <= d - 1.0]
                if k.size:
                    fx = np.clip(lx + ux * self.DRIBBLE_V * k, -ph.hl + 1.5, ph.hl - 1.5)
                    fy = np.clip(ly + uy * self.DRIBBLE_V * k, -ph.hw + 1.5, ph.hw - 1.5)
                    MX = np.array([m[1] + m[4] * 0.6 for m in mates])
                    MY = np.array([m[2] + m[5] * 0.6 for m in mates])
                    dist = np.hypot(fx[:, None] - MX, fy[:, None] - MY)
                    steps = np.searchsorted(ph.DASH, np.maximum(dist - ph.kick_r, 0.0))
                    ok = steps <= (ctx["opp_min"] + k)[:, None]
                    if self.FIX_BLOCK:
                        gs = np.array([m[1] <= lx + 1.0 for m in mates])
                        if (ok & gs[None, :]).any():
                            ok = ok & gs[None, :]
                    rows = np.nonzero(ok.any(axis=1))[0]
                    if rows.size:
                        r = int(rows[0])
                        cand = [(int(steps[r, j]), mates[j][0]) for j in range(len(mates)) if ok[r, j]]
                        who = min(cand)[1]
                        sy = float(fy[r])
                        plan = (who, float(fx[r]), sy - math.copysign(min(abs(sy), self.SHADE), sy))
        ctx["block"] = plan
        return plan

    def _dash_power(self, agent, urgent, dist=None, far=False, t_need=None, serious=False,
                    attack=False):
        """Dash power for a positioning run.

        Full power when the situation is serious or late in a half; the
        stamina-neutral FREE_POWER when stamina is down to the floor plus
        RESERVE; otherwise scaled by stamina, urgency and distance, and
        paced so the player arrives just in time (t_need) rather than early.
        """
        ph = self.ph
        c = agent.wm.cycle
        left = (HALF_CYCLES if c < HALF_CYCLES else MATCH_CYCLES) - c
        st = agent.wm.self.stamina
        if (self.STAMINA_MODE == 1 or (self.STAMINA_MODE == 2 and attack)) and left > LATE_CYCLES:
            return self._dash_power_h(agent, urgent, dist, far, t_need, serious)
        ratio = st / ph.stamina_max
        spare = st - ph.st_floor - RESERVE
        if serious:
            power = 100.0
        elif spare <= 0.0:
            power = FREE_POWER
        elif left <= LATE_CYCLES:
            power = 100.0
        elif urgent:
            power = 100.0 if ratio > 0.40 else 80.0 if ratio > 0.32 else 55.0
        else:
            power = 100.0 if ratio > 0.60 else 75.0 if ratio > 0.42 else FREE_POWER
        hurry = self.FIX_DEF_PACE and (urgent or serious)
        if dist is not None and not hurry:
            power = min(power, 45.0 if dist < 2.0 else 75.0 if dist < 5.0 else 100.0)
        if t_need is not None and dist is not None and not serious and not hurry:
            power = min(power, max(100.0 * dist / max(t_need - PACE_SLACK, 1.0), FREE_POWER))
        if far and not serious and not hurry:
            power = min(power, 65.0)
        return power

    def _save_power(self, agent, label, power, dist, serious, ball):
        """Extra power caps per run type (STAMINA_SAVE, off)."""
        wm = agent.wm
        c = wm.cycle
        if (HALF_CYCLES if c < HALF_CYCLES else MATCH_CYCLES) - c <= LATE_CYCLES:
            return power
        st = wm.self.stamina
        if label == "rest" and dist < 12.0:
            return min(power, 60.0)
        if label in ("diag", "support"):
            return min(power, 75.0)
        if label == "run" and st < 5000.0:
            return min(power, 75.0)
        if self.STAMINA_SAVE >= 2 and label == "defend" and not serious and ball.x() > 10.0:
            return min(power, 70.0)
        return power

    def _dash_power_h(self, agent, urgent, dist, far, t_need, serious):
        """Hysteresis recovery mode (STAMINA_MODE 1/2, off): below 50% stamina
        walk until back above 70%, except in serious situations."""
        wm = agent.wm
        ph = self.ph
        st = wm.self.stamina
        if st < 0.5 * ph.stamina_max:
            self._recover = True
        elif st > 0.7 * ph.stamina_max:
            self._recover = False
        inc = FREE_POWER * max(float(wm.self.recovery), 0.1)
        line = wm.our_defense_line_x
        if serious or (wm.self.position.x() < line + 0.5 and wm.ball.position.x() < line + 20.0):
            return 100.0
        if self._recover:
            power = FREE_POWER if urgent else max(inc - 25.0, 0.0)
        elif urgent:
            power = 100.0
        elif wm.kickable_teammate_existance and wm.ball.dist_from_self < 20.0:
            power = min(1.1 * inc, 100.0)
        else:
            power = min(1.7 * inc, 100.0)
        if dist is not None:
            power = min(power, 40.0 if dist < 3.0 else 70.0 if dist < 6.0 else 100.0)
        if t_need is not None and dist is not None:
            power = min(power, max(100.0 * dist / max(t_need - PACE_SLACK, 1.0), FREE_POWER))
        if far:
            power = min(power, 60.0)
        return power

    def _serious(self, ctx):
        """True when the opponents will win the ball somewhere dangerous: past
        x = -10, or within 36 m of our goal with at most one of us behind the
        ball (22 m: at most one covering the line to goal). Defenders then
        run at full power."""
        if "serious" in ctx:
            return ctx["serious"]
        ph = self.ph
        out = False
        if not ctx["attack"] and ctx["opp_min"] < NEVER and ctx["opp_min"] <= ctx["our_min"]:
            cx, cy = self._carrier(ctx)
            d = math.hypot(cx + ph.hl, cy)
            behind = [m for m in ctx["mates"] if m[1] < cx - 1.5]
            close = sum(1 for m in behind
                        if abs(m[2] - cy * (m[1] + ph.hl) / max(cx + ph.hl, 1.0)) < 12.0)
            out = (d < 36.0 and len(behind) <= 1) or (d < 22.0 and close <= 1) or cx < -10.0
        ctx["serious"] = out
        return out

    def _support_point(self, agent, ctx):
        """Where to stand in attack.

        Special roles first (rest defence, far-post / cut-back finish), then
        either an outlet near a pressed carrier, a forward or diagonal run,
        or the home area. Candidate points are scored by how open they are
        for a pass (race again), the value of receiving there, distance to
        home/self and crowding by teammates; the previous target gets a
        small bonus so players don't dither.
        """
        ph = self.ph
        wm = agent.wm
        me = wm.self.uniform_number
        home = self._home(agent, ctx)
        hx, hy = home.x(), home.y()
        if wm.kickable_teammate_existance:
            ox, oy = ctx["ball"][0], ctx["ball"][1]
        else:
            ox, oy = ctx["our_loc"]
        rest = (self._shell_spot(ctx, me, ox) if self.LOW_BLOCK
                else self._rest_spot(ctx, me, ox) if self.REST_DEFENSE else None)
        if rest is not None:
            self._running = "rest"
            self._support_target = None
            return rest
        fin = self._finish_spot(ctx, me, ox, oy) if self.FINISH else None
        if fin is not None:
            self._running = "run"
            self._support_target = None
            return fin
        outlet = wm.kickable_teammate_existance and self._is_outlet(ctx, me, ox, oy)
        run = None if outlet else self._run_spot(ctx, me, ox, oy)
        self._running = run[2] if run else None
        if outlet:
            cx, cy, rings = ox, oy, ((5.0, 12), (8.0, 12), (11.0, 12))
            pts = []
        elif run and run[2] == "run":
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

    def _finish_spot(self, ctx, me, ox, oy):
        """Near their box, the two most advanced players take the far post
        and the cut-back spot (whichever pairing means less running)."""
        if ox < self.FINISH_X:
            return None
        ph = self.ph
        rest = [m for m in ctx["mates"] if m[0] != ctx["chaser"]]
        anchor = (min(rest, key=lambda m: (m[1], m[0]))[0]
                  if self.REST_DEFENSE and len(rest) > 2 else None)
        cand = sorted((m for m in rest if m[0] != anchor), key=lambda m: (-m[1], m[0]))[:2]
        if len(cand) < 2 or me not in [m[0] for m in cand]:
            return None
        line = max(ctx["offside_x"], ox) - 0.7
        s = 1.0 if oy >= 0.0 else -1.0
        far = (min(line, 46.5), -s * 5.5)
        cut = (min(line, max(ox - 4.5, 36.5)), -s * 1.0)
        a, b = cand
        d_ab = math.hypot(a[1] - far[0], a[2] - far[1]) + math.hypot(b[1] - cut[0], b[2] - cut[1])
        d_ba = math.hypot(a[1] - cut[0], a[2] - cut[1]) + math.hypot(b[1] - far[0], b[2] - far[1])
        spot = ({a[0]: far, b[0]: cut} if d_ab <= d_ba else {a[0]: cut, b[0]: far})[me]
        return Vector2D(min(spot[0], ph.hl - 2.0), spot[1])

    def _shell_spot(self, ctx, me, ox):
        if ox >= 25.0:
            return None
        rest = sorted((m for m in ctx["mates"] if m[0] != ctx["chaser"]),
                      key=lambda m: (m[1], m[0]))[:2]
        ids = [m[0] for m in rest]
        if me not in ids:
            return None
        lo = -self.ph.hl + 3.0
        if ids.index(me) == 0:
            return Vector2D(max(min(ox - 14.0, -5.0), lo), -6.0)
        return Vector2D(max(min(ox - 10.0, 0.0), lo), 6.0)

    def _low_block_spot(self, ctx, me):
        busy = self._busy(ctx)
        ids = [m[0] for m in sorted((m for m in ctx["mates"] if m[0] not in busy),
                                    key=lambda m: (m[2], m[0]))]
        n = len(ids)
        k = ids.index(me) if me in ids else 0
        ys = (0.0,) if n <= 1 else tuple(-9.0 + 18.0 * i / (n - 1) for i in range(n))
        y = ys[min(k, len(ys) - 1)] + 0.3 * ctx["ball"][1]
        return Vector2D(-12.0, max(min(y, self.ph.hw - 2.0), -self.ph.hw + 2.0))

    def _rest_spot(self, ctx, me, ox):
        """Rest defence: the deepest non-chaser stays behind the play."""
        if ox <= -10.0:
            return None
        rest = [m for m in ctx["mates"] if m[0] != ctx["chaser"]]
        if not rest or min(rest, key=lambda m: (m[1], m[0]))[0] != me:
            return None
        fx, fy = min(((o[1], o[2]) for o in ctx["opps"] if not o[4]), default=(0.0, 0.0))
        x = max(min(ox - 15.0, fx - 3.5, 2.0), -self.ph.hl + 4.0)
        return Vector2D(x, max(min(fy * 0.7, 10.0), -10.0))

    def _run_spot(self, ctx, me, ox, oy):
        """Forward runs for the two most advanced players: into the box
        ("run") high up the pitch, otherwise a diagonal ahead of the ball."""
        if ox < self.DIAG_X0:
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
        if ox > self.RUN_SPLIT_X or (ox > 0.0 and ids.index(me) == 0):
            x = min(ctx["offside_x"] - 1.2, max(ox + 10.0, 36.0), ph.hl - 7.0)
            return x, side * 8.0, "run"
        x = min(ox + 8.0, max(ctx["offside_x"], ox) - 1.0)
        y = max(min(oy + side * 8.0, ph.hw - 3.0), -ph.hw + 3.0)
        return x, y, "diag"

    def _is_outlet(self, ctx, me, ox, oy):
        """Whether this player is one of the two nearest outlets for a pressed carrier."""
        if not any(math.hypot(o[1] - ox, o[2] - oy) < self.OUTLET_D for o in ctx["opps"]):
            return False
        near = sorted((math.hypot(m[1] - ox, m[2] - oy), m[0]) for m in ctx["mates"])
        near = [u for d, u in near if d > self.ph.kick_r + 0.3][:2]
        return me in near