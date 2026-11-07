"""Decisions for the player on the ball (`on_ball`), plus per-cycle context.

Order of play each on-ball cycle:
  1. continue a committed two-kick pass or shot (unless it now looks bad);
  2. deep in our half, pressed, with our keeper off his line: kick the
     ball out over the touchline so he can get back;
  3. shot decision (SearchMixin._on_ball_chain);
  4. otherwise gather options (shots, one-kick passes, dribbles, hold,
     clear), run the expectimax beam over passes and dribbles, and take
     the best; if nothing is viable, a channel kick, bounce pass or
     evasive dribble.

`_context` is also computed here: one race for the ball as it moves now,
shared by every on- and off-ball decision of the cycle.
"""
import math
import time
import numpy as np
from pyrusgeom.geom_2d import Vector2D
from src.student.interface import Action, Tools, note

from .config import (
    HORIZON,
    MATCH_CYCLES,
    NEVER,
    GAMMA,
    GIVE_GO,
    GO_X0,
    _norm_deg,
    _sig,
)
from .physics import _Phys, _race, _go_point
from .state import _State


class OnBallMixin:
    """The on_ball entry point, shared context, and action execution."""

    def _phys(self, agent):
        """The _Phys tables, built once from the server parameters."""
        if self.ph is None:
            self.ph = _Phys(agent.server_params)
        return self.ph

    def _allow_ms(self, agent, on_ball):
        """Milliseconds this decision may take: a share of the remaining CPU
        budget spread over the remaining cycles (on-ball decisions get
        ONBALL_SHARE times the average, capped), never eating into the
        harness's per-decision limit."""
        rem = agent.budget.remaining_ms
        if rem == float("inf"):
            rem = 60000.0
        left = max(60, MATCH_CYCLES - agent.wm.cycle)
        per = rem / left
        cap = self.ONBALL_CAP_MS if on_ball else 30.0
        allow = min(per * (self.ONBALL_SHARE if on_ball else 0.8), cap,
                    agent.budget.decision_remaining_ms - 280.0)
        return max(allow, 1.5)

    def _context(self, agent):
        """Shared picture of this cycle, computed once and cached.

        Races the ball's current path against all 10 players and derives:
        the chaser (our first arrival, with stickiness to last cycle's
        chaser; none if our keeper will catch it), first arrival cycles for
        each side and where, whether we are attacking (we reach the ball
        clearly first), and whether it is a counter-attack. Every player
        computes the same thing from the same world model, which is how the
        team coordinates without communicating.
        """
        wm = agent.wm
        key = (wm.cycle, wm.ball.position.x(), wm.ball.position.y())
        if self._ctx is not None and self._ctx["key"] == key:
            return self._ctx
        ph = self._phys(agent)

        me = wm.self
        mates, goalie = [], None
        for p in wm.teammates:
            if p.uniform_number <= 0:
                continue
            row = (p.uniform_number, p.position.x(), p.position.y(),
                   p.body_direction, p.velocity.x(), p.velocity.y())
            if p.is_goalie:
                goalie = row
            else:
                mates.append(row)
        if not me.is_goalie:
            mates.append((me.uniform_number, me.position.x(), me.position.y(),
                          me.body_direction, me.velocity.x(), me.velocity.y()))
        mates.sort(key=lambda r: r[0])
        opps = []
        for p in wm.opponents:
            if p.uniform_number <= 0:
                continue
            opps.append((p.uniform_number, p.position.x(), p.position.y(),
                         p.body_direction, p.is_goalie,
                         p.velocity.x(), p.velocity.y()))
        opps.sort(key=lambda r: r[0])

        bx, by = wm.ball.position.x(), wm.ball.position.y()
        bvx, bvy = wm.ball.velocity.x(), wm.ball.velocity.y()

        rows = mates + ([goalie] if goalie else [])
        P = [[r[1] + r[4] * 0.6, r[2] + r[5] * 0.6] for r in rows]
        P += [[o[1] + o[5] * 0.6, o[2] + o[6] * 0.6] for o in opps]
        body = [r[3] for r in rows] + [o[3] for o in opps]
        gk = [0] * len(mates) + ([-1] if goalie else []) + [1 if o[4] else 0 for o in opps]
        ctrl = [ph.kick_r] * (len(rows) + len(opps))
        T = HORIZON
        t = np.arange(T + 1)
        B = np.stack([bx + bvx * ph.G[t], by + bvy * ph.G[t]], axis=1)[None]
        first = _race(ph, B, np.array(P, dtype=float).reshape(-1, 2),
                      np.array(body, dtype=float), np.array(ctrl, dtype=float),
                      np.array(gk, dtype=np.int64),
                      np.zeros(len(P), dtype=np.int64),
                      tired=self._race_budgets(ph, mates, goalie, opps))[0]
        nm = len(mates)
        mate_t = first[:nm]
        gk_t = first[nm] if goalie else NEVER
        opp_t = first[len(rows):]
        opp_min = int(opp_t.min()) if opp_t.size else NEVER

        chaser = None
        if nm:
            if self.IMP_CHASER:
                score = []
                for i in range(nm):
                    ix, iy = B[0, min(int(mate_t[i]), T)]
                    dx, dy = ix - P[i][0], iy - P[i][1]
                    d = math.hypot(dx, dy)
                    sp = math.hypot(mates[i][4], mates[i][5])
                    align = ((mates[i][4] * dx + mates[i][5] * dy) / (sp * d)
                             if sp > 0.2 and d > 0.1 else 0.0)
                    stick = 0.35 if mates[i][0] == self._prev_chaser else 0.0
                    score.append((mate_t[i] + 0.01 * d - 0.3 * max(align, 0.0) - stick,
                                  mates[i][0]))
            else:
                score = [(mate_t[i] - (0.5 if mates[i][0] == self._prev_chaser else 0.0),
                          mates[i][0]) for i in range(nm)]
            best = min(score)
            chaser = best[1]
            best_t = int(mate_t[[m[0] for m in mates].index(chaser)])
            if gk_t < best_t:
                gi = min(int(gk_t), T)
                behind = self.FIX_GK_BACK and B[0, gi, 0] < goalie[1] - 0.5
                if B[0, gi, 0] < -ph.pen_x and abs(B[0, gi, 1]) < ph.pen_hw and not behind:
                    chaser = None
        self._prev_chaser = chaser
        our_min = min(int(mate_t.min()) if nm else NEVER, int(gk_t))

        oi = min(opp_min, T)
        opp_first = None
        if opp_t.size and opp_min < NEVER:
            opp_first = opps[int(opp_t.argmin())][0]
        ui = min(our_min, T)

        attack = (bool(wm.kickable_teammate_existance)
                  or (not wm.kickable_opponent_existance and nm > 0
                      and int(mate_t.min()) + (1 if self.ATTACK_MARGIN1 and bx > 0.0
                                               else 2 if bx > -25.0 else 5) <= opp_min))
        opps_ahead = sum(1 for o in opps if not o[4] and o[1] > bx + 2.0)
        counter = attack and (self._opp_poss > 3 or opps_ahead <= 1)

        self._ctx = {
            "key": key, "mates": mates, "goalie": goalie, "opps": opps,
            "mate_t": {mates[i][0]: int(mate_t[i]) for i in range(nm)},
            "ball": (bx, by, bvx, bvy), "chaser": chaser,
            "our_min": our_min, "opp_min": opp_min, "opp_first": opp_first,
            "our_loc": (float(B[0, ui, 0]), float(B[0, ui, 1])),
            "opp_loc": (float(B[0, oi, 0]), float(B[0, oi, 1])),
            "attack": attack,
            "counter": counter,
            "offside_x": wm.offside_line_x,
            "kickable_mate": bool(wm.kickable_teammate_existance),
            "kickable_opp": bool(wm.kickable_opponent_existance),
        }
        if attack:
            self._opp_poss = 0
        elif wm.cycle != self._opp_poss_cycle:
            self._opp_poss += 1
        self._opp_poss_cycle = wm.cycle
        self._ctx["midblock"] = (self.MIDBLOCK and not attack
                                 and self._opp_poss > self.MB_T and bx > self.MB_X)
        return self._ctx

    def _root_state(self, agent, ctx):
        """The search root: the current world as a _State."""
        wm = agent.wm
        mates = [(m[0], m[1], m[2], m[3]) for m in ctx["mates"]]
        opps = [(o[0], o[1] + o[5] * 0.6, o[2] + o[6] * 0.6, o[3], o[4])
                for o in ctx["opps"]]
        bx, by = ctx["ball"][0], ctx["ball"][1]
        S = _State(self.ph, bx, by, wm.self.uniform_number, mates, opps,
                   ctx["offside_x"], opp_vel=[(o[5], o[6]) for o in ctx["opps"]],
                   mate_vel={m[0]: (m[4], m[5]) for m in ctx["mates"]}, lead=0.6,
                   tired_map=self._budget if self.STAM_RACE else None)
        S.adv = self.CHILD_ADV * (1.25 if ctx.get("counter") else 1.0)
        g = ctx.get("goalie")
        if g:
            S.own_gk = (g[1], g[2])
        return S

    def _risk(self, agent):
        """(pass margin m0, shot weight) for the score situation: after cycle
        1150 play safer when leading, riskier and shoot more when behind."""
        wm = agent.wm
        lead = wm.our_team_score - wm.their_team_score
        late = wm.cycle > 1150
        if late and lead > 0:
            return self.PASS_M0 + 0.35, 1.0
        if late and lead < 0:
            return self.PASS_M0 - 0.20, 1.12
        return self.PASS_M0, 1.0

    def on_ball(self, agent) -> None:
        """Framework entry point when this player can kick the ball. Any
        exception falls back to a clearance instead of crashing."""
        try:
            self._stam_cycle_start(agent, True)
            self._on_ball(agent)
        except Exception as e:
            note(agent, "on_ball_error", err=str(e)[:120])
            Action.clear(agent)

    def _on_ball(self, agent):
        """The on-ball decision (see the module docstring for the order)."""
        wm = agent.wm
        ph = self._phys(agent)
        if wm.self.is_goalie:
            Action.clear(agent)
            return
        start = time.perf_counter()
        deadline = start + self._allow_ms(agent, True) / 1000.0
        ctx = self._context(agent)
        S = self._root_state(agent, ctx)
        m0, shot_w = self._risk(agent)

        c = self._commit
        if (c is not None and self.FIX_COMMIT and wm.cycle - c["cycle"] == 1
                and self._pressure(S)[0] < 1.5):
            ang = math.degrees(math.atan2(c["ty"] - S.oy, c["tx"] - S.ox))
            if not self._cone_mask(S, [ang])[0]:
                note(agent, "abort_" + c["kind"])
                c = None
        if (c is not None and self.FIX_LANE and c["kind"] == "pass"
                and wm.cycle - c["cycle"] == 1):
            D = math.hypot(c["tx"] - S.ox, c["ty"] - S.oy)
            n_left = max(c["arrive"] - 1, 1)
            s0 = min(ph.bmax, D * (1.0 - ph.decay) / (1.0 - ph.decay ** n_left))
            r = self._eval_kicks(S, np.array([c["tx"]]), np.array([c["ty"]]),
                                 np.array([s0]), np.zeros(1, dtype=np.int64))
            if not r.succ[0] or r.p[0] < 0.55:
                note(agent, "abort_lane", p=round(float(r.p[0]), 2))
                c = None
        if (c is not None and wm.cycle - c["cycle"] == 1 and c["age"] < 2
                and wm.last_kicker_uniform_number == wm.self.uniform_number):
            c["cycle"], c["age"] = wm.cycle, c["age"] + 1
            note(agent, "continue_" + c["kind"])
            Action.kick(agent, Vector2D(c["tx"], c["ty"]), c["arrive"],
                        accept_late=self.SHOT_KICK_LATE if c["kind"] == "shot" else self.KICK_LATE)
            if self.FIX_CONT_SHOT and c["kind"] == "shot":
                Action.kick_one_step(agent, Vector2D(c["tx"], c["ty"]))
            Action.hold(agent)
            return
        self._commit = None

        g = ctx["goalie"]
        if (self.IMP_KEEPER_RESET and g is not None and S.ox < -25.0
                and g[1] > -ph.hl + 8.0 and self._pressure(S)[0] < 3.0):
            note(agent, "keeper_reset", gx=round(g[1], 1))
            side = 1.0 if S.oy >= 0.0 else -1.0
            Action.kick_one_step(agent, Vector2D(S.ox + 8.0, side * (ph.hw + 5.0)))
            Action.clear(agent)
            return

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
        if math.hypot(ph.hl - bx, by) < 30.0:
            best_shot = None
            v1 = self._vmax(agent, math.degrees(math.atan2(-by, ph.hl - bx)))
            v1s = self._shot1_speeds(agent, S) if self.PER_AIM_V1 else None
            if (v1s.max() if v1s is not None else v1) >= 1.8:
                sv, pg, gy = self._eval_shots(S, v1s if v1s is not None else min(v1, ph.bmax), 0)
                i = int(np.argmax(sv))
                best_shot = (float(sv[i]) * shot_w, "shot1", (float(gy[i]), float(pg[i])))
            sv, pg, gy = self._eval_shots(S, ph.bmax, 1 if v1 > 1.2 else 2)
            i = int(np.argmax(sv))
            cand = (float(sv[i]) * shot_w, "shot", (float(gy[i]), float(pg[i])))
            if best_shot is None or cand[0] > best_shot[0]:
                best_shot = cand
            min_pg = 0.22 if (heavy and bx > 35.0) else (0.55 if math.hypot(ph.hl - bx, by) > 20.0 else 0.38)
            if best_shot[2][1] >= min_pg:
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
            drib = [o for o, c in zip(drib, clear) if c and o[2][2] >= 0.55]
        options.extend(drib)

        if d_opp >= self.HOLD_BAN_D and t_opp > 2:
            options.append(self._eval_hold(agent, S))
        if S.ox < self.CLEAR_X:
            options.append(self._eval_clear(S))

        if self.EXP_BEAM:
            values = None
            if res is not None:
                values = res.value.copy()
                if heavy:
                    values = np.where(self._live(res), values, -np.inf)
            options = [o for o in options if o[1] != "dribble"]
            options.extend(self._beam_exp(S, res, values, drib, deadline))
            if values is not None:
                i = int(np.argmax(values))
                if np.isfinite(values[i]):
                    options.append((float(values[i]), "pass", i))
        elif res is not None:
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
            away = self._evasive_angles(S)[:1]
            if away.size:
                base = away[0] + np.array([0.0, 30.0, -30.0])
                for p, ex, ey, K, lx, ly in self._dribble_cands(S, (2.5,), base):
                    v = (p * GAMMA ** K * float(ph.v_keep(ex, ey, S.space(ex, ey)))
                         + (1.0 - p) * float(ph.v_lose(lx, ly)))
                    last.append((v, "dribble", (ex, ey, p)))
            options.extend(last or [(-1.0, "clear", None)])

        tops = {}
        for o in options:
            tops[o[1]] = max(tops.get(o[1], -9.0), round(o[0], 3))
        note(agent, "fb", d=round(d_opp, 1), last=int(cornered), **tops)
        best = max(options, key=lambda o: o[0])
        self._execute(agent, S, best, res)

    def _pressure(self, S):
        """(distance to the nearest opponent, cycles until one can reach the ball)."""
        if S.nop == 0:
            return 99.0, NEVER
        d = float(np.hypot(S.op_pos[:, 0] - S.ox, S.op_pos[:, 1] - S.oy).min())
        T = 4
        B = np.tile(np.array([S.ox, S.oy]), (1, T + 1, 1))
        first = _race(S.ph, B, S.op_pos, S.op_body, np.full(S.nop, S.ph.kick_r + 0.5),
                      S.op_gk, np.zeros(S.nop, dtype=np.int64))
        return d, int(first.min())

    def _multi_keep(self, S):
        """Success multiplier for two-kick passes, shrinking as an opponent
        gets closer (he may steal the ball between the kicks)."""
        if not self.MULTI_PRESS:
            return 1.0
        d = self._pressure(S)[0]
        return (0.32 if d < 1.8 else 0.48 if d < 2.2 else 0.72 if d < 2.6
                else 0.87 if d < 3.5 else 0.94 if d < 4.0 else 1.0)

    @staticmethod
    def _live(r):
        return r.succ & (r.t_th > r.t_us)

    def _channel_kick(self, agent, S):
        """Last resort: a one-kick ball 20 m down a wing channel."""
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

    def _cone_mask(self, S, angs):
        """Which kick directions are not blocked by an opponent standing
        close (within 3 m and CONE degrees) to the ball."""
        angs = np.asarray(angs, dtype=float)
        if S.nop == 0:
            return np.ones(angs.size, dtype=bool)
        d = np.hypot(S.op_pos[:, 0] - S.ox, S.op_pos[:, 1] - S.oy)
        a = np.degrees(np.arctan2(S.op_pos[:, 1] - S.oy, S.op_pos[:, 0] - S.ox))
        off = np.abs(_norm_deg(angs[:, None] - a[None, :]))
        ok = ((off > self.CONE) | (d[None, :] >= 3.0)).all(axis=1)
        wide = ok & ((off > 90.0) | (d[None, :] >= S.ph.kick_r + 0.3)).all(axis=1)
        return wide if wide.any() else ok

    def _open_ahead(self, S, dist):
        if S.nop == 0:
            return True
        dx, dy = S.op_pos[:, 0] - S.ox, S.op_pos[:, 1] - S.oy
        d = np.hypot(dx, dy)
        ahead = np.abs(np.degrees(np.arctan2(dy, dx))) < 45.0
        return not bool(np.any(ahead & (d < dist)))

    def _bounce_pass(self, agent, S):
        """Last resort: a one-kick pass straight to a teammate 5-11 m away."""
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

    def _slot_angles(self, S):
        if S.ox >= self.SLOT_X - 3.0:
            return []
        return [math.degrees(math.atan2(sy - S.oy, self.SLOT_X - S.ox)) for sy in (-7.0, 7.0)]

    def _drib_options(self, S, dists, angs, pressed):
        """Dribbles that clear the success threshold, scored for the chain
        search, best first: [score, x, y, cycles, direction]."""
        attack = S.ox > 0.0
        p_min = self.P_DRIB_ATT if attack else self.P_DRIB
        out = []
        for p, ex, ey, K, _, _ in self._dribble_cands(S, dists, angs):
            if p < p_min:
                continue
            score = float(self._field_eval(ex, ey)) - 0.05 * K
            if attack:
                score -= self.LOSS_ATT * (1.0 - p)
            if S.ox > self.BACK_X and not pressed:
                score -= self.BACK_W * max(0.0, S.ox - 4.0 - ex)
            out.append([score, ex, ey, K, math.degrees(math.atan2(ey - S.oy, ex - S.ox))])
        out.sort(key=lambda d: -d[0])
        return out

    def _evasive_angles(self, S):
        if S.nop == 0:
            return np.zeros(0)
        d = np.hypot(S.op_pos[:, 0] - S.ox, S.op_pos[:, 1] - S.oy)
        k = int(np.argmin(d))
        away = math.degrees(math.atan2(S.oy - S.op_pos[k, 1], S.ox - S.op_pos[k, 0]))
        return away + np.array([0.0, 45.0, -45.0, 90.0, -90.0])

    def _dribble_cands(self, S, dists, base=None):
        """Simulate dribbles of each length in `dists` along each direction in `base`.

        The ball is carried at 0.72 m/cycle (after a turn if the direction
        is more than 35 deg off the body); opponents race the carry path.
        Success comes from the race margin, recalibrated with CARRY_FIT for
        short dribbles. Dribbles leaving the pitch or passing next to our own
        keeper in our box are dropped.
        Returns [(p, end x, end y, cycles, loss x, loss y)].
        """
        ph = S.ph
        body = S.kbody
        if base is None:
            goal_ang = math.degrees(math.atan2(-S.oy, ph.hl - S.ox))
            base = np.array([-90.0, -60.0, -30.0, -15.0, 0.0, 15.0, 30.0, 60.0, 90.0, goal_ang])
        speed = 0.72
        out = []
        for dist in dists:
            T = int(math.ceil(dist / speed)) + 5
            ra = np.radians(base)
            ux, uy = np.cos(ra), np.sin(ra)
            ex, ey = S.ox + ux * dist, S.oy + uy * dist
            turn = (np.abs(_norm_deg(base - body)) > 35.0).astype(np.int64)
            t = np.arange(T + 1)[None, :]
            travel = np.clip((t - turn[:, None]) * speed, 0.0, dist)
            B = np.stack([S.ox + ux[:, None] * travel, S.oy + uy[:, None] * travel], axis=2)
            first = _race(ph, B, S.op_pos, S.op_body,
                          np.full(S.nop, ph.kick_r + 0.5), S.op_gk,
                          np.zeros(S.nop, dtype=np.int64))
            t_th = first.min(axis=1) if S.nop else np.full(base.size, NEVER)
            K = turn + int(math.ceil(dist / speed))
            p = np.where(t_th > T, 0.97, _sig((t_th - K + 0.5) / 0.8))
            theirs = getattr(S, "theirs", False)
            if S.nop and dist <= 10.0 and (not theirs or -10.0 <= S.ox <= 35.0):
                od = np.hypot(S.op_pos[:, 0] - S.ox, S.op_pos[:, 1] - S.oy)
                a, b_d, b_n, b_r = (1.71, 0.09, -0.87, 0.17) if theirs else self.CARRY_FIT
                pr = np.clip(p, 0.01, 0.99)
                p = _sig(a + b_d * min(float(od.min()), 8.0) + b_n * float((od < 4.0).sum())
                         + b_r * np.log(pr / (1.0 - pr)))
            lt = B[np.arange(base.size), np.minimum(t_th, T)]
            inside = (np.abs(ex) < ph.hl - 1.0) & (np.abs(ey) < ph.hw - 1.0)
            if S.own_gk is not None:
                gd = np.hypot(B[:, :, 0] - S.own_gk[0], B[:, :, 1] - S.own_gk[1])
                box = (B[:, :, 0] < -ph.pen_x) & (np.abs(B[:, :, 1]) < ph.pen_hw)
                inside &= ~(box & (gd < ph.catch_r + 2.0)).any(axis=1)
            for i in range(base.size):
                if inside[i]:
                    out.append((float(p[i]), float(ex[i]), float(ey[i]), int(K[i]),
                                float(lt[i, 0]), float(lt[i, 1])))
        return out

    def _eval_dribbles(self, agent, S):
        """Dribble options valued like passes: p * GAMMA^K * V(end) + (1 - p) * V(loss)."""
        ph = S.ph
        goal_ang = math.degrees(math.atan2(-S.oy, ph.hl - S.ox))
        base = np.concatenate([[-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0, goal_ang],
                               self._evasive_angles(S)])
        out = []
        for p, ex, ey, K, lx, ly in self._dribble_cands(S, self.DRIB_DISTS, base):
            value = (p * GAMMA ** K * float(ph.v_keep(ex, ey, S.space(ex, ey)))
                     + (1.0 - p) * float(ph.v_lose(lx, ly)))
            out.append((value, "dribble", (ex, ey, p, K, lx, ly)))
        return out

    def _eval_hold(self, agent, S):
        """Value of keeping the ball this cycle (riskier the closer the nearest opponent)."""
        ph = S.ph
        d, _ = self._pressure(S)
        p = 0.0 if d < 1.0 else 0.45 if d < 2.0 else 0.8 if d < 3.0 else 0.95 if d < 5.0 else 0.98
        v = (p * (float(ph.v_keep(S.ox, S.oy, S.space(S.ox, S.oy))) - 0.012)
             + (1.0 - p) * float(ph.v_lose(S.ox, S.oy)))
        return (v, "hold", None)

    def _eval_clear(self, S):
        """Value of a long clearance upfield and wide."""
        ph = S.ph
        lx = min(ph.hl - 5.0, S.ox + 30.0)
        ly = math.copysign(min(ph.hw - 3.0, abs(S.oy) + 12.0), S.oy if S.oy else 1.0)
        v = 0.35 * float(ph.v_keep(lx, ly, 4.0)) + 0.65 * float(ph.v_out(lx, ly))
        return (v, "clear", None)

    def _start_go(self, wm, S):
        """After passing, remember a give-and-go run target for off_ball."""
        if GIVE_GO and S.ox > GO_X0:
            gx, gy, _, _ = _go_point(S.ph, S.ox, S.oy, S.offside_x, self.GO_RUN)
            self._go = (wm.cycle, gx, gy)

    def _touch_out(self, agent, S, best, res):
        """When heavily pressed in our own half and 2+ teammates are low on stamina,
        kick the ball out over the nearest touchline to prevent a dangerous turnover."""
        ph = self.ph
        if S.ox >= 0.0 or self._pressure(S)[0] > getattr(self, "TOUCH_PRESS", 2.0):
            return False
        value, kind, payload = best
        if kind in ("shot", "shot1", "clear"):
            return False
        if kind == "pass":
            p = float(res.p[payload])
        elif kind in ("pass1", "channel"):
            p = float(getattr(payload[0], "p")[payload[1]])
        elif kind == "dribble":
            p = float(payload[2]) if len(payload) > 2 else 0.5
        else:
            p = 0.5
        if p >= getattr(self, "TOUCH_P", 0.80):
            return False
        me = agent.wm.self.uniform_number
        st = {m[0]: (agent.wm.self.stamina if m[0] == me else
                     (self._stam.estimate(("us", m[0])) if self._stam is not None else ph.stamina_max))
              for m in self._context(agent)["mates"]}
        tired = sum(1 for v in st.values() if v < getattr(self, "TOUCH_ST", 3000.0))
        if tired < getattr(self, "TOUCH_N", 2):
            return False
        side = 1.0 if S.oy >= 0.0 else -1.0
        tx = min(S.ox + 10.0, ph.hl - 2.0)
        note(agent, "touch_out", k=kind, p=round(p, 2), tired=tired, x=round(S.ox, 1))
        Action.kick_one_step(agent, Vector2D(tx, side * (ph.hw + 5.0)))
        return True

    def _execute(self, agent, S, best, res):
        """Carry out the chosen (value, kind, payload) option."""
        if self.TOUCH_OUT and self._touch_out(agent, S, best, res):
            return
        self._execute_base(agent, S, best, res)

    def _execute_base(self, agent, S, best, res):
        """Issue the kick/dribble/hold actions. Passes and shots that need
        two kicks are stored in self._commit so the next cycle continues
        them. Fallback actions (hold, then clear) are queued after the main
        one."""
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
            self._start_go(wm, S)
            Action.kick(agent, Vector2D(tx, ty), arrive, accept_late=self.KICK_LATE)
            Action.hold(agent)
        elif kind == "pass1":
            r, i = payload
            note(agent, "pass", v=round(value, 3), p=round(float(r.p[i]), 2),
                 t=int(r.t_us[i]), m=int(min(r.t_th[i], 99)) - int(r.t_us[i]),
                 d=0, s0=round(float(r.s0[i]), 2), one=1)
            self._start_go(wm, S)
            Action.kick_one_step(agent, Vector2D(float(r.tx[i]), float(r.ty[i])))
        elif kind == "channel":
            r, i = payload
            note(agent, "channel", v=round(value, 3), s0=round(float(r.s0[i]), 2))
            Action.kick_one_step(agent, Vector2D(float(r.tx[i]), float(r.ty[i])))
        elif kind in ("shot", "shot1"):
            gy, pg = payload
            if self.SHOT_RAY_FIX:
                gy = S.oy + (gy - S.oy) * (ph.hl - S.ox) / max(ph.hl + 2.0 - S.ox, 1.0)
            target = Vector2D(ph.hl, gy)
            note(agent, kind, v=round(value, 3), pg=round(pg, 2))
            if kind == "shot1":
                Action.kick_one_step(agent, target)
            elif self.FIX_SHOT:
                D = math.hypot(ph.hl - S.ox, gy - S.oy)
                arrive = 2
                while arrive < 60 and D * (1.0 - ph.decay) / (1.0 - ph.decay ** arrive) >= ph.bmax:
                    arrive += 1
                self._commit = {"cycle": wm.cycle, "age": 0, "kind": "shot",
                                "tx": ph.hl, "ty": gy, "arrive": arrive}
                Action.kick(agent, target, arrive, accept_late=self.SHOT_KICK_LATE)
                Action.kick_one_step(agent, target)
            else:
                self._commit = {"cycle": wm.cycle, "age": 0, "kind": "shot",
                                "tx": ph.hl, "ty": gy, "arrive": 1}
                Action.kick(agent, target, 1)
        elif kind == "dribble":
            note(agent, "dribble", v=round(value, 3))
            Action.dribble(agent, Vector2D(payload[0], payload[1]))
        elif kind == "hold":
            note(agent, "hold", v=round(value, 3))
            Action.hold(agent)
        else:
            note(agent, "clear", v=round(value, 3))
        if self.FIX_FALLBACK and kind != "clear" and S.ox > self.CLEAR_X:
            Action.hold(agent)
        Action.clear(agent)