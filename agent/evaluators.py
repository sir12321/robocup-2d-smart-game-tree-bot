"""Scoring candidate actions: passes, shots, and the stamina bookkeeping.

`_eval_kicks` is the workhorse: for hundreds of candidate kicks at once it
races the ball against every player and turns the race margin into a
success probability and an expected value

    value = p * GAMMA^t * V(reception) + (1 - p) * V(loss).
"""
import math
import time
import numpy as np
from pyrusgeom.angle_deg import AngleDeg
from pyrusgeom.geom_2d import Vector2D
from src.student.interface import Tools, Action, note

from .config import (
    HORIZON,
    NEVER,
    GAMMA,
    HALF_CYCLES,
    SPRINT_NET,
    SIG_BASE,
    SIG_LO,
    SIG_STEP,
    _sig,
)
from .physics import _kick_traj, _race, _first_true
from .state import _KickResult
from .stamina import StaminaPredictor


class EvaluatorMixin:
    """Action evaluation shared by the on-ball search and the defence model."""

    def _pass_targets(self, S, rings=None, speeds=None, space=True):
        """Candidate pass kicks from state S as arrays (tx, ty, s0).

        Targets: each receiver's feet plus points on `rings` around him, kicks
        into space in a fan in front of the ball (if `space`), and through
        balls ahead of forward receivers (THROUGH). Each target is tried at
        every arrival speed in `speeds`; s0 is the initial ball speed that
        arrives at the target with that speed.
        """
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
        if self.THROUGH:
            th = np.array([(x + dx, y + dy) for (_, x, y, _) in S.recv if x > -18.0
                           for dx in (7.0, 13.0) for dy in (-4.5, 0.0, 4.5)]).reshape(-1, 2)
            if th.size:
                Dt = np.hypot(th[:, 0] - S.ox, th[:, 1] - S.oy)
                ok = ((np.abs(th[:, 0]) < ph.hl - 1.0) & (np.abs(th[:, 1]) < ph.hw - 1.0)
                      & (Dt > 3.0) & (Dt < 40.0))
                for e in (0.3, 0.65):
                    TX.append(th[ok, 0])
                    TY.append(th[ok, 1])
                    S0.append(np.minimum(ph.bmax, Dt[ok] * (1.0 - ph.decay) + e))
        return np.concatenate(TX), np.concatenate(TY), np.concatenate(S0)

    def _eval_kicks(self, S, tx, ty, s0, delay, m0=None, T=HORIZON, k_multi=1.0):
        """Race and value C candidate kicks at once; returns a _KickResult.

        1. Simulate every ball path and race it against all players.
        2. Success probability from the margin (their arrival - ours): two
           logistics (hand-set and fitted), times a penalty if an opponent
           can close down the receiver on arrival, then recalibrated with
           PASS_FIT. Two-kick passes are scaled by `k_multi` under pressure.
        3. Value = p * GAMMA^t * v_keep(reception) + (1 - p) * v_lose. A
           pass our own keeper would collect in his box is a back-pass foul
           and gets V_BACKPASS.
        m0 shifts the first logistic (more/less risk late in the match).
        """
        ph = S.ph
        m0 = self.PASS_M0 if m0 is None else m0
        C = tx.size
        dx, dy = tx - S.ox, ty - S.oy
        D = np.maximum(np.hypot(dx, dy), 1e-6)
        ux, uy = dx / D, dy / D
        B = _kick_traj(ph, S.ox, S.oy, ux, uy, s0, delay, T)
        first = _race(ph, B, S.P_race, S.body, S.ctrl, S.gk, S.react, vel=S.vel, omni=S.omni,
                      tired=S.tired)
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

        mates_behind = sum(1 for m in S.mates if m[0] != S.kicker and m[1] < S.ox - 2.0)
        if mates_behind == 0 and S.ox < 15.0:
            v_fail = v_fail - 0.08

        out_margin = getattr(self, "PASS_OUT_MARGIN", True)
        margin = np.clip((np.minimum(t_th, out) if out_margin else t_th) - t_us, -6, 6).astype(float)
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

        # Calibrate pass survival against opponent 2.0m tackle reach (from RN8p.py)
        pass_fit = getattr(self, "PASS_FIT", (1.11, 0.16, 0.10))
        if pass_fit is not None and S.nop and not getattr(S, "theirs", False):
            first2 = _race(ph, B, S.P_race[nus:], S.body[nus:], np.maximum(S.ctrl[nus:], 2.0),
                           S.gk[nus:], S.react[nus:], vel=S.vel[nus:], omni=S.omni[nus:])
            m2 = np.clip(first2.min(axis=1) - t_us, -6, 8).astype(float)
            a, b, c = pass_fit
            pc = np.clip(p, 1e-3, 1.0 - 1e-3)
            p = _sig(a + b * np.log(pc / (1.0 - pc)) + c * m2)

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
        return r

    def _eval_shots(self, S, s0, delay, n=13, T=HORIZON):
        """Evaluate shots at n points across the goal mouth.

        P(goal) = P(on target, given kick noise growing with distance)
                  x P(no field player intercepts first, from the race)
                  x P(the keeper does not save, fitted GK_* logistic of his
                    distance from the shot line and the time he has).
        Returns (value, P(goal), aim y) arrays.
        """
        ph = S.ph
        gy = np.linspace(-ph.gh + 0.55, ph.gh - 0.55, n)
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
        t_th = first.min(axis=1) if first.shape[1] else np.full(n, NEVER)
        dist = np.hypot(ph.hl - S.ox, gy - S.oy)
        e = 0.03 * dist + 0.3
        lo = np.maximum(gy - e, -ph.gh + 0.15)
        hi = np.minimum(gy + e, ph.gh - 0.15)
        p_on = np.clip((hi - lo) / (2.0 * e), 0.0, 1.0)
        rel = np.clip(t_th - t_goal, -6, 6).astype(float)
        p_beat = np.where(t_goal > T, 0.0, _sig((rel + 1.0) / 0.45))
        if gk.size and getattr(self, "KEEPER_RACE", False):
            k = int(gk[0])
            t_k = _race(ph, B, S.P_race[S.nus + k][None], S.op_body[k:k + 1], np.array([ph.kick_r]),
                        np.ones(1, dtype=np.int64), np.zeros(1, dtype=np.int64),
                        vel=S.ov[k:k + 1], omni=np.ones(1, dtype=bool))[:, 0]
            m = t_goal - t_k
            kr_p1 = getattr(self, "KR_P1", 0.67)
            kr_p2 = getattr(self, "KR_P2", 0.33)
            p_beat = p_beat * np.where(m <= 0, 1.0, np.where(m == 1, kr_p1, kr_p2))
        elif gk.size:
            kx, ky = S.op_pos[gk[0]]
            ux, uy = dx / D, dy / D
            rx, ry = kx - S.ox, ky - S.oy
            along = rx * ux + ry * uy
            perp = np.where(along > 0, np.abs(rx * uy - ry * ux), math.hypot(rx, ry))
            frac = np.clip(np.maximum(along, 0.1) * (1.0 - ph.decay) / s0, 0.0, 0.999)
            n_keep = np.log(1.0 - frac) / math.log(ph.decay) + delay
            p_gk = _sig(self.GK_A + self.GK_P * perp + self.GK_N * n_keep)
            if self.KIN_SHOT:
                avail = np.clip(n_keep.astype(np.int64) - 2, 0, ph.DASH.size - 1)
                p_gk = np.maximum(p_gk, _sig((perp - ph.catch_r - ph.DASH[avail]) / 0.45))
            p_beat = p_beat * p_gk
        pg = p_on * p_beat
        return pg - (1.0 - pg) * 0.03, pg, gy

    def _deep_value(self, S, res, i, depth, deadline):
        """Depth-limited look-ahead after pass i (used only when EXP_BEAM is 0)."""
        best = float(res.vs[i])
        if depth <= 1 or time.perf_counter() > deadline:
            return best
        recv = S.recv[int(res.who[i])][0]
        child = S.child(recv, float(res.lx[i]), float(res.ly[i]), int(res.t_us[i]))
        if child.ox > 10.0:
            sv, _, _ = self._eval_shots(child, self.ph.bmax, 1)
            best = max(best, 0.94 * float(sv.max()))
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
            best = max(best, 0.90 * float(cv.max()))
        return best

    def _kick_delay(self, agent, S, tx, ty, s0):
        """0 if one kick can reach speed s0 in that direction, else MULTI_DELAY."""
        ang = np.degrees(np.arctan2(ty - S.oy, tx - S.ox))
        vmax = np.array([self._vmax(agent, a) for a in ang])
        return np.where(vmax >= s0 * 0.98, 0, self.MULTI_DELAY).astype(np.int64)

    def _shot1_speeds(self, agent, S, n=13):
        ph = self.ph
        gy = np.linspace(-ph.gh + 0.55, ph.gh - 0.55, n)
        return np.array([min(self._vmax(agent, math.degrees(math.atan2(y - S.oy, ph.hl + 2.0 - S.ox))),
                             ph.bmax) for y in gy])

    def _vmax(self, agent, angle_deg):
        """Max ball speed one kick can give in this direction (cached per cycle)."""
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

    def _stam_cycle_start(self, agent, on_ball):
        """Once per cycle: update the stamina model and teammates' sprint budgets.

        The sprint budget (full-power dashes left before the stamina floor)
        feeds the race model, so tired teammates are raced at the slower
        floor pace. With STAM_FLOOR, a teammate who should be running flat
        out but is seen dashing at exactly 45 must be at the floor. Returns
        True if this player spent the cycle sending a stamina signal.
        """
        if not (self.STAM_TRACK or self.STAM_RACE or self.STAM_SIGNAL or self.STAM_FLOOR):
            return False
        wm = agent.wm
        if wm.self.is_goalie:
            return False
        ph = self._phys(agent)
        if wm.cycle != self._stam_cycle:
            self._stam_cycle = wm.cycle
            if self._stam is None:
                self._stam = StaminaPredictor(agent.server_params)
                self._stam.half_cycles = float(HALF_CYCLES)
            P = self._stam
            prev = P.last
            P.update(agent)
            if self.STAM_FLOOR and self._floor_roles[0] == prev and wm.cycle == (prev or -9) + 1:
                for u in self._floor_roles[1]:
                    need = P.last_need.get(("us", u))
                    if (need is not None and abs(need - 45.0) <= 0.3 and ("us", u) in P.state
                            and P.state[("us", u)][0] > ph.st_floor + 1.0):
                        P.state[("us", u)][0] = ph.st_floor
                        P.var[("us", u)] = 0.0
                        note(agent, "floor_hit", u=u)
            self._budget = {k[1]: int(min(max((P.estimate(k) - ph.st_floor) / SPRINT_NET, 0.0), ph.J))
                            for k in P.state if k[0] == "us"}
            if wm.cycle % 25 == 0:
                note(agent, "stam", **{f"u{k[1]}": round(P.estimate(k)) for k in sorted(P.state)
                                       if k[0] == "us"})
        if self.STAM_FLOOR and self._floor_roles[0] != wm.cycle:
            self._floor_roles = (wm.cycle, self._flat_out(self._context(agent)))
        if not self.STAM_SIGNAL:
            return False
        ctx = self._context(agent)
        senders = self._sig_senders(agent, ctx)
        if self.STAM_FLOOR:
            self._floor_roles = (wm.cycle, self._floor_roles[1] - set(senders))
        for u in senders:
            self._sig_pending[u] = wm.cycle
        if not on_ball and wm.self.uniform_number in senders:
            self._send_signal(agent)
            return True
        return False

    def _sig_senders(self, agent, ctx):
        c = agent.wm.cycle
        h, hc = (0, c) if c < HALF_CYCLES else (1, c - HALF_CYCLES)
        if hc < self.SIG_AT:
            return []
        busy = set(self._busy(ctx)) | {ctx["chaser"]}
        bx, by = ctx["ball"][0], ctx["ball"][1]
        return [m[0] for m in ctx["mates"]
                if (h, m[0]) not in self._sig_done and m[0] not in self._sig_pending
                and m[0] not in busy and math.hypot(m[1] - bx, m[2] - by) >= 3.0]

    def _send_signal(self, agent):
        wm = agent.wm
        ph = self.ph
        st, eff, rec = wm.self.stamina, max(wm.self.effort, 0.05), wm.self.recovery
        inc = float(getattr(agent.server_params, "stamina_inc_max", 0.0)) or 45.0
        top = SIG_BASE + (ph.stamina_max - SIG_LO) / SIG_STEP
        p = SIG_BASE
        for _ in range(6):
            cmd = min(p / eff, 100.0)
            nxt = min(st - cmd + inc * rec, ph.stamina_max)
            p = min(max(SIG_BASE + (nxt - SIG_LO) / SIG_STEP, SIG_BASE), top)
        cmd = min(p / eff, 100.0)
        ahead = wm.self.position + Vector2D.polar2vector(30.0, wm.self.body_direction)
        note(agent, "sig", p=round(p, 3), cmd=round(cmd, 3), s=round(st))
        Action.go_to_point(agent, ahead, max_dash_power=cmd, distance_threshold=1.0)

    def _flat_out(self, ctx):
        """Teammates who should be sprinting this cycle (chaser, or everyone
        free when the situation is serious)."""
        out = set()
        ch = ctx["chaser"]
        if ch is not None and not ctx["kickable_mate"] and (
                ctx["attack"] or ctx["mate_t"].get(ch, NEVER) <= ctx["opp_min"] + 1):
            out.add(ch)
        if not ctx["attack"] and self._serious(ctx):
            busy = set(self._busy(ctx)) | {ch}
            out |= {m[0] for m in ctx["mates"] if m[0] not in busy}
        return out

    def _race_budgets(self, ph, mates, goalie, opps):
        """Per-player sprint budgets in `_race` order (None if STAM_RACE is off)."""
        if not self.STAM_RACE:
            return None
        return np.array([self._budget.get(m[0], ph.J) for m in mates]
                        + ([ph.J] if goalie else []) + [ph.J] * len(opps), dtype=np.int64)