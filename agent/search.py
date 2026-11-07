"""On-ball look-ahead: anytime beam searches over chains of passes and dribbles.

Two searches live here:

* the expectimax beam (`_beam_exp`, the one used with EXP_BEAM = 2). Every
  pass or dribble is a chance node: with probability p the receiver gets
  the ball and play continues from the child state, otherwise we take the
  loss value. A node's value is A + Q * v, where A collects the discounted
  loss terms along the path and Q the product of success probabilities and
  discounts, so a deep line is only worth what it is likely to deliver.
* the "chain" beam (`_on_ball_chain` / `_beam_chain`), scored with the
  field evaluation `_field_eval`. With EXP_BEAM = 2 only its shot decision
  is used.

Both are anytime: they stop at a deadline set from the remaining CPU budget
and return the best root action found so far.
"""
import math
import time
import numpy as np
from src.student.interface import note

from .config import GAMMA, NEVER, _sig, _norm_deg


class SearchMixin:
    """Look-ahead search over action chains for the player on the ball."""

    def _field_eval(self, x, y):
        """Chain-search score of having the ball at (x, y): progress (x),
        plus 100x the scoring threat, plus a bonus for the central slot."""
        slot = self.SLOT_W * _sig((x - 31.0) / 2.0) * _sig((12.0 - np.abs(y)) / 2.0)
        return x + 100.0 * (self.ph.threat(x, y) - 0.05) + slot

    def _chain_scores(self, S, r, pressed):
        """Score kicks for the chain search; kicks below the success
        threshold (P_PASS, or P_ATT in attack) are marked unsafe (-inf)."""
        att = (S.ox > 15.0) | (r.lx > 22.0)
        safe = r.succ & (r.p >= np.where(att, self.P_ATT, self.P_PASS))
        mid = self.MID_LOSS if self.FIX_CHAIN_RISK and self.FIX_RISK_SYM else 0.0
        score = (self._field_eval(r.lx, r.ly) - 0.05 * r.t_us
                 - np.where(att, self.LOSS_ATT, mid) * (1.0 - r.p))
        if S.ox > self.BACK_X and not pressed:
            score = score - self.BACK_W * np.maximum(0.0, S.ox - 4.0 - r.lx)
        return safe, np.where(safe, score, -np.inf)

    def _step_cost(self, S, r, base):
        att = (S.ox > 15.0) | (r.lx > 22.0)
        if self.FIX_RISK_SYM:
            return 0.05 * r.t_us + np.where(att, self.LOSS_ATT, self.MID_LOSS) * (1.0 - r.p)
        return ((self._field_eval(r.lx, r.ly) - base)
                + np.where(att, 0.0, self.MID_LOSS * (1.0 - r.p)))

    def _on_ball_chain(self, agent, S, deadline, shot_w):
        """Shot decision first, then (if EXP_BEAM < 2) the chain search.

        Within 30 m of goal, take the best one- or two-kick shot if its goal
        probability clears the threshold for the situation (SHOT_P, lower in
        the box or under pressure, higher from far out), unless a short
        dribble would give a clearly better shot (`_carry_in`). Returns True
        if an action was executed, False to fall through to the main
        evaluator in OnBallMixin._on_ball.
        """
        ph = self.ph
        opts = []

        if math.hypot(ph.hl - S.ox, S.oy) < 30.0:
            v1 = self._vmax(agent, math.degrees(math.atan2(-S.oy, ph.hl - S.ox)))
            v1s = self._shot1_speeds(agent, S) if self.PER_AIM_V1 else None
            if (v1s.max() if v1s is not None else v1) >= 1.8:
                sv, pg, gy = self._eval_shots(S, v1s if v1s is not None else min(v1, ph.bmax), 0)
                kind = "shot1"
                d_opp, t_opp = self._pressure(S)
                if self.SHOT_BOTH_STRICT:
                    two = (pg.max() * shot_w < self.SHOT_P and d_opp >= self.HOLD_BAN_D
                           and t_opp > 2)
                else:
                    two = d_opp >= self.PRESS_D
                if self.FIX_SHOT_BOTH and two:
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
            if (self.FAR_SHOT_D and math.hypot(ph.hl - S.ox, S.oy) > self.FAR_SHOT_D
                    and self._pressure(S)[0] >= self.HOLD_BAN_D):
                p_need = max(p_need, self.FAR_SHOT_P)
            if pg[i] * shot_w >= p_need:
                carry = self._carry_in(agent, S, float(pg[i]))
                if carry is not None:
                    note(agent, "carry_in", pg=round(float(pg[i]), 2), pg2=round(carry[2], 2))
                    self._execute(agent, S, (carry[2], "dribble", (carry[0], carry[1])), None)
                    return True
                shot = (float(sv[i]), kind, (float(gy[i]), float(pg[i])))
                if (not self.SHOT_COMPARE or pg[i] * shot_w >= self.SHOT_SURE
                        or self._pressure(S)[0] < self.HOLD_BAN_D):
                    self._execute(agent, S, shot, None)
                    return True
                opts.append((self.SHOT_V0_US + self.SHOT_V1 * float(pg[i]) * shot_w,
                             kind, shot[2]))
        if self.EXP_BEAM == 2:
            return False

        t0 = time.perf_counter()
        self._drib_wins = 0
        pressed = self._pressure(S)[0] < self.PRESS_D
        tx, ty, s0 = self._pass_targets(S)
        res = None
        if tx.size:
            delay = self._kick_delay(agent, S, tx, ty, s0)
            res = self._eval_kicks(S, tx, ty, s0, delay, k_multi=self._multi_keep(S))
            safe, base = self._chain_scores(S, res, pressed)
            score = base.copy()
        goal_ang = math.degrees(math.atan2(-S.oy, ph.hl - S.ox))
        angs = np.concatenate([np.arange(-157.5, 180.0, 22.5), [goal_ang], self._slot_angles(S),
                                self._evasive_angles(S)])
        long = S.ox > 0.0 or (S.ox > self.BACK_X and self._open_ahead(S, self.OPEN_D))
        dists = self.CHAIN_DRIB + ((15.0,) if long else ())
        drib = self._drib_options(S, dists, angs, pressed)
        level = 0
        cost = (self._step_cost(S, res, base) if res is not None and self.FIX_CHAIN_RISK
                else np.zeros(res.tx.size) if res is not None else None)
        dcost = [float(self._field_eval(d[1], d[2])) - d[0] if self.FIX_CHAIN_RISK else 0.0
                 for d in drib]
        if self.BEAM:
            level = self._beam_chain(S, res, safe if res is not None else None,
                                     base if res is not None else None, cost,
                                     score if res is not None else None, drib, dcost, long,
                                     deadline)
        for n_lv, (depth, k, light) in enumerate(() if self.BEAM else self.CHAIN_PLAN, 1):
            if time.perf_counter() > deadline:
                break
            level = n_lv
            if res is not None:
                for i in np.argsort(-score)[:k]:
                    if not safe[i] or time.perf_counter() > deadline:
                        break
                    child = self._child(S, res, i)
                    score[i] = max(score[i], self._chain_value(child, depth, deadline, light=light)
                                   - cost[i])
            if long and depth <= self.DRIB_DEPTH:
                for n_d, d in enumerate(drib[:self.DRIB_K]):
                    if time.perf_counter() > deadline:
                        break
                    child = S.child(S.kicker, d[1], d[2], d[3], body=d[4])
                    d[0] = max(d[0], self._chain_value(child, depth, deadline, drib=False,
                                                       light=light) - dcost[n_d])
        if res is not None:
            j = int(np.argmax(score))
            if np.isfinite(score[j]):
                opts.append((float(score[j]), "pass", j))
        for score_d, ex, ey, _, _ in drib:
            opts.append((score_d, "dribble", (ex, ey)))

        T = 4
        B = np.tile(np.array([S.ox, S.oy]), (1, T + 1, 1))
        first = _race(ph, B, S.op_pos, S.op_body, np.full(S.nop, ph.kick_r + 0.5),
                      S.op_gk, np.zeros(S.nop, dtype=np.int64))
        t_th = float(first.min()) if S.nop else NEVER
        if t_th > T:
            opts.append((float(self._field_eval(S.ox, S.oy)) - 1.5, "hold", None))

        if not opts:
            return False
        best = max(opts, key=lambda o: o[0])
        if (self.CHAIN_FALLBACK and S.ox > 0.0
                and best[0] < float(self._field_eval(S.ox, S.oy)) - 12.0):
            note(agent, "chain_fallback", v=round(best[0], 1))
            return False
        note(agent, "chain", k=best[1], dw=self._drib_wins, lv=level,
             ms=round(1000.0 * (time.perf_counter() - t0), 1),
             cut=int(time.perf_counter() > deadline))
        self._execute(agent, S, best, res)
        return True

    def _beam_chain(self, S, res, safe, base, cost, score, drib, dcost, long, deadline):
        """Seed the chain beam with the safe root passes and dribbles; values
        found deeper are written back into `score` / `drib` of their root."""
        nodes = []
        if res is not None:
            for i in np.flatnonzero(safe):
                nodes.append((float(base[i]), float(cost[i]), int(i),
                              lambda i=i: self._child(S, res, i), self.BEAM_CARRIES,
                              (-1, int(res.who[i]))))
        if long or self.BEAM_ROOT_DRIB:
            for n_d, d in enumerate(drib[:self.DRIB_K]):
                nodes.append((d[0], dcost[n_d], -1 - n_d,
                              lambda d=d: S.child(S.kicker, d[1], d[2], d[3], body=d[4]),
                              self.BEAM_CARRIES - 1,
                              (-1, -1 - n_d) if self.BEAM_ROOT_DRIB else (-1, -1)))
        def raise_root(root, v):
            if root >= 0:
                score[root] = max(score[root], v)
            else:
                drib[-1 - root][0] = max(drib[-1 - root][0], v)

        return self._beam_run(nodes, self._beam_expand, raise_root, self.BEAM_W, self.BEAM_D,
                              deadline)

    def _beam_expand(self, C, g, light, may_drib, b):
        ph = self.ph
        out = []
        if C.ox > 20.0:
            _, pg, _ = self._eval_shots(C, ph.bmax, 1)
            if pg.max() >= self.SHOT_P:
                out.append((self.SHOT_V0_US + self.SHOT_V1 * float(pg.max()) - g, 0.0, None,
                            False, None))
        pressed = self._pressure(C)[0] < self.PRESS_D
        tx, ty, s0 = self._pass_targets(C, rings=() if light else self.CHILD_RINGS,
                                        speeds=self.CHILD_SPEEDS, space=not light)
        if tx.size:
            r = self._eval_kicks(C, tx, ty, s0, np.ones(tx.size, dtype=np.int64))
            safe, sc = self._chain_scores(C, r, pressed)
            if safe.any():
                step = (self._step_cost(C, r, sc) if self.FIX_CHAIN_RISK
                        else np.zeros(sc.size))
                for j in np.flatnonzero(safe):
                    out.append((float(sc[j]) - self.HOP_COST - g,
                                g + self.HOP_COST + float(step[j]),
                                lambda j=j: self._child(C, r, j), self.BEAM_CARRIES,
                                (b, C.recv[int(r.who[j])][0])))
        if may_drib > 0:
            goal_ang = math.degrees(math.atan2(-C.oy, ph.hl - C.ox))
            angs = goal_ang + np.array([-30.0, 0.0, 30.0])
            if not light:
                angs = np.concatenate([angs, self._slot_angles(C), self._evasive_angles(C)[:3]])
            for d in self._drib_options(C, self.CHILD_DRIB, angs, pressed)[:self.CHILD_DRIB_K]:
                step = float(self._field_eval(d[1], d[2])) - d[0] if self.FIX_CHAIN_RISK else 0.0
                out.append((d[0] - 1.0 - g, g + 1.0 + step,
                            lambda d=d: C.child(C.kicker, d[1], d[2], d[3], body=d[4]),
                            may_drib - 1, (b, -1)))
        return out

    def _beam_run(self, nodes, expand, raise_root, width, depth, deadline):
        """Generic anytime beam search.

        nodes: (value, accumulator, root index, make_state, dribbles left,
        diversity key). Each layer expands every kept node with `expand`,
        reports each child's value to its root action via `raise_root`
        (a root's value is the best line found below it), and keeps the
        best `width` children. Stops after `depth` layers or at `deadline`.
        Returns the number of layers completed.
        """
        beam = self._beam_keep(nodes, width)
        layer = 0
        while beam and layer < depth and time.perf_counter() < deadline:
            layer += 1
            light = self.DEEP_LIGHT and layer > 1
            nxt = []
            for b, (_, g, root, make, may_drib, _) in enumerate(beam):
                if time.perf_counter() > deadline:
                    break
                for v, g2, make2, may2, key in expand(make(), g, light, may_drib, b):
                    raise_root(root, v)
                    if make2 is not None:
                        nxt.append((v, g2, root, make2, may2, key))
            beam = self._beam_keep(nxt, width)
        return layer

    def _beam_keep(self, nodes, width):
        """Best `width` nodes, at most BEAM_DIV per (parent, receiver) key so
        the beam does not fill up with near-identical passes."""
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

    def _beam_exp(self, S, res, values, drib, deadline):
        """Expectimax beam search from the root (the search Agent1 uses).

        Root nodes are every feasible pass in `res` and every dribble in
        `drib`; each carries its accumulator (A, Q) = ((1 - p) * V(loss),
        p * GAMMA^t). Deeper values found by `_beam_expand_exp` raise the
        root's value in place (`values` for passes). Returns the dribble
        options with their backed-up values.
        """
        ph = self.ph
        nodes = []
        if values is not None:
            for i in np.flatnonzero(np.isfinite(values) & res.succ & (res.p > 0.0)):
                p = float(res.p[i])
                nodes.append((float(values[i]),
                              ((1.0 - p) * float(res.v_fail[i]), p * GAMMA ** int(res.t_us[i])),
                              int(i), lambda i=i: self._child(S, res, i), True,
                              (-1, int(res.who[i]))))
        dv = [o[0] for o in drib]
        for k, (_, _, (ex, ey, p, K, lx, ly)) in enumerate(drib):
            nodes.append((dv[k], ((1.0 - p) * float(ph.v_lose(lx, ly)), p * GAMMA ** K), -1 - k,
                          lambda ex=ex, ey=ey, K=K: S.child(
                              S.kicker, ex, ey, K,
                              body=math.degrees(math.atan2(ey - S.oy, ex - S.ox))),
                          False, (-1, -1)))

        def raise_root(root, v):
            if root >= 0:
                values[root] = max(values[root], v)
            else:
                dv[-1 - root] = max(dv[-1 - root], v)

        self._beam_run(nodes, self._beam_expand_exp, raise_root, self.EXP_W, self.EXP_D, deadline)
        return [(dv[k], "dribble", o[2]) for k, o in enumerate(drib)]

    def _beam_expand_exp(self, C, g, light, may_drib, b):
        """Children of state C for the expectimax beam.

        With g = (A, Q), each option at C is valued A + Q' * V where Q' folds
        in EXP_DEEP, the option's success probability and its discount; a
        shot is a leaf. Passes and dribbles both expand further, but a
        dribble may not directly follow another dribble (`may_drib`).
        """
        ph = self.ph
        A, Q = g
        q = Q * self.EXP_DEEP
        out = []
        if C.ox > 12.0:
            sv, _, _ = self._eval_shots(C, ph.bmax, 1)
            out.append((A + Q * self.EXP_SHOT * float(sv.max()), None, None, False, None))
        tx, ty, s0 = self._pass_targets(C, rings=() if light else self.CHILD_RINGS,
                                        speeds=self.CHILD_SPEEDS, space=not light)
        if tx.size:
            r = self._eval_kicks(C, tx, ty, s0, np.ones(tx.size, dtype=np.int64))
            for j in np.flatnonzero(r.succ & (r.p > 0.0)):
                A2 = A + q * (1.0 - float(r.p[j])) * float(r.v_fail[j])
                Q2 = q * float(r.p[j]) * GAMMA ** int(r.t_us[j])
                out.append((A2 + Q2 * float(r.vs[j]), (A2, Q2),
                            lambda j=j: self._child(C, r, j), True,
                            (b, C.recv[int(r.who[j])][0])))
        if may_drib:
            goal_ang = math.degrees(math.atan2(-C.oy, ph.hl - C.ox))
            angs = goal_ang + np.array([-30.0, 0.0, 30.0])
            if not light:
                angs = np.concatenate([angs, self._slot_angles(C), self._evasive_angles(C)[:3]])
            carries = []
            for p, ex, ey, K, lx, ly in self._dribble_cands(C, self.CHILD_DRIB, angs):
                A2 = A + q * (1.0 - p) * float(ph.v_lose(lx, ly))
                Q2 = q * p * GAMMA ** K
                carries.append((A2 + Q2 * float(ph.v_keep(ex, ey, C.space(ex, ey))), (A2, Q2),
                                lambda ex=ex, ey=ey, K=K: C.child(
                                    C.kicker, ex, ey, K,
                                    body=math.degrees(math.atan2(ey - C.oy, ex - C.ox))),
                                False, (b, -1)))
            carries.sort(key=lambda c: -c[0])
            out.extend(carries[:self.CHILD_DRIB_K])
        return out

    def _carry_in(self, agent, S, pg):
        """A short, safe dribble towards goal that improves on a shot with
        goal probability pg by at least 0.05, or None."""
        ph = self.ph
        if pg >= self.CARRY_IN_P or S.ox > ph.hl - 12.0 or S.nop == 0:
            return None
        if self._pressure(S)[0] < self.CARRY_IN_D:
            return None
        base = np.array([math.degrees(math.atan2(gy - S.oy, ph.hl - 2.0 - S.ox))
                         for gy in (-4.0, 0.0, 4.0)])
        best = None
        for p, ex, ey, K, _, _ in self._dribble_cands(S, (3.0, 6.0), base):
            if p < 0.9:
                continue
            _, pg2, _ = self._eval_shots(S.child(S.kicker, ex, ey, K), ph.bmax, 1)
            v = p * float(pg2.max())
            if v > pg + 0.05 and (best is None or v > best[2]):
                best = (ex, ey, v)
        return best

    def _child(self, S, res, i):
        """The state after candidate kick i of `res` is received."""
        recv = S.recv[int(res.who[i])][0]
        return S.child(recv, float(res.lx[i]), float(res.ly[i]), int(res.t_us[i]))

    def _chain_value(self, S, depth, deadline, drib=True, light=False, shoot=True):
        """Depth-limited chain value of state S (field-evaluation units).

        Used to look one step into the opponents' follow-ups in defence
        (always with light=True). The dribble branch below refers to an
        undefined name `C` when light=False; the current settings never get
        there (BEAM is on, so the non-beam chain loop never runs).
        """
        best = -np.inf
        if shoot and S.ox > 20.0:
            _, pg, _ = self._eval_shots(S, self.ph.bmax, 1)
            if pg.max() >= self.SHOT_P:
                best = self.SHOT_V0 + self.SHOT_V1 * float(pg.max())
        if depth <= 0 or time.perf_counter() > deadline:
            return best
        pressed = self._pressure(S)[0] < self.PRESS_D
        tx, ty, s0 = self._pass_targets(S, rings=() if light else self.CHILD_RINGS,
                                        speeds=self.CHILD_SPEEDS, space=not light)
        if tx.size:
            r = self._eval_kicks(S, tx, ty, s0, np.ones(tx.size, dtype=np.int64))
            safe, sc = self._chain_scores(S, r, pressed)
            if safe.any():
                if depth > 1:
                    cost = (self._step_cost(S, r, sc) if self.FIX_CHAIN_RISK
                            else np.zeros(sc.size))
                    for j in np.argsort(-sc)[:self.CHILD_K]:
                        if not safe[j] or time.perf_counter() > deadline:
                            break
                        sc[j] = max(sc[j], self._chain_value(self._child(S, r, j), depth - 1,
                                                             deadline,
                                                             light=light or self.DEEP_LIGHT)
                                    - cost[j])
                best = max(best, float(sc.max()) - 2.0)
        if drib:
            goal_ang = math.degrees(math.atan2(-S.oy, self.ph.hl - S.ox))
            angs = goal_ang + np.array([-30.0, 0.0, 30.0])
            if not light:
                angs = np.concatenate([angs, self._slot_angles(S), self._evasive_angles(C)[:3]])
            for d in self._drib_options(S, self.CHILD_DRIB, angs, pressed)[:self.CHILD_DRIB_K]:
                if time.perf_counter() > deadline:
                    break
                child = S.child(S.kicker, d[1], d[2], d[3], body=d[4])
                step = float(self._field_eval(d[1], d[2])) - d[0] if self.FIX_CHAIN_RISK else 0.0
                v = max(d[0], self._chain_value(child, depth, deadline, drib=False,
                                                light=light) - step) - 1.0
                if v > best:
                    best = v
                    self._drib_wins += 1
        return best