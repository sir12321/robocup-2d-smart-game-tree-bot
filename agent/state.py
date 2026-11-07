"""Search states: who is about to kick, from where, and where everyone is.

A `_State` is a node of the on-ball search. `child()` imagines the world
after a pass (or a dribble) has been received: the receiver has the ball,
opponents have closed in on it, teammates have drifted and advanced, and the
passer may be making a give-and-go run.
"""
import math
import numpy as np

from .config import (
    OPP_MARGIN,
    CHILD_DRIFT,
    CHILD_LEAD_FIX,
    CHILD_OPP_VEL,
    CHILD_OPP_SPEED,
    OPP_PURSUIT,
    OPP_UNC,
    OPP_UNC_MAX,
    GK_REACH,
    GIVE_GO,
    GO_PURSUIT,
    GO_X0,
)
from .physics import _go_point


class _State:
    """One node of the on-ball search.

    ox, oy     ball position; `kicker` is the uniform number on the ball.
    mates      [(unum, x, y, body)] of our field players.
    opps       [(unum, x, y, body, is_goalie)].
    recv       teammates who can receive: everyone but the kicker, minus
               anyone offside.
    P, body, ctrl, gk, react, vel, omni, tired
               receivers then opponents, packed as the arrays `_race` takes.
    lead       opponents are raced from position + lead * velocity.
    adv        how far teammates push up in child states (CHILD_ADV).
    theirs     True when the state is mirrored to model the opponents'
               attack (see DefenseMixin._their_root).
    """

    def __init__(self, ph, ox, oy, kicker, mates, opps, offside_x, opp_vel=None, mate_vel=None,
                 opp_unc=None, lead=0.0, tired_map=None):
        self.ph = ph
        self.ox, self.oy = float(ox), float(oy)
        self.kicker = kicker
        self.mates = mates
        self.opps = opps
        self.offside_x = offside_x
        self.kbody = next((m[3] for m in mates if m[0] == kicker), 0.0)
        self.lead = lead
        self.adv = 0.0
        self.theirs = False
        line = max(offside_x, ox)
        self.recv = [m for m in mates
                     if m[0] != kicker and not (m[1] > line and m[1] > 0.0)]

        nus, nop = len(self.recv), len(opps)
        self.nus, self.nop = nus, nop
        self.op_pos = (np.array([[o[1], o[2]] for o in opps])
                       if nop else np.zeros((0, 2)))
        op_body = np.array([o[3] for o in opps]) if nop else np.zeros(0)
        op_gk = np.array([1 if o[4] else 0 for o in opps], dtype=np.int64)
        us_pos = (np.array([[m[1], m[2]] for m in self.recv])
                  if nus else np.zeros((0, 2)))
        us_body = np.array([m[3] for m in self.recv]) if nus else np.zeros(0)

        unc = np.zeros(nop) if opp_unc is None else np.asarray(opp_unc, dtype=float)
        self.P = np.vstack([us_pos, self.op_pos])
        self.body = np.concatenate([us_body, op_body])
        self.ctrl = np.concatenate([np.full(nus, ph.kick_r),
                                    np.full(nop, ph.kick_r + OPP_MARGIN) + unc])
        self.gk = np.concatenate([np.zeros(nus, dtype=np.int64), op_gk])
        self.react = np.ones(nus + nop, dtype=np.int64)
        self.op_ctrl = np.full(nop, ph.kick_r + OPP_MARGIN) + unc
        self.op_gk = op_gk
        self.op_body = op_body
        self.own_gk = None
        ov = np.asarray(opp_vel, dtype=float).reshape(-1, 2) if opp_vel is not None else np.zeros((nop, 2))
        mv = mate_vel or {}
        self.ov, self.mv = ov, mv
        uv = np.array([mv.get(m[0], (0.0, 0.0)) for m in self.recv], dtype=float).reshape(-1, 2)
        self.vel = np.vstack([uv, ov])
        self.P_race = self.P - np.vstack([np.zeros((nus, 2)), self.lead * ov])
        self.omni = np.concatenate([np.zeros(nus, dtype=bool), np.ones(nop, dtype=bool)])
        self.tired_map = tired_map
        self.tired = (None if tired_map is None else
                      np.array([tired_map.get(m[0], ph.J) for m in self.recv] + [ph.J] * nop,
                               dtype=np.int64))

    def space(self, x, y):
        """Distance from (x, y) to the nearest opponent (20 if there are none)."""
        if self.nop == 0:
            return np.full(np.shape(x), 20.0)
        d = np.hypot(np.asarray(x)[..., None] - self.op_pos[:, 0],
                     np.asarray(y)[..., None] - self.op_pos[:, 1])
        return d.min(axis=-1)

    def child(self, receiver_unum, lx, ly, t, body=None):
        """The state after `receiver_unum` controls the ball at (lx, ly) at cycle t.

        Opponents are the opponents' reply: the one nearest the reception
        point presses it, the others move to cover the space between it and
        the goal they defend, the keeper guards that goal, and each gets extra control
        radius for the uncertainty of the extrapolation. Teammates coast on
        their velocity and push up by `adv`; the passer runs a give-and-go if
        GIVE_GO applies. Passing the kicker's own number models a dribble.
        """
        ph = self.ph
        drift = (1.0 - ph.pdecay ** t) / (1.0 - ph.pdecay) if CHILD_DRIFT else 0.0
        reach = (float(ph.DASH[min(max(t - 1, 0), ph.DASH.size - 1)]) if CHILD_DRIFT
                 else float(t))

        # Identify the primary pressing opponent (closest field player to (lx, ly))
        field_dists = []
        for k, (u, x, y, b, g) in enumerate(self.opps):
            if not g:
                coast = drift - (self.lead if CHILD_LEAD_FIX else 0.0)
                cx, cy = x + self.ov[k, 0] * coast, y + self.ov[k, 1] * coast
                field_dists.append((math.hypot(lx - cx, ly - cy), u))
        presser_u = min(field_dists)[1] if field_dists else None

        opps, unc, opp_vel = [], [], []
        for k, (u, x, y, b, g) in enumerate(self.opps):
            coast = drift - (self.lead if CHILD_LEAD_FIX else 0.0)
            x, y = x + self.ov[k, 0] * coast, y + self.ov[k, 1] * coast
            r = reach * (GK_REACH if g else 1.0)
            if g:
                tx = max(ph.hl - 1.5, lx)
                ty = float(np.clip(ly * 0.35, -ph.gh + 1.0, ph.gh - 1.0))
            elif u == presser_u or getattr(self, "theirs", False):
                tx, ty = lx, ly
            else:
                gx_goal = ph.hl
                d_goal = math.hypot(gx_goal - lx, -ly) or 1.0
                slot_x = lx + (gx_goal - lx) / d_goal * min(7.0, 0.35 * d_goal)
                slot_y = ly * 0.65
                tx = 0.55 * slot_x + 0.45 * max(x, lx + 2.0)
                ty = 0.55 * slot_y + 0.45 * y

            dx, dy = tx - x, ty - y
            d = math.hypot(dx, dy)
            step = min(max(d - (0.8 if u == presser_u else 0.0), 0.0), OPP_PURSUIT * r)
            v = (0.0, 0.0)
            if d > 1e-6:
                x, y = x + dx / d * step, y + dy / d * step
                b = math.degrees(math.atan2(ly - y, lx - x))
                if step > 0.5:
                    v = (CHILD_OPP_SPEED * dx / d, CHILD_OPP_SPEED * dy / d)
            opps.append((u, x, y, b, g))
            opp_vel.append(v)
            unc.append(min(OPP_UNC * r, OPP_UNC_MAX))

        mates, mate_vel = [], {}
        keep = ph.pdecay ** t
        cap = max(self.offside_x, lx) - 0.5
        for m in self.mates:
            if m[0] == receiver_unum:
                mates.append((m[0], lx, ly, m[3] if body is None else body))
                continue
            if m[0] == self.kicker and GIVE_GO and self.ox > GO_X0:
                gx, gy, ux, uy = _go_point(ph, self.ox, self.oy, self.offside_x,
                                           GO_PURSUIT * reach)
                mates.append((m[0], gx, gy, math.degrees(math.atan2(uy, ux))))
                mate_vel[m[0]] = (0.8 * ux, 0.8 * uy)
                continue
            if m[0] == self.kicker:
                x, y = self.ox, self.oy
            else:
                vx, vy = self.mv.get(m[0], (0.0, 0.0))
                x, y = m[1] + vx * drift, m[2] + vy * drift
                mate_vel[m[0]] = (vx * keep, vy * keep)
            if self.adv:
                x = max(x, min(x + self.adv * reach, cap))
            mates.append((m[0], x, y, m[3]))

        offside_x = self.offside_x
        if self.adv and len(opps) > 1:
            offside_x = max(offside_x, sorted((o[1] for o in opps), reverse=True)[1])

        s = _State(ph, lx, ly, receiver_unum, mates, opps, offside_x,
                   opp_vel=opp_vel if CHILD_OPP_VEL else None,
                   mate_vel=mate_vel, opp_unc=unc, lead=0.0,
                   tired_map=self.tired_map)
        s.own_gk = self.own_gk
        s.adv = self.adv
        s.theirs = getattr(self, "theirs", False)
        return s


class _KickResult:
    """Vectorised evaluation of C candidate kicks (see EvaluatorMixin._eval_kicks).

    value   expected value of each kick; p: success probability.
    t_us / t_th   first cycle we / they reach the ball; who: index into recv.
    lx, ly  where we receive it; succ: we get there before it goes out.
    v_fail  value if the pass fails; vs: value if it succeeds.
    tx, ty, s0, delay   the kick itself (target, speed, extra kick cycles).
    """
    __slots__ = ("value", "p", "t_us", "t_th", "who", "lx", "ly", "succ", "v_fail",
                 "vs", "tx", "ty", "s0", "delay")