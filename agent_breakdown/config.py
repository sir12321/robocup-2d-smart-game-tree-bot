"""Agent1 class header and tuning constants, per-cycle context, risk, time budget.

Part view of agent.py, written by `agent_breakdown/parts.py split`.
Edit the code between the begin/end markers, then run `agent_breakdown/parts.py build`
to write it back into agent.py. Text outside the markers is ignored."""

# ==== begin Agent1 ====


class Agent1(StarterAgent):

    PASS_M0 = 0.8
    PASS_S = 0.6
    FIT_M0 = -1.9
    FIT_S = 2.8
    RINGS = ((3.0, 8), (6.0, 8), (9.0, 12))
    END_SPEEDS = (1.0, 1.4, 1.8)
    CHILD_RINGS = ((3.0, 8), (6.0, 8))
    CHILD_SPEEDS = (1.0, 1.8)
    MULTI_DELAY = 1
    KICK_LATE = 2
    P_PASS = 0.8
    P_DRIB = 0.9
    P_ATT = 0.65
    P_DRIB_ATT = 0.75
    LOSS_ATT = 30.0
    BACK_W = 1.0
    SLOT_W = 15.0
    DRIB_K = 6
    CHANNEL_V = 1.7
    CONE = 35.0
    BACK_X = -10.0
    DIAG_X0 = -15.0
    ENGAGE_D = 3.5
    ENGAGE_CONE = 25.0
    HOLD_ENGAGE = 15.0
    HOLD_V = 0.08
    CARRY_IN_P = 0.8
    CARRY_IN_D = 3.0
    V_BACKPASS = -0.85
    SHADE = 0.6
    SHOT_P = 0.7
    CHILD_K = 3
    BEAM_W = 10
    BEAM_D = 3
    BEAM_DIV = 3
    CHILD_ADV = 0.35
    BEAM_CARRIES = 2
    HOP_COST = 0.5
    SHOT_V0_US = 82.0
    BOX_SHOT_P = 0.7
    MARK_BEHIND = 8.0
    SHOT_KICK_LATE = 2
    FOUL_P = 0.8
    TOUCH_ST = 3000.0
    TOUCH_N = 2
    TOUCH_PRESS = 2.0
    TOUCH_P = 0.8
    CHAIN_MS = 300.0
    ONBALL_CAP_MS = 400.0
    ONBALL_SHARE = 10.0
    CHILD_DRIB_K = 2
    MID_LOSS = 18.0
    SHOT_V1 = 40.0
    SHOT_P_PRESSED = 0.3
    RUN_SPLIT_X = 20.0
    CLEAR_X = -30.0
    PRESS_D = 1.8
    HOLD_BAN_D = 2.2
    OUTLET_D = 4.0
    DRIBBLE_V = 0.73
    COMPACT = 0.7
    KB_GAP = 9.0
    KB_MIN_R = 7.0
    KB_MAX_R = 34.0
    ROLE_DEF = False
    ROLE_H = 5
    ROLE_K = 6
    ROLE_RUN = 0.05
    GK_M = 1.0
    GK_S = 0.5
    KICK_SCEN = False
    KICK_CONF = 0.6
    KS_BAND = 4
    KS_CHILD = False
    KN_RATE = 1.25
    CTRL_PHYS = False
    V_CTRL = 2.0
    TACKLE_REACH = False
    TACKLE_P = 0.8
    NOISE_SHOT = False
    SHOT_CONF = 0.71
    CENTER_GUARD = False
    GUARD_GAP = 2.0
    GUARD_GAP_ATT = 3.0
    GUARD_WG = 0.7
    GUARD_STICK = 4.0
    GUARD_URGENT = 2.5
    GUARD_REST = True
    TACKLE_ZONE = True
    TZ_X = 1.0
    FOUL_MAX_X = 20.0
    TZ_P_BOX = 0.6
    BACKPASS_X = 30.0
    SP_T = 30
    SP_BALL_D = 35.0
    SP_R = 25.0
    TZ_Y = 0.0
    CIRC_W = 0.5
    SELF_GOAL_W = 0.0
    CIRC_K = 1.0
    DEF_ROLES = False
    MARK_X = -5.0
    MARK_VIEW = 0.45
    COVER_FRAC = 0.35
    COVER_NEAR = 3.0
    COVER_FAR = 5.0
    JOB_STICK = 3.0
    SHAPE_BACK = 14.0
    SHAPE_SIDE_D = 6.0
    SHAPE_W = 7.0
    SHAPE_GOAL = 11.0
    REST_AHEAD = 10.0
    REST_BEHIND = 10.0
    REST_SPACE = 25.0
    LANE_GAP = 2.0
    BACK_STAY = 10.0
    JOCKEY_GAP = 1.2
    JOCKEY_GAP_BOX = 0.8
    FRONT_CLOSE = 4.0
    FRONT_STICK = 2
    HARRY_D = 6.0
    FRONT_STEP = 6.0
    MAN_LEAD = 4.0
    FRONT_AHEAD = 8.0
    CONTEST_T = 12
    CONTEST_D = 3
    STAM_MARGIN = 600.0
    OUTLET = False
    OUTLET_AHEAD = 15.0
    OUTLET_MIN = 8.0
    OUTLET_ATT = 2
    OUTLET_W = 16.0
    TURN_D = 2.2
    TURN_V = 0.5
    COUNTER = False
    COUNTER_T = 40
    COUNTER_DEF = 8
    CTR_RUN = 25.0
    CTR_LANE = 10.0
    FWD_CTR = 0.4
    BACK_BIAS_CTR = 1.0
    HOLD_CTR = 3.0
    OUT_Q = 0.5
    BACK_BIAS = 0.3
    TZ_P = 0.8
    RECOVER = True
    REC_ON = 0.5
    REC_OFF = 0.7
    REC_GAIN = 25.0
    DF_LINE_D = 20.0
    REC_SAFE_R = 25.0
    REC_SIGNAL = True
    REC_SIG_TOL = 0.5
    REC_SIG_N = 2
    REC_SIG_OFF = 15.0
    REC_SIG_GAP = 10
    KICK2_REAL = False
    KICK2_MID = 0.78
    KICK2_LOW = 0.42
    KICK2_ANG = 7.0
    REC_P_W = 25
    REC_P_WOFF = 40
    REC_P_MON = 0.0
    REC_P_MOFF = -300.0
    BLOCK_FRONT = False
# ==== end Agent1 ====

    # ==== begin Agent1.__init__ ====
    def __init__(self):
        super().__init__()
        self.ph = None
        self._ctx = None
        self._commit = None
        self._support_target = None
        self._prev_chaser = None
        self._running = None
        self._vmax_cache = {}
        self._stam = None
        self._stam_cycle = -1
        self._budget = {}
        self._floor_roles = (-1, set())
        self._guard_prev = None
        self._front_prev = None
        self._front_on = None
        self._job_prev = {}
        self._roles_c = -10
        self._outlet_y = None
        self._def_c0 = None
        self._counter_c = None
        self._ctx_c = -10
        self._counter_now = False
        self._budget_low = {}
        self._last_att_c = -100
        self._sp_c = -100
        self._recover = False
        self._rec_seen = {}
    # ==== end Agent1.__init__ ====

    # ==== begin Agent1._phys ====


    def _phys(self, agent):
        if self.ph is None:
            self.ph = _Phys(agent.server_params)
        return self.ph
    # ==== end Agent1._phys ====

    # ==== begin Agent1._allow_ms ====

    def _allow_ms(self, agent, on_ball):
        rem = agent.budget.remaining_ms
        if rem == float("inf"):
            rem = 60000.0
        left = max(60, MATCH_CYCLES - agent.wm.cycle)
        per = rem / left
        cap = self.ONBALL_CAP_MS if on_ball else 25.0
        allow = min(per * (self.ONBALL_SHARE if on_ball else 0.7), cap,
                    agent.budget.decision_remaining_ms - 300.0)
        return max(allow, 1.0)
    # ==== end Agent1._allow_ms ====

    # ==== begin Agent1._context ====

    def _context(self, agent):
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
            best = min((mate_t[i] - (0.5 if mates[i][0] == self._prev_chaser else 0.0),
                        mates[i][0]) for i in range(nm))
            chaser = best[1]
            best_t = int(mate_t[[m[0] for m in mates].index(chaser)])
            if gk_t < best_t:
                gi = min(int(gk_t), T)
                behind = B[0, gi, 0] < goalie[1] - 0.5
                if B[0, gi, 0] < -ph.pen_x and abs(B[0, gi, 1]) < ph.pen_hw and not behind:
                    chaser = None
        self._prev_chaser = chaser
        our_min = min(int(mate_t.min()) if nm else NEVER, int(gk_t))

        oi = min(opp_min, T)
        opp_first = None
        if opp_t.size and opp_min < NEVER:
            opp_first = opps[int(opp_t.argmin())][0]
        ui = min(our_min, T)

        self._ctx = {
            "key": key, "mates": mates, "goalie": goalie, "opps": opps,
            "mate_t": {mates[i][0]: int(mate_t[i]) for i in range(nm)},
            "ball": (bx, by, bvx, bvy), "chaser": chaser,
            "our_min": our_min, "opp_min": opp_min, "opp_first": opp_first,
            "our_loc": (float(B[0, ui, 0]), float(B[0, ui, 1])),
            "opp_loc": (float(B[0, oi, 0]), float(B[0, oi, 1])),
            "attack": (bool(wm.kickable_teammate_existance)
                       or (not wm.kickable_opponent_existance and nm > 0
                           and int(mate_t.min()) + (2 if bx > -25.0 else 5) <= opp_min)),
            "offside_x": wm.offside_line_x,
            "kickable_mate": bool(wm.kickable_teammate_existance),
            "kickable_opp": bool(wm.kickable_opponent_existance),
        }
        gd = None
        self._ctx["guard"], self._ctx["guard_spot"], self._ctx["guard_target"] = gd or (None, None, None)
        self._ctx["counter"] = self._counter_state(wm, self._ctx["attack"])
        return self._ctx
    # ==== end Agent1._context ====

    # ==== begin Agent1._root_state ====

    def _root_state(self, agent, ctx):
        wm = agent.wm
        mates = [(m[0], m[1], m[2], m[3]) for m in ctx["mates"]]
        opps = [(o[0], o[1] + o[5] * 0.6, o[2] + o[6] * 0.6, o[3], o[4])
                for o in ctx["opps"]]
        bx, by = ctx["ball"][0], ctx["ball"][1]
        S = _State(self.ph, bx, by, wm.self.uniform_number, mates, opps,
                   ctx["offside_x"], opp_vel=[(o[5], o[6]) for o in ctx["opps"]],
                   mate_vel={m[0]: (m[4], m[5]) for m in ctx["mates"]}, lead=0.6,
                   tired_map=self._budget_low or self._budget)
        S.adv = self.CHILD_ADV
        kr = float(getattr(wm.self, "kick_rate", 0.0) or 0.0)
        if kr > 0.0:
            bvx, bvy = ctx["ball"][2], ctx["ball"][3]
            rate = (1.5 - kr / self.ph.kick_power_rate
                    + 0.5 + 0.5 * math.hypot(bvx, bvy) / (self.ph.bmax * self.ph.decay))
            S.kick_cond = (rate, kr, bvx, bvy)
        g = ctx.get("goalie")
        if g:
            S.own_gk = (g[1], g[2])
        return S
    # ==== end Agent1._root_state ====

    # ==== begin Agent1._risk ====

    def _risk(self, agent):
        wm = agent.wm
        lead = wm.our_team_score - wm.their_team_score
        late = wm.cycle > 1150
        if late and lead > 0:
            return self.PASS_M0 + 0.4, 1.0
        if late and lead < 0:
            return self.PASS_M0 - 0.2, 1.15
        return self.PASS_M0, 1.0
    # ==== end Agent1._risk ====

    # ==== begin Agent1._counter_state ====

    def _counter_state(self, wm, attack):
        cyc = wm.cycle
        if cyc - self._ctx_c > 2:
            self._def_c0, self._counter_c = None, None
            self._sp_c = cyc
        self._ctx_c = cyc
        if attack:
            self._last_att_c = cyc
            if self._def_c0 is not None:
                spell = cyc - self._def_c0
                if spell >= self.COUNTER_DEF:
                    self._counter_c = cyc
                elif spell >= 3:
                    self._counter_c = None
                self._def_c0 = None
        else:
            if self._def_c0 is None:
                self._def_c0 = cyc
            if cyc - self._def_c0 >= 3:
                self._counter_c = None
        return bool(self.COUNTER and attack and self._counter_c is not None and int(wm.game_mode_type) == 2
                    and cyc - self._counter_c <= self.COUNTER_T)
    # ==== end Agent1._counter_state ====

