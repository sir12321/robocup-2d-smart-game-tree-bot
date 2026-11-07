"""The agent class: Agent1 is the mixins combined."""
from src.student.interface import StarterAgent

from .config import AgentConfig
from .on_ball import OnBallMixin
from .evaluators import EvaluatorMixin
from .search import SearchMixin
from .off_ball import OffBallMixin
from .defense import DefenseMixin


class Agent1(StarterAgent, AgentConfig, OnBallMixin, EvaluatorMixin, SearchMixin, OffBallMixin, DefenseMixin):
    """Search-based 5-a-side player.

    The framework calls `on_ball` (OnBallMixin) when this player can kick
    the ball and `off_ball` (OffBallMixin) otherwise, only in normal play;
    set pieces and the goalkeeper are run by the framework. Players never
    communicate: they coordinate because each runs the same deterministic
    computation on the same (exact) world model.
    """

    def __init__(self):
        super().__init__()
        self.ph = None                # _Phys tables, built on first use
        self._ctx = None              # per-cycle shared context (OnBallMixin._context)
        self._commit = None           # two-kick pass/shot to continue next cycle
        self._support_target = None   # last support spot (hysteresis)
        self._prev_chaser = None      # last cycle's chaser (stickiness)
        self._running = None          # current attacking run type
        self._vmax_cache = {}         # max one-kick speed per direction, this cycle
        self._drib_wins = 0           # debug counter: dribbles chosen in the chain search
        self._recover = False         # hysteresis recovery mode state
        self._opp_poss = 0            # cycles the opponents have had the ball
        self._opp_poss_cycle = -1
        self._go = None               # active give-and-go run (cycle, x, y)
        self._stam = None             # StaminaPredictor
        self._stam_cycle = -1
        self._budget = {}             # teammates' sprint budgets (full dashes left)
        self._sig_done = set()        # stamina signalling bookkeeping (off)
        self._sig_pending = {}
        self._floor_roles = (-1, set())  # teammates expected to sprint this cycle


class Agent2:
    """Secondary agent stub."""
    pass
