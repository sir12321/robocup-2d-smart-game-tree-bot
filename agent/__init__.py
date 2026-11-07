"""Agent1, split into readable modules.

The course harness loads a single file, so this package is the source and
`agent.py` at the repository root is generated from it by
`python3 tools/bundle.py`. The modules import from the starter framework
(`src.student.interface`), so the package only imports when the starter
(py2d/) is on the path.

    config.py      tuning constants and feature flags
    physics.py     server physics tables, value maps, the race model
    stamina.py     stamina estimates for all players
    state.py       search states and how the world moves after a pass
    evaluators.py  pass and shot evaluation, stamina bookkeeping
    search.py      beam / expectimax search over passes and dribbles
    on_ball.py     on_ball decision, shared per-cycle context, execution
    defense.py     defensive assignment against the opponents' options
    off_ball.py    off_ball rules: tackles, chasing, blocking, positioning
    player.py      Agent1 = all of the above
"""
from .player import Agent1, Agent2
from .config import AgentConfig
from .stamina import StaminaPredictor
from .physics import _Phys
from .state import _State, _KickResult

__all__ = ['Agent1', 'Agent2', 'AgentConfig', 'StaminaPredictor', '_Phys', '_State', '_KickResult']
