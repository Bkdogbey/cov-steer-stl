"""Planning factory.

Usage:
    from planning import get_planner
    planner = get_planner(cfg, dynamics, steerer, env)
"""

from planning.base import BasePlanner, PlanResult
from planning.single_shot import SingleShotPlanner
from planning.receding_horizon import RecedingHorizonPlanner
from planning.environment import Environment, build_environment

_REGISTRY = {
    "single_shot": SingleShotPlanner,
    "receding_horizon": RecedingHorizonPlanner,
}


def get_planner(cfg, dynamics, steerer, env):
    """Instantiate a planner from the 'planner.type' config key."""
    ptype = cfg.get("planner", {}).get("type", "single_shot")
    return _REGISTRY[ptype](dynamics, steerer, env, cfg)
