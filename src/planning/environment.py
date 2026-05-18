"""Environment: scenario geometry → STL specification.

Reads obstacle / goal / bounds definitions and constructs the
pdSTL formula. Decoupled from dynamics and steering.
"""

import torch
from stl import (
    Always, Eventually, And,
    RectangularGoalPredicate,
    RectangularObstaclePredicate,
    CircularObstaclePredicate,
    MovingRectangularObstaclePredicate,
)


class Environment:
    """Workspace geometry and constraint builder."""

    def __init__(self, device="cpu"):
        self.obstacles = []
        self.circle_obstacles = []
        self.moving_obstacles = []
        self.visit_regions = []
        self.lane_markings = []
        self.goal = None
        self.bounds = None
        self.device = device

    # ── Geometry mutators ────────────────────────────────────────────

    def add_obstacle(self, x_range, y_range):
        self.obstacles.append({"x": x_range, "y": y_range})

    def add_circle_obstacle(self, center, radius):
        self.circle_obstacles.append({"center": center, "radius": radius})

    def add_moving_obstacle(self, x_traj, y_traj, width, height):
        self.moving_obstacles.append(
            {"x_traj": x_traj, "y_traj": y_traj, "width": width, "height": height}
        )

    def add_visit_region(self, x_range, y_range):
        self.visit_regions.append({"x": x_range, "y": y_range})

    def add_lane_marking(self, x_range, y_pos, style="dashed"):
        self.lane_markings.append({"x": x_range, "y": y_pos, "style": style})

    def set_goal(self, x_range, y_range):
        self.goal = {"x": x_range, "y": y_range}

    def set_bounds(self, x_range, y_range):
        self.bounds = {"x": x_range, "y": y_range}

    # ── STL specification ────────────────────────────────────────────

    def get_specification(self, T, t_goal_start=0, t_safety_start=1):
        """Build the combined pdSTL formula for this environment.

        Args:
            T: planning horizon (steps)
            t_goal_start: start of goal liveness window
            t_safety_start: start of safety window (1 skips initial state)

        Returns:
            STL_Formula producing [B, T+1, 2] traces.
        """
        specs = []

        # Goal (liveness)
        if self.goal:
            pred = RectangularGoalPredicate(self.goal)
            specs.append(Eventually(pred, interval=[t_goal_start, T]))

        # Visit regions (liveness)
        for region in self.visit_regions:
            pred = RectangularGoalPredicate(region)
            specs.append(Eventually(pred, interval=[0, T]))

        # Obstacles (safety)
        obs_preds = []
        for obs in self.obstacles:
            obs_preds.append(RectangularObstaclePredicate(obs))
        for obs in self.circle_obstacles:
            obs_preds.append(CircularObstaclePredicate(obs, device=self.device))
        for obs in self.moving_obstacles:
            obs_preds.append(MovingRectangularObstaclePredicate(obs, device=self.device))

        if obs_preds:
            safe = obs_preds[0]
            for p in obs_preds[1:]:
                safe = And(safe, p)
            specs.append(Always(safe, interval=[t_safety_start, T]))

        # Workspace bounds (safety)
        if self.bounds:
            pred = RectangularGoalPredicate(self.bounds)
            specs.append(Always(pred, interval=[t_safety_start, T]))

        if not specs:
            raise ValueError("No constraints defined in environment.")

        combined = specs[0]
        for s in specs[1:]:
            combined = And(combined, s)
        return combined.to(self.device)


# ═════════════════════════════════════════════════════════════════════
# Config → Environment builder
# ═════════════════════════════════════════════════════════════════════

def build_environment(cfg, device="cpu"):
    """Construct an Environment from a scenario config dict."""
    env = Environment(device=device)
    if "goal" in cfg:
        env.set_goal(**cfg["goal"])
    if "bounds" in cfg:
        env.set_bounds(**cfg["bounds"])
    for vr in cfg.get("visit_regions", []):
        env.add_visit_region(**vr)
    for obs in cfg.get("obstacles", []):
        kind = obs.get("type", "rectangle")
        if kind == "circle":
            env.add_circle_obstacle(center=obs["center"], radius=obs["radius"])
        else:
            env.add_obstacle(x_range=obs["x_range"], y_range=obs["y_range"])
    for lm in cfg.get("lane_markings", []):
        env.add_lane_marking(
            x_range=lm["x_range"],
            y_pos=lm["y_pos"],
            style=lm.get("style", "dashed"),
        )
    return env
