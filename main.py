
"""
Covariance Steering + pdSTL
============================
Three experiments — toggle "run" / "skip" to select which ones to execute.
All options (device, save_dir, weights, ...) live in the YAML configs:
  configs/defaults.yaml                     shared defaults
  configs/scenarios/double_slit.yaml        single-shot: open-loop vs closed-loop
  configs/scenarios/narrow_gap.yaml         MPC: receding-horizon, double integrator
  configs/scenarios/obstacle_field.yaml     MPC: receding-horizon, single integrator
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from utils import skip_run
from experiments import (
    run_comparison, run_mpc_scenario, run_double_slit_live,
    run_lane_change_normal, run_lane_change_aggressive,
)


# ── 1. Double Slit — live iteration visualisation ────────────────────
with skip_run("skip", "Double Slit — Live") as check, check():
    run_double_slit_live("configs/scenarios/double_slit.yaml")

# ── 1b. Double Slit — static open-loop vs closed-loop comparison ─────
with skip_run("skip", "Double Slit — Single-Shot Comparison") as check, check():
    run_comparison("configs/scenarios/double_slit.yaml")


# ── 2. Narrow Gap — MPC, double integrator ───────────────────────────
with skip_run("skip", "Narrow Gap — MPC") as check, check():
    run_mpc_scenario("configs/scenarios/narrow_gap.yaml", mc_samples=1000)


# ── 3. Obstacle Field — MPC, single integrator ───────────────────────
with skip_run("skip", "Obstacle Field — MPC") as check, check():
    run_mpc_scenario("configs/scenarios/obstacle_field.yaml")


# ── 4. Lane Merge — single-shot OL vs CL comparison ─────────────────
with skip_run("skip", "Lane Merge — Single-Shot Comparison") as check, check():
    run_comparison("configs/scenarios/lane_merge.yaml", mc_samples=500)


# ── 5. Lane Change — MPC with covariance steering (pdstl-style) ──────
with skip_run("run", "Lane Change — Normal") as check, check():
    run_lane_change_normal()

with skip_run("skip", "Lane Change — Aggressive") as check, check():
    run_lane_change_aggressive()