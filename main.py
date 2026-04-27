"""
Covariance Steering + pdSTL — Main Entry Point
================================================
Toggle "run" / "skip" to select which experiments to execute.
All options (device, save_dir, animate, weights, ...) live in the YAML configs:
  configs/defaults.yaml       ← shared defaults for every scenario
  configs/scenarios/*.yaml    ← per-scenario overrides

To run tests:
  make test        # fast tests only (excludes @pytest.mark.slow)
  make test-all    # full suite including gradient-flow tests
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from utils import skip_run
from experiments import run_comparison, run_scenario_plot, run_joint_noise_sweep, run_mpc_scenario


# ── 1. Narrow Gap: MPC Covariance Steering ───────────────────────────
with skip_run("skip", "Narrow Gap — MPC Cov Steering") as check, check():
    run_mpc_scenario("configs/scenarios/narrow_gap.yaml", mc_samples=1000)

# ── 1b. Narrow Gap: Open-Loop vs Single-Shot Comparison ──────────────
with skip_run("run", "Narrow Gap — Single-Shot Comparison") as check, check():
    run_comparison("configs/scenarios/narrow_gap_single_shot.yaml")


# ── 2. Obstacle Field ────────────────────────────────────────────────
with skip_run("skip", "Obstacle Field") as check, check():
    run_scenario_plot("configs/scenarios/obstacle_field.yaml")


# ── 3. Double Slit: MPC Covariance Steering ──────────────────────────
with skip_run("skip", "Double Slit — MPC Cov Steering") as check, check():
    run_scenario_plot("configs/scenarios/double_slit.yaml")


# ── 4. Joint Noise Sweep: where OL fails and CL holds ───────────────
with skip_run("skip", "Joint Noise Sweep — Double Slit") as check, check():
    run_joint_noise_sweep(
        "configs/scenarios/double_slit.yaml",
        noise_levels=[0.0001, 0.001, 0.01, 0.1, 0.5],
        mc_samples=1000,
        max_iters_sweep=800,
    )
