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

import os
import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
from utils import skip_run
from experiments import run_comparison, run_scenario_plot


# ── 1. Narrow Gap: Open-Loop vs Covariance Steering ─────────────────
with skip_run("skip", "Narrow Gap — Open-Loop vs Cov Steering") as check, check():
    run_comparison("configs/scenarios/narrow_gap.yaml")


# ── 2. Obstacle Field ────────────────────────────────────────────────
with skip_run("run", "Obstacle Field") as check, check():
    run_scenario_plot("configs/scenarios/obstacle_field.yaml")


# ── 3. Double Slit: Open-Loop vs Covariance Steering ────────────────
with skip_run("skip", "Double Slit — Open-Loop vs Cov Steering") as check, check():
    run_comparison("configs/scenarios/double_slit.yaml")
