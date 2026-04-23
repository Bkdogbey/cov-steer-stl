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

from utils import skip_run
from experiments import run_comparison, run_scenario_plot, run_covariance_sweep


# ── 1. Narrow Gap: Open-Loop vs Covariance Steering ─────────────────
with skip_run("skip", "Narrow Gap — Open-Loop vs Cov Steering") as check, check():
    run_scenario_plot("configs/scenarios/narrow_gap.yaml", mc_samples=500)
    run_comparison("configs/scenarios/narrow_gap.yaml")


# ── 2. Obstacle Field ────────────────────────────────────────────────
with skip_run("skip", "Obstacle Field") as check, check():
    run_scenario_plot("configs/scenarios/obstacle_field.yaml")


# ── 3. Double Slit: MPC Covariance Steering ──────────────────────────
with skip_run("run", "Double Slit — MPC Cov Steering") as check, check():
    run_scenario_plot("configs/scenarios/double_slit.yaml")


# ── 4. Covariance Sweep: where OL fails and CL holds ────────────────
with skip_run("skip", "Covariance Sweep — Narrow Gap") as check, check():
    run_covariance_sweep(
        "configs/scenarios/narrow_gap.yaml",
        sigma0_values=[0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5],
        D_values=[0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2],
        mc_samples=200,
        max_iters_sweep=300,
    )
