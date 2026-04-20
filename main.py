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
import matplotlib.pyplot as plt

from utils import skip_run
from experiments import run_comparison, run_scenario
from visualization import plot_trajectory


# ── 1. Narrow Gap: Open-Loop vs Covariance Steering ─────────────────
with skip_run("skip", "Narrow Gap — Open-Loop vs Cov Steering") as check, check():
    run_comparison("configs/scenarios/narrow_gap.yaml")


# ── 2. Obstacle Field (single run) ──────────────────────────────────
with skip_run("skip", "Obstacle Field") as check, check():
    result, env, cfg = run_scenario("configs/scenarios/obstacle_field.yaml")
    mu_np = result.mu_trace.detach().cpu().squeeze().numpy()
    S_np = result.Sigma_trace.detach().cpu().squeeze().numpy()
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_trajectory(ax, mu_np, S_np, env, cfg["horizon"],
                    title=f"Obstacle Field  |  P(φ)={result.best_p:.3f}")
    plt.tight_layout()
    plt.show()


# ── 3. Lane Change MPC ───────────────────────────────────────────────
with skip_run("run", "Lane Change MPC") as check, check():
    result, env, cfg = run_scenario("configs/scenarios/lane_change.yaml")
    print(f"  Lane change done. P(φ) = {result.best_p:.4f}")


# ── 4. Double Slit: primary test scenario (matches Okamoto 2019 Fig 2) ─
with skip_run("skip", "Double Slit — Open-Loop vs Cov Steering") as check, check():
    run_comparison("configs/scenarios/double_slit.yaml")
