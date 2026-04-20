"""Experiment helpers: setup, single run, comparison, animation.

These functions were previously defined inline in main.py. Moving them here
keeps main.py to ~35 lines of experiment-block declarations.

All configuration (device, save_dir, animate) is read from the merged YAML
config — the single entry point for every option.
"""

import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from utils import load_scenario
from dynamics import get_dynamics
from steering import get_steerer
from planning import get_planner, build_environment
from planning.single_shot import SingleShotPlanner
from visualization import (
    plot_comparison,
    plot_convergence,
    plot_trajectory,
    plot_control_sequence,
    animate_trajectory,
)


def setup_scenario(scenario_path):
    """Load config → build dynamics, steerer, env, initial belief.

    Device comes from cfg["device"] (resolved once in load_scenario).

    Returns:
        (cfg, dyn_cfg, dynamics, steerer, env, mu0, Sigma0)
    """
    cfg, dyn_cfg = load_scenario(scenario_path)
    device = torch.device(cfg["device"])

    dynamics = get_dynamics(dyn_cfg, device)
    steering_mode = cfg.get("planner", {}).get("steering", "closed_loop")
    steerer = get_steerer(steering_mode, dynamics)
    env = build_environment(cfg, device)

    init = cfg["initial_state"]
    mu0 = torch.tensor(init["mean"], dtype=torch.float32, device=device)
    Sigma0 = torch.diag(torch.tensor(init["cov_diag"], dtype=torch.float32, device=device))

    return cfg, dyn_cfg, dynamics, steerer, env, mu0, Sigma0


def run_scenario(scenario_path, verbose=True):
    """Run a single scenario end-to-end.

    Returns:
        (PlanResult, env, cfg)
    """
    cfg, dyn_cfg, dynamics, steerer, env, mu0, Sigma0 = setup_scenario(scenario_path)
    planner = get_planner(cfg, dynamics, steerer, env)
    result = planner.solve(mu0, Sigma0, verbose=verbose)
    return result, env, cfg


def run_comparison(scenario_path):
    """Run open-loop vs closed-loop on the same scenario.

    All options (save_dir, animate, dt) are read from the merged config YAML.
    Toggle them in configs/defaults.yaml or in the scenario's own YAML.

    Returns:
        (result_ol, result_cl)
    """
    cfg, dyn_cfg = load_scenario(scenario_path)
    device = torch.device(cfg["device"])
    save_dir = cfg.get("save_dir", "data/results")
    do_animate = cfg.get("animate", False)
    dt = dyn_cfg.get("dt", 0.2)

    dynamics = get_dynamics(dyn_cfg, device)
    env = build_environment(cfg, device)
    T = cfg["horizon"]

    init = cfg["initial_state"]
    mu0 = torch.tensor(init["mean"], dtype=torch.float32, device=device)
    Sigma0 = torch.diag(torch.tensor(init["cov_diag"], dtype=torch.float32, device=device))

    # ── Open-loop baseline (K ≡ 0) ──────────────────────────────────
    print("\n── Open-Loop (K ≡ 0) ──")
    steerer_ol = get_steerer("open_loop", dynamics)
    cfg_ol = {**cfg, "optimizer": {**cfg["optimizer"], "lr_k": 0.0}}
    planner_ol = SingleShotPlanner(dynamics, steerer_ol, env, cfg_ol)
    result_ol = planner_ol.solve(mu0, Sigma0, T=T, verbose=True)

    # ── Closed-loop covariance steering ─────────────────────────────
    print("\n── Closed-Loop (K optimised) ──")
    steerer_cl = get_steerer("closed_loop", dynamics)
    planner_cl = SingleShotPlanner(dynamics, steerer_cl, env, cfg)
    result_cl = planner_cl.solve(mu0, Sigma0, T=T, verbose=True)

    # ── Summary ──────────────────────────────────────────────────────
    S_end_ol = result_ol.Sigma_trace[0, -1, :2, :2].detach().cpu().numpy()
    S_end_cl = result_cl.Sigma_trace[0, -1, :2, :2].detach().cpu().numpy()
    det_ol = np.linalg.det(S_end_ol)
    det_cl = np.linalg.det(S_end_cl)
    print("\n  Summary:")
    print(f"    Open-loop  P(φ) = {result_ol.best_p:.4f},  det(Σ_end) = {det_ol:.2e}")
    print(f"    Cov-steer  P(φ) = {result_cl.best_p:.4f},  det(Σ_end) = {det_cl:.2e}")
    if det_cl > 1e-15:
        print(f"    Covariance reduction: {det_ol / det_cl:.1f}x")
    print(f"    ||K||_F = {result_cl.K.norm().item():.4f}")

    # ── Plots ─────────────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    label = cfg.get("label", "comparison").lower().replace(" ", "_")

    fig = plot_comparison(
        result_ol, result_cl, env, T,
        save_path=os.path.join(save_dir, f"{label}_trajectories.png"),
    )
    plt.show()

    # Convergence: p_sat and loss are tracked separately to avoid confusion
    fig = plot_convergence(
        [
            {"p_sat": result_ol.p_history, "loss": result_ol.history},
            {"p_sat": result_cl.p_history, "loss": result_cl.history},
        ],
        labels=["Open-Loop", "Cov Steering"],
        save_path=os.path.join(save_dir, f"{label}_convergence.png"),
    )
    plt.show()

    # Control sequence plot for the closed-loop result
    fig = plot_control_sequence(
        result_cl, dt=dt,
        save_path=os.path.join(save_dir, f"{label}_controls.png"),
    )
    plt.show()

    # Animation (controlled by cfg["animate"], default false)
    if do_animate:
        gif_path = os.path.join(save_dir, f"{label}.gif")
        print(f"  Saving animation → {gif_path}")
        animate_trajectory(result_cl, env, filename=gif_path, dt=dt)

    return result_ol, result_cl
