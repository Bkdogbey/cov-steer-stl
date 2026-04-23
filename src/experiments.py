"""Experiment helpers: setup, single run, comparison, animation.

These functions were previously defined inline in main.py. Moving them here
keeps main.py to ~35 lines of experiment-block declarations.

All configuration (device, save_dir, animate) is read from the merged YAML
config — the single entry point for every option.
"""

import copy
from pathlib import Path

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
    plot_covariance_sweep,
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
    """Run a single scenario end-to-end. Returns (PlanResult, env, cfg)."""
    cfg, dyn_cfg, dynamics, steerer, env, mu0, Sigma0 = setup_scenario(scenario_path)
    planner = get_planner(cfg, dynamics, steerer, env)
    result = planner.solve(mu0, Sigma0, verbose=verbose)
    return result, env, cfg


def run_scenario_plot(scenario_path, verbose=True, mc_samples=0):
    """Run a single scenario and save/show a trajectory plot.

    Args:
        scenario_path: path to scenario YAML
        verbose:       print optimisation progress
        mc_samples:    if > 0, run Monte Carlo verification with this many samples
    """
    cfg, dyn_cfg, dynamics, steerer, env, mu0, Sigma0 = setup_scenario(scenario_path)
    device = torch.device(cfg["device"])
    T = cfg["horizon"]
    planner = get_planner(cfg, dynamics, steerer, env)
    result = planner.solve(mu0, Sigma0, verbose=verbose)

    save_dir = Path(cfg.get("save_dir", "data/results"))
    label = cfg.get("label", "scenario").lower().replace(" ", "_")
    save_dir.mkdir(parents=True, exist_ok=True)

    mu_np = result.mu_trace.detach().cpu().squeeze().numpy()
    S_np = result.Sigma_trace.detach().cpu().squeeze().numpy()

    fig, ax = plt.subplots(figsize=(10, 10))
    plot_trajectory(ax, mu_np, S_np, env, T,
                    title=f"{cfg.get('label', 'Scenario')}  |  P(φ)={result.best_p:.3f}")
    plt.tight_layout()
    fig.savefig(save_dir / f"{label}_trajectory.png", dpi=150)
    plt.show()

    if mc_samples > 0:
        _run_mc_and_plot(result, dynamics, env, cfg, mu0, Sigma0, T,
                         mc_samples, save_dir, label, device)

    return result, env, cfg


def _mode_cfg(base_cfg, mode):
    """Merge a per-mode YAML section (open_loop or closed_loop) into the base config.

    Scenario YAML can contain an optional top-level key matching *mode* with
    'weights' and/or 'optimizer' subsections that override the base values:

        open_loop:
          weights:
            w_du: 0.2       # allow sharper bends
            w_repulsion: 0.5
          optimizer:
            lr_v: 0.06

    Keys not present in the mode section fall back to the base config.
    The mode section itself is stripped so the returned config is clean.
    """
    overrides = base_cfg.get(mode, {})
    # strip both mode sections before copying so they don't pollute the result
    result = {k: v for k, v in base_cfg.items() if k not in ("open_loop", "closed_loop")}
    if "weights" in overrides:
        result["weights"] = {**result.get("weights", {}), **overrides["weights"]}
    if "optimizer" in overrides:
        result["optimizer"] = {**result.get("optimizer", {}), **overrides["optimizer"]}
    return result


def _run_mc_and_plot(result, dynamics, env, cfg, mu0, Sigma0, T,
                     mc_samples, save_dir, label, device, suffix=""):
    """Internal helper: run MC verification and save plot."""
    from monte_carlo import mc_verify
    from visualization.monte_carlo import plot_mc_verification

    spec = env.get_specification(T)
    print(f"\n── Monte Carlo Verification (N={mc_samples}) ──")
    mc_result = mc_verify(result, dynamics, spec, mu0, Sigma0,
                          n_samples=mc_samples, device=str(device))
    n_ok = int(mc_result["successes"].sum())
    print(f"  Analytic  P(φ) = {mc_result['p_analytic']:.4f}")
    print(f"  Empirical P̂(φ) = {mc_result['p_empirical']:.4f}"
          f"  ({n_ok}/{mc_samples} samples satisfied)")

    fig = plot_mc_verification(
        mc_result, env, cfg, result,
        save_path=Path(save_dir) / f"{label}{suffix}_mc_verification.png",
    )
    plt.show()
    return mc_result


def run_comparison(scenario_path, mc_samples=0):
    """Run open-loop vs closed-loop on the same scenario.

    Each mode can have its own weight/optimizer overrides via top-level
    'open_loop:' and 'closed_loop:' sections in the scenario YAML.
    Keys not overridden fall back to the shared base config.

    Args:
        scenario_path: path to scenario YAML
        mc_samples:    if > 0, run Monte Carlo verification on both results

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
    cfg_ol = _mode_cfg(cfg, "open_loop")
    cfg_ol["optimizer"] = {**cfg_ol["optimizer"], "lr_k": 0.0}  # K never updated
    planner_ol = SingleShotPlanner(dynamics, steerer_ol, env, cfg_ol)
    result_ol = planner_ol.solve(mu0, Sigma0, T=T, verbose=True)

    # ── Closed-loop covariance steering ─────────────────────────────
    print("\n── Closed-Loop (K optimised) ──")
    steerer_cl = get_steerer("closed_loop", dynamics)
    cfg_cl = _mode_cfg(cfg, "closed_loop")
    planner_cl = SingleShotPlanner(dynamics, steerer_cl, env, cfg_cl)
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
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    label = cfg.get("label", "comparison").lower().replace(" ", "_")

    fig = plot_comparison(
        result_ol, result_cl, env, T,
        save_path=save_dir / f"{label}_trajectories.png",
    )
    plt.show()

    # Convergence: p_sat and loss are tracked separately to avoid confusion
    fig = plot_convergence(
        [
            {"p_sat": result_ol.p_history, "loss": result_ol.history},
            {"p_sat": result_cl.p_history, "loss": result_cl.history},
        ],
        labels=["Open-Loop", "Cov Steering"],
        save_path=save_dir / f"{label}_convergence.png",
    )
    plt.show()

    # Control sequence plot for the closed-loop result
    fig = plot_control_sequence(
        result_cl, dt=dt,
        save_path=save_dir / f"{label}_controls.png",
    )
    plt.show()

    # Animation (controlled by cfg["animate"], default false)
    if do_animate:
        gif_path = save_dir / f"{label}.gif"
        print(f"  Saving animation → {gif_path}")
        animate_trajectory(result_cl, env, filename=str(gif_path), dt=dt)

    # Monte Carlo verification (opt-in via mc_samples > 0)
    if mc_samples > 0:
        print("\n── Open-Loop MC ──")
        _run_mc_and_plot(result_ol, dynamics, env, cfg, mu0, Sigma0, T,
                         mc_samples, save_dir, label, device, suffix="_open_loop")
        print("\n── Closed-Loop MC ──")
        _run_mc_and_plot(result_cl, dynamics, env, cfg, mu0, Sigma0, T,
                         mc_samples, save_dir, label, device, suffix="_closed_loop")

    return result_ol, result_cl


def _build_steerers_and_env(dyn_cfg, cfg, device):
    """Instantiate dynamics, both steerers, and env from configs."""
    dynamics = get_dynamics(dyn_cfg, device)
    steerer_ol = get_steerer("open_loop", dynamics)
    steerer_cl = get_steerer("closed_loop", dynamics)
    env = build_environment(cfg, device)
    return dynamics, steerer_ol, steerer_cl, env


def _sweep_one_point(dyn_cfg, cfg, mu0, Sigma0, T, device, max_iters, mc_samples):
    """Run OL + CL for a single sweep configuration. Returns metrics dict."""
    from monte_carlo import mc_verify

    dynamics, steerer_ol, steerer_cl, env = _build_steerers_and_env(dyn_cfg, cfg, device)

    cfg_ol = _mode_cfg(cfg, "open_loop")
    cfg_ol["optimizer"] = {**cfg_ol["optimizer"], "lr_k": 0.0, "max_iters": max_iters}
    cfg_cl = _mode_cfg(cfg, "closed_loop")
    cfg_cl["optimizer"] = {**cfg_cl["optimizer"], "max_iters": max_iters}

    planner_ol = SingleShotPlanner(dynamics, steerer_ol, env, cfg_ol)
    result_ol = planner_ol.solve(mu0, Sigma0, T=T, verbose=False)

    planner_cl = SingleShotPlanner(dynamics, steerer_cl, env, cfg_cl)
    result_cl = planner_cl.solve(mu0, Sigma0, T=T, verbose=False)

    row = {
        "p_ol_analytic": result_ol.best_p,
        "p_cl_analytic": result_cl.best_p,
        "p_ol_mc": None,
        "p_cl_mc": None,
    }

    if mc_samples > 0:
        spec = env.get_specification(T)
        mc_ol = mc_verify(result_ol, dynamics, spec, mu0, Sigma0, mc_samples, str(device))
        mc_cl = mc_verify(result_cl, dynamics, spec, mu0, Sigma0, mc_samples, str(device))
        row["p_ol_mc"] = mc_ol["p_empirical"]
        row["p_cl_mc"] = mc_cl["p_empirical"]

    return row


def run_covariance_sweep(
    scenario_path,
    sigma0_values=None,
    D_values=None,
    mc_samples=200,
    max_iters_sweep=300,
):
    """Sweep initial covariance Σ₀ and process noise D; plot P(φ) for OL vs CL.

    Args:
        scenario_path:  path to scenario YAML
        sigma0_values:  list of variance values for cov_diag (all entries set equal)
        D_values:       list of D_diag scalar values
        mc_samples:     MC samples per point (0 to skip MC)
        max_iters_sweep: optimizer iterations per sweep point (fewer than full run)
    """
    if sigma0_values is None:
        sigma0_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
    if D_values is None:
        D_values = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2]

    base_cfg, base_dyn_cfg = load_scenario(scenario_path)
    device = torch.device(base_cfg["device"])
    T = base_cfg["horizon"]
    save_dir = base_cfg.get("save_dir", "data/results")
    label = base_cfg.get("label", "sweep")
    n_state = len(base_cfg["initial_state"]["mean"])

    # ── Σ₀ sweep ────────────────────────────────────────────────────────
    print(f"\n── Σ₀ sweep ({len(sigma0_values)} points) ──")
    sigma0_rows = []
    for var in sigma0_values:
        sigma = var ** 0.5
        print(f"  σ₀={sigma:.4f} ...", end=" ", flush=True)

        cfg = copy.deepcopy(base_cfg)
        cfg["initial_state"]["cov_diag"] = [var] * n_state

        init = cfg["initial_state"]
        mu0 = torch.tensor(init["mean"], dtype=torch.float32, device=device)
        Sigma0 = torch.diag(torch.tensor(init["cov_diag"], dtype=torch.float32, device=device))

        row = _sweep_one_point(base_dyn_cfg, cfg, mu0, Sigma0, T, device,
                               max_iters_sweep, mc_samples)
        row["sigma"] = sigma
        sigma0_rows.append(row)
        print(f"OL={row['p_ol_analytic']:.3f}  CL={row['p_cl_analytic']:.3f}"
              + (f"  MC-OL={row['p_ol_mc']:.3f}  MC-CL={row['p_cl_mc']:.3f}"
                 if mc_samples > 0 else ""))

    # ── D sweep ─────────────────────────────────────────────────────────
    print(f"\n── D sweep ({len(D_values)} points) ──")
    D_rows = []

    init = base_cfg["initial_state"]
    mu0_base = torch.tensor(init["mean"], dtype=torch.float32, device=device)
    Sigma0_base = torch.diag(torch.tensor(init["cov_diag"], dtype=torch.float32, device=device))

    for d_val in D_values:
        print(f"  D={d_val:.4f} ...", end=" ", flush=True)

        dyn_cfg = copy.deepcopy(base_dyn_cfg)
        dyn_cfg["D_diag"] = d_val

        row = _sweep_one_point(dyn_cfg, base_cfg, mu0_base, Sigma0_base, T, device,
                               max_iters_sweep, mc_samples)
        row["d"] = d_val
        D_rows.append(row)
        print(f"OL={row['p_ol_analytic']:.3f}  CL={row['p_cl_analytic']:.3f}"
              + (f"  MC-OL={row['p_ol_mc']:.3f}  MC-CL={row['p_cl_mc']:.3f}"
                 if mc_samples > 0 else ""))

    # ── Plot ─────────────────────────────────────────────────────────────
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fig = plot_covariance_sweep(sigma0_rows, D_rows, label, save_dir)
    plt.show()
    stem = label.lower().replace(" ", "_")
    print(f"\n  Saved → {save_dir / f'{stem}_covariance_sweep.png'}")

    return sigma0_rows, D_rows
