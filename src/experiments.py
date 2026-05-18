"""Experiment entry points: single-shot comparison and MPC.

  run_comparison(scenario_path)     open-loop vs closed-loop single-shot
  run_mpc_scenario(scenario_path)   receding-horizon MPC with live plot
  run_joint_noise_sweep(...)        optional: sweep noise levels, OL vs CL
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
from planning.environment import Environment
from planning.base import PlanResult
from planning.single_shot import SingleShotPlanner
from visualization import (
    plot_comparison,
    plot_covariance_trace,
    plot_convergence,
    plot_trajectory,
    plot_control_sequence,
    animate_trajectory,
    plot_joint_noise_sweep,
)


def _canvas_frame(fig):
    """Capture a matplotlib figure as an RGB numpy array."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    return np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)[:, :, 1:]


def _build_params_dict(cfg, dyn_cfg, T):
    """Build the parameter display dict used in comparison figure captions."""
    init_cov = cfg.get("initial_state", {}).get("cov_diag", [0, 0])
    sigma0 = float(np.sqrt(max(init_cov[0], init_cov[1]))) if init_cov else 0.0
    w   = cfg.get("weights", {})
    opt = cfg.get("optimizer", {})
    return {
        "horizon":     T,
        "dt":          dyn_cfg.get("dt", 0.2),
        "D_diag":      dyn_cfg.get("D_diag", "?"),
        "sigma0":      sigma0,
        "w_phi":       w.get("w_phi", "?"),
        "w_repulsion": w.get("w_repulsion", "?"),
        "max_iters":   opt.get("max_iters", "?"),
        "lr_v":        opt.get("lr_v", "?"),
    }


def _print_result_summary(result_ol, result_cl):
    """Print open-loop vs closed-loop P(phi) and covariance summary."""
    S_end_ol = result_ol.Sigma_trace[0, -1, :2, :2].detach().cpu().numpy()
    S_end_cl = result_cl.Sigma_trace[0, -1, :2, :2].detach().cpu().numpy()
    det_ol = np.linalg.det(S_end_ol)
    det_cl = np.linalg.det(S_end_cl)
    print("\n  Summary:")
    print(f"    Open-loop  P(phi) = {result_ol.best_p:.4f},  det(Sigma_end) = {det_ol:.2e}")
    print(f"               route  = {getattr(result_ol, 'route_label', 'n/a')}")
    print(f"    Cov-steer  P(phi) = {result_cl.best_p:.4f},  det(Sigma_end) = {det_cl:.2e}")
    print(f"               route  = {getattr(result_cl, 'route_label', 'n/a')}")
    if det_cl > 1e-15:
        print(f"    Covariance reduction: {det_ol / det_cl:.1f}x")
    print(f"    ||K||_F = {result_cl.K.norm().item():.4f}")


def _set_seed(cfg, offset=0):
    """Make a scenario run repeatable when the YAML provides seed."""
    seed = cfg.get("seed")
    if seed is None:
        return
    seed = int(seed) + int(offset)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _warm_start_from_waypoints(cfg, dyn, mu0, T, waypoints=None):
    """Build a simple single-integrator feedforward warm start from XY waypoints."""
    waypoints = waypoints or cfg.get("warm_start_waypoints")
    if not waypoints:
        return None
    if dyn.nx != 2 or dyn.nu != 2:
        return None

    device = mu0.device
    pts = torch.tensor(waypoints, dtype=torch.float32, device=device)
    if torch.norm(pts[0] - mu0[:2]) > 1e-5:
        pts = torch.cat([mu0[:2].unsqueeze(0), pts], dim=0)

    seg = pts[1:] - pts[:-1]
    seg_len = torch.norm(seg, dim=1)
    total_len = seg_len.sum()
    distances = torch.linspace(0.0, total_len, T + 1, device=device)
    cumulative = torch.cat([torch.zeros(1, device=device), torch.cumsum(seg_len, dim=0)])

    ref = []
    for distance in distances:
        idx = int(torch.searchsorted(cumulative[1:], distance).item())
        idx = min(idx, len(seg_len) - 1)
        frac = ((distance - cumulative[idx]) / (seg_len[idx] + 1e-6)).clamp(0.0, 1.0)
        ref.append(pts[idx] + frac * seg[idx])

    ref = torch.stack(ref)
    u = (ref[1:] - ref[:-1]) / dyn.dt
    u = torch.clamp(u, -0.95 * dyn.u_max, 0.95 * dyn.u_max)
    z = torch.clamp(u / dyn.u_max, -0.95, 0.95)
    return 0.5 * torch.log((1.0 + z) / (1.0 - z))


def _warm_start_options(cfg, dyn, mu0, T):
    """Return named warm starts for route-choice scenarios."""
    candidates = cfg.get("warm_start_candidates")
    if candidates:
        return [
            (name, _warm_start_from_waypoints(cfg, dyn, mu0, T, waypoints))
            for name, waypoints in candidates.items()
        ]

    init_v = _warm_start_from_waypoints(cfg, dyn, mu0, T)
    if init_v is not None:
        return [("warm start", init_v)]
    return [("random", None)]


def setup_scenario(scenario_path):
    """Load config and build dynamics, steerer, env, and initial belief.

    Returns:
        (cfg, dyn_cfg, dynamics, steerer, env, mu0, Sigma0)
    """
    cfg, dyn_cfg = load_scenario(scenario_path)
    device = torch.device(cfg["device"])

    dynamics = get_dynamics(dyn_cfg, device)
    steering_mode = cfg.get("planner", {}).get("steering", "closed_loop")
    steerer = get_steerer(steering_mode, dynamics)
    T = cfg["horizon"]
    env = build_environment(cfg, device, T=T, dt=dyn_cfg.get("dt", 0.2))

    init = cfg["initial_state"]
    mu0 = torch.tensor(init["mean"], dtype=torch.float32, device=device)
    Sigma0 = torch.diag(torch.tensor(init["cov_diag"], dtype=torch.float32, device=device))

    return cfg, dyn_cfg, dynamics, steerer, env, mu0, Sigma0


def _mode_cfg(base_cfg, mode):
    """Merge a per-mode section (open_loop or closed_loop) into the base config.

    The scenario YAML can have top-level 'open_loop:' / 'closed_loop:' keys
    with 'weights' and/or 'optimizer' subsections that override the base.
    Both mode sections are stripped from the returned config.
    """
    overrides = base_cfg.get(mode, {})
    result = {k: v for k, v in base_cfg.items() if k not in ("open_loop", "closed_loop")}
    if "weights" in overrides:
        result["weights"] = {**result.get("weights", {}), **overrides["weights"]}
    if "optimizer" in overrides:
        result["optimizer"] = {**result.get("optimizer", {}), **overrides["optimizer"]}
    return result


def _solve_best_route_candidate(
    cfg, mode_key, dynamics, env, mu0, Sigma0, T, verbose=True, iter_callback=None
):
    """Optimize all configured route candidates and keep the best probability."""
    best_result = None
    best_name = None

    options = _warm_start_options(cfg, dynamics, mu0, T)
    multi_candidate = len(options) > 1

    for candidate_name, init_v in options:
        _set_seed(cfg)
        cfg_mode = _mode_cfg(cfg, mode_key)
        steerer = get_steerer(mode_key, dynamics)
        planner = SingleShotPlanner(dynamics, steerer, env, cfg_mode)

        label = candidate_name
        callback = iter_callback if not multi_candidate else None
        if verbose and multi_candidate:
            print(f"  candidate: {label}")

        result = planner.solve(
            mu0, Sigma0, T=T, init_V=init_v,
            verbose=verbose and not multi_candidate,
            iter_callback=callback,
        )
        result.route_label = label

        if verbose and multi_candidate:
            print(f"    P(phi)={result.best_p:.4f}")

        if best_result is None or result.best_p > best_result.best_p:
            best_result = result
            best_name = label

    if best_result is not None:
        best_result.route_label = best_name
        if verbose and multi_candidate:
            print(f"  selected route: {best_name}  P(phi)={best_result.best_p:.4f}")

    return best_result


def _solve_named_route_candidate(
    cfg, mode_key, route_name, dynamics, env, mu0, Sigma0, T, verbose=True, iter_callback=None
):
    """Solve one named warm-start route, used for the live closed-loop display."""
    options = dict(_warm_start_options(cfg, dynamics, mu0, T))
    init_v = options.get(route_name)
    if init_v is None:
        raise ValueError(f"Unknown route candidate '{route_name}'")

    _set_seed(cfg)
    cfg_mode = _mode_cfg(cfg, mode_key)
    steerer = get_steerer(mode_key, dynamics)
    planner = SingleShotPlanner(dynamics, steerer, env, cfg_mode)
    result = planner.solve(
        mu0, Sigma0, T=T, init_V=init_v,
        verbose=verbose, iter_callback=iter_callback,
    )
    result.route_label = route_name
    return result


def _run_mc_and_plot(result, dynamics, env, cfg, mu0, Sigma0, T,
                     mc_samples, save_dir, label, device, suffix=""):
    """Run Monte Carlo verification and save the plot."""
    from monte_carlo import mc_verify
    from visualization.monte_carlo import plot_mc_verification

    spec = env.get_specification(T)
    print(f"\n-- Monte Carlo Verification (N={mc_samples}) --")
    mc_result = mc_verify(result, dynamics, spec, mu0, Sigma0,
                          n_samples=mc_samples, device=str(device))
    n_ok = int(mc_result["successes"].sum())
    print(f"  Analytic   P(phi) = {mc_result['p_analytic']:.4f}")
    print(f"  Empirical  P_hat  = {mc_result['p_empirical']:.4f}"
          f"  ({n_ok}/{mc_samples} satisfied)")

    fig = plot_mc_verification(
        mc_result, env, cfg, result,
        save_path=Path(save_dir) / f"{label}{suffix}_mc_verification.png",
    )
    plt.show()
    return mc_result


def run_comparison(scenario_path, mc_samples=0):
    """Open-loop vs closed-loop single-shot on the same scenario.

    Per-mode weight/optimizer overrides come from 'open_loop:' and
    'closed_loop:' sections in the scenario YAML.

    Returns:
        (result_ol, result_cl)
    """
    cfg, dyn_cfg = load_scenario(scenario_path)
    _set_seed(cfg)
    device = torch.device(cfg["device"])
    save_dir = cfg.get("save_dir", "data/results")
    do_animate = cfg.get("animate", False)
    dt = dyn_cfg.get("dt", 0.2)

    dynamics = get_dynamics(dyn_cfg, device)
    T = cfg["horizon"]
    env = build_environment(cfg, device, T=T, dt=dt)

    init = cfg["initial_state"]
    mu0 = torch.tensor(init["mean"], dtype=torch.float32, device=device)
    Sigma0 = torch.diag(torch.tensor(init["cov_diag"], dtype=torch.float32, device=device))

    # ── Open-loop baseline (K ≡ 0) ──────────────────────────────────
    print("\n-- Open-Loop (K == 0) --")
    result_ol = _solve_best_route_candidate(
        cfg, "open_loop", dynamics, env, mu0, Sigma0, T, verbose=True)

    # ── Closed-loop covariance steering ─────────────────────────────
    print("\n-- Closed-Loop (K optimised) --")
    result_cl = _solve_best_route_candidate(
        cfg, "closed_loop", dynamics, env, mu0, Sigma0, T, verbose=True)

    # ── Summary ──────────────────────────────────────────────────────
    _print_result_summary(result_ol, result_cl)

    # ── Plots ─────────────────────────────────────────────────────────
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    label = cfg.get("label", "comparison").lower().replace(" ", "_")
    params = _build_params_dict(cfg, dyn_cfg, T)

    fig = plot_comparison(
        result_ol, result_cl, env, T,
        save_path=save_dir / f"{label}_trajectories.png",
        params=params,
    )
    plt.show()

    fig = plot_covariance_trace(
        result_ol, result_cl, T, dt=dyn_cfg.get("dt", 0.2),
        save_path=save_dir / f"{label}_covariance_trace.png",
    )
    plt.show()

    fig = plot_convergence(
        [
            {"p_sat": result_ol.p_history, "loss": result_ol.history},
            {"p_sat": result_cl.p_history, "loss": result_cl.history},
        ],
        labels=["Open-Loop", "Cov Steering"],
        save_path=save_dir / f"{label}_convergence.png",
    )
    plt.show()

    fig = plot_control_sequence(
        result_cl, dt=dt,
        save_path=save_dir / f"{label}_controls.png",
    )
    plt.show()

    if do_animate:
        gif_path = save_dir / f"{label}.gif"
        print(f"  Saving animation -> {gif_path}")
        animate_trajectory(result_cl, env, filename=str(gif_path), dt=dt)

    if mc_samples > 0:
        print("\n-- Open-Loop MC --")
        _run_mc_and_plot(result_ol, dynamics, env, cfg, mu0, Sigma0, T,
                         mc_samples, save_dir, label, device, suffix="_open_loop")
        print("\n-- Closed-Loop MC --")
        _run_mc_and_plot(result_cl, dynamics, env, cfg, mu0, Sigma0, T,
                         mc_samples, save_dir, label, device, suffix="_closed_loop")

    return result_ol, result_cl


def _setup_ss_live_plot(env, cfg, mode_label=""):
    """Live figure for single-shot: map, satisfaction probability, and loss."""
    import matplotlib.patches as patches

    plt.ion()
    fig = plt.figure(figsize=(16, 5.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1, 1])
    ax_map = fig.add_subplot(gs[0])
    ax_p   = fig.add_subplot(gs[1])
    ax_loss = fig.add_subplot(gs[2])

    if env.bounds:
        ax_map.set_xlim(env.bounds["x"][0] - 0.5, env.bounds["x"][1] + 0.5)
        ax_map.set_ylim(env.bounds["y"][0] - 0.5, env.bounds["y"][1] + 0.5)
    ax_map.set_aspect("equal")
    ax_map.grid(True, alpha=0.3)
    ax_map.set_title(mode_label, fontsize=12)

    from visualization import draw_env
    draw_env(ax_map, env)

    line_traj, = ax_map.plot([], [], "b-", lw=2, alpha=0.9, label="Current plan")
    ax_map.legend(loc="upper left", fontsize=9)

    max_iters = cfg.get("optimizer", {}).get("max_iters", 1000)
    ax_p.set_xlim(0, max_iters)
    ax_p.set_ylim(0, 1.1)
    ax_p.set_title("P(φ) vs Optimizer Iteration", fontsize=12)
    ax_p.set_xlabel("Iteration")
    ax_p.set_ylabel("P(φ)")
    ax_p.axhline(0.95, color="gray", ls="--", lw=1, alpha=0.5)
    ax_p.grid(True, alpha=0.3)
    line_p, = ax_p.plot([], [], color="#2ca02c", lw=1.5)

    ax_loss.set_xlim(0, max_iters)
    ax_loss.set_title("Gradient Descent Progress", fontsize=12)
    ax_loss.set_xlabel("Iteration")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, alpha=0.3)
    line_loss, = ax_loss.plot([], [], color="#d62728", lw=1.5)

    return fig, ax_map, ax_p, ax_loss, line_traj, line_p, line_loss


def _save_double_slit_slide_outputs(result_ol, result_cl, env, cfg, dyn_cfg, T, save_dir, label):
    """Save a compact set of presentation-ready double-slit figures."""
    trace_ol = float(torch.trace(result_ol.Sigma_trace[0, -1, :2, :2]).item())
    trace_cl = float(torch.trace(result_cl.Sigma_trace[0, -1, :2, :2]).item())
    params = _build_params_dict(cfg, dyn_cfg, T)

    fig = plot_comparison(
        result_ol, result_cl, env, T,
        save_path=save_dir / f"{label}_slide_trajectories.png",
        params=params,
    )
    plt.close(fig)

    fig = plot_covariance_trace(
        result_ol, result_cl, T, dt=dyn_cfg.get("dt", 0.2),
        save_path=save_dir / f"{label}_slide_covariance.png",
    )
    plt.close(fig)

    fig = plot_convergence(
        [
            {"p_sat": result_ol.p_history, "loss": result_ol.history},
            {"p_sat": result_cl.p_history, "loss": result_cl.history},
        ],
        labels=["Open-Loop", "Cov Steering"],
        save_path=save_dir / f"{label}_slide_convergence.png",
    )
    plt.close(fig)

    summary_path = save_dir / f"{label}_slide_summary.txt"
    summary_path.write_text(
        "\n".join([
            "Double Slit summary",
            f"Open-loop P(phi): {result_ol.best_p:.4f}",
            f"Open-loop route: {getattr(result_ol, 'route_label', 'n/a')}",
            f"Cov-steering P(phi): {result_cl.best_p:.4f}",
            f"Cov-steering route: {getattr(result_cl, 'route_label', 'n/a')}",
            f"Improvement: {result_cl.best_p - result_ol.best_p:+.4f}",
            f"Open-loop final tr(Sigma_pos): {trace_ol:.4f}",
            f"Cov-steering final tr(Sigma_pos): {trace_cl:.4f}",
            f"Final covariance reduction: {trace_ol / trace_cl:.2f}x" if trace_cl > 1e-12 else "Final covariance reduction: n/a",
            f"Closed-loop ||K||_F: {result_cl.K.norm().item():.4f}",
            f"Seed: {cfg.get('seed', 'none')}",
            "",
        ]),
        encoding="utf-8",
    )

    print("\n  Presentation outputs:")
    print(f"    {save_dir / f'{label}_slide_trajectories.png'}")
    print(f"    {save_dir / f'{label}_slide_covariance.png'}")
    print(f"    {save_dir / f'{label}_slide_convergence.png'}")
    print(f"    {summary_path}")


def run_double_slit_live(scenario_path, verbose=True):
    """Run double slit with one live closed-loop optimisation display.

    Open-loop and closed-loop route candidates are scored first for the
    summary/slide figures. The live figure then replays only the selected
    closed-loop route with map, P(phi), and loss panels.
    """
    cfg, dyn_cfg = load_scenario(scenario_path)
    _set_seed(cfg)
    device = torch.device(cfg["device"])
    T = cfg["horizon"]
    dt = dyn_cfg.get("dt", 0.2)
    save_dir = Path(cfg.get("save_dir", "data/results"))
    save_dir.mkdir(parents=True, exist_ok=True)
    label = cfg.get("label", "double_slit").lower().replace(" ", "_")

    dynamics = get_dynamics(dyn_cfg, device)
    env = build_environment(cfg, device, T=T, dt=dt)

    init = cfg["initial_state"]
    mu0 = torch.tensor(init["mean"], dtype=torch.float32, device=device)
    Sigma0 = torch.diag(torch.tensor(init["cov_diag"], dtype=torch.float32, device=device))

    print("\n-- Open-Loop route choice --")
    result_ol = _solve_best_route_candidate(
        cfg, "open_loop", dynamics, env, mu0, Sigma0, T, verbose=verbose)

    print("\n-- Closed-Loop route choice --")
    selected_cl = _solve_best_route_candidate(
        cfg, "closed_loop", dynamics, env, mu0, Sigma0, T, verbose=verbose)
    selected_route = getattr(selected_cl, "route_label", "n/a")

    print(f"\n-- Closed-Loop live optimisation ({selected_route}) --")
    cfg_cl = _mode_cfg(cfg, "closed_loop")
    max_iters = cfg_cl.get("optimizer", {}).get("max_iters", 1000)
    update_freq = max(1, max_iters // 50)
    capture_frames = cfg.get("animate", False)
    frames = []
    p_trace_k = []
    p_trace_v = []
    loss_trace_k = []
    loss_trace_v = []

    fig_live, ax_map, ax_p, ax_loss, line_traj, line_p, line_loss = _setup_ss_live_plot(
        env, cfg_cl,
        mode_label=f"Closed-Loop | Route: {selected_route} | Iter 0/{max_iters}")

    def _capture():
        frames.append(_canvas_frame(fig_live))

    def _on_iter(k, p_sat, mu_trace, loss=None):
        if k % update_freq != 0:
            return
        plan_np = mu_trace.cpu().squeeze(0).numpy()
        line_traj.set_data(plan_np[:, 0], plan_np[:, 1])

        p_trace_k.append(k)
        p_trace_v.append(p_sat)
        line_p.set_data(p_trace_k, p_trace_v)

        if loss is not None:
            loss_trace_k.append(k)
            loss_trace_v.append(loss)
            line_loss.set_data(loss_trace_k, loss_trace_v)
            ymin = min(loss_trace_v)
            ymax = max(loss_trace_v)
            pad = max(1e-3, 0.08 * (ymax - ymin))
            ax_loss.set_ylim(ymin - pad, ymax + pad)

        ax_map.set_title(
            f"Closed-Loop | Route: {selected_route} | Iter {k}/{max_iters} | P={p_sat:.3f}",
            fontsize=11)
        if plt.get_backend().lower() != "agg":
            plt.pause(0.01)
        if capture_frames:
            _capture()

    result_cl = _solve_named_route_candidate(
        cfg, "closed_loop", selected_route, dynamics, env, mu0, Sigma0, T,
        verbose=verbose, iter_callback=_on_iter)

    if p_trace_k:
        final_iter = max(50, p_trace_k[-1] + update_freq)
        ax_p.set_xlim(0, final_iter)
        ax_loss.set_xlim(0, final_iter)

    live_panel_path = save_dir / f"{label}_closed_loop_live_panel.png"
    fig_live.savefig(live_panel_path, dpi=200, bbox_inches="tight")
    plt.ioff()
    plt.close(fig_live)

    if cfg.get("animate", False) and frames:
        from PIL import Image
        live_path = save_dir / f"{label}_closed_loop_live.gif"
        print(f"  Saving live animation -> {live_path}")
        pil_frames = [Image.fromarray(f) for f in frames]
        pil_frames[0].save(
            str(live_path), save_all=True, append_images=pil_frames[1:],
            loop=0, duration=100, optimize=False)  # 10 fps
        print(f"  {len(frames)} frames saved")
    print(f"  Saved live panel -> {live_panel_path}")

    _print_result_summary(result_ol, result_cl)
    print(f"    Improvement        = {result_cl.best_p - result_ol.best_p:+.4f}")

    if cfg.get("save_slide_outputs", True):
        _save_double_slit_slide_outputs(
            result_ol, result_cl, env, cfg, dyn_cfg, T, save_dir, label)

    return result_ol, result_cl


def _setup_mpc_live_plot(env, cfg):
    """Two-panel live figure: map with executed path + planned window, and P(φ) per step."""
    import matplotlib.patches as patches

    plt.ion()
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1])
    ax_map = fig.add_subplot(gs[0])
    ax_p = fig.add_subplot(gs[1])

    if env.bounds:
        ax_map.set_xlim(env.bounds["x"][0] - 0.5, env.bounds["x"][1] + 0.5)
        ax_map.set_ylim(env.bounds["y"][0] - 0.5, env.bounds["y"][1] + 0.5)
    ax_map.set_aspect("equal")
    ax_map.grid(True, alpha=0.3)
    ax_map.set_title(f"MPC — {cfg.get('label', '')}", fontsize=12)

    from visualization import draw_env
    draw_env(ax_map, env)

    line_exec, = ax_map.plot([], [], "b-o", ms=4, lw=1.5, label="Executed path")
    line_plan, = ax_map.plot([], [], "--", color="#ff7f0e", lw=2, alpha=0.8, label="Planned window")
    ax_map.legend(loc="upper left", fontsize=9)

    ax_p.set_xlim(0, max(10, cfg.get("horizon", 30)))
    ax_p.set_ylim(0, 1.1)
    ax_p.set_title("Window P(φ) per step", fontsize=12)
    ax_p.set_xlabel("MPC step")
    ax_p.set_ylabel("P(φ)")
    ax_p.axhline(0.95, color="gray", ls="--", lw=1, alpha=0.5)
    ax_p.grid(True, alpha=0.3)
    line_p, = ax_p.plot([], [], color="#2ca02c", marker="o", ms=3)

    return fig, ax_map, ax_p, line_exec, line_plan, line_p


def run_mpc_scenario(scenario_path, verbose=True, mc_samples=0):
    """Run the MPC planner with a live two-panel visualisation.

    Left panel: map with executed path and current planned window.
    Right panel: per-step P(φ). Saves a static PNG and optionally a GIF.
    """
    cfg, dyn_cfg, dynamics, steerer, env, mu0, Sigma0 = setup_scenario(scenario_path)
    device = torch.device(cfg["device"])
    T = cfg["horizon"]
    dt = dyn_cfg.get("dt", 0.2)

    planner = get_planner(cfg, dynamics, steerer, env)

    save_dir = Path(cfg.get("save_dir", "data/results"))
    save_dir.mkdir(parents=True, exist_ok=True)
    label = cfg.get("label", "mpc").lower().replace(" ", "_")

    fig_live, ax_map, ax_p, line_exec, line_plan, line_p = _setup_mpc_live_plot(env, cfg)

    mpc_iters = cfg.get("mpc", {}).get("iters", 100)
    iter_update_freq = max(1, mpc_iters // 15)  # ~15 live updates per MPC step
    frames = []

    def _capture():
        frames.append(_canvas_frame(fig_live))

    def _on_iter(step_t, i, p_sat, mu_trace):
        if i % iter_update_freq != 0:
            return
        plan_np = mu_trace.cpu().squeeze(0).numpy()
        line_plan.set_data(plan_np[:, 0], plan_np[:, 1])
        ax_map.set_title(
            f"MPC Step {step_t+1}/{T} | Iter {i}/{mpc_iters} | P={p_sat:.3f}",
            fontsize=11)
        plt.pause(0.01)
        _capture()

    def _on_step(t, mu_list, plan_traces, p_history):
        path = np.array([[m[0].item(), m[1].item()] for m in mu_list])
        line_exec.set_data(path[:, 0], path[:, 1])
        if plan_traces:
            plan_np = plan_traces[-1].cpu().squeeze(0).numpy()
            line_plan.set_data(plan_np[:, 0], plan_np[:, 1])
        line_p.set_data(range(len(p_history)), p_history)
        ax_p.set_xlim(0, max(len(p_history) + 5, 10))
        ax_map.set_title(
            f"MPC Step {t+1}/{T} | Done | P={p_history[-1]:.3f}", fontsize=11)
        plt.pause(0.05)
        for _ in range(3):  # hold 3 frames so the step transition is visible in the GIF
            _capture()

    result = planner.solve(mu0, Sigma0, verbose=verbose,
                           step_callback=_on_step, iter_callback=_on_iter)

    plt.ioff()
    plt.close(fig_live)

    mu_np = result.mu_trace.detach().cpu().squeeze().numpy()
    S_np = result.Sigma_trace.detach().cpu().squeeze().numpy()

    fig, ax = plt.subplots(figsize=(10, 8))
    plot_trajectory(ax, mu_np, S_np, env, T,
                    title=f"{cfg.get('label', 'MPC')}  |  P(φ)={result.best_p:.3f}")
    plt.tight_layout()
    fig.savefig(save_dir / f"{label}_trajectory.png", dpi=150)
    plt.show()

    if cfg.get("animate", False):
        gif_path = save_dir / f"{label}.gif"
        print(f"  Saving animation -> {gif_path}")
        animate_trajectory(result, env, filename=str(gif_path), dt=dt,
                           plan_traces=result.plan_traces)

        if frames:
            from PIL import Image
            live_path = save_dir / f"{label}_live.gif"
            print(f"  Saving live animation -> {live_path}")
            pil_frames = [Image.fromarray(f) for f in frames]
            pil_frames[0].save(
                str(live_path), save_all=True, append_images=pil_frames[1:],
                loop=0, duration=67, optimize=False)  # ~15 fps
            print(f"  {len(frames)} frames saved")

    if mc_samples > 0:
        _run_mc_and_plot(result, dynamics, env, cfg, mu0, Sigma0, T,
                         mc_samples, save_dir, label, device)

    return result, env, cfg


def run_joint_noise_sweep(
    scenario_path,
    noise_levels=None,
    mc_samples=1000,
    max_iters_sweep=800,
):
    """Sweep a joint noise level v (sets both σ₀² and D = v simultaneously).

    For each v: initial position variance = v, process noise D_diag = v.
    Runs OL vs CL at each level and plots P(φ) vs v.
    """
    if noise_levels is None:
        noise_levels = [0.0001, 0.001, 0.01, 0.1, 0.5]

    base_cfg, base_dyn_cfg = load_scenario(scenario_path)
    device = torch.device(base_cfg["device"])
    T = base_cfg["horizon"]
    save_dir = Path(base_cfg.get("save_dir", "data/results"))
    label = base_cfg.get("label", "sweep")
    vel_cov = list(base_cfg["initial_state"]["cov_diag"][2:])

    print(f"\n-- Joint noise sweep ({len(noise_levels)} points) --")
    rows = []
    for v in noise_levels:
        print(f"  v={v:.4f} ...", end=" ", flush=True)

        cfg = copy.deepcopy(base_cfg)
        dyn_cfg = copy.deepcopy(base_dyn_cfg)
        cfg["initial_state"]["cov_diag"] = [v, v] + vel_cov
        dyn_cfg["D_diag"] = v

        mu0 = torch.tensor(cfg["initial_state"]["mean"], dtype=torch.float32, device=device)
        Sigma0 = torch.diag(torch.tensor(cfg["initial_state"]["cov_diag"], dtype=torch.float32, device=device))

        row = _sweep_one_point(dyn_cfg, cfg, mu0, Sigma0, T, device, max_iters_sweep, mc_samples)
        row["noise_level"] = v
        rows.append(row)
        print(f"OL={row['p_ol_analytic']:.3f}  CL={row['p_cl_analytic']:.3f}"
              + (f"  MC-OL={row['p_ol_mc']:.3f}  MC-CL={row['p_cl_mc']:.3f}"
                 if mc_samples > 0 else ""))

    save_dir.mkdir(parents=True, exist_ok=True)
    fig = plot_joint_noise_sweep(rows, label, save_dir)
    plt.show()
    stem = label.lower().replace(" ", "_")
    print(f"\n  Saved -> {save_dir / f'{stem}_joint_noise_sweep.png'}")

    return rows


def _sweep_one_point(dyn_cfg, cfg, mu0, Sigma0, T, device, max_iters, mc_samples):
    """Run OL + CL for a single sweep configuration. Returns a metrics dict."""
    from monte_carlo import mc_verify

    dynamics = get_dynamics(dyn_cfg, device)
    env = build_environment(cfg, device, T=T, dt=dyn_cfg.get("dt", 0.2))

    cfg_ol = _mode_cfg(cfg, "open_loop")
    cfg_ol["optimizer"] = {**cfg_ol["optimizer"], "max_iters": max_iters}
    cfg_cl = _mode_cfg(cfg, "closed_loop")
    cfg_cl["optimizer"] = {**cfg_cl["optimizer"], "max_iters": max_iters}

    planner_ol = SingleShotPlanner(dynamics, get_steerer("open_loop", dynamics), env, cfg_ol)
    result_ol = planner_ol.solve(mu0, Sigma0, T=T, verbose=False)

    planner_cl = SingleShotPlanner(dynamics, get_steerer("closed_loop", dynamics), env, cfg_cl)
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


# ═══════════════════════════════════════════════════════════════════════
# Lane Change (pdstl-style MPC with covariance steering)
# ═══════════════════════════════════════════════════════════════════════

def _stack_traces(mean_list, cov_list, u_list):
    """Stack trajectory lists into [1, T, D] / [1, T, D, D] tensors."""
    return (
        torch.stack(mean_list).unsqueeze(0),
        torch.stack(cov_list).unsqueeze(0),
        torch.stack(u_list).unsqueeze(0),
    )


def _step_with_noise(dynamics, mean, cov, u):
    """Step belief forward one timestep and inject process noise sample."""
    next_mean, next_cov = dynamics.step(mean, cov, u)
    noise = torch.distributions.MultivariateNormal(
        torch.zeros(dynamics.nx, device=dynamics.device), dynamics.DDT
    ).sample()
    return next_mean + noise, next_cov


def _step_with_noise_cl(dynamics, mean, cov, V0, K0):
    """Closed-loop belief step: applies K to covariance (same as ClosedLoopSteerer).

    Control law:  u = sat(V0)
    Cov update:   A_cl = A + B @ K_eff,  K_eff = K0 * sech²(V0) * u_max
                  Σ_next = A_cl Σ A_cl^T + DDT
    """
    u_ff = dynamics.bound_control(V0)
    next_mean = dynamics.A @ mean + dynamics.B @ u_ff
    noise = torch.distributions.MultivariateNormal(
        torch.zeros(dynamics.nx, device=dynamics.device), dynamics.DDT
    ).sample()
    next_mean = next_mean + noise

    gain_scale = dynamics.u_max * (1.0 - torch.tanh(V0) ** 2)  # sech²(V0) * u_max
    K_eff = K0 * gain_scale.unsqueeze(-1)                        # [nu, nx]
    A_cl  = dynamics.A + dynamics.B @ K_eff                      # [nx, nx]
    next_cov = A_cl @ cov @ A_cl.T + dynamics.DDT
    return next_mean, next_cov


def _plot_lc_final(mu_np, S_np, env_global, road, xlim, dt, label,
                   K_trace=None, p_sat_trace=None, save_path=None):
    """Static final plot in animation style: wide road view, no aspect constraint."""
    import matplotlib.patches as mp
    from visualization.trajectory import cov_ellipse_params

    has_K = K_trace is not None and len(K_trace) > 0
    fig, axes = plt.subplots(1 + int(has_K), 1,
                             figsize=(13, 7 if has_K else 5),
                             gridspec_kw={"height_ratios": [3, 1]} if has_K else None)
    ax = axes[0] if has_K else axes

    # Road backdrop
    ax.axhspan(road["y_min"], road["y_max"], color="#F2F2F7", zorder=0)
    ax.axhspan(road["lane_divider"], road["y_max"], color="#98df8a", alpha=0.18,
               zorder=1, label="Goal Lane")
    ax.axhline(road["lane_divider"], color="#7f7f7f", ls="--", lw=1.5, alpha=0.9, zorder=2)
    ax.axhline(road["y_min"],        color="#7f7f7f", ls="-",  lw=2.0, alpha=0.9, zorder=2)
    ax.axhline(road["y_max"],        color="#7f7f7f", ls="-",  lw=2.0, alpha=0.9, zorder=2)

    # Moving obstacle: dashed path + 5 snapshot rectangles
    obs = env_global.moving_obstacles[0]
    xt = obs["x_traj"].cpu().numpy() if isinstance(obs["x_traj"], torch.Tensor) else np.asarray(obs["x_traj"])
    yt = obs["y_traj"].cpu().numpy() if isinstance(obs["y_traj"], torch.Tensor) else np.asarray(obs["y_traj"])
    ax.plot(xt, yt, "--", color="#d62728", alpha=0.35, lw=1.0, zorder=3, label="Obstacle path")
    snap = max(1, len(xt) // 5)
    w_o, h_o = obs["width"], obs["height"]
    for k in range(0, len(xt), snap):
        ax.add_patch(mp.Rectangle(
            (xt[k] - w_o / 2, yt[k] - h_o / 2), w_o, h_o,
            fc="#ff9896", ec="#d62728", alpha=0.30, lw=1.0, zorder=4,
        ))

    # Ghost full-path line
    ax.plot(mu_np[:, 0], mu_np[:, 1], "b-", lw=2.0, alpha=0.9, zorder=5, label="Ego trajectory")

    # Covariance ellipses every ~5% of steps
    T = mu_np.shape[0] - 1
    step = max(1, T // 20)
    for t in range(0, T + 1, step):
        theta, w_e, h_e = cov_ellipse_params(S_np[t, :2, :2], k=2.0)
        ax.add_patch(mp.Ellipse(
            (mu_np[t, 0], mu_np[t, 1]), w_e, h_e, angle=theta,
            fc="#1f77b4", ec="#1f77b4", alpha=0.22, zorder=6,
        ))

    # Start marker
    ax.plot(mu_np[0, 0], mu_np[0, 1], "ko", ms=8, zorder=8, label="Start")

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(road["y_min"] - 0.3, road["y_max"] + 0.3)
    ax.set_xlabel("x [m]", fontsize=12, fontweight="bold")
    ax.set_ylabel("y [m]", fontsize=12, fontweight="bold")
    ax.set_title(f"Lane Change: {label}  |  T={T} steps  |  dt={dt}s", fontsize=13)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    if has_K:
        import numpy as np_inner
        K_arr = np_inner.stack(K_trace)           # [T_mpc, nu, nx]
        K_frob = np_inner.linalg.norm(K_arr, axis=(1, 2))   # Frobenius norm per step
        # lateral gains: K[u_y, p_y] = K[:, 1, 1] and K[u_y, v_y] = K[:, 1, 3]
        K_py = K_arr[:, 1, 1]   # lateral control ← lateral position error
        K_vy = K_arr[:, 1, 3]   # lateral control ← lateral velocity error

        steps = np_inner.arange(len(K_trace))
        ax_k = axes[1]
        ax_k.plot(steps, K_frob, "k-",  lw=1.8, label=r"$\|K_t\|_F$")
        ax_k.plot(steps, K_py,   "--",  color="#1f77b4", lw=1.4, label=r"$K_{u_y, p_y}$")
        ax_k.plot(steps, K_vy,   ":",   color="#ff7f0e", lw=1.4, label=r"$K_{u_y, v_y}$")
        ax_k.axhline(0, color="#aaaaaa", lw=0.8)
        ax_k.set_xlim(0, len(K_trace))
        ax_k.set_xlabel("MPC step", fontsize=11)
        ax_k.set_ylabel("Gain", fontsize=11)
        ax_k.set_title("Feedback gains  $K_t[0]$", fontsize=11)
        ax_k.legend(loc="upper right", fontsize=9, framealpha=0.85)
        ax_k.grid(True, alpha=0.25)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.show()
    plt.close(fig)


def _setup_lc_live_plot(road, obs_cfg, obs_x0, obs_y0, success_cfg, label, xlim):
    """Build the lane-change live figure (mirrors pdstl setup_lane_change_live_plot)."""
    import matplotlib.patches as mp

    plt.ion()
    fig, ax = plt.subplots(figsize=(13, 5))

    # road backdrop
    ax.axhspan(road["y_min"], road["y_max"], color="#F2F2F7", zorder=0)
    # goal lane shade
    ax.axhspan(2.0, road["y_max"], color="#98df8a", alpha=0.18, zorder=1, label="Goal Lane")
    # success region shade
    ax.axhspan(success_cfg["y_min"], success_cfg["y_max"],
               color="#2ca02c", alpha=0.10, zorder=1, label="Success Zone")
    # lane markings
    ax.axhline(road["lane_divider"], color="#7f7f7f", ls="--", lw=1.5, alpha=0.9, zorder=2)
    ax.axhline(road["y_min"], color="#7f7f7f", ls="-",  lw=2.0, alpha=0.9, zorder=2)
    ax.axhline(road["y_max"], color="#7f7f7f", ls="-",  lw=2.0, alpha=0.9, zorder=2)

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(road["y_min"] - 0.5, road["y_max"] + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]", fontsize=12)
    ax.set_ylabel("y [m]", fontsize=12)
    ax.set_title(f"Lane Change: {label}", fontsize=13)
    ax.grid(True, alpha=0.25)

    w, h = obs_cfg["width"], obs_cfg["height"]
    obs_rect = mp.Rectangle(
        (obs_x0 - w / 2, obs_y0 - h / 2), w, h,
        fc="#ff9896", ec="#d62728", lw=1.5, alpha=0.8, zorder=5,
    )
    ax.add_patch(obs_rect)

    ego_dot,  = ax.plot([], [], "o", color="#1f77b4", ms=8, zorder=10)
    ego_trail, = ax.plot([], [], "-", color="#1f77b4", lw=1.5, alpha=0.7, zorder=9)
    plan_line, = ax.plot([], [], "--", color="#ff7f0e", lw=1.8, alpha=0.85, zorder=8)

    ego_cov_patch = mp.Ellipse(
        (0, 0), 0.1, 0.1, angle=0,
        fc="#1f77b4", ec="#1f77b4", alpha=0.30, zorder=7,
    )
    ax.add_patch(ego_cov_patch)

    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    return fig, ax, ego_dot, ego_trail, plan_line, ego_cov_patch, obs_rect


def run_lane_change(scenario_path):
    """Lane change MPC with covariance steering — exact pdstl scenario structure."""
    from visualization.trajectory import cov_ellipse_params, plot_trajectory

    cfg, dyn_cfg = load_scenario(scenario_path)
    device = torch.device(cfg["device"])
    dt = dyn_cfg.get("dt", 0.2)
    dynamics = get_dynamics(dyn_cfg, device)

    H = cfg["H"]
    T_SIM = cfg["T_SIM"]
    road = cfg["road"]
    obs_cfg = cfg["obstacle"]
    success_cfg = cfg["success"]
    label = cfg.get("label", "lane_change")

    save_dir = Path(cfg.get("save_dir", "data/results"))
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Global obstacle trajectory ──────────────────────────────────────
    total_pts = T_SIM + H + 10
    times_np = np.arange(total_pts) * dt
    obs_x_global = obs_cfg["x0"] + obs_cfg["speed"] * times_np
    obs_y_global = np.ones_like(times_np) * obs_cfg["y"]

    # ── Global env (road markings + full obs traj — for final viz only) ─
    marking_x = road["marking_x_range"]
    env_global = Environment(device=device)
    env_global.add_lane_marking(marking_x, road["lane_divider"], style="dashed")
    env_global.add_lane_marking(marking_x, road["y_min"], style="solid")
    env_global.add_lane_marking(marking_x, road["y_max"], style="solid")
    env_global.set_goal(**cfg["goal"])
    env_global.set_bounds(
        x_range=[road["marking_x_range"][0], road["marking_x_range"][1]],
        y_range=[road["y_min"], road["y_max"]],
    )
    env_global.add_moving_obstacle(
        x_traj=obs_x_global[: T_SIM + 1],
        y_traj=obs_y_global[: T_SIM + 1],
        width=obs_cfg["width"],
        height=obs_cfg["height"],
    )

    # ── Initial belief ──────────────────────────────────────────────────
    mu0 = torch.tensor(cfg["x0_mean"], dtype=torch.float32, device=device)
    Sigma0 = torch.eye(len(cfg["x0_mean"]), device=device) * cfg["x0_cov_scale"]
    curr_mean, curr_cov = mu0.clone(), Sigma0.clone()

    real_mean_trace = [curr_mean]
    real_cov_trace = [curr_cov]
    real_u_trace = []
    p_sat_trace = []
    K_trace = []       # K[0] at each MPC step  [nu, nx]
    all_plans = []
    prev_V = None
    success_counter = 0

    # ── MPC local-window params ─────────────────────────────────────────
    goal_lookahead   = cfg.get("mpc_goal_lookahead", 4.0)
    goal_window_width = cfg.get("mpc_goal_window_width", 60.0)
    lane_margin      = cfg.get("lane_boundary_margin", 0.5)
    goal_y_inset     = cfg.get("goal_y_inset", 0.1)

    # ── Live plot ───────────────────────────────────────────────────────
    xlim = cfg.get("plot_xlim", [-3, 35])
    fig, ax, ego_dot, ego_trail, plan_line, ego_cov_patch, obs_rect = \
        _setup_lc_live_plot(road, obs_cfg, obs_x_global[0], obs_y_global[0],
                            success_cfg, label, xlim)

    # ── MPC loop ────────────────────────────────────────────────────────
    for t in range(T_SIM):
        ego_pos = curr_mean.cpu().numpy()
        curr_x = float(ego_pos[0])

        # local environment
        env_local = Environment(device=device)
        goal_x_lo = curr_x + goal_lookahead
        goal_x_hi = curr_x + goal_lookahead + goal_window_width
        env_local.set_goal(
            x_range=[goal_x_lo, goal_x_hi],
            y_range=[cfg["goal"]["y_range"][0] + goal_y_inset,
                     cfg["goal"]["y_range"][1] - goal_y_inset],
        )

        y_min_bound = road["y_min"] + lane_margin
        if curr_mean[1].item() > road["lane_divider"] - lane_margin:
            y_min_bound = road["lane_divider"]
        env_local.set_bounds(
            x_range=[-100.0, 200.0],
            y_range=[y_min_bound, road["y_max"]],
        )

        idx_end = t + H + 1
        if idx_end <= len(obs_x_global):
            sl_x = obs_x_global[t:idx_end]
            sl_y = obs_y_global[t:idx_end]
        else:
            pad = idx_end - len(obs_x_global)
            sl_x = np.concatenate([obs_x_global[t:], np.full(pad, obs_x_global[-1])])
            sl_y = np.concatenate([obs_y_global[t:], np.full(pad, obs_y_global[-1])])
        env_local.add_moving_obstacle(
            x_traj=sl_x, y_traj=sl_y,
            width=obs_cfg["width"], height=obs_cfg["height"],
        )

        # warm start
        init_V = None
        if prev_V is not None:
            init_V = torch.cat([prev_V[1:], prev_V[-1:]], dim=0)

        # covariance steering solve
        steerer = get_steerer("closed_loop", dynamics)
        planner = SingleShotPlanner(dynamics, steerer, env_local, cfg)
        result = planner.solve(curr_mean, curr_cov, T=H, verbose=False, init_V=init_V)

        prev_V = result.V.detach()
        all_plans.append(result.mu_trace)
        p_sat_trace.append(result.best_p)
        K_trace.append(result.K[0].detach().cpu().numpy())   # [nu, nx]

        # apply first control and step (closed-loop covariance steering)
        curr_mean, curr_cov = _step_with_noise_cl(
            dynamics, curr_mean, curr_cov,
            result.V[0].detach(), result.K[0].detach()
        )
        u_curr = dynamics.bound_control(result.V[0].detach())

        real_mean_trace.append(curr_mean)
        real_cov_trace.append(curr_cov)
        real_u_trace.append(u_curr)

        # live plot update (use post-step position for dot + ellipse)
        ex, ey = curr_mean[0].item(), curr_mean[1].item()
        ego_dot.set_data([ex], [ey])
        ego_trail.set_data(
            [m[0].item() for m in real_mean_trace],
            [m[1].item() for m in real_mean_trace],
        )
        plan_np = result.mu_trace.detach().cpu().squeeze(0).numpy()
        plan_line.set_data(plan_np[:, 0], plan_np[:, 1])

        theta, w_e, h_e = cov_ellipse_params(curr_cov[:2, :2].cpu().numpy())
        ego_cov_patch.set_center((ex, ey))
        ego_cov_patch.set_width(w_e)
        ego_cov_patch.set_height(h_e)
        ego_cov_patch.set_angle(theta)

        obs_rect.set_xy((obs_x_global[t] - obs_cfg["width"] / 2,
                         obs_y_global[t] - obs_cfg["height"] / 2))

        if plt.get_backend().lower() != "agg":
            plt.draw()
            plt.pause(cfg.get("live_plot_pause", 0.001))

        if t % 5 == 0:
            print(f"  step {t:3d} | ego=({ex:.2f}, {ey:.2f}) | P={result.best_p:.3f} | "
                  f"tr(Σ)={float(torch.trace(curr_cov[:2,:2]).item()):.4f}")

        # success check
        if success_cfg["y_min"] <= ey <= success_cfg["y_max"]:
            success_counter += 1
        else:
            success_counter = 0
        if success_counter >= success_cfg["consecutive_steps"]:
            print(f"  Lane change complete at step {t} ({label})")
            break

    plt.ioff()
    plt.close(fig)

    # ── Stack traces ────────────────────────────────────────────────────
    full_mean, full_cov, full_u = _stack_traces(
        real_mean_trace, real_cov_trace, real_u_trace
    )
    actual_steps = len(real_mean_trace)
    env_global.moving_obstacles[0]["x_traj"] = \
        torch.as_tensor(obs_x_global[:actual_steps], dtype=torch.float32)
    env_global.moving_obstacles[0]["y_traj"] = \
        torch.as_tensor(obs_y_global[:actual_steps], dtype=torch.float32)

    # ── Final static plot (animation style) ────────────────────────────
    mu_np = full_mean.detach().cpu().squeeze().numpy()
    S_np  = full_cov.detach().cpu().squeeze().numpy()
    T_actual = mu_np.shape[0] - 1
    stem = label.lower().replace(" ", "_")
    _plot_lc_final(
        mu_np, S_np, env_global, road, xlim, dt, label,
        K_trace=K_trace, p_sat_trace=p_sat_trace,
        save_path=save_dir / f"lane_change_{stem}_trajectory.png",
    )

    if cfg.get("animate", False):
        anim_cfg = cfg.get("animation", {})
        gif_path = save_dir / anim_cfg.get("filename", f"lane_change_{stem}.gif")
        print(f"  Saving animation → {gif_path}")
        animate_trajectory(
            PlanResult(
                mu_trace=full_mean, Sigma_trace=full_cov,
                V=real_u_trace[0].unsqueeze(0) if real_u_trace else torch.zeros(1, dynamics.nu),
                K=torch.zeros(1, dynamics.nu, dynamics.nx),
                best_p=p_sat_trace[-1] if p_sat_trace else 0.0,
            ),
            env_global, filename=str(gif_path),
            dt=dt, plan_traces=all_plans,
        )

    return full_mean, full_cov, full_u, p_sat_trace


def run_lane_change_normal():
    run_lane_change("configs/scenarios/lane_change.yaml")


def run_lane_change_aggressive():
    run_lane_change("configs/scenarios/lane_change_aggressive.yaml")
