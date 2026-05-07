"""Shared loss computation for all planner types.

J = w_phi            * -log(p_sat)               (1) pdSTL satisfaction
  + w_trace_terminal * tr(Σ_T[:2,:2])             (2) terminal covariance
  + w_dist           * ||μ_T - goal||²             (3) goal distance
  + w_du             * ||Δu||²                     (4) control smoothness
  + w_K              * ||K||²_F                    (5) gain regularization
  + w_repulsion      * Σ ReLU(margin - dist)²      (6) obstacle repulsion
"""

import torch


def _rect_repulsion(pts, obs, margin, device):
    """SDF-based repulsion from an axis-aligned rectangle.

    Returns sum of ReLU(margin - sdf)^2 over all trajectory points,
    where sdf is negative inside the box and positive outside.
    """
    x_lo, x_hi = obs["x"]
    y_lo, y_hi = obs["y"]
    cx = (x_lo + x_hi) / 2.0
    cy = (y_lo + y_hi) / 2.0
    hx = (x_hi - x_lo) / 2.0
    hy = (y_hi - y_lo) / 2.0
    p = pts[..., :2] - torch.tensor([cx, cy], dtype=torch.float32, device=device)
    q = torch.abs(p) - torch.tensor([hx, hy], dtype=torch.float32, device=device)
    # SDF: positive outside, 0 on surface, negative inside
    dist = (torch.clamp(q, min=0.0).norm(dim=-1)
            + torch.clamp(torch.amax(q, dim=-1), max=0.0))
    return torch.sum(torch.relu(margin - dist) ** 2)


def _circle_repulsion(pts, obs, margin, device):
    """Repulsion from a circle: ReLU(margin - (||x - c|| - r))^2."""
    center = torch.tensor(obs["center"], dtype=torch.float32, device=device)
    dist = torch.norm(pts[..., :2] - center, dim=-1) - obs["radius"]
    return torch.sum(torch.relu(margin - dist) ** 2)


def compute_loss(p_sat, V, K, mu_trace, Sigma_trace, env, dyn, weights):
    """Return scalar loss J (differentiable).

    p_sat: [1] — pdSTL satisfaction probability at t=0
    V:     [T, nu],        K: [T, nu, nx]
    mu_trace: [1, T+1, nx], Sigma_trace: [1, T+1, nx, nx]
    """
    w = weights
    device = V.device

    loss_phi = -torch.log(p_sat + 1e-4)

    loss_trace_terminal = Sigma_trace[0, -1, 0, 0] + Sigma_trace[0, -1, 1, 1]

    loss_dist = torch.tensor(0.0, device=device)
    if env.goal is not None:
        gx = (env.goal["x"][0] + env.goal["x"][1]) / 2.0
        gy = (env.goal["y"][0] + env.goal["y"][1]) / 2.0
        goal_xy = torch.tensor([gx, gy], device=device)
        loss_dist = torch.sum((mu_trace[0, -1, :2] - goal_xy) ** 2)

    u_seq = dyn.bound_control(V)
    loss_du = torch.sum((u_seq[1:] - u_seq[:-1]) ** 2) + torch.sum(u_seq[0] ** 2)
    loss_K = torch.sum(K ** 2)

    loss_repulsion = torch.tensor(0.0, device=device)
    base_margin = float(w.get("obs_margin", 0.5))
    cov_scale   = float(w.get("cov_margin_scale", 0.0))

    if cov_scale > 0.0:
        # Per-timestep margin: base + cov_scale * max std-dev in x/y at that step.
        # Larger covariance -> larger clearance required, keeping the mean further
        # from obstacles when the state is most uncertain.
        varxy     = torch.diagonal(Sigma_trace[0, :, :2, :2], dim1=-2, dim2=-1)  # [T+1, 2]
        sigma_max = varxy.amax(dim=-1).clamp(min=0.0).sqrt()                      # [T+1]
        for t in range(mu_trace.shape[1]):
            margin_t = base_margin + cov_scale * sigma_max[t]
            pt = mu_trace[0, t:t+1]
            for obs in env.obstacles:
                loss_repulsion += _rect_repulsion(pt, obs, margin_t, device)
            for obs in env.circle_obstacles:
                loss_repulsion += _circle_repulsion(pt, obs, margin_t, device)
    else:
        for obs in env.obstacles:
            loss_repulsion += _rect_repulsion(mu_trace[0], obs, base_margin, device)
        for obs in env.circle_obstacles:
            loss_repulsion += _circle_repulsion(mu_trace[0], obs, base_margin, device)

    return (
        float(w.get("w_phi",             1.0)) * loss_phi
        + float(w.get("w_trace_terminal", 0.0)) * loss_trace_terminal
        + float(w.get("w_dist",           0.0)) * loss_dist
        + float(w.get("w_du",             0.0)) * loss_du
        + float(w.get("w_K",              0.0)) * loss_K
        + float(w.get("w_repulsion",      0.0)) * loss_repulsion
    )
