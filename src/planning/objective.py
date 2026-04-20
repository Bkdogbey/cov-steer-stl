"""Shared loss computation for all planner types.

J = w_phi           * L_phi                   (1) pdSTL: -log(p_sat)
  + w_trace         * sum_k tr(Σ_k[:2,:2])    (2) trajectory covariance trace
  + w_trace_terminal * tr(Σ_T[:2,:2])         (3) terminal covariance trace
  + w_dist          * ||μ_T - goal||²          (4) terminal goal distance
  + w_u             * ||u||²                   (5) control effort
  + w_du            * ||Δu||²                  (6) control smoothness
  + w_K             * ||K||²_F                 (7) feedback gain regularization
  + w_repulsion     * Σ ReLU(margin - dist)²   (8) obstacle repulsion

Note on trace vs K-regularization:
  loss_trace / loss_trace_terminal measure ACTUAL uncertainty (tr(Σ)).
  They directly penalize distribution spread and are the theoretically
  principled uncertainty-reduction objectives (cf. eq. 17 in Okamoto 2019,
  the tr(...ΣY) term). loss_K penalizes gain magnitude as a regularizer to
  prevent degenerate K solutions — it is NOT an uncertainty measure.
  Both are needed: trace terms for the learning signal, K-reg for stability.

Obstacle repulsion uses ReLU(margin - dist)^2 — zero gradient when the
trajectory is farther than obs_margin from an obstacle, quadratic penalty
when inside the margin or penetrating. This provides a smooth directional
gradient even when the trajectory is deep inside an obstacle, preventing
the planner from getting stuck when the STL term alone has vanishing gradient.
Rectangle distances use the exact signed distance field (SDF).
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
    """Compute the total scalar objective.

    Args:
        p_sat:        scalar tensor — SRM(φ, B) at t=0
        V:            [T, nu]          unconstrained feedforward params
        K:            [T, nu, nx]      feedback gains
        mu_trace:     [1, T+1, nx]     mean trajectory
        Sigma_trace:  [1, T+1, nx, nx] covariance trajectory
        env:          Environment — for goal center
        dyn:          BaseDynamics    — for bound_control
        weights:      dict with all w_* keys

    Returns:
        J: scalar tensor (differentiable)
    """
    w = weights
    device = V.device

    # ── (1) pdSTL satisfaction ───────────────────────────────────────
    loss_phi = -torch.log(p_sat + 1e-4)

    # ── (2) Trajectory covariance trace ─────────────────────────────
    # Vectorized: tr(Σ_k[:2,:2]) = Σ[0,k,0,0] + Σ[0,k,1,1]  (position block)
    # Sum over t=1..T (exclude t=0 initial condition)
    loss_trace = (Sigma_trace[0, 1:, 0, 0] + Sigma_trace[0, 1:, 1, 1]).sum()

    # ── (3) Terminal covariance trace ────────────────────────────────
    loss_trace_terminal = Sigma_trace[0, -1, 0, 0] + Sigma_trace[0, -1, 1, 1]

    # ── (4) Terminal distance to goal ────────────────────────────────
    loss_dist = torch.tensor(0.0, device=device)
    if env.goal is not None:
        gx = (env.goal["x"][0] + env.goal["x"][1]) / 2.0
        gy = (env.goal["y"][0] + env.goal["y"][1]) / 2.0
        goal_xy = torch.tensor([gx, gy], device=device)
        loss_dist = torch.sum((mu_trace[0, -1, :2] - goal_xy) ** 2)

    # ── (5) Control effort ───────────────────────────────────────────
    u_seq = dyn.bound_control(V)
    loss_u = torch.sum(u_seq ** 2)

    # ── (6) Control smoothness ───────────────────────────────────────
    u_diff = u_seq[1:] - u_seq[:-1]
    loss_du = torch.sum(u_diff ** 2) + torch.sum(u_seq[0] ** 2)

    # ── (7) Feedback gain regularization ────────────────────────────
    loss_K = torch.sum(K ** 2)

    # ── (8) Obstacle Repulsion ───────────────────────────────────────
    # ReLU(margin - dist)^2 gives nonzero gradient even when the trajectory
    # penetrates an obstacle, preventing the planner from getting stuck.
    loss_repulsion = torch.tensor(0.0, device=device)
    obs_margin = float(w.get("obs_margin", 0.5))
    for obs in env.obstacles:
        loss_repulsion += _rect_repulsion(mu_trace[0], obs, obs_margin, device)
    for obs in env.circle_obstacles:
        loss_repulsion += _circle_repulsion(mu_trace[0], obs, obs_margin, device)

    return (
        float(w.get("w_phi", 1.0))              * loss_phi
        + float(w.get("w_trace", 0.0))          * loss_trace
        + float(w.get("w_trace_terminal", 0.0)) * loss_trace_terminal
        + float(w.get("w_dist", 0.0))           * loss_dist
        + float(w.get("w_u", 0.0))              * loss_u
        + float(w.get("w_du", 0.0))             * loss_du
        + float(w.get("w_K", 0.0))              * loss_K
        + float(w.get("w_repulsion", 0.0))      * loss_repulsion
    )
