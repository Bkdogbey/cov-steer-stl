"""Tests for the planning loss function and obstacle repulsion."""

import torch
import pytest
from dynamics.double_integrator import DoubleIntegrator
from dynamics.single_integrator import SingleIntegrator
from steering.closed_loop import ClosedLoopSteerer
from planning.objective import _rect_repulsion, _circle_repulsion, compute_loss
from planning.environment import Environment
from planning.single_shot import SingleShotPlanner


# ── SDF helper unit tests ────────────────────────────────────────────


def test_rect_sdf_inside_is_negative(device):
    """SDF is negative inside the box."""
    obs = {"x": [1.0, 3.0], "y": [1.0, 3.0]}
    # Point at box center — well inside
    pts = torch.tensor([[[2.0, 2.0, 0.0, 0.0]]], device=device)
    loss = _rect_repulsion(pts[0], obs, margin=0.0, device=device)
    # With margin=0 and point inside, ReLU(-sdf)^2 > 0
    assert loss.item() > 0.0


def test_rect_sdf_outside_beyond_margin(device):
    """No repulsion when trajectory is far beyond obs_margin."""
    obs = {"x": [0.0, 1.0], "y": [0.0, 1.0]}
    # Point at (5, 5) — far from [0,1]x[0,1] with margin 0.3
    pts = torch.tensor([[5.0, 5.0, 0.0, 0.0]], device=device)
    loss = _rect_repulsion(pts, obs, margin=0.3, device=device)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_circle_repulsion_inside(device):
    """Repulsion is positive when point is inside circle."""
    obs = {"center": [0.0, 0.0], "radius": 1.0}
    # Point at origin — dead centre of circle
    pts = torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=device)
    loss = _circle_repulsion(pts, obs, margin=0.3, device=device)
    assert loss.item() > 0.0


def test_circle_repulsion_far(device):
    """No repulsion when far from circle."""
    obs = {"center": [0.0, 0.0], "radius": 0.5}
    # Point at (10, 0) — far away with margin 0.5
    pts = torch.tensor([[10.0, 0.0, 0.0, 0.0]], device=device)
    loss = _circle_repulsion(pts, obs, margin=0.5, device=device)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


# ── compute_loss repulsion integration ───────────────────────────────


@pytest.fixture
def simple_env(device):
    """Environment with one rectangle and one circle."""
    env = Environment(device=str(device))
    env.set_goal(x_range=[8.0, 10.0], y_range=[-1.0, 1.0])
    env.add_obstacle(x_range=[3.0, 5.0], y_range=[-1.0, 1.0])
    env.add_circle_obstacle(center=[6.5, 0.0], radius=0.5)
    return env


def test_repulsion_near_rectangle(device, simple_env):
    """compute_loss repulsion term > 0 when trajectory passes through rectangle."""
    dyn = DoubleIntegrator(dt=0.2, u_max=2.5, D_diag=0.03, device=device)
    T = 5
    # Trajectory stuck at centre of rectangle [3,5]x[-1,1]
    mu_trace = torch.zeros(1, T + 1, 4, device=device)
    mu_trace[0, :, 0] = 4.0  # x = 4 (inside rect)
    mu_trace[0, :, 1] = 0.0  # y = 0 (inside rect)
    Sigma_trace = torch.zeros(1, T + 1, 4, 4, device=device)
    for t in range(T + 1):
        Sigma_trace[0, t] = torch.eye(4, device=device) * 0.01

    V = torch.zeros(T, 2, device=device, requires_grad=False)
    K = torch.zeros(T, 2, 4, device=device, requires_grad=False)
    weights = {"w_phi": 0.0, "w_trace": 0.0, "w_trace_terminal": 0.0,
               "w_dist": 0.0, "w_u": 0.0, "w_du": 0.0, "w_K": 0.0,
               "w_repulsion": 1.0, "obs_margin": 0.5}

    # p_sat = 0.5 (arbitrary, only repulsion matters here)
    p_sat = torch.tensor(0.5, device=device)
    loss = compute_loss(p_sat, V, K, mu_trace, Sigma_trace, simple_env, dyn, weights)
    assert loss.item() > 0.0, "Expected positive repulsion loss near rectangle"


def test_repulsion_near_circle(device, simple_env):
    """compute_loss repulsion term > 0 when trajectory passes through circle."""
    dyn = DoubleIntegrator(dt=0.2, u_max=2.5, D_diag=0.03, device=device)
    T = 5
    # Trajectory at centre of circle at (6.5, 0)
    mu_trace = torch.zeros(1, T + 1, 4, device=device)
    mu_trace[0, :, 0] = 6.5
    mu_trace[0, :, 1] = 0.0
    Sigma_trace = torch.zeros(1, T + 1, 4, 4, device=device)
    for t in range(T + 1):
        Sigma_trace[0, t] = torch.eye(4, device=device) * 0.01

    V = torch.zeros(T, 2, device=device)
    K = torch.zeros(T, 2, 4, device=device)
    weights = {"w_phi": 0.0, "w_trace": 0.0, "w_trace_terminal": 0.0,
               "w_dist": 0.0, "w_u": 0.0, "w_du": 0.0, "w_K": 0.0,
               "w_repulsion": 1.0, "obs_margin": 0.5}
    p_sat = torch.tensor(0.5, device=device)
    loss = compute_loss(p_sat, V, K, mu_trace, Sigma_trace, simple_env, dyn, weights)
    assert loss.item() > 0.0, "Expected positive repulsion loss near circle"


def test_repulsion_zero_when_far(device, simple_env):
    """Repulsion is zero when trajectory is far from all obstacles."""
    dyn = DoubleIntegrator(dt=0.2, u_max=2.5, D_diag=0.03, device=device)
    T = 5
    # Trajectory at (0.0, 3.0) — far from both obstacles
    mu_trace = torch.zeros(1, T + 1, 4, device=device)
    mu_trace[0, :, 0] = 0.0
    mu_trace[0, :, 1] = 3.0
    Sigma_trace = torch.zeros(1, T + 1, 4, 4, device=device)
    for t in range(T + 1):
        Sigma_trace[0, t] = torch.eye(4, device=device) * 0.01

    V = torch.zeros(T, 2, device=device)
    K = torch.zeros(T, 2, 4, device=device)
    weights = {"w_phi": 0.0, "w_trace": 0.0, "w_trace_terminal": 0.0,
               "w_dist": 0.0, "w_u": 0.0, "w_du": 0.0, "w_K": 0.0,
               "w_repulsion": 1.0, "obs_margin": 0.5}
    p_sat = torch.tensor(0.5, device=device)
    loss = compute_loss(p_sat, V, K, mu_trace, Sigma_trace, simple_env, dyn, weights)
    assert loss.item() == pytest.approx(0.0, abs=1e-5), "Expected zero repulsion when far from obstacles"


def test_repulsion_gradient_flows(device, simple_env):
    """Gradient of repulsion loss w.r.t. V is nonzero when near an obstacle."""
    dyn = DoubleIntegrator(dt=0.2, u_max=2.5, D_diag=0.03, device=device)
    steerer = ClosedLoopSteerer(dyn)
    T = 5
    Sigma0 = torch.eye(4, device=device) * 0.05

    # Start off-centre so abs() has nonzero derivative (center gives grad=0)
    mu0 = torch.tensor([3.5, 0.3, 0.0, 0.0], device=device)

    V = torch.nn.Parameter(torch.zeros(T, 2, device=device))
    K = torch.zeros(T, 2, 4, device=device)

    result = steerer(V, K, mu0, Sigma0)
    weights = {"w_phi": 0.0, "w_trace": 0.0, "w_trace_terminal": 0.0,
               "w_dist": 0.0, "w_u": 0.0, "w_du": 0.0, "w_K": 0.0,
               "w_repulsion": 1.0, "obs_margin": 0.5}
    p_sat = torch.tensor(0.5, device=device)
    loss = compute_loss(p_sat, V, K, result.mu_trace, result.Sigma_trace,
                        simple_env, dyn, weights)
    loss.backward()

    assert V.grad is not None
    assert V.grad.abs().sum().item() > 0.0, "Repulsion gradient w.r.t. V should be nonzero near obstacle"


# ── Integration test ─────────────────────────────────────────────────


@pytest.mark.slow
def test_planner_avoids_single_obstacle(device):
    """Single-shot planner should find a path around a rectangle."""
    dyn = SingleIntegrator(dt=0.3, u_max=2.0, D_diag=0.02, device=device)
    steerer = ClosedLoopSteerer(dyn)

    env = Environment(device=str(device))
    env.set_goal(x_range=[4.5, 5.5], y_range=[-0.5, 0.5])
    env.add_obstacle(x_range=[2.0, 3.5], y_range=[-1.5, 1.5])
    env.set_bounds(x_range=[-1.0, 7.0], y_range=[-3.0, 3.0])

    cfg = {
        "horizon": 20,
        "alpha": 0.7,
        "optimizer": {"lr_v": 0.05, "lr_k": 0.005, "max_iters": 300,
                      "converge_patience": 15},
        "weights": {"w_phi": 1.0, "w_trace": 0.01, "w_trace_terminal": 0.1,
                    "w_dist": 0.1, "w_u": 0.001, "w_du": 0.0005, "w_K": 0.0005,
                    "w_repulsion": 1.0, "obs_margin": 0.4},
    }

    planner = SingleShotPlanner(dyn, steerer, env, cfg)
    mu0 = torch.tensor([0.0, 0.0], device=device)
    Sigma0 = torch.eye(2, device=device) * 0.05

    result = planner.solve(mu0, Sigma0, verbose=False)
    assert result.best_p > 0.5, (
        f"Planner should find a viable path around obstacle, got P(φ)={result.best_p:.3f}"
    )
