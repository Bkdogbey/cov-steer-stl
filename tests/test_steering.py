"""Tests for the steering module."""

import torch
import pytest
from dynamics.double_integrator import DoubleIntegrator
from steering.open_loop import OpenLoopSteerer
from steering.closed_loop import ClosedLoopSteerer


@pytest.fixture
def setup(device):
    dyn = DoubleIntegrator(dt=0.2, u_max=2.5, D_diag=0.03, device=device)
    T, nx, nu = 10, 4, 2
    mu0 = torch.tensor([0.0, 0.0, 1.0, 0.0], device=device)
    Sigma0 = torch.diag(torch.tensor([0.05, 0.05, 0.01, 0.01], device=device))
    V = torch.randn(T, nu, device=device) * 0.1
    K_zero = torch.zeros(T, nu, nx, device=device)
    return dyn, T, nx, nu, mu0, Sigma0, V, K_zero


def test_closed_loop_k_zero_matches_open_loop(setup):
    """With K=0, closed-loop and open-loop must produce identical traces."""
    dyn, T, nx, nu, mu0, Sigma0, V, K_zero = setup
    ol = OpenLoopSteerer(dyn)
    cl = ClosedLoopSteerer(dyn)

    r_ol = ol(V, K_zero, mu0, Sigma0)
    r_cl = cl(V, K_zero, mu0, Sigma0)

    torch.testing.assert_close(r_ol.mu_trace, r_cl.mu_trace, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(r_ol.Sigma_trace, r_cl.Sigma_trace, atol=1e-6, rtol=1e-5)


def test_closed_loop_output_shapes(setup):
    """Output shapes must be [1, T+1, nx] and [1, T+1, nx, nx]."""
    dyn, T, nx, nu, mu0, Sigma0, V, K_zero = setup
    cl = ClosedLoopSteerer(dyn)
    K = torch.randn(T, nu, nx) * 0.01
    result = cl(V, K, mu0, Sigma0)
    assert result.mu_trace.shape == (1, T + 1, nx)
    assert result.Sigma_trace.shape == (1, T + 1, nx, nx)


def test_closed_loop_covariance_is_psd(setup):
    """Covariance must stay positive semi-definite at every step."""
    dyn, T, nx, nu, mu0, Sigma0, V, K_zero = setup
    cl = ClosedLoopSteerer(dyn)
    K = torch.randn(T, nu, nx) * 0.05
    result = cl(V, K, mu0, Sigma0)
    for t in range(T + 1):
        eigs = torch.linalg.eigvalsh(result.Sigma_trace[0, t])
        assert (eigs >= -1e-6).all(), f"Negative eigenvalue at t={t}: {eigs}"


@pytest.mark.slow
def test_gradient_flows_through_K(setup):
    """∂(loss)/∂K must be nonzero — gradients flow through covariance steering."""
    dyn, T, nx, nu, mu0, Sigma0, V_data, _ = setup
    cl = ClosedLoopSteerer(dyn)

    K = torch.nn.Parameter(torch.randn(T, nu, nx) * 0.01)
    V = V_data.clone()  # V is not a Parameter here — Σ doesn't depend on V

    result = cl(V, K, mu0, Sigma0)
    loss = torch.trace(result.Sigma_trace[0, -1])
    loss.backward()

    assert K.grad is not None, "K.grad is None — no gradient path"
    assert K.grad.abs().sum() > 0, "K.grad is all zeros"


@pytest.mark.slow
def test_gradient_flows_through_V(setup):
    """∂(loss)/∂V must be nonzero — gradients flow through mean propagation."""
    dyn, T, nx, nu, mu0, Sigma0, V_data, K_zero = setup
    cl = ClosedLoopSteerer(dyn)

    V = torch.nn.Parameter(V_data.clone())

    result = cl(V, K_zero, mu0, Sigma0)
    loss = result.mu_trace[0, -1].sum()  # depends on V via mean
    loss.backward()

    assert V.grad is not None, "V.grad is None"
    assert V.grad.abs().sum() > 0, "V.grad is all zeros"


@pytest.mark.slow
def test_nonzero_K_shrinks_covariance(setup):
    """A stabilising K should produce smaller terminal covariance than K=0."""
    dyn, T, nx, nu, mu0, Sigma0, V, K_zero = setup
    cl = ClosedLoopSteerer(dyn)

    # Small negative feedback (stabilising)
    K_stab = torch.zeros(T, nu, nx)
    K_stab[:, 0, 0] = -0.1  # ax feeds back on px
    K_stab[:, 1, 1] = -0.1  # ay feeds back on py

    r_zero = cl(V, K_zero, mu0, Sigma0)
    r_stab = cl(V, K_stab, mu0, Sigma0)

    det_zero = torch.det(r_zero.Sigma_trace[0, -1, :2, :2])
    det_stab = torch.det(r_stab.Sigma_trace[0, -1, :2, :2])
    assert det_stab < det_zero, (
        f"Stabilising K should shrink covariance: det_stab={det_stab:.6f} >= det_zero={det_zero:.6f}"
    )
