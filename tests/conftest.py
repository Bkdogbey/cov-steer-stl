"""Shared pytest fixtures for the test suite."""

import pytest
import torch


@pytest.fixture(scope="session")
def device():
    """Shared torch device for all tests. CPU for reproducibility."""
    return torch.device("cpu")


@pytest.fixture(scope="session")
def double_integrator_dyn(device):
    """Pre-built DoubleIntegrator for tests that need dynamics."""
    from dynamics.double_integrator import DoubleIntegrator
    return DoubleIntegrator(dt=0.2, u_max=2.5, D_diag=0.03, device=device)
