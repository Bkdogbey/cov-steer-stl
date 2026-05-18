"""Shared pytest fixtures for the test suite."""

import pytest
import torch


@pytest.fixture(scope="session")
def device():
    """Shared torch device for all tests. CPU for reproducibility."""
    return torch.device("cpu")
