"""Base planner — shared infrastructure for all planner types.

Subclasses implement `solve()` which may call `_optimize_step()` once
(single-shot) or repeatedly in a receding-horizon loop (MPC).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from stl.base import GaussianBelief, BeliefTrajectory
from planning.objective import compute_loss


@dataclass
class PlanResult:
    """Output of a planner solve call."""
    mu_trace: torch.Tensor          # [1, T+1, nx]
    Sigma_trace: torch.Tensor       # [1, T+1, nx, nx]
    V: torch.Tensor                 # [T, nu]     best feedforward
    K: torch.Tensor                 # [T, nu, nx] best gains
    best_p: float                   # best P(φ) achieved
    history: List[float] = field(default_factory=list)    # loss per iteration
    p_history: List[float] = field(default_factory=list)  # P(φ) per iteration


class BasePlanner:
    """Shared planner infrastructure. Subclasses implement solve()."""

    def __init__(self, dynamics, steerer, env, cfg):
        self.dyn = dynamics
        self.steerer = steerer
        self.env = env
        self.cfg = cfg
        self.device = dynamics.device
        self.weights = cfg["weights"]
        self.opt_cfg = cfg["optimizer"]

    # ── Parameter initialization ─────────────────────────────────────

    def _init_params(self, T, init_V=None):
        """Create (V, K) as nn.Parameters.

        Args:
            T: horizon length
            init_V: optional warm-start for V [T, nu]

        Returns:
            V, K as nn.Parameters
        """
        if init_V is not None:
            # Invert tanh for warm-start
            u_norm = torch.clamp(init_V / (self.dyn.u_max + 1e-6), -0.99, 0.99)
            v_init = 0.5 * torch.log((1 + u_norm) / (1 - u_norm))
            V = nn.Parameter(v_init.to(self.device))
        else:
            V = nn.Parameter(torch.randn(T, self.dyn.nu, device=self.device) * 0.1)

        K = nn.Parameter(torch.zeros(T, self.dyn.nu, self.dyn.nx, device=self.device))
        return V, K

    def _build_optimizer(self, V, K):
        return optim.Adam([
            {"params": V, "lr": self.opt_cfg["lr_v"]},
            {"params": K, "lr": self.opt_cfg["lr_k"]},
        ])

    # ── Belief wrapping ──────────────────────────────────────────────

    @staticmethod
    def _wrap_beliefs(mu_trace, Sigma_trace, T):
        """Convert (μ, Σ) traces to a BeliefTrajectory for STL evaluation."""
        beliefs = []
        for t in range(T + 1):
            var_diag = torch.diagonal(Sigma_trace[0, t], dim1=-2, dim2=-1)
            beliefs.append(GaussianBelief(mu_trace[:, t, :], var_diag.unsqueeze(0)))
        return BeliefTrajectory(beliefs)

    # ── Single optimization step ─────────────────────────────────────

    def _optimize_step(self, V, K, mu0, Sigma0, spec, optimizer):
        """One forward-backward-update cycle.

        Returns:
            p_sat (float), loss (float)
        """
        optimizer.zero_grad()

        result = self.steerer(V, K, mu0, Sigma0)
        traj = self._wrap_beliefs(result.mu_trace, result.Sigma_trace, V.shape[0])

        stl_trace = spec(traj)
        p_sat = stl_trace[0, 0, 0]

        J = compute_loss(p_sat, V, K, result.mu_trace, result.Sigma_trace, self.env, self.dyn, self.weights)
        J.backward()
        optimizer.step()

        return p_sat.item(), J.item(), result

    # ── Interface ────────────────────────────────────────────────────

    @abstractmethod
    def solve(self, mu0, Sigma0, spec=None, init_V=None, verbose=True):
        """Find optimal (V, K).

        Args:
            mu0:    [nx] initial mean
            Sigma0: [nx, nx] initial covariance
            spec:   STL_Formula (defaults to env.get_specification)
            init_V: optional warm-start [T, nu]
            verbose: print progress

        Returns:
            PlanResult
        """
        ...
