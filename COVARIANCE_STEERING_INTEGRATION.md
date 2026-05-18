# Covariance Steering Integration Guide for `probabilistic-dstl`

This document specifies exactly how to port covariance steering (K-gain optimization) from `cov-steer-stl` into the `probabilistic-dstl` repo. All changes go on a dedicated branch: **`feat/covariance-steering`**.

---

## Background

`probabilistic-dstl`'s `ProbabilisticSTLPlanner` currently optimizes only the feedforward control sequence **V** `[T, nu]`. Covariance grows monotonically via the open-loop update `Σ_{k+1} = A Σ A^T + Q`.

Covariance steering adds a feedback gain matrix **K** `[T, nu, nx]` so the actual control applied is:

```
u_k = sat(v_k + K_k (x_k − μ_k))
```

After linearizing the tanh saturation around the mean, the effective gain is:

```
K_eff_k = diag(sech²(v_k) · u_max) @ K_k
```

and the closed-loop covariance propagation becomes:

```
A_cl_k  = A + B @ K_eff_k
Σ_{k+1} = A_cl_k Σ_k A_cl_k^T + Q
```

This allows the optimizer to actively steer the uncertainty distribution — shrinking it near obstacles and shaping it toward a target.

**Convention note:** `probabilistic-dstl` uses `.Q` for process noise. `cov-steer-stl` uses `.DDT`. The steerer below uses `.Q`.

---

## Branch Strategy

```bash
cd /path/to/probabilistic-dstl
git checkout -b feat/covariance-steering
```

Suggested commit sequence:
1. `add: ClosedLoopSteerer to src/planning/steering.py`
2. `extend: ProbabilisticSTLPlanner to accept optional steerer`
3. `config: add lr_k, w_K, w_trace_terminal to configs/planning.yaml`
4. `test: smoke-test steerer rollout and planner with steerer`

---

## File 1 — Create `src/planning/steering.py` (new file)

```python
"""Covariance steering via feedback gains K (Okamoto et al., Theorem 1).

Control law: u_k = sat(v_k + K_k (x_k − μ_k))
Covariance:  Σ_{k+1} = (A + B K_eff_k) Σ_k (A + B K_eff_k)^T + Q
where K_eff_k = diag(sech²(v_k) · u_max) @ K_k.

Key difference from cov-steer-stl: uses dynamics.Q (not dynamics.DDT).
"""

import torch


class ClosedLoopSteerer:
    """Rolls out mean + covariance under the closed-loop saturating policy."""

    def __init__(self, dynamics):
        self.dyn = dynamics

    def rollout(self, V, K, mu0, Sigma0):
        """Differentiable rollout for T steps.

        Args:
            V:      [T, nu]     unconstrained feedforward controls (pre-tanh)
            K:      [T, nu, nx] feedback gain matrices
            mu0:    [nx]        initial mean
            Sigma0: [nx, nx]    initial covariance

        Returns:
            means: [T+1, nx]
            covs:  [T+1, nx, nx]
        """
        T = V.shape[0]
        means = [mu0]
        covs = [Sigma0]
        mu, Sigma = mu0, Sigma0

        for t in range(T):
            u = self.dyn.bound_control(V[t])                          # [nu]
            mu = self.dyn.A @ mu + self.dyn.B @ u                     # [nx]

            gain_scale = self.dyn.u_max * (1.0 - torch.tanh(V[t]) ** 2)  # [nu]
            K_eff = K[t] * gain_scale.unsqueeze(-1)                   # [nu, nx]
            A_cl = self.dyn.A + self.dyn.B @ K_eff                    # [nx, nx]
            Sigma = A_cl @ Sigma @ A_cl.T + self.dyn.Q                # [nx, nx]

            means.append(mu)
            covs.append(Sigma)

        return torch.stack(means), torch.stack(covs)
```

---

## File 2 — Modify `src/planning/planner.py`

### 2a. Add `_init_gains` method (after `_init_controls`):

```python
def _init_gains(self, nx, nu):
    """Initialise feedback gain parameters K [T, nu, nx] near zero."""
    return torch.nn.Parameter(
        torch.zeros(self.T, nu, nx, device=self.device),
        requires_grad=True,
    )
```

### 2b. Extend `_compute_loss` signature and body:

Change:
```python
def _compute_loss(self, mean_trace, u_seq, p_all, loss_fn):
```
To:
```python
def _compute_loss(self, mean_trace, cov_trace, u_seq, p_all, loss_fn, K_params=None):
```

At the end of the existing return expression, add the optional K terms:

```python
    loss = (
        self.cfg["w_u"]    * loss_u
        + self.cfg["w_du"]   * loss_du
        + self.cfg["w_phi"]  * loss_phi
        + self.cfg["w_dist"] * self._goal_dist_loss(mean_trace)
        + self.cfg["w_obs"]  * self._obs_repulsion_loss(mean_trace)
        + self.cfg["w_visit"]* self._visit_loss(mean_trace)
    )
    if K_params is not None:
        loss = loss + self.cfg.get("w_K", 0.01) * torch.sum(K_params ** 2)
        loss = loss + self.cfg.get("w_trace_terminal", 0.1) * torch.trace(cov_trace[-1, :2, :2])
    return loss
```

**Note:** The existing call site inside `solve` passes `mean_trace` but not `cov_trace`. Update the call to pass both. When `steerer=None`, pass `cov_trace=None` and `K_params=None` so the guard is a no-op.

### 2c. Extend `solve` to accept an optional steerer:

Change signature:
```python
def solve(self, x0_mean, x0_cov, render=False, verbose=True, spec=None, init_guess=None, loss_fn=None):
```
To:
```python
def solve(self, x0_mean, x0_cov, render=False, verbose=True, spec=None,
          init_guess=None, loss_fn=None, steerer=None):
```

After `v_params = self._init_controls(init_guess)`, add:

```python
K_params = None
if steerer is not None:
    nx = x0_mean.shape[0]
    nu = 2  # or derive from dynamics: self.dyn.B.shape[1]
    K_params = self._init_gains(nx, nu)
    optimizer = optim.Adam([
        {"params": [v_params], "lr": self.cfg["lr"]},
        {"params": [K_params], "lr": self.cfg.get("lr_k", 0.005)},
    ])
else:
    optimizer = optim.Adam([v_params], lr=self.cfg["lr"])
```

Inside the optimisation loop, replace:
```python
mean_trace, cov_trace = self.dyn(v_params, x0_mean, x0_cov)
```
With:
```python
if steerer is not None:
    means_raw, covs_raw = steerer.rollout(v_params, K_params, x0_mean, x0_cov)
    mean_trace = means_raw.unsqueeze(0)    # [1, T+1, nx]
    cov_trace  = covs_raw                  # [T+1, nx, nx]  (no batch dim)
else:
    mean_trace, cov_trace = self.dyn(v_params, x0_mean, x0_cov)
```

Update `_compute_loss` call:
```python
J = self._compute_loss(mean_trace, cov_trace, u_seq, p_all, loss_fn, K_params)
```

**Note:** When `steerer is not None`, `cov_trace` has shape `[T+1, nx, nx]` (no batch dim) — ensure `_goal_dist_loss`, `_obs_repulsion_loss`, `_visit_loss` only use `mean_trace` (they already do). The `w_trace_terminal` term indexes `cov_trace[-1, :2, :2]` directly.

Track best K alongside best_u:
```python
best_K = None
# ... inside loop, where best_p is updated:
if steerer is not None:
    best_K = K_params.detach().clone()
```

Return statement:
```python
if steerer is not None:
    return best_mean, best_cov, best_u, best_p, history, best_K
return best_mean, best_cov, best_u, best_p, history
```

---

## File 3 — Append to `configs/planning.yaml`

```yaml
# Covariance steering (used only when a ClosedLoopSteerer is passed to solve)
lr_k: 0.005              # Adam lr for feedback gains K
w_K: 0.01                # L2 regularization weight on K entries
w_trace_terminal: 0.1    # Weight on terminal position covariance trace tr(Σ_T[:2,:2])
```

---

## Usage Example

```python
import torch
from planning.dynamics import DoubleIntegrator
from planning.steering import ClosedLoopSteerer
from planning.planner import ProbabilisticSTLPlanner

dyn = DoubleIntegrator(dt=0.2, u_max=1.5, q_std=0.01)
steerer = ClosedLoopSteerer(dyn)

# env and spec built the usual pdstl way
planner = ProbabilisticSTLPlanner(dyn, env, T=40)
x0_mean = torch.tensor([0.0, 0.0, 3.5, 0.0])
x0_cov  = torch.eye(4) * 0.01

result = planner.solve(x0_mean, x0_cov, steerer=steerer)
best_mean, best_cov, best_u, best_p, history, best_K = result

print(f"P(sat) = {best_p:.4f}")
print(f"best_K shape: {best_K.shape}")  # [T, 2, 4]
```

---

## Smoke Test

Run this from the pdstl repo root to verify the steerer alone:

```python
import torch
import sys; sys.path.insert(0, "src")
from planning.dynamics import DoubleIntegrator
from planning.steering import ClosedLoopSteerer

dyn = DoubleIntegrator()
steerer = ClosedLoopSteerer(dyn)

V  = torch.zeros(10, 2)
K  = torch.zeros(10, 2, 4)
mu = torch.zeros(4)
S  = torch.eye(4)

means, covs = steerer.rollout(V, K, mu, S)
assert means.shape == (11, 4),  f"got {means.shape}"
assert covs.shape  == (11, 4, 4), f"got {covs.shape}"
print("Steerer smoke test passed.")
```

---

## Backwards Compatibility

- When `steerer=None` (default), `solve` returns the **unchanged 5-tuple** `(best_mean, best_cov, best_u, best_p, history)`. No existing caller is broken.
- `_compute_loss` is called with `cov_trace` and `K_params=None` when not steering — the extra terms are skipped.
- `SingleIntegrator` callers are unaffected (they don't have `.A`/`.B` so they can't be passed to `ClosedLoopSteerer` anyway).
