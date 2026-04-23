"""Receding-horizon MPC planner for covariance steering.

At each time step t:
  1. Solve (V, K) over the next H steps from the current belief (μ_t, Σ_t).
  2. Apply only the first control — advance the belief one step.
  3. Warm-start the next solve by shifting the previous solution forward by one.

The full trajectory is assembled step-by-step. The final P(φ) is evaluated
on the complete T-step belief trace using the full-horizon specification.
"""

import torch
from planning.base import BasePlanner, PlanResult


class RecedingHorizonPlanner(BasePlanner):

    def _solve_one_step(self, h, mu_cur, Sigma_cur, spec_t, warm_V, mpc_iters):
        """Run one MPC local solve; return (best_p, best_V, best_K, history)."""
        warm = warm_V[:h] if warm_V is not None else None
        V, K = self._init_params(h, warm)
        optimizer = self._build_optimizer(V, K)

        best_p = -1.0
        best_V = V.data.clone()
        best_K = K.data.clone()
        hist = []

        for _ in range(mpc_iters):
            p_sat, loss, _ = self._optimize_step(V, K, mu_cur, Sigma_cur, spec_t, optimizer)
            hist.append(loss)
            if p_sat > best_p:
                best_p = p_sat
                best_V = V.data.clone()
                best_K = K.data.clone()

        return best_p, best_V, best_K, hist

    def solve(self, mu0, Sigma0, T=None, spec=None, init_V=None, verbose=True):
        T = T or self.cfg["horizon"]
        mpc_cfg = self.cfg.get("mpc", {})
        H = mpc_cfg.get("horizon", min(10, T))
        mpc_iters = mpc_cfg.get("iters", 100)
        n_starts = mpc_cfg.get("n_starts", 1)   # random restarts per step

        mu_list = [mu0]
        Sigma_list = [Sigma0]
        V_list = []
        K_list = []
        history = []
        p_history = []

        V_warm = None

        for t in range(T):
            h = min(H, T - t)
            spec_t = self.env.get_specification(h)

            mu_cur = mu_list[-1]
            Sigma_cur = Sigma_list[-1]

            # Warm-start candidate
            best_p_t, best_V_t, best_K_t, hist = self._solve_one_step(
                h, mu_cur, Sigma_cur, spec_t, V_warm, mpc_iters)
            history.extend(hist)
            p_history.append(best_p_t)

            # Random restarts — each ignores the warm start, helps escape local minima
            for _ in range(n_starts - 1):
                p_s, V_s, K_s, hist_s = self._solve_one_step(
                    h, mu_cur, Sigma_cur, spec_t, None, mpc_iters)
                history.extend(hist_s)
                if p_s > best_p_t:
                    best_p_t, best_V_t, best_K_t = p_s, V_s, K_s

            # Warm-start next step: drop the first entry, pad with zero at the end
            if h > 1:
                pad = torch.zeros(1, self.dyn.nu, device=self.device)
                V_warm = torch.cat([best_V_t[1:], pad], dim=0)
            else:
                V_warm = None

            # Advance belief one step using best first control
            v0 = best_V_t[0:1]   # [1, nu]
            k0 = best_K_t[0:1]   # [1, nu, nx]
            with torch.no_grad():
                step = self.steerer(v0, k0, mu_cur, Sigma_cur)

            V_list.append(best_V_t[0])
            K_list.append(best_K_t[0])
            mu_list.append(step.mu_trace[0, 1])
            Sigma_list.append(step.Sigma_trace[0, 1])

            if verbose and t % 5 == 0:
                print(f"  MPC step {t:3d}/{T} | local P(φ)={best_p_t:.4f}")

        # Assemble full trajectory
        mu_trace = torch.stack(mu_list).unsqueeze(0)       # [1, T+1, nx]
        Sigma_trace = torch.stack(Sigma_list).unsqueeze(0) # [1, T+1, nx, nx]
        V_full = torch.stack(V_list)                       # [T, nu]
        K_full = torch.stack(K_list)                       # [T, nu, nx]

        # Evaluate global P(φ) on the full assembled trajectory
        spec_full = spec or self.env.get_specification(T)
        traj = self._wrap_beliefs(mu_trace, Sigma_trace, T)
        stl_trace = spec_full(traj)
        best_p = stl_trace[0, 0, 0].item()

        if verbose:
            print(f"  MPC done | global P(φ)={best_p:.4f}")

        return PlanResult(
            mu_trace=mu_trace.detach(),
            Sigma_trace=Sigma_trace.detach(),
            V=V_full.detach(),
            K=K_full.detach(),
            best_p=best_p,
            history=history,
            p_history=p_history,
        )
