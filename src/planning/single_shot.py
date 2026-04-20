"""Single-shot planner: optimize (V, K) over a fixed horizon."""

from planning.base import BasePlanner, PlanResult


class SingleShotPlanner(BasePlanner):

    def solve(self, mu0, Sigma0, T=None, spec=None, init_V=None, verbose=True):
        T = T or self.cfg["horizon"]
        spec = spec or self.env.get_specification(T)
        V, K = self._init_params(T, init_V)
        optimizer = self._build_optimizer(V, K)

        best_p = -1.0
        best_V, best_K = V.data.clone(), K.data.clone()
        best_result = None
        history = []
        p_history = []
        converged_iters = 0
        patience = self.opt_cfg.get("converge_patience", 20)
        alpha = self.cfg.get("alpha", 0.95)

        for k in range(self.opt_cfg["max_iters"]):
            p_sat, loss, result = self._optimize_step(V, K, mu0, Sigma0, spec, optimizer)
            history.append(loss)
            p_history.append(p_sat)

            if p_sat > best_p:
                best_p = p_sat
                best_V = V.data.clone()
                best_K = K.data.clone()
                best_result = result

            if verbose and k % 50 == 0:
                K_norm = K.data.norm().item()
                print(f"  iter {k:4d} | loss={loss:.4f} | P(φ)={p_sat:.4f} | ||K||={K_norm:.3f}")

            if best_p >= alpha:
                converged_iters += 1
                if converged_iters >= patience:
                    if verbose:
                        print(f"  Converged at iter {k}, P(φ)={best_p:.4f}")
                    break
            else:
                converged_iters = 0

        return PlanResult(
            mu_trace=best_result.mu_trace.detach(),
            Sigma_trace=best_result.Sigma_trace.detach(),
            V=best_V,
            K=best_K,
            best_p=best_p,
            history=history,
            p_history=p_history,
        )
