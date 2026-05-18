"""Single-shot planner: optimize (V, K) over a fixed horizon."""

from planning.base import BasePlanner, PlanResult


class SingleShotPlanner(BasePlanner):

    def _run_one_solve(self, T, mu0, Sigma0, spec, init_V, verbose, label="", iter_callback=None):
        """One full optimization pass. Returns (best_p, best_V, best_K, best_result, history, p_history)."""
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
                print(f"  {label}iter {k:4d} | loss={loss:.4f} | P(phi)={p_sat:.4f} | ||K||={K_norm:.3f}")

            if iter_callback is not None:
                iter_callback(k, p_sat, result.mu_trace.detach(), loss)

            if best_p >= alpha:
                converged_iters += 1
                if converged_iters >= patience:
                    if verbose:
                        print(f"  {label}Converged at iter {k}, P(phi)={best_p:.4f}")
                    break
            else:
                converged_iters = 0

        return best_p, best_V, best_K, best_result, history, p_history

    def solve(self, mu0, Sigma0, T=None, spec=None, init_V=None, verbose=True, iter_callback=None):
        T = T or self.cfg["horizon"]
        spec = spec or self.env.get_specification(T)
        n_starts = self.opt_cfg.get("n_starts", 1)

        # First solve uses init_V (warm-start or None); subsequent restarts are random.
        best_p, best_V, best_K, best_result, history, p_history = \
            self._run_one_solve(T, mu0, Sigma0, spec, init_V, verbose,
                                label=f"[1/{n_starts}] " if n_starts > 1 else "",
                                iter_callback=iter_callback)

        for s in range(1, n_starts):
            p_s, V_s, K_s, res_s, hist_s, ph_s = self._run_one_solve(
                T, mu0, Sigma0, spec, None, verbose,
                label=f"[{s+1}/{n_starts}] ")
            history.extend(hist_s)
            p_history.extend(ph_s)
            if p_s > best_p:
                best_p, best_V, best_K, best_result = p_s, V_s, K_s, res_s

        return PlanResult(
            mu_trace=best_result.mu_trace.detach(),
            Sigma_trace=best_result.Sigma_trace.detach(),
            V=best_V,
            K=best_K,
            best_p=best_p,
            history=history,
            p_history=p_history,
        )
