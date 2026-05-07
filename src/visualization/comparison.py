"""Side-by-side comparison and covariance-trace plots for OL vs CL."""

import numpy as np
import matplotlib.pyplot as plt
from visualization.trajectory import plot_trajectory, TITLE_FS, LABEL_FS, TICK_FS, LEGEND_FS, LINE_LW, SAVE_DPI


def plot_comparison(result_ol, result_cl, env, T, save_path=None, params=None):
    """Two-panel trajectory comparison: open-loop (left) vs closed-loop (right).

    Args:
        params: optional dict with keys horizon, dt, D_diag, sigma0,
                w_phi, w_repulsion, max_iters — shown as a caption strip.
    """
    mu_ol = result_ol.mu_trace.cpu().squeeze().numpy()
    S_ol  = result_ol.Sigma_trace.cpu().squeeze().numpy()
    mu_cl = result_cl.mu_trace.cpu().squeeze().numpy()
    S_cl  = result_cl.Sigma_trace.cpu().squeeze().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    plot_trajectory(
        axes[0], mu_ol, S_ol, env, T,
        title=(
            f"Open-Loop (K ≡ 0)   P(φ) = {result_ol.best_p:.3f}"
            f"   Route: {getattr(result_ol, 'route_label', 'n/a')}"
        ),
    )
    plot_trajectory(
        axes[1], mu_cl, S_cl, env, T,
        title=(
            f"Covariance Steering   P(φ) = {result_cl.best_p:.3f}"
            f"   Route: {getattr(result_cl, 'route_label', 'n/a')}"
        ),
    )

    trace_ol = float(np.trace(S_ol[-1, :2, :2]))
    trace_cl = float(np.trace(S_cl[-1, :2, :2]))
    if trace_cl > 1e-12:
        axes[1].text(
            0.02, 0.92,
            f"Final covariance vs open-loop:\n{trace_ol / trace_cl:.2f}x lower trace",
            transform=axes[1].transAxes, ha="left", va="top",
            fontsize=11, fontweight="bold",
            bbox=dict(fc="#e8f4ff", ec="#1f77b4", alpha=0.92, boxstyle="round,pad=0.35"),
        )

    fig.suptitle("pdSTL Covariance Steering", fontsize=TITLE_FS + 2, fontweight="bold", y=1.01)

    if params is not None:
        parts = [
            f"T = {params.get('horizon', '—')}  |  dt = {params.get('dt', '—')} s",
            f"σ₀ = {params.get('sigma0', 0.0):.3f} m  |  D = {params.get('D_diag', '—')}",
            f"w_φ = {params.get('w_phi', '—')}  |  w_rep = {params.get('w_repulsion', '—')}",
            f"Iterations = {params.get('max_iters', '—')}  |  lr = {params.get('lr_v', '—')}",
        ]
        fig.text(0.5, -0.02, "     ".join(parts), ha="center", va="top",
                 fontsize=12, style="italic",
                 bbox=dict(fc="lightyellow", ec="#aaaaaa", alpha=0.9,
                           boxstyle="round,pad=0.5"))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=SAVE_DPI, bbox_inches="tight")
    return fig


def plot_covariance_trace(result_ol, result_cl, T, dt, save_path=None):
    """Position covariance trace tr(Σ_pos) over time for OL vs CL.

    Shows how covariance evolves: OL grows unchecked (process noise accumulates);
    CL uses feedback K to keep uncertainty smaller.
    """
    S_ol = result_ol.Sigma_trace.cpu().squeeze().numpy()   # [T+1, nx, nx]
    S_cl = result_cl.Sigma_trace.cpu().squeeze().numpy()

    time = np.arange(T + 1) * dt
    trace_ol = np.array([np.trace(S_ol[k, :2, :2]) for k in range(T + 1)])
    trace_cl = np.array([np.trace(S_cl[k, :2, :2]) for k in range(T + 1)])

    # Feedback gain norms for CL
    K_np = result_cl.K.detach().cpu().numpy()   # [T, nu, nx]
    K_norms = np.linalg.norm(K_np, axis=(1, 2))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(time, trace_ol, color="#d62728", lw=LINE_LW, label="Open-Loop")
    ax1.plot(time, trace_cl, color="#1f77b4", lw=LINE_LW, label="Cov Steering")
    ax1.set_ylabel("tr(Σ_pos)  [m²]", fontsize=LABEL_FS, fontweight="bold")
    ax1.legend(fontsize=LEGEND_FS)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=TICK_FS)
    ax1.set_title("Position Covariance Growth", fontsize=TITLE_FS, fontweight="bold")

    if K_norms.max() > 1e-8:
        t_K = np.arange(len(K_norms)) * dt
        ax2.plot(t_K, K_norms, color="#2ca02c", lw=LINE_LW, label=r"$\|K_t\|_F$  (CL)")
        ax2.set_ylabel(r"Feedback gain $\|K_t\|_F$", fontsize=LABEL_FS, fontweight="bold")
        ax2.legend(fontsize=LEGEND_FS)
    else:
        ax2.text(0.5, 0.5, "K ≡ 0  (open-loop only)", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=LABEL_FS, color="gray")
        ax2.set_ylabel("Feedback gain", fontsize=LABEL_FS, fontweight="bold")

    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=TICK_FS)
    ax2.set_xlabel("Time [s]", fontsize=LABEL_FS, fontweight="bold")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=SAVE_DPI, bbox_inches="tight")
    return fig
