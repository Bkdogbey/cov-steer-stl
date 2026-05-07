"""Static trajectory plots with covariance ellipses."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── Slide-ready defaults ──────────────────────────────────────────────
TITLE_FS  = 16
LABEL_FS  = 14
TICK_FS   = 12
LEGEND_FS = 12
LINE_LW   = 2.5
SAVE_DPI  = 200


def cov_ellipse_params(cov_2x2, k=1.96):
    """(angle_deg, width, height) for a 2D covariance ellipse."""
    vals, vecs = np.linalg.eigh(cov_2x2)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w = 2 * k * np.sqrt(max(vals[0], 1e-10))
    h = 2 * k * np.sqrt(max(vals[1], 1e-10))
    return theta, w, h


def covariance_change_text(Sigma_np):
    """Short label for initial/final position covariance trace."""
    trace0 = float(np.trace(Sigma_np[0, :2, :2]))
    trace_t = float(np.trace(Sigma_np[-1, :2, :2]))
    if trace0 > 1e-12:
        ratio = trace_t / trace0
        change = f"{ratio:.2f}x initial"
    else:
        change = "from near-zero initial"
    return f"trΣ₀={trace0:.3f}  →  trΣT={trace_t:.3f}\n{change}"


def draw_env(ax, env):
    """Draw obstacles, goal, and visit regions onto axes."""
    if env.goal:
        gx, gy = env.goal["x"], env.goal["y"]
        ax.add_patch(patches.Rectangle(
            (gx[0], gy[0]), gx[1] - gx[0], gy[1] - gy[0],
            fc="#98df8a", ec="#2ca02c", alpha=0.5, lw=2,
        ))
        ax.text((gx[0] + gx[1]) / 2, (gy[0] + gy[1]) / 2, "G",
                fontsize=18, fontweight="bold", ha="center", va="center", color="#2ca02c")

    for obs in env.obstacles:
        ox, oy = obs["x"], obs["y"]
        ax.add_patch(patches.Rectangle(
            (ox[0], oy[0]), ox[1] - ox[0], oy[1] - oy[0],
            fc="#ff9896", ec="#d62728", alpha=0.75, hatch="//", lw=1.5,
        ))

    for obs in env.circle_obstacles:
        ax.add_patch(patches.Circle(
            obs["center"], obs["radius"],
            fc="#ff9896", ec="#d62728", alpha=0.65, hatch="//", lw=1.5,
        ))

    for region in env.visit_regions:
        vx, vy = region["x"], region["y"]
        ax.add_patch(patches.Rectangle(
            (vx[0], vy[0]), vx[1] - vx[0], vy[1] - vy[0],
            fc="#c5b0d5", ec="#9467bd", alpha=0.4, lw=1.5,
        ))


def plot_trajectory(ax, mu_np, Sigma_np, env, T, title=None, ellipse_step=2, k=2.0,
                    annotate_covariance=True):
    """Mean trajectory with covariance ellipses on a given axes."""
    if env.bounds:
        ax.set_xlim(env.bounds["x"][0] - 0.5, env.bounds["x"][1] + 0.5)
        ax.set_ylim(env.bounds["y"][0] - 0.5, env.bounds["y"][1] + 0.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x [m]", fontsize=LABEL_FS, fontweight="bold")
    ax.set_ylabel("y [m]", fontsize=LABEL_FS, fontweight="bold")
    ax.tick_params(labelsize=TICK_FS)
    if title:
        ax.set_title(title, fontsize=TITLE_FS, fontweight="bold", pad=8)

    draw_env(ax, env)

    ax.plot(mu_np[:, 0], mu_np[:, 1], "b-", linewidth=LINE_LW, alpha=0.9, zorder=3)

    for t in range(0, T + 1, max(1, ellipse_step)):
        theta, w, h = cov_ellipse_params(Sigma_np[t, :2, :2], k=k)
        ell = patches.Ellipse(
            (mu_np[t, 0], mu_np[t, 1]), w, h, angle=theta,
            fc="#1f77b4", ec="#1f77b4", alpha=0.18, zorder=2,
        )
        ax.add_patch(ell)

    ax.plot(mu_np[0, 0], mu_np[0, 1], "ko", ms=10, zorder=4, label="Start")
    ax.plot(mu_np[-1, 0], mu_np[-1, 1], "bs", ms=10, zorder=4, label="End")

    theta0, w0, h0 = cov_ellipse_params(Sigma_np[0, :2, :2], k=k)
    ax.add_patch(patches.Ellipse(
        (mu_np[0, 0], mu_np[0, 1]), w0, h0, angle=theta0,
        fc="none", ec="#111111", lw=2.0, ls="-", zorder=5,
    ))
    theta_t, wt, ht = cov_ellipse_params(Sigma_np[-1, :2, :2], k=k)
    ax.add_patch(patches.Ellipse(
        (mu_np[-1, 0], mu_np[-1, 1]), wt, ht, angle=theta_t,
        fc="none", ec="#0057b8", lw=2.0, ls="--", zorder=5,
    ))

    if annotate_covariance:
        ax.text(
            0.02, 0.02, covariance_change_text(Sigma_np),
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=10,
            bbox=dict(fc="white", ec="#777777", alpha=0.9, boxstyle="round,pad=0.35"),
        )


def plot_control_sequence(result, dt, dyn=None, save_path=None):
    """Feedforward controls and feedback gain norms over time."""
    V_np = result.V.detach().cpu().numpy()   # [T, nu]
    K_np = result.K.detach().cpu().numpy()   # [T, nu, nx]
    T, nu = V_np.shape
    time = [t * dt for t in range(T)]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    fig, ax1 = plt.subplots(figsize=(10, 4))

    for i in range(nu):
        ax1.step(time, V_np[:, i], where="post",
                 color=colors[i % len(colors)], lw=LINE_LW,
                 label=f"$u_{i}$")
    ax1.axhline(0, color="k", lw=0.8, ls="--")
    ax1.set_xlabel("Time [s]", fontsize=LABEL_FS, fontweight="bold")
    ax1.set_ylabel("Control input u", fontsize=LABEL_FS, fontweight="bold")
    ax1.tick_params(labelsize=TICK_FS)
    ax1.legend(loc="upper left", fontsize=LEGEND_FS)
    ax1.grid(True, alpha=0.3)

    K_norms = np.linalg.norm(K_np, axis=(1, 2))
    if K_norms.max() > 1e-8:
        ax2 = ax1.twinx()
        ax2.plot(time, K_norms, "k--", lw=LINE_LW - 0.5, alpha=0.65,
                 label=r"$\|K_t\|_F$")
        ax2.set_ylabel(r"$\|K_t\|_F$", fontsize=LABEL_FS)
        ax2.tick_params(labelsize=TICK_FS)
        ax2.legend(loc="upper right", fontsize=LEGEND_FS)

    ax1.set_title("Control Sequence", fontsize=TITLE_FS, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=SAVE_DPI, bbox_inches="tight")
    return fig
