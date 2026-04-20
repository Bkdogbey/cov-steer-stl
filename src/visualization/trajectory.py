"""Static trajectory plots with covariance ellipses."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def cov_ellipse_params(cov_2x2, k=1.96):
    """(angle_deg, width, height) for a 2D covariance ellipse."""
    vals, vecs = np.linalg.eigh(cov_2x2)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w = 2 * k * np.sqrt(max(vals[0], 1e-10))
    h = 2 * k * np.sqrt(max(vals[1], 1e-10))
    return theta, w, h


def draw_env(ax, env):
    """Draw obstacles, goal, bounds onto an axes."""
    if env.goal:
        gx, gy = env.goal["x"], env.goal["y"]
        ax.add_patch(patches.Rectangle(
            (gx[0], gy[0]), gx[1]-gx[0], gy[1]-gy[0],
            fc="#98df8a", ec="#2ca02c", alpha=0.4,
        ))
        ax.text((gx[0]+gx[1])/2, (gy[0]+gy[1])/2, "G",
                fontsize=16, fontweight="bold", ha="center", va="center", color="#2ca02c")

    for obs in env.obstacles:
        ox, oy = obs["x"], obs["y"]
        ax.add_patch(patches.Rectangle(
            (ox[0], oy[0]), ox[1]-ox[0], oy[1]-oy[0],
            fc="#ff9896", ec="#d62728", alpha=0.7, hatch="//",
        ))

    for obs in env.circle_obstacles:
        ax.add_patch(patches.Circle(
            obs["center"], obs["radius"],
            fc="#ff9896", ec="#d62728", alpha=0.6, hatch="//",
        ))

    for region in env.visit_regions:
        vx, vy = region["x"], region["y"]
        ax.add_patch(patches.Rectangle(
            (vx[0], vy[0]), vx[1]-vx[0], vy[1]-vy[0],
            fc="#c5b0d5", ec="#9467bd", alpha=0.4,
        ))


def plot_trajectory(ax, mu_np, Sigma_np, env, T, title=None, ellipse_step=2, k=2.0):
    """Plot mean trajectory with covariance ellipses on a given axes.

    Args:
        ax: matplotlib Axes
        mu_np: [T+1, nx] numpy array
        Sigma_np: [T+1, nx, nx] numpy array
        env: Environment
        T: horizon
        title: optional title string
        ellipse_step: draw ellipse every N steps
        k: confidence interval scale
    """
    if env.bounds:
        ax.set_xlim(env.bounds["x"][0] - 0.5, env.bounds["x"][1] + 0.5)
        ax.set_ylim(env.bounds["y"][0] - 0.5, env.bounds["y"][1] + 0.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x [m]", fontsize=12)
    ax.set_ylabel("y [m]", fontsize=12)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")

    draw_env(ax, env)

    # Trajectory line
    ax.plot(mu_np[:, 0], mu_np[:, 1], "b-", linewidth=2, alpha=0.9)

    # Covariance ellipses
    for t in range(0, T + 1, max(1, ellipse_step)):
        theta, w, h = cov_ellipse_params(Sigma_np[t, :2, :2], k=k)
        ell = patches.Ellipse(
            (mu_np[t, 0], mu_np[t, 1]), w, h, angle=theta,
            fc="#1f77b4", ec="#1f77b4", alpha=0.2,
        )
        ax.add_patch(ell)

    ax.plot(mu_np[0, 0], mu_np[0, 1], "ko", ms=8)
    ax.plot(mu_np[-1, 0], mu_np[-1, 1], "bs", ms=8)


def plot_control_sequence(result, dt, dyn=None, save_path=None):
    """Plot the feedforward control sequence and feedback gain norms over time.

    Args:
        result:    PlanResult — uses result.V [T, nu] and result.K [T, nu, nx]
        dt:        float — timestep in seconds (x-axis label)
        dyn:       optional BaseDynamics — if provided, shows bounded u = u_max*tanh(V)
                   instead of raw V params
        save_path: optional str/Path — saved as PNG if provided

    Returns:
        fig: matplotlib Figure
    """
    import torch

    V_np = result.V.detach().cpu().numpy()   # [T, nu]
    K_np = result.K.detach().cpu().numpy()   # [T, nu, nx]
    T, nu = V_np.shape
    time = [t * dt for t in range(T)]

    if dyn is not None:
        with torch.no_grad():
            u_bounded = dyn.bound_control(result.V).cpu().numpy()
        y_data = u_bounded
        y_label = "Control input u"
        legend_prefix = "u"
    else:
        y_data = V_np
        y_label = "Feedforward param V (raw)"
        legend_prefix = "v"

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    fig, ax1 = plt.subplots(figsize=(10, 4))

    for i in range(nu):
        ax1.step(time, y_data[:, i], where="post",
                 color=colors[i % len(colors)], lw=1.8,
                 label=f"${legend_prefix}_{i}$")
    ax1.axhline(0, color="k", lw=0.5, ls="--")
    ax1.set_xlabel("Time [s]", fontsize=12)
    ax1.set_ylabel(y_label, fontsize=12)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # K gain norms on secondary axis (only if non-trivial)
    K_norms = np.linalg.norm(K_np, axis=(1, 2))   # [T]
    if K_norms.max() > 1e-8:
        ax2 = ax1.twinx()
        ax2.plot(time, K_norms, "k--", lw=1.5, alpha=0.6, label=r"$\|K_t\|_F$")
        ax2.set_ylabel(r"$\|K_t\|_F$", fontsize=12)
        ax2.legend(loc="upper right", fontsize=10)

    ax1.set_title("Control Sequence", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
