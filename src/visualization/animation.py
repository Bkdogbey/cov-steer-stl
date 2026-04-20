"""Trajectory animation (GIF/MP4) using FuncAnimation."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

from visualization.trajectory import draw_env, cov_ellipse_params


def animate_trajectory(result, env, filename="trajectory.gif", dt=0.2):
    """Animate a PlanResult as a GIF or MP4.

    Each frame draws the current mean position and covariance ellipse; the
    last FADE_STEPS ellipses are kept at decreasing opacity to show history.

    Args:
        result:   PlanResult — uses mu_trace [1, T+1, nx] and
                               Sigma_trace [1, T+1, nx, nx]
        env:      Environment object (obstacles, goal, bounds)
        filename: output filename — ".gif" uses PillowWriter,
                  ".mp4" uses FFMpegWriter (requires ffmpeg on PATH)
        dt:       timestep in seconds (controls animation speed)

    Returns:
        anim: FuncAnimation object (keep a reference to prevent GC)
    """
    mu_np = result.mu_trace.detach().cpu().squeeze(0).numpy()       # [T+1, nx]
    Sigma_np = result.Sigma_trace.detach().cpu().squeeze(0).numpy() # [T+1, nx, nx]
    T = mu_np.shape[0] - 1

    fig, ax = plt.subplots(figsize=(9, 7))
    if env.bounds:
        ax.set_xlim(env.bounds["x"][0] - 0.5, env.bounds["x"][1] + 0.5)
        ax.set_ylim(env.bounds["y"][0] - 0.5, env.bounds["y"][1] + 0.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x [m]", fontsize=12)
    ax.set_ylabel("y [m]", fontsize=12)
    ax.set_title("Covariance Steering Trajectory", fontsize=13, fontweight="bold")
    draw_env(ax, env)

    # Ghost trail of the full mean path (static, low opacity)
    ax.plot(mu_np[:, 0], mu_np[:, 1], "b-", lw=1, alpha=0.2, zorder=1)

    dot, = ax.plot([], [], "bo", ms=9, zorder=5)
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes,
                        fontsize=11, va="top")

    FADE_STEPS = 8
    CURRENT_ALPHA = 0.45
    MIN_ALPHA = 0.05
    history_ellipses: list = []

    def init():
        dot.set_data([], [])
        time_text.set_text("")
        return dot, time_text

    def update(frame):
        # Fade existing ellipses
        for i, ell in enumerate(history_ellipses):
            age = len(history_ellipses) - i   # 1 = most recently added past ellipse
            alpha = max(MIN_ALPHA, CURRENT_ALPHA * (1.0 - age / FADE_STEPS))
            ell.set_alpha(alpha)

        # Prune ellipses beyond the fade window
        while len(history_ellipses) > FADE_STEPS:
            history_ellipses.pop(0).remove()

        # Add current ellipse
        theta, w, h = cov_ellipse_params(Sigma_np[frame, :2, :2], k=2.0)
        ell = patches.Ellipse(
            (mu_np[frame, 0], mu_np[frame, 1]), w, h, angle=theta,
            fc="#1f77b4", ec="#1f77b4", alpha=CURRENT_ALPHA, zorder=3,
        )
        ax.add_patch(ell)
        history_ellipses.append(ell)

        dot.set_data([mu_np[frame, 0]], [mu_np[frame, 1]])
        time_text.set_text(f"t = {frame * dt:.1f} s")
        return [dot, time_text] + history_ellipses

    interval_ms = max(50, int(dt * 1000))
    anim = FuncAnimation(
        fig, update, frames=range(T + 1),
        init_func=init, blit=False, interval=interval_ms,
    )

    ext = filename.rsplit(".", 1)[-1].lower()
    fps = max(1, int(1.0 / dt))
    if ext == "gif":
        anim.save(filename, writer=PillowWriter(fps=fps))
    elif ext == "mp4":
        anim.save(filename, writer=FFMpegWriter(
            fps=fps, metadata={"title": "Covariance Steering"}
        ))
    else:
        raise ValueError(f"Unknown animation format: {ext!r}. Use '.gif' or '.mp4'.")

    plt.close(fig)
    return anim
