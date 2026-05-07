"""Convergence curve plots."""

import numpy as np
import matplotlib.pyplot as plt
from visualization.trajectory import TITLE_FS, LABEL_FS, TICK_FS, LEGEND_FS, LINE_LW, SAVE_DPI


def plot_convergence(histories, labels=None, colors=None, save_path=None):
    """P(φ) and loss convergence for one or more runs.

    Args:
        histories: list of dicts with keys 'p_sat' and/or 'loss'
        labels:    list of display names
        colors:    list of colour strings
    """
    n = len(histories)
    labels = labels or [f"Run {i}" for i in range(n)]
    colors = colors or ["#2ca02c", "#d62728", "#1f77b4", "#ff7f0e"][:n]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for h, lbl, c in zip(histories, labels, colors):
        if "p_sat" in h:
            axes[0].plot(h["p_sat"], label=lbl, color=c, lw=LINE_LW)
        if "loss" in h:
            axes[1].plot(h["loss"], label=lbl, color=c, lw=LINE_LW)

    axes[0].axhline(0.95, color="gray", ls="--", lw=1.2, alpha=0.7, label="α = 0.95")
    axes[0].set_ylabel("P(φ)", fontsize=LABEL_FS, fontweight="bold")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(fontsize=LEGEND_FS)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(labelsize=TICK_FS)
    axes[0].set_title("Optimisation Convergence", fontsize=TITLE_FS, fontweight="bold")

    axes[1].set_ylabel("Loss", fontsize=LABEL_FS, fontweight="bold")
    axes[1].legend(fontsize=LEGEND_FS)
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(labelsize=TICK_FS)
    axes[1].set_xlabel("Iteration", fontsize=LABEL_FS, fontweight="bold")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=SAVE_DPI, bbox_inches="tight")
    return fig
