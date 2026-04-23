"""Covariance sweep plots: P(φ) vs σ₀ and P(φ) vs D for OL vs CL."""

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_covariance_sweep(sigma0_rows, D_rows, label, save_dir):
    """Plot P(φ) sweep over Σ₀ and D for open-loop vs closed-loop.

    Args:
        sigma0_rows: list of dicts with keys:
            sigma, p_ol_analytic, p_cl_analytic, p_ol_mc, p_cl_mc
        D_rows: same structure but with key 'd' instead of 'sigma'
        label: scenario label string (used in file names)
        save_dir: directory to save outputs

    Returns:
        matplotlib Figure
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _draw_sweep_ax(
        axes[0], sigma0_rows, x_key="sigma",
        xlabel="Initial uncertainty  σ₀  (std dev)",
        title="P(φ) vs Initial Uncertainty  Σ₀",
    )
    _draw_sweep_ax(
        axes[1], D_rows, x_key="d",
        xlabel="Process noise  D  (std dev)",
        title="P(φ) vs Process Noise  D",
    )

    fig.suptitle(f"{label} — Covariance Sweep", fontsize=13, fontweight="bold")
    plt.tight_layout()

    png_path = Path(save_dir) / f"{label.lower().replace(' ', '_')}_covariance_sweep.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")

    _export_csv(sigma0_rows, D_rows, save_dir, label)

    return fig


def _draw_sweep_ax(ax, rows, x_key, xlabel, title):
    """Draw one sweep subplot."""
    if not rows:
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("P(φ)")
        return

    xs = np.array([r[x_key] for r in rows])
    p_ol_a = np.array([r["p_ol_analytic"] for r in rows])
    p_cl_a = np.array([r["p_cl_analytic"] for r in rows])

    ax.plot(xs, p_ol_a, color="#1f77b4", lw=2, label="Open-loop analytic")
    ax.plot(xs, p_cl_a, color="#d62728", lw=2, label="Cov-steering analytic")

    has_mc = rows[0].get("p_ol_mc") is not None
    if has_mc:
        p_ol_mc = np.array([r["p_ol_mc"] for r in rows])
        p_cl_mc = np.array([r["p_cl_mc"] for r in rows])
        ax.plot(xs, p_ol_mc, color="#1f77b4", lw=1.5, ls="--",
                marker="o", ms=5, label="Open-loop empirical (MC)")
        ax.plot(xs, p_cl_mc, color="#d62728", lw=1.5, ls="--",
                marker="s", ms=5, label="Cov-steering empirical (MC)")
        ax.fill_between(xs, p_ol_a, p_ol_mc, alpha=0.10, color="#1f77b4")
        ax.fill_between(xs, p_cl_a, p_cl_mc, alpha=0.10, color="#d62728")

    ax.axhline(0.95, color="gray", lw=1, ls=":", alpha=0.7, label="α = 0.95")
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("P(φ)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def _export_csv(sigma0_rows, D_rows, save_dir, label):
    """Write sweep results to CSV."""
    csv_path = Path(save_dir) / f"{label.lower().replace(' ', '_')}_covariance_sweep.csv"
    fieldnames = ["sweep_type", "value", "p_ol_analytic", "p_cl_analytic", "p_ol_mc", "p_cl_mc"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sigma0_rows:
            writer.writerow({
                "sweep_type": "sigma0",
                "value": r["sigma"],
                "p_ol_analytic": r["p_ol_analytic"],
                "p_cl_analytic": r["p_cl_analytic"],
                "p_ol_mc": r.get("p_ol_mc", ""),
                "p_cl_mc": r.get("p_cl_mc", ""),
            })
        for r in D_rows:
            writer.writerow({
                "sweep_type": "D",
                "value": r["d"],
                "p_ol_analytic": r["p_ol_analytic"],
                "p_cl_analytic": r["p_cl_analytic"],
                "p_ol_mc": r.get("p_ol_mc", ""),
                "p_cl_mc": r.get("p_cl_mc", ""),
            })
