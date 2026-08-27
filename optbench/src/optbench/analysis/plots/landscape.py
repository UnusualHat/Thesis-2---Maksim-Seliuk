"""RQ3 figures: filter-normalized 2D loss slices, the naive-vs-filter comparison, and
1D interpolation barriers between optimizers' solutions."""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from optbench.analysis.plots import ensure_parent  # noqa: E402


def plot_surface_2d(coords, Z, save_path: str, title: str = "Loss surface") -> str:
    X, Y = np.meshgrid(coords, coords, indexing="ij")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cf = ax.contourf(X, Y, Z, levels=30, cmap="viridis")
    ax.contour(X, Y, Z, levels=12, colors="k", linewidths=0.3, alpha=0.5)
    ax.plot(0, 0, "r*", markersize=12, label="solution")
    ax.set_xlabel("direction 1"); ax.set_ylabel("direction 2")
    ax.set_title(title); ax.legend(loc="upper right")
    fig.colorbar(cf, ax=ax, label="loss")
    fig.tight_layout()
    fig.savefig(ensure_parent(save_path), dpi=150)
    plt.close(fig)
    return save_path


def plot_compare_surfaces(panels, save_path: str, suptitle: str = "") -> str:
    """panels: list of (title, coords, Z). Side-by-side contours, each on its own scale."""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 4.3))
    if n == 1:
        axes = [axes]
    for ax, (title, coords, Z) in zip(axes, panels):
        X, Y = np.meshgrid(coords, coords, indexing="ij")
        cf = ax.contourf(X, Y, Z, levels=30, cmap="viridis")
        ax.contour(X, Y, Z, levels=12, colors="k", linewidths=0.3, alpha=0.5)
        ax.plot(0, 0, "r*", markersize=12)
        ax.set_xlabel("direction 1"); ax.set_ylabel("direction 2")
        ax.set_title(title)
        fig.colorbar(cf, ax=ax, label="loss")
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(ensure_parent(save_path), dpi=150)
    plt.close(fig)
    return save_path


def plot_interpolation(ts, series: dict, save_path: str,
                       title: str = "1D interpolation between solutions") -> str:
    """series: {label: ys}. Vertical guides mark the two endpoints (t=0, t=1)."""
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    for label, ys in series.items():
        ax.plot(ts, ys, marker="o", markersize=3, label=label)
    ax.axvline(0.0, color="gray", ls="--", lw=0.8)
    ax.axvline(1.0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("interpolation coefficient t  (0 = solution A, 1 = solution B)")
    ax.set_ylabel("loss")
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ensure_parent(save_path), dpi=150)
    plt.close(fig)
    return save_path
