"""RQ1 figure: validation curves of the best config, mean +/- std across seeds."""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from optbench.analysis.plots import ensure_parent  # noqa: E402


def plot_val_curves(out: dict, save_path: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for opt, d in out.items():
        seeds = d["seeds"]
        if not seeds:
            continue
        length = min(len(r.history) for r in seeds)
        if length == 0:
            continue
        steps = [seeds[0].history[i]["step"] for i in range(length)]
        mat = np.array([[r.history[i]["val_loss"] for i in range(length)] for r in seeds])
        mean, std = mat.mean(axis=0), mat.std(axis=0)
        ax.plot(steps, mean, label=opt)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel("step")
    ax.set_ylabel("validation loss")
    ax.set_title("Best-config validation curves (mean ± std over seeds)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ensure_parent(save_path), dpi=150)
    plt.close(fig)
    return save_path
