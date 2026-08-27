"""RQ2 figure: validation quality versus learning rate (one line per optimizer)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from optbench.analysis.plots import ensure_parent  # noqa: E402
from optbench.tuning.selection import tunability_table  # noqa: E402


def plot_tunability(out: dict, save_path: str, zoom_span: float = 0.20) -> str:
    """Two panels: full range (left) shows divergence at bad LRs; the zoomed-Y panel
    (right) reveals the basin shape near the optima, which the full scale compresses
    into an unreadable band when one optimizer blows up."""
    series = {}
    for opt, d in out.items():
        rows = tunability_table(d["sweep"])
        xs = [r["lr"] for r in rows if r["lr"] is not None]
        ys = [r["best_val"] for r in rows if r["lr"] is not None]
        series[opt] = (xs, ys)

    ymin = min(min(ys) for _, ys in series.values() if ys)

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(12, 4.5))
    for opt, (xs, ys) in series.items():
        ax_full.plot(xs, ys, marker="o", markersize=4, label=opt)
        ax_zoom.plot(xs, ys, marker="o", markersize=4, label=opt)
    for ax in (ax_full, ax_zoom):
        ax.set_xscale("log")
        ax.set_xlabel("learning rate")
        ax.set_ylabel("best validation loss")
        ax.grid(True, which="both", alpha=0.3)
    ax_full.set_title("Full range (divergence at bad LR)")
    ax_full.legend()
    ax_zoom.set_ylim(ymin - 0.005, ymin + zoom_span)
    ax_zoom.set_title(f"Zoom near optima (y ≤ {ymin + zoom_span:.2f})")
    fig.suptitle("Tunability: quality vs learning rate")
    fig.tight_layout()
    fig.savefig(ensure_parent(save_path), dpi=150)
    plt.close(fig)
    return save_path
