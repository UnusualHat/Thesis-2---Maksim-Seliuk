"""Aggregate K-seed results: mean, std, bootstrap CI, and a pairwise test.

K is small (5), so we lean on confidence intervals and report the Mann-Whitney
p-value with an explicit caveat rather than over-claiming significance.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def bootstrap_ci(values, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0):
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return (float("nan"), float("nan"))
    if v.size == 1:
        return (float(v[0]), float(v[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def mann_whitney(a, b) -> dict:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return {"U": float("nan"), "p": float("nan")}
    try:
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        return {"U": float(u), "p": float(p)}
    except ValueError:
        return {"U": float("nan"), "p": float("nan")}


def steps_to_target_from_history(history, target, metric: str = "val_loss"):
    """First eval step where `metric <= target`, derived from a run's eval history.

    Recomputing time-to-target from the stored history (rather than the value baked
    into RunResult at fit time) lets the target be recalibrated without re-running:
    the table reflects whatever `target_metric_value` the config currently declares.
    Returns None if the target is never reached (honest "did not reach").
    """
    if target is None or not history:
        return None
    for h in history:
        val = h.get(metric)
        if val is not None and val <= target:
            return h.get("step")
    return None


def aggregate_metric(results, getter) -> dict:
    vals = np.asarray([getter(r) for r in results], dtype=float)
    lo, hi = bootstrap_ci(vals)
    return {
        "mean": float(vals.mean()) if vals.size else float("nan"),
        "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
        "n": int(vals.size),
        "ci_low": lo,
        "ci_high": hi,
    }
