"""Bootstrap CI and Mann-Whitney behave sanely."""

from __future__ import annotations

from optbench.analysis.aggregate import (
    aggregate_metric,
    bootstrap_ci,
    mann_whitney,
    steps_to_target_from_history,
)

_HIST = [
    {"step": 100, "val_loss": 3.40},
    {"step": 200, "val_loss": 3.25},
    {"step": 300, "val_loss": 3.18},
]


def test_bootstrap_ci_constant():
    assert bootstrap_ci([2.0, 2.0, 2.0]) == (2.0, 2.0)


def test_bootstrap_ci_brackets_mean():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    lo, hi = bootstrap_ci(vals, seed=0)
    assert lo <= 3.0 <= hi


def test_mann_whitney_separated():
    res = mann_whitney([1, 2, 3], [10, 11, 12])
    assert 0.0 <= res["p"] <= 1.0


def test_steps_to_target_first_crossing():
    # first eval step where val_loss <= target, not the best/last
    assert steps_to_target_from_history(_HIST, 3.20) == 300
    assert steps_to_target_from_history(_HIST, 3.30) == 200
    assert steps_to_target_from_history(_HIST, 3.50) == 100


def test_steps_to_target_never_reached():
    assert steps_to_target_from_history(_HIST, 3.0) is None
    assert steps_to_target_from_history([], 3.2) is None
    assert steps_to_target_from_history(_HIST, None) is None


def test_aggregate_metric():
    class R:
        def __init__(self, v):
            self.best_val = v

    agg = aggregate_metric([R(1.0), R(2.0), R(3.0)], lambda r: r.best_val)
    assert agg["n"] == 3
    assert abs(agg["mean"] - 2.0) < 1e-9
