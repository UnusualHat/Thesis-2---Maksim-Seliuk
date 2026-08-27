"""LR sweep produces the right number of points, in range, ascending, deterministic."""

from __future__ import annotations

from optbench.config.schema import HParamRange, SearchSpace
from optbench.tuning.search_space import sample_trials


def _space(n=5, low=1e-4, high=1e-1):
    return SearchSpace(
        optimizer="adamw", n_trials=n,
        ranges=(HParamRange(name="lr", kind="log_uniform", low=low, high=high),),
    )


def test_grid_count_range_and_order():
    trials = sample_trials(_space(n=5, low=1e-4, high=1e-1))
    assert len(trials) == 5
    lrs = [t["lr"] for t in trials]
    assert lrs[0] == 1e-4 and abs(lrs[-1] - 1e-1) < 1e-9
    assert all(lrs[i] < lrs[i + 1] for i in range(len(lrs) - 1))
    assert all(1e-4 <= v <= 1e-1 + 1e-9 for v in lrs)


def test_deterministic():
    assert sample_trials(_space()) == sample_trials(_space())
