"""Turn a SearchSpace into concrete trials.

For the LR-only sweep we use a deterministic log-spaced grid over the declared
[low, high] range. Even coverage of the LR axis gives both a clean tunability
curve (RQ2) and an equal tuning budget per optimizer (RQ1).
"""

from __future__ import annotations

import math

import numpy as np

from optbench.config.schema import HParamRange, SearchSpace


def _lr_range(space: SearchSpace) -> HParamRange:
    for r in space.ranges:
        if r.name == "lr":
            return r
    raise ValueError(f"search space for '{space.optimizer}' has no 'lr' range")


def sample_trials(space: SearchSpace, seed: int = 0) -> list[dict]:
    """Return n_trials hyperparameter dicts. LR varies on a log grid; any
    kind='fixed' ranges contribute constants. ``seed`` is accepted for API
    symmetry (the grid is deterministic)."""
    lr = _lr_range(space)
    n = max(1, space.n_trials)
    fixed = {r.name: r.value for r in space.ranges if r.kind == "fixed"}

    if lr.kind in ("log_uniform", "log_grid"):
        if n == 1:
            lrs = [lr.low]
        else:
            lrs = np.logspace(math.log10(lr.low), math.log10(lr.high), n)
    elif lr.kind == "uniform":
        lrs = np.linspace(lr.low, lr.high, n)
    else:
        raise ValueError(f"unsupported lr kind for grid: {lr.kind}")

    return [{"lr": float(v), **fixed} for v in lrs]
