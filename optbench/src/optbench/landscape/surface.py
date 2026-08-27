"""Evaluate the loss on 2D slices and 1D interpolations of parameter space.

All functions restore the model's original parameters before returning, so the caller's
trained solution is never left perturbed.
"""

from __future__ import annotations

import numpy as np
import torch

from optbench.eval.metrics import evaluate
from optbench.landscape.params import set_offset_, set_params_


def make_eval_loss(model, loader, task: str, model_kind: str, device="cpu", autocast=None):
    """A zero-arg closure returning mean loss over ``loader`` at the model's current params."""

    def _f() -> float:
        return float(evaluate(model, loader, task, model_kind, device, autocast)["val_loss"])

    return _f


@torch.no_grad()
def loss_surface_2d(model, eval_loss, theta, dir_x, dir_y, coords):
    """Loss on the grid ``theta + a*dir_x + b*dir_y`` for a, b in ``coords``.

    Returns (coords, Z) where ``Z[i, j]`` is the loss at ``(coords[i], coords[j])``.
    """
    coords = np.asarray(coords, dtype=float)
    Z = np.empty((len(coords), len(coords)), dtype=float)
    for i, a in enumerate(coords):
        for j, b in enumerate(coords):
            set_offset_(model, theta, [dir_x, dir_y], [float(a), float(b)])
            Z[i, j] = eval_loss()
    set_params_(model, theta)  # restore
    return coords, Z


@torch.no_grad()
def loss_interpolation_1d(model, eval_loss, theta_a, theta_b, ts):
    """Loss along the line ``(1 - t) * theta_a + t * theta_b`` for t in ``ts``.

    t=0 is solution A, t=1 is solution B; t outside [0, 1] extrapolates. A bump between
    the endpoints is a loss barrier (the two solutions sit in different basins).
    """
    delta = [b - a for a, b in zip(theta_a, theta_b)]
    ts = np.asarray(ts, dtype=float)
    ys = np.empty(len(ts), dtype=float)
    for k, t in enumerate(ts):
        set_offset_(model, theta_a, [delta], [float(t)])
        ys[k] = eval_loss()
    set_params_(model, theta_a)  # restore
    return ts, ys
