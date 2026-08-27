"""Read/write a model's parameters as an ordered list of tensors.

A "point" theta and a "direction" d are both represented as a list of tensors aligned
to ``model.parameters()`` order, which keeps the slicing/interpolation code shape-safe
without ever flattening to one giant vector.
"""

from __future__ import annotations

import torch


def get_params(model) -> list[torch.Tensor]:
    """Detached clone of every parameter, in module order (a landscape point)."""
    return [p.detach().clone() for p in model.parameters()]


@torch.no_grad()
def set_params_(model, point) -> None:
    for p, v in zip(model.parameters(), point):
        p.copy_(v)


@torch.no_grad()
def set_offset_(model, base, dirs, coeffs) -> None:
    """Set params to ``base + sum_k coeffs[k] * dirs[k]`` (in place).

    ``base`` is a point (list of tensors); ``dirs`` is a list of directions (each a list
    of tensors aligned to base); ``coeffs`` the matching scalar for each direction.
    """
    for i, p in enumerate(model.parameters()):
        val = base[i].clone()
        for c, d in zip(coeffs, dirs):
            if c != 0.0:
                val = val + c * d[i]
        p.copy_(val)
