"""Random directions for landscape slices, with filter normalization (Li et al., 2018).

The methodological point of RQ3 lives here. A raw random direction has a scale unrelated
to the local weight magnitudes, so a slice along it says nothing comparable about flatness.
Filter normalization rescales the direction *per parameter tensor* to match that tensor's
norm, which is what makes flatness comparisons between solutions meaningful. The naive
(globally normalized) direction is kept as the deliberately-misleading baseline.
"""

from __future__ import annotations

import torch


def random_direction(params, generator=None) -> list[torch.Tensor]:
    """Gaussian noise shaped like ``params`` (a raw, unnormalized direction)."""
    return [torch.randn(p.shape, generator=generator, dtype=torch.float32) for p in params]


def filter_normalize(direction, params, ignore_dims_lt: int = 2) -> list[torch.Tensor]:
    """Filter normalization (Li et al., 2018), applied per parameter tensor.

    For each tensor ``d_i`` corresponding to parameter ``theta_i`` with dim >=
    ``ignore_dims_lt``, rescale so ``||d_i|| == ||theta_i||`` (Frobenius norm of the whole
    tensor -- the standard adaptation for non-convolutional weights, where each weight
    matrix is treated as one filter group). Low-dimensional parameters (biases, norm
    scales; dim < ignore_dims_lt) get a zero direction, as Li et al. recommend, so they
    cannot dominate the slice.
    """
    out = []
    for d, p in zip(direction, params):
        if p.dim() < ignore_dims_lt:
            out.append(torch.zeros_like(d))
            continue
        pn, dn = p.norm(), d.norm()
        if dn.item() == 0.0 or pn.item() == 0.0:
            out.append(torch.zeros_like(d))
        else:
            out.append(d * (pn / dn))
    return out


def naive_normalize(direction, params) -> list[torch.Tensor]:
    """Global normalization: scale the whole direction so its total (concatenated) norm
    equals the total parameter norm. No per-tensor matching -- this is the baseline whose
    flatness reading is misleading, and the contrast against ``filter_normalize`` is the
    RQ3 result."""
    total_p = torch.sqrt(sum((p.float() ** 2).sum() for p in params))
    total_d = torch.sqrt(sum((d.float() ** 2).sum() for d in direction))
    scale = 0.0 if total_d.item() == 0.0 else (total_p / total_d).item()
    return [d * scale for d in direction]
