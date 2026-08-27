"""Optimizer factory. One entry point so the sweep/runner builds any optimizer
from a name + hyperparameters, spanning the inclusion hierarchy (SGD -> AdamW)
plus modern methods (Lion sign-momentum, Sophia diagonal-curvature).
"""

from __future__ import annotations

import torch

from optbench.optimizers.sophia import SophiaG

# Optimizers that maintain a diagonal curvature estimate refreshed every k steps.
HESSIAN_OPTIMIZERS = {"sophia"}


def make_optimizer(name: str, params, lr: float, **kwargs):
    key = name.lower()
    if key in ("sgd_momentum", "sgd"):
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=kwargs.get("momentum", 0.9),
            nesterov=kwargs.get("nesterov", True),
            weight_decay=kwargs.get("weight_decay", 0.0),
        )
    if key == "adamw":
        return torch.optim.AdamW(
            params,
            lr=lr,
            betas=tuple(kwargs.get("betas", (0.9, 0.999))),
            eps=kwargs.get("eps", 1e-8),
            weight_decay=kwargs.get("weight_decay", 0.01),
        )
    if key == "lion":
        from lion_pytorch import Lion

        return Lion(
            params,
            lr=lr,
            betas=tuple(kwargs.get("betas", (0.9, 0.99))),
            weight_decay=kwargs.get("weight_decay", 0.0),
        )
    if key == "sophia":
        return SophiaG(
            params,
            lr=lr,
            betas=tuple(kwargs.get("betas", (0.965, 0.99))),
            rho=kwargs.get("rho", 0.04),
            weight_decay=kwargs.get("weight_decay", 0.1),
        )
    raise ValueError(f"unknown optimizer: {name}")
