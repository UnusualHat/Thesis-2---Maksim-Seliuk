"""Small MLP for the toy regression task (and full-Hessian curvature ground truth)."""

from __future__ import annotations

import torch.nn as nn


class ToyMLP(nn.Module):
    def __init__(self, in_dim: int = 8, hidden: int = 32, out_dim: int = 1, depth: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
