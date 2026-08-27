"""Learning-rate schedule fixed to the shared step budget (same for all optimizers)."""

from __future__ import annotations

import math

from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(optimizer, budget):
    """Return a LambdaLR (warmup + cosine/linear), or None when scheduler == 'none'."""
    if budget.scheduler == "none":
        return None
    total = max(1, budget.max_steps)
    warmup = int(budget.warmup_ratio * total)

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        if budget.scheduler == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        if budget.scheduler == "linear":
            return max(0.0, 1.0 - progress)
        return 1.0

    return LambdaLR(optimizer, lr_lambda)
