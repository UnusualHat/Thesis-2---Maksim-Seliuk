"""SophiaG-style optimizer (Liu et al., 2024, "Sophia").

Sophia preconditions a clipped momentum update by a diagonal curvature estimate
that is refreshed only every k steps (cheap). The original uses a Gauss-Newton-
Bartlett estimator requiring an extra sampled backward. Here we use the diagonal
of the *empirical Fisher* (an EMA of grad*grad) computed from the gradients the
trainer already produces, refreshed every k steps via ``update_hessian()``. This
keeps Sophia a pure optimizer with no extra forward/backward; the simplification
is stated explicitly in the thesis methodology (it is a Gauss-Newton diagonal).
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer


class SophiaG(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 2e-4,
        betas: tuple[float, float] = (0.965, 0.99),
        rho: float = 0.04,
        weight_decay: float = 1e-1,
        eps: float = 1e-12,
    ):
        if lr <= 0:
            raise ValueError(f"invalid lr: {lr}")
        defaults = dict(lr=lr, betas=betas, rho=rho, weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def update_hessian(self) -> None:
        """Refresh the diagonal curvature EMA from current gradients (call every k steps)."""
        for group in self.param_groups:
            beta2 = group["betas"][1]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state.setdefault(p, {})
                if "hessian" not in state:
                    state["hessian"] = torch.zeros_like(p)
                state["hessian"].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1 = group["betas"][0]
            lr, rho, wd, eps = (
                group["lr"], group["rho"], group["weight_decay"], group["eps"],
            )
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state.setdefault(p, {})
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p)
                if "hessian" not in state:
                    state["hessian"] = torch.zeros_like(p)
                exp_avg, hess = state["exp_avg"], state["hessian"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                if wd != 0:
                    p.mul_(1 - lr * wd)
                denom = (rho * hess).clamp(min=eps)
                ratio = (exp_avg.abs() / denom).clamp(max=1.0)  # Sophia clips per-coord update
                p.addcmul_(exp_avg.sign(), ratio, value=-lr)
        return loss
