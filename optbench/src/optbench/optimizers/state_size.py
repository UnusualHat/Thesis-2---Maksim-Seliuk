"""Measure optimizer state memory (momentum/variance/hessian buffers) in bytes.

This is one of the practical comparison axes: Lion keeps only momentum (~half of
AdamW's state), Sophia keeps momentum + a diagonal hessian, etc.
"""

from __future__ import annotations

import torch


def optimizer_state_bytes(optimizer) -> int:
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                total += value.numel() * value.element_size()
    return total
