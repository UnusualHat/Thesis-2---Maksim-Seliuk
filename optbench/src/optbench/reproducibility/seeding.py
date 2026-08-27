"""Global seed control. Multi-seed runs + CIs are how we handle the residual
nondeterminism of some CUDA kernels; this function pins everything we can.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Required for some deterministic CUDA matmul paths.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            # Older/newer torch may not support warn_only; determinism is best-effort.
            pass
