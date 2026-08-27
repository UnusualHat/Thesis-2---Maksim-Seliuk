"""Capture the runtime environment for reproducibility metadata (saved per run)."""

from __future__ import annotations

import platform


def capture_env() -> dict:
    import numpy
    import torch

    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        import transformers

        env["transformers"] = transformers.__version__
    except Exception:
        pass
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
        env["cuda_capability"] = list(torch.cuda.get_device_capability(0))
        env["cuda"] = torch.version.cuda
    return env
