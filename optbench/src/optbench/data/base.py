"""DataModule: a task plus its train/val/test loaders and metadata."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DataModule:
    task: str                 # "regression" | "causal_lm"
    train_loader: object
    val_loader: object
    test_loader: object
    meta: dict = field(default_factory=dict)
