"""Two identically-seeded CPU runs must produce identical loss trajectories."""

from __future__ import annotations

import torch

from optbench.builders import build_data, build_model
from optbench.config.schema import BudgetConfig, WorkloadConfig
from optbench.reproducibility.seeding import set_global_seed
from optbench.training.trainer import Trainer


def _run_once():
    set_global_seed(123)
    w = WorkloadConfig(
        name="t", task="regression", model_kind="toy_mlp",
        micro_batch_size=32, extras={"in_dim": 8, "hidden": 16},
    )
    data = build_data(w, seed=123)
    model = build_model(w, data.meta, seed=123)
    trainer = Trainer(model, data, device="cpu")
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    res = trainer.fit(opt, BudgetConfig(max_steps=60, eval_every_steps=20))
    return [h["train_loss"] for h in res.history], res.best_val


def test_determinism():
    a_hist, a_best = _run_once()
    b_hist, b_best = _run_once()
    assert a_hist == b_hist
    assert a_best == b_best
