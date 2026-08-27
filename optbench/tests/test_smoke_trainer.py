"""Smoke tests: the trainer must actually reduce loss on both toy tasks."""

from __future__ import annotations

import torch

from optbench.builders import build_data, build_model
from optbench.config.schema import BudgetConfig, WorkloadConfig
from optbench.reproducibility.seeding import set_global_seed
from optbench.training.trainer import Trainer


def _train(workload, *, max_steps, eval_every, lr):
    set_global_seed(0)
    data = build_data(workload, seed=0)
    model = build_model(workload, data.meta, seed=0)
    trainer = Trainer(model, data, device="cpu", precision="fp32")
    init_val = trainer.evaluate()["val_loss"]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    res = trainer.fit(
        opt, BudgetConfig(max_steps=max_steps, eval_every_steps=eval_every), opt_name="adam"
    )
    return init_val, res


def test_regression_loss_decreases():
    w = WorkloadConfig(
        name="toy_reg", task="regression", model_kind="toy_mlp",
        micro_batch_size=32, extras={"in_dim": 8, "hidden": 32},
    )
    init_val, res = _train(w, max_steps=150, eval_every=50, lr=1e-2)
    assert res.best_val < init_val * 0.5
    assert len(res.history) >= 1


def test_causal_lm_loss_decreases():
    w = WorkloadConfig(
        name="toy_lm", task="causal_lm", model_kind="toy_transformer",
        seq_len=64, micro_batch_size=16,
    )
    init_val, res = _train(w, max_steps=120, eval_every=40, lr=3e-3)
    assert res.best_val < init_val
    assert "perplexity" in res.history[-1]
