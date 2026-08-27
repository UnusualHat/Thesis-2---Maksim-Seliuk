"""Every optimizer in the factory must (a) construct, (b) reduce loss on the toy
task, and (c) report non-zero state memory when it is stateful.
"""

from __future__ import annotations

import pytest

from optbench.builders import build_data, build_model
from optbench.config.schema import BudgetConfig, WorkloadConfig
from optbench.optimizers.factory import make_optimizer
from optbench.optimizers.state_size import optimizer_state_bytes
from optbench.reproducibility.seeding import set_global_seed
from optbench.training.trainer import Trainer

OPTIMIZERS = [
    ("sgd_momentum", 5e-2, {}),
    ("adamw", 1e-2, {}),
    ("lion", 3e-3, {}),
    ("sophia", 1e-2, {"rho": 0.05}),
]


@pytest.mark.parametrize("name,lr,kwargs", OPTIMIZERS)
def test_optimizer_reduces_loss(name, lr, kwargs):
    set_global_seed(0)
    w = WorkloadConfig(
        name="reg", task="regression", model_kind="toy_mlp",
        micro_batch_size=32, extras={"in_dim": 8, "hidden": 32},
    )
    data = build_data(w, seed=0)
    model = build_model(w, data.meta, seed=0)
    trainer = Trainer(model, data, device="cpu")
    init_val = trainer.evaluate()["val_loss"]
    opt = make_optimizer(name, model.parameters(), lr, **kwargs)
    res = trainer.fit(
        opt, BudgetConfig(max_steps=200, eval_every_steps=100), opt_name=name
    )
    assert res.best_val < init_val * 0.7, f"{name} failed to reduce loss"
    # All four maintain state (momentum / variance / hessian).
    assert optimizer_state_bytes(opt) > 0
