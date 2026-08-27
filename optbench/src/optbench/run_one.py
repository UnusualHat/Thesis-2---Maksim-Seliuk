"""Run a single training job under a given workload/budget/optimizer/seed.

Shared by the sweep (vary LR, fix seed) and the multi-seed runner (fix best LR,
vary seed). Frees GPU memory after each run so long sweeps don't accumulate.
"""

from __future__ import annotations

import torch

from optbench.builders import build_experiment_unit
from optbench.optimizers.factory import make_optimizer
from optbench.reproducibility.seeding import set_global_seed
from optbench.training.scheduler import build_scheduler
from optbench.training.trainer import Trainer


def _trainable(model):
    return [p for p in model.parameters() if p.requires_grad]


def train_one(
    workload, budget, opt_spec, hparams, runtime, *,
    seed, target_value=None, eval_metric="val_loss",
):
    set_global_seed(seed, deterministic=runtime.deterministic)
    model, data = build_experiment_unit(workload, seed=seed)
    trainer = Trainer(
        model, data, device=runtime.device, precision=runtime.precision, grad_clip=budget.grad_clip
    )
    hp = dict(hparams)
    lr = hp.pop("lr")
    optimizer = make_optimizer(opt_spec.name, _trainable(model), lr, **{**opt_spec.defaults, **hp})
    scheduler = build_scheduler(optimizer, budget)
    result = trainer.fit(
        optimizer, budget,
        target_value=target_value, eval_metric=eval_metric, scheduler=scheduler,
        opt_name=opt_spec.name, seed=seed, hparams=dict(hparams),
    )
    del model, data, trainer, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
