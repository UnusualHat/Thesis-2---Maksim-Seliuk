"""RQ3: loss-landscape visualization on a toy char-level Transformer.

Trains 2-3 optimizers from the SAME initialization and data order (the fairness
invariant, reused here), then produces three figures for the Findings chapter:

  A. rq3_filternorm_slices.png   -- filter-normalized 2D slices of each solution
  B. rq3_naive_vs_filternorm.png -- same solution, naive vs filter-normalized directions
                                    (the "naive visualization misleads" result)
  C. rq3_interpolation.png       -- 1D interpolation between two solutions (basin / barrier)

Toy-only by design (project design note): full-data slices are cheap and meaningful here.
Run: .venv\\Scripts\\python.exe scripts\\run_landscape.py
"""

from __future__ import annotations

import numpy as np

from optbench.builders import build_experiment_unit
from optbench.config.schema import BudgetConfig, RuntimeConfig, WorkloadConfig
from optbench.analysis.plots.landscape import (
    plot_compare_surfaces,
    plot_interpolation,
    plot_surface_2d,
)
from optbench.landscape.directions import filter_normalize, naive_normalize, random_direction
from optbench.landscape.params import get_params, set_params_
from optbench.landscape.surface import loss_interpolation_1d, loss_surface_2d, make_eval_loss
from optbench.optimizers.factory import make_optimizer
from optbench.reproducibility.seeding import set_global_seed
from optbench.training.scheduler import build_scheduler
from optbench.training.trainer import Trainer

import torch

SEED = 0
FIG_DIR = "results/figures"

WORKLOAD = WorkloadConfig(
    name="toy_char_landscape", task="causal_lm", model_kind="toy_transformer",
    seq_len=48, micro_batch_size=16,
    extras={"d_model": 64, "nhead": 2, "num_layers": 2},
)
BUDGET = BudgetConfig(max_steps=400, eval_every_steps=200, warmup_ratio=0.05,
                      scheduler="cosine", grad_clip=1.0)
RUNTIME = RuntimeConfig(device="cpu", precision="fp32")

# Same toy LRs are fine here -- we want trained solutions to slice, not a fair benchmark.
OPTS = [
    ("adamw", 3e-3, {"betas": (0.9, 0.999), "weight_decay": 0.01}),
    ("sgd_momentum", 0.2, {"momentum": 0.9, "nesterov": True}),
    ("lion", 1e-3, {"betas": (0.9, 0.99), "weight_decay": 0.0}),
]


def train_solution(opt_name, lr, defaults):
    set_global_seed(SEED)
    model, data = build_experiment_unit(WORKLOAD, seed=SEED)  # same init + data order
    trainer = Trainer(model, data, device=RUNTIME.device, precision=RUNTIME.precision,
                      grad_clip=BUDGET.grad_clip)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = make_optimizer(opt_name, params, lr, **defaults)
    scheduler = build_scheduler(optimizer, BUDGET)
    res = trainer.fit(optimizer, BUDGET, scheduler=scheduler, opt_name=opt_name, seed=SEED)
    return model, data, res.best_val


def main():
    solutions = {}
    host_model = host_data = None
    for name, lr, defaults in OPTS:
        model, data, best = train_solution(name, lr, defaults)
        solutions[name] = get_params(model)
        print(f"[{name}] trained: best_val={best:.4f}")
        if host_model is None:
            host_model, host_data = model, data  # evaluate every slice on one host

    eval_loss = make_eval_loss(host_model, host_data.val_loader, host_data.task, "toy_transformer")
    gen = torch.Generator().manual_seed(123)
    coords = np.linspace(-0.5, 0.5, 25)

    # --- Figure A: filter-normalized slice per optimizer (fair, comparable) ---
    panels = []
    for name in solutions:
        theta = solutions[name]
        dx = filter_normalize(random_direction(theta, gen), theta)
        dy = filter_normalize(random_direction(theta, gen), theta)
        _, Z = loss_surface_2d(host_model, eval_loss, theta, dx, dy, coords)
        panels.append((name, coords, Z))
    fa = plot_compare_surfaces(panels, f"{FIG_DIR}/rq3_filternorm_slices.png",
                               suptitle="Filter-normalized loss slices (Li et al., 2018)")

    # --- Figure B: naive vs filter-normalized directions on the SAME solution ---
    theta = solutions["adamw"]
    raw_x = random_direction(theta, gen)
    raw_y = random_direction(theta, gen)
    fn_x, fn_y = filter_normalize(raw_x, theta), filter_normalize(raw_y, theta)
    nv_x, nv_y = naive_normalize(raw_x, theta), naive_normalize(raw_y, theta)
    _, Z_fn = loss_surface_2d(host_model, eval_loss, theta, fn_x, fn_y, coords)
    _, Z_nv = loss_surface_2d(host_model, eval_loss, theta, nv_x, nv_y, coords)
    fb = plot_compare_surfaces(
        [("naive (global) directions", coords, Z_nv),
         ("filter-normalized directions", coords, Z_fn)],
        f"{FIG_DIR}/rq3_naive_vs_filternorm.png",
        suptitle="Same AdamW solution, same total step size -- only direction normalization differs",
    )

    # --- Figure C: 1D interpolation between two optimizers' solutions ---
    ts = np.linspace(-0.25, 1.25, 31)
    _, ys = loss_interpolation_1d(host_model, eval_loss, solutions["adamw"],
                                  solutions["sgd_momentum"], ts)
    fc = plot_interpolation(ts, {"AdamW -> SGD+momentum": ys},
                            f"{FIG_DIR}/rq3_interpolation.png",
                            title="1D interpolation between optimizer solutions")
    set_params_(host_model, solutions["adamw"])  # leave host clean

    print(f"\nfigures -> {fa}\n           {fb}\n           {fc}")


if __name__ == "__main__":
    main()
