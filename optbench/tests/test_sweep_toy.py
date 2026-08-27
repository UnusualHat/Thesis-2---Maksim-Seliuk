"""Equal-budget sweep on a toy task: N trials run, selection and tunability work."""

from __future__ import annotations

from optbench.config.schema import (
    BudgetConfig,
    HParamRange,
    OptimizerSpec,
    RuntimeConfig,
    SearchSpace,
    WorkloadConfig,
)
from optbench.tuning.selection import select_best_by_val, tunability_table
from optbench.tuning.sweep_runner import run_equal_budget_sweep


def test_sweep_select_tunability():
    workload = WorkloadConfig(
        name="toy_reg", task="regression", model_kind="toy_mlp",
        micro_batch_size=32, extras={"in_dim": 8, "hidden": 16},
    )
    budget = BudgetConfig(max_steps=60, eval_every_steps=30, scheduler="none")
    space = SearchSpace(
        optimizer="adamw", n_trials=3,
        ranges=(HParamRange(name="lr", kind="log_uniform", low=1e-3, high=1e-1),),
    )
    opt_spec = OptimizerSpec(name="adamw", defaults={})
    runtime = RuntimeConfig(device="cpu", precision="fp32")

    results = run_equal_budget_sweep(
        workload, budget, opt_spec, space, runtime, sweep_seed=0, store=None
    )
    assert len(results) == 3
    assert all("lr" in r.hparams for r in results)

    best = select_best_by_val(results)
    assert best.best_val == min(r.best_val for r in results)

    rows = tunability_table(results)
    assert len(rows) == 3
    assert [r["lr"] for r in rows] == sorted(r["lr"] for r in rows)
