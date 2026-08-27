"""RQ1/RQ2 orchestrator: for each optimizer, sweep LR, select best, re-run K seeds."""

from __future__ import annotations

from optbench.experiments.multiseed import run_multiseed
from optbench.tuning.selection import select_best_by_val
from optbench.tuning.sweep_runner import run_equal_budget_sweep


def run_rq1(cfg, store, log=print):
    out = {}
    for opt_spec in cfg.optimizers:
        space = cfg.search_spaces[opt_spec.name]
        sweep = run_equal_budget_sweep(
            cfg.workload, cfg.budget, opt_spec, space, cfg.runtime,
            sweep_seed=cfg.seeds.sweep_seed, target_value=cfg.target_metric_value,
            eval_metric=cfg.workload.eval_metric, store=store, run_group=cfg.name, log=log,
        )
        best = select_best_by_val(sweep)
        if log:
            log(f"[{opt_spec.name}] BEST lr={best.hparams.get('lr'):.2e} best_val={best.best_val:.4f}")
        seeds = run_multiseed(
            cfg.workload, cfg.budget, opt_spec, best.hparams, cfg.runtime, cfg.seeds.final_seeds,
            store=store, target_value=cfg.target_metric_value,
            eval_metric=cfg.workload.eval_metric, run_group=cfg.name, log=log,
        )
        out[opt_spec.name] = {"sweep": sweep, "best": best, "seeds": seeds}
    return out
