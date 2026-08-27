"""Equal-budget LR sweep for one optimizer.

The fairness invariant lives here: every trial uses the SAME workload, budget and
sweep_seed (so identical init and data order); only the learning rate changes.
Each trial is saved immediately, so a re-run skips finished trials (resumable).
"""

from __future__ import annotations

from optbench.logging.artifacts import load_json
from optbench.run_one import train_one
from optbench.serialize import result_from_dict, result_to_dict
from optbench.tuning.search_space import sample_trials


def run_equal_budget_sweep(
    workload, budget, opt_spec, space, runtime, *,
    sweep_seed=0, target_value=None, eval_metric="val_loss",
    store=None, run_group="rq1", log=None,
):
    trials = sample_trials(space, seed=sweep_seed)
    sweep_dir = store.run_dir(run_group, opt_spec.name, "sweep") if store is not None else None
    results = []
    for i, hp in enumerate(trials):
        path = (sweep_dir / f"trial{i:02d}.json") if sweep_dir is not None else None
        if path is not None and path.exists():
            res = result_from_dict(load_json(path))
            results.append(res)
            if log:
                log(f"[{opt_spec.name}] trial {i+1}/{len(trials)} lr={hp['lr']:.2e} (cached) best_val={res.best_val:.4f}")
            continue
        res = train_one(
            workload, budget, opt_spec, hp, runtime,
            seed=sweep_seed, target_value=target_value, eval_metric=eval_metric,
        )
        results.append(res)
        if path is not None:
            store.save_json(path, result_to_dict(res))
        if log:
            log(f"[{opt_spec.name}] trial {i+1}/{len(trials)} lr={hp['lr']:.2e} best_val={res.best_val:.4f}")
    return results
