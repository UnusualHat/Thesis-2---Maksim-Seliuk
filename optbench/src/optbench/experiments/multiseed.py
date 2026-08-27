"""Re-run the best config with K seeds to get mean +/- spread and CIs (RQ1)."""

from __future__ import annotations

from optbench.logging.artifacts import load_json
from optbench.run_one import train_one
from optbench.serialize import result_from_dict, result_to_dict


def run_multiseed(
    workload, budget, opt_spec, best_hparams, runtime, seeds, *,
    store=None, target_value=None, eval_metric="val_loss", run_group="rq1", log=None,
):
    seed_dir = store.run_dir(run_group, opt_spec.name, "seeds") if store is not None else None
    results = []
    for seed in seeds:
        path = (seed_dir / f"seed{seed}.json") if seed_dir is not None else None
        if path is not None and path.exists():
            res = result_from_dict(load_json(path))
            results.append(res)
            if log:
                log(f"[{opt_spec.name}] seed {seed} (cached) best_val={res.best_val:.4f}")
            continue
        res = train_one(
            workload, budget, opt_spec, best_hparams, runtime,
            seed=seed, target_value=target_value, eval_metric=eval_metric,
        )
        results.append(res)
        if path is not None:
            store.save_json(path, result_to_dict(res))
        if log:
            log(f"[{opt_spec.name}] seed {seed} best_val={res.best_val:.4f}")
    return results
