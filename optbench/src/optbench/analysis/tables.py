"""Turn RQ1 results into the three CSV tables for the Results chapter."""

from __future__ import annotations

import os

import numpy as np

from optbench.analysis.aggregate import (
    bootstrap_ci,
    mann_whitney,
    steps_to_target_from_history,
)
from optbench.logging.artifacts import ArtifactStore
from optbench.tuning.selection import tunability_table


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def write_tables(out: dict, tables_dir: str, cfg) -> dict:
    os.makedirs(tables_dir, exist_ok=True)
    main_rows, tun_rows = [], []
    seed_best = {}

    # Time-to-target is recomputed from history at the config's current target, so the
    # target can be recalibrated without re-running (see aggregate.steps_to_target_*).
    target = cfg.target_metric_value
    metric = cfg.workload.eval_metric

    for opt, d in out.items():
        seeds = d["seeds"]
        best = d["best"]
        bvals = [r.best_val for r in seeds]
        seed_best[opt] = bvals
        lo, hi = bootstrap_ci(bvals)
        n_reached = sum(
            steps_to_target_from_history(r.history, target, metric) is not None
            for r in seeds
        )
        main_rows.append({
            "optimizer": opt,
            "best_lr": best.hparams.get("lr"),
            "val_mean": float(np.mean(bvals)) if bvals else None,
            "val_std": float(np.std(bvals, ddof=1)) if len(bvals) > 1 else 0.0,
            "val_ci_low": lo,
            "val_ci_high": hi,
            "steps_to_target_mean": _mean(
                [steps_to_target_from_history(r.history, target, metric) for r in seeds]
            ),
            "seeds_reaching_target": f"{n_reached}/{len(seeds)}",
            "peak_mem_mb": _mean([r.peak_gpu_mem_bytes / 1e6 for r in seeds]),
            "step_time_ms": _mean([r.mean_step_time_s * 1e3 for r in seeds]),
            "opt_state_mb": _mean([r.optimizer_state_bytes / 1e6 for r in seeds]),
            "n_seeds": len(bvals),
        })
        for row in tunability_table(d["sweep"], target=target, metric=metric):
            tun_rows.append({"optimizer": opt, **row})

    ref = "adamw" if "adamw" in seed_best else next(iter(seed_best))
    sig_rows = []
    for opt, vals in seed_best.items():
        if opt == ref:
            continue
        sig_rows.append({"pair": f"{ref}_vs_{opt}", **mann_whitney(seed_best[ref], vals)})

    ArtifactStore.save_csv(os.path.join(tables_dir, f"{cfg.name}_rq1_main.csv"), main_rows)
    ArtifactStore.save_csv(os.path.join(tables_dir, f"{cfg.name}_rq2_tunability.csv"), tun_rows)
    ArtifactStore.save_csv(os.path.join(tables_dir, f"{cfg.name}_rq1_significance.csv"), sig_rows)
    return {"main": main_rows, "tunability": tun_rows, "significance": sig_rows}
