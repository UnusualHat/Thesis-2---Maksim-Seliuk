"""Pick the best trial and extract the tunability curve from a sweep."""

from __future__ import annotations

from optbench.analysis.aggregate import steps_to_target_from_history


def select_best_by_val(results, mode: str = "min"):
    """Best trial by RunResult.best_val (which already tracks the eval metric)."""
    if not results:
        raise ValueError("no results to select from")
    return (min if mode == "min" else max)(results, key=lambda r: r.best_val)


def tunability_table(results, target=None, metric: str = "val_loss") -> list[dict]:
    """Rows {lr, best_val, steps_to_target} sorted by lr, for the quality-vs-LR plot.

    If `target` is given, steps_to_target is recomputed from each trial's history at
    that target; otherwise the value baked into the RunResult is used (back-compat).
    """
    rows = [
        {
            "lr": r.hparams.get("lr"),
            "best_val": r.best_val,
            "steps_to_target": (
                steps_to_target_from_history(r.history, target, metric)
                if target is not None
                else r.steps_to_target
            ),
        }
        for r in results
    ]
    rows.sort(key=lambda x: (x["lr"] is None, x["lr"]))
    return rows
