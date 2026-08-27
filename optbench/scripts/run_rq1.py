"""Run the RQ1/RQ2 experiment from a YAML config: equal-budget LR sweeps over the
optimizers, select best per optimizer, K-seed reruns, then write tables + figures.

Resumable: finished trials/seeds are cached under results/runs and skipped.

Run: .venv\\Scripts\\python.exe scripts\\run_rq1.py --config configs\\experiments\\rq1_gpt2_lora.yaml
"""

from __future__ import annotations

import argparse

from optbench.analysis.plots.curves import plot_val_curves
from optbench.analysis.plots.tunability import plot_tunability
from optbench.analysis.tables import write_tables
from optbench.config.loader import load_experiment
from optbench.experiments.rq1_benchmark import run_rq1
from optbench.logging.artifacts import ArtifactStore
from optbench.reproducibility.env import capture_env


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--figures", default="results/figures")
    args = ap.parse_args()

    cfg = load_experiment(args.config)
    store = ArtifactStore(cfg.runtime.results_root)
    print(f"env: {capture_env()}")
    print(f"experiment: {cfg.name} | optimizers: {[o.name for o in cfg.optimizers]}")

    out = run_rq1(cfg, store, log=print)

    tables = write_tables(out, args.tables, cfg)
    tun = plot_tunability(out, f"{args.figures}/{cfg.name}_rq2_tunability.png")
    curves = plot_val_curves(out, f"{args.figures}/{cfg.name}_rq1_val_curves.png")

    print("\n=== RQ1 main (per optimizer) ===")
    for row in tables["main"]:
        print(
            f"{row['optimizer']:>13}  best_lr={row['best_lr']:.2e}  "
            f"val={row['val_mean']:.4f} +/- {row['val_std']:.4f}  "
            f"CI[{row['val_ci_low']:.4f}, {row['val_ci_high']:.4f}]  "
            f"steps_to_target={row['steps_to_target_mean']}"
        )
    print(f"\ntables -> {args.tables}\nfigures -> {tun} , {curves}")


if __name__ == "__main__":
    main()
