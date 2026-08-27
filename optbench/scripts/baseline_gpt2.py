"""GPT-2 / WikiText-2 LoRA baseline with AdamW.

Purpose: see where validation loss lands under a realistic-but-cheap budget so we
can pick ``target_metric_value`` for the time-to-target metric used in the sweeps.
Saves the run (config, env, history) under results/runs/ via ArtifactStore.

Run: .venv\\Scripts\\python.exe scripts\\baseline_gpt2.py
"""

from __future__ import annotations

from dataclasses import asdict

import torch

from optbench.builders import build_experiment_unit
from optbench.config.schema import BudgetConfig, LoraSpec, WorkloadConfig
from optbench.logging.artifacts import ArtifactStore
from optbench.models.hf_causal import count_parameters, trainable_parameters
from optbench.optimizers.factory import make_optimizer
from optbench.optimizers.state_size import optimizer_state_bytes
from optbench.reproducibility.env import capture_env
from optbench.reproducibility.seeding import set_global_seed
from optbench.training.scheduler import build_scheduler
from optbench.training.trainer import Trainer


def main() -> None:
    set_global_seed(0)
    workload = WorkloadConfig(
        name="gpt2_wikitext2_lora_baseline",
        task="causal_lm",
        model_kind="hf_causal",
        model_name="gpt2",
        dataset="wikitext2",
        seq_len=256,
        micro_batch_size=8,
        grad_accum_steps=1,
        lora=LoraSpec(r=8, alpha=16, dropout=0.05, target_modules=("c_attn",)),
        extras={"max_train_blocks": 1024, "max_eval_blocks": 128},
    )
    budget = BudgetConfig(
        max_steps=300, eval_every_steps=50, warmup_ratio=0.05, scheduler="cosine"
    )

    model, data = build_experiment_unit(workload, seed=0)
    trainable, total = count_parameters(model)
    print(f"trainable {trainable:,}/{total:,} ({100*trainable/total:.3f}%) | data {data.meta}")

    trainer = Trainer(model, data, device="cuda", precision="bf16", grad_clip=1.0)
    init = trainer.evaluate()
    opt = make_optimizer("adamw", trainable_parameters(model), lr=3e-4)
    sched = build_scheduler(opt, budget)
    res = trainer.fit(opt, budget, scheduler=sched, opt_name="adamw", seed=0)

    print(f"\nstep |  val_loss  |  ppl")
    for h in [{"step": 0, **init}] + res.history:
        print(f"{h['step']:>4} |  {h['val_loss']:.4f}  | {h['perplexity']:.2f}")
    print(
        f"\nbest_val_loss={res.best_val:.4f} (step {res.best_step}) | "
        f"peak_mem_GB={res.peak_gpu_mem_bytes/1e9:.2f} | s/step={res.mean_step_time_s:.3f} | "
        f"opt_state_MB={optimizer_state_bytes(opt)/1e6:.2f}"
    )
    # A sensible time-to-target threshold: ~90% of the way from init to best.
    target = round(init["val_loss"] - 0.9 * (init["val_loss"] - res.best_val), 2)
    print(f"suggested target_metric_value (val_loss) for sweeps: {target}")

    store = ArtifactStore("results/runs")
    run_dir = store.run_dir("baseline", "gpt2_wikitext2_lora", "adamw_seed0")
    store.save_json(run_dir / "config.json", asdict(workload))
    store.save_json(run_dir / "env.json", capture_env())
    store.save_jsonl(run_dir / "history.jsonl", [{"step": 0, **init}] + res.history)
    store.save_json(
        run_dir / "summary.json",
        {
            "best_val": res.best_val,
            "best_step": res.best_step,
            "peak_gpu_mem_bytes": res.peak_gpu_mem_bytes,
            "mean_step_time_s": res.mean_step_time_s,
            "opt_state_bytes": optimizer_state_bytes(opt),
            "suggested_target": target,
        },
    )
    print(f"saved run artifacts -> {run_dir}")


if __name__ == "__main__":
    main()
