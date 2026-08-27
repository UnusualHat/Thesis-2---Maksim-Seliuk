"""End-to-end smoke for the HF + LoRA path on GPT-2 / WikiText-2.

Tiny budget + capped data so it runs in well under a minute on the GPU. Verifies:
loading, LoRA wrapping, bf16 autocast, grad-accum, scheduler, eval, peak-mem.

Run: .venv\\Scripts\\python.exe scripts\\smoke_hf.py
"""

from __future__ import annotations

import torch

from optbench.builders import build_experiment_unit
from optbench.config.schema import BudgetConfig, LoraSpec, WorkloadConfig
from optbench.models.hf_causal import count_parameters, trainable_parameters
from optbench.reproducibility.env import capture_env
from optbench.reproducibility.seeding import set_global_seed
from optbench.training.scheduler import build_scheduler
from optbench.training.trainer import Trainer


def main() -> None:
    set_global_seed(0)
    workload = WorkloadConfig(
        name="gpt2_wikitext2_lora_smoke",
        task="causal_lm",
        model_kind="hf_causal",
        model_name="gpt2",
        dataset="wikitext2",
        seq_len=256,
        micro_batch_size=4,
        grad_accum_steps=2,  # effective batch 8
        lora=LoraSpec(r=8, alpha=16, dropout=0.05, target_modules=("c_attn",)),
        extras={"max_train_blocks": 128, "max_eval_blocks": 32},
    )
    model, data = build_experiment_unit(workload, seed=0)
    trainable, total = count_parameters(model)
    print(f"env: {capture_env()}")
    print(f"trainable params: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")
    print(f"data: {data.meta}")

    trainer = Trainer(model, data, device="cuda", precision="bf16", grad_clip=1.0)
    init = trainer.evaluate()
    budget = BudgetConfig(
        max_steps=20, eval_every_steps=10, warmup_ratio=0.1, scheduler="cosine"
    )
    opt = torch.optim.AdamW(trainable_parameters(model), lr=2e-4)
    sched = build_scheduler(opt, budget)
    res = trainer.fit(opt, budget, scheduler=sched, opt_name="adamw", seed=0)

    print(
        f"init_val_loss={init['val_loss']:.3f} init_ppl={init['perplexity']:.1f} "
        f"-> best_val_loss={res.best_val:.3f} best_ppl={res.history[-1]['perplexity']:.1f}"
    )
    print(
        f"peak_mem_GB={res.peak_gpu_mem_bytes/1e9:.2f} "
        f"s/step={res.mean_step_time_s:.3f} steps={budget.max_steps}"
    )
    assert res.best_val < init["val_loss"], "loss did not decrease"
    print("SMOKE OK")


if __name__ == "__main__":
    main()
