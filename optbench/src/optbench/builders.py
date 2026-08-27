"""Dispatch from a WorkloadConfig to concrete (model, data) units.

``build_experiment_unit`` is the single entry point used by runners; it covers
both the toy path (in-process synthetic data) and the HF causal-LM + LoRA path.
HF/peft/datasets are imported lazily so the toy path stays fast.
"""

from __future__ import annotations

from optbench.config.schema import WorkloadConfig
from optbench.data.base import DataModule
from optbench.data.toy import build_toy_char_lm, build_toy_regression
from optbench.models.toy_mlp import ToyMLP
from optbench.models.toy_transformer import ToyCharTransformer


def build_data(workload: WorkloadConfig, seed: int = 0) -> DataModule:
    if workload.model_kind == "toy_mlp" or workload.dataset == "toy_regression":
        dm = build_toy_regression(
            in_dim=workload.extras.get("in_dim", 8),
            batch_size=workload.micro_batch_size,
            seed=seed,
        )
    elif workload.model_kind == "toy_transformer" or workload.dataset == "toy_char":
        dm = build_toy_char_lm(
            seq_len=workload.seq_len,
            batch_size=workload.micro_batch_size,
            seed=seed,
        )
    else:
        raise ValueError(f"no toy data builder for workload: {workload.name}")
    dm.meta["model_kind"] = workload.model_kind
    dm.meta["grad_accum_steps"] = workload.grad_accum_steps
    return dm


def build_model(workload: WorkloadConfig, data_meta: dict, seed: int = 0):
    if workload.model_kind == "toy_mlp":
        return ToyMLP(
            in_dim=data_meta["in_dim"],
            hidden=workload.extras.get("hidden", 32),
            out_dim=data_meta.get("out_dim", 1),
            depth=workload.extras.get("depth", 2),
        )
    if workload.model_kind == "toy_transformer":
        return ToyCharTransformer(
            vocab_size=data_meta["vocab_size"],
            d_model=workload.extras.get("d_model", 64),
            nhead=workload.extras.get("nhead", 2),
            num_layers=workload.extras.get("num_layers", 2),
            max_len=max(workload.seq_len, 64),
        )
    raise ValueError(f"no toy model builder for workload: {workload.name}")


def build_experiment_unit(workload: WorkloadConfig, seed: int = 0):
    """Return (model, data_module) for any supported workload."""
    if workload.model_kind == "hf_causal":
        return _build_hf_unit(workload, seed)
    data = build_data(workload, seed)
    model = build_model(workload, data.meta, seed)
    return model, data


def _build_hf_unit(workload: WorkloadConfig, seed: int):
    from optbench.data.causal_lm import build_wikitext2_causal_lm
    from optbench.models.hf_causal import apply_lora, load_hf_causal_lm

    model, tokenizer = load_hf_causal_lm(
        workload.model_name,
        gradient_checkpointing=workload.extras.get("gradient_checkpointing", False),
    )
    if workload.lora.enabled and not workload.full_finetune:
        model = apply_lora(model, workload.lora)

    if workload.dataset in ("wikitext2", "wikitext-2", ""):
        data = build_wikitext2_causal_lm(
            tokenizer,
            seq_len=workload.seq_len,
            batch_size=workload.micro_batch_size,
            seed=seed,
            max_train_blocks=workload.extras.get("max_train_blocks"),
            max_eval_blocks=workload.extras.get("max_eval_blocks"),
        )
    else:
        raise ValueError(f"no HF data builder for dataset: {workload.dataset}")

    data.meta["model_kind"] = "hf_causal"
    data.meta["grad_accum_steps"] = workload.grad_accum_steps
    return model, data
