"""Declarative configuration contracts for the whole harness.

These dataclasses are the single source of truth; YAML files (configs/) are just
their serialization. The fair-comparison protocol is encoded here: a fixed
``WorkloadConfig`` + a fixed ``BudgetConfig`` are shared across all optimizers,
while only ``SearchSpace`` hyperparameters vary during tuning.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LoraSpec:
    enabled: bool = True
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("c_attn",)
    bias: str = "none"


@dataclass(frozen=True)
class WorkloadConfig:
    """A fixed unit of work shared identically across optimizers."""

    name: str
    task: str            # "regression" | "causal_lm"
    model_kind: str      # "toy_mlp" | "toy_transformer" | "hf_causal"
    model_name: str = ""  # HF id when model_kind == "hf_causal"
    dataset: str = ""     # "toy_regression" | "toy_char" | "wikitext2" | "sst2"
    seq_len: int = 128
    micro_batch_size: int = 32
    grad_accum_steps: int = 1
    lora: LoraSpec = field(default_factory=LoraSpec)
    full_finetune: bool = False
    eval_metric: str = "val_loss"   # "val_loss" | "perplexity" | "accuracy"
    extras: dict = field(default_factory=dict)  # model/data knobs (hidden, d_model, ...)


@dataclass(frozen=True)
class BudgetConfig:
    """Identical compute budget for every optimizer (fairness invariant)."""

    kind: str = "steps"           # "steps" | "tokens"
    max_steps: int = 200
    max_tokens: int | None = None
    warmup_ratio: float = 0.0
    eval_every_steps: int = 50
    scheduler: str = "none"       # "none" | "cosine" | "linear"
    grad_clip: float | None = 1.0


@dataclass(frozen=True)
class HParamRange:
    name: str
    kind: str                     # "log_uniform" | "uniform" | "choice" | "fixed"
    low: float | None = None
    high: float | None = None
    choices: tuple = ()
    value: float | None = None


@dataclass(frozen=True)
class SearchSpace:
    optimizer: str
    n_trials: int = 12            # equal tuning budget per optimizer
    ranges: tuple[HParamRange, ...] = ()


@dataclass(frozen=True)
class OptimizerSpec:
    name: str                     # "sgd_momentum" | "adamw" | "lion" | "sophia"
    defaults: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SeedConfig:
    sweep_seed: int = 0
    final_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)   # K seeds for the final runs


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "auto"          # "auto" | "cuda" | "cpu"
    precision: str = "fp32"       # "fp32" | "bf16" | "fp16"
    gradient_checkpointing: bool = False
    deterministic: bool = True
    results_root: str = "results/runs"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    rq: str = ""                  # "rq1" | "rq2" | "rq3"
    workload: "WorkloadConfig | None" = None
    optimizers: tuple[OptimizerSpec, ...] = ()
    search_spaces: dict = field(default_factory=dict)   # name -> SearchSpace
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    seeds: SeedConfig = field(default_factory=SeedConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    target_metric_value: float | None = None  # for steps/tokens-to-target
