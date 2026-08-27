"""HuggingFace causal-LM loading + LoRA wrapping.

We keep base weights in fp32 and use bf16 autocast at compute time (set via
RuntimeConfig.precision). This keeps optimizer states and the HVP path in fp32,
which matters for the curvature analysis (RQ3).
"""

from __future__ import annotations


def load_hf_causal_lm(model_name: str, *, gradient_checkpointing: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False  # correct for training; avoids warnings
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # needed so checkpointing works with frozen base
    return model, tokenizer


def apply_lora(model, lora_spec):
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=lora_spec.r,
        lora_alpha=lora_spec.alpha,
        lora_dropout=lora_spec.dropout,
        target_modules=list(lora_spec.target_modules),
        bias=lora_spec.bias,
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


def count_parameters(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
