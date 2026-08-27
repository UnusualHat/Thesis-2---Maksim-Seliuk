"""Task-aware loss and evaluation, shared by the trainer and standalone analysis."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def compute_loss(model, batch: dict, task: str, model_kind: str):
    """Return (scalar loss, n_tokens) for one batch.

    n_tokens drives token-weighted eval, perplexity, and the token budget.
    """
    if model_kind == "hf_causal":
        out = model(**batch)
        ntok = int((batch["labels"] != -100).sum().item())
        return out.loss, ntok
    if task == "regression":
        pred = model(batch["x"])
        return F.mse_loss(pred, batch["y"]), batch["y"].numel()
    if task == "causal_lm":
        logits = model(batch["input_ids"])
        vocab = logits.size(-1)
        labels = batch["labels"]
        loss = F.cross_entropy(
            logits.reshape(-1, vocab), labels.reshape(-1), ignore_index=-100
        )
        return loss, int((labels != -100).sum().item())
    raise ValueError(f"unknown task/model_kind: {task}/{model_kind}")


@torch.no_grad()
def evaluate(model, loader, task: str, model_kind: str, device, autocast=None) -> dict:
    was_training = model.training
    model.eval()
    total, ntok_total = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        ctx = autocast() if autocast is not None else _nullctx()
        with ctx:
            loss, ntok = compute_loss(model, batch, task, model_kind)
        total += float(loss.item()) * ntok
        ntok_total += ntok
    mean = total / max(ntok_total, 1)
    out = {"val_loss": mean}
    if task == "causal_lm" or model_kind == "hf_causal":
        out["perplexity"] = math.exp(min(mean, 20.0))
    if was_training:
        model.train()
    return out


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
