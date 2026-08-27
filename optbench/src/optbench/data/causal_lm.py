"""WikiText-2 causal-LM data: tokenize the corpus and pack into fixed-length
blocks (no padding). Labels equal input_ids; the HF model shifts internally.

``max_train_blocks`` / ``max_eval_blocks`` cap the data so a single tuning trial
stays within a few minutes on one GPU (the fairness/feasibility trade-off).
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from optbench.data.base import DataModule


class _BlockDataset(Dataset):
    def __init__(self, input_ids: torch.Tensor):
        self.input_ids = input_ids  # (N, seq_len) long

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, i: int) -> dict:
        ids = self.input_ids[i]
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": ids.clone(),
        }


def _tokenize_and_block(texts, tokenizer, seq_len: int, max_blocks=None) -> torch.Tensor:
    joined = "\n\n".join(t for t in texts if t and t.strip())
    ids = tokenizer(joined, return_attention_mask=False)["input_ids"]
    n_blocks = len(ids) // seq_len
    if max_blocks is not None:
        n_blocks = min(n_blocks, max_blocks)
    ids = ids[: n_blocks * seq_len]
    return torch.tensor(ids, dtype=torch.long).view(n_blocks, seq_len)


def build_wikitext2_causal_lm(
    tokenizer,
    seq_len: int = 512,
    batch_size: int = 8,
    seed: int = 0,
    max_train_blocks=None,
    max_eval_blocks=None,
) -> DataModule:
    from datasets import load_dataset

    # "wikitext" (legacy script dataset) no longer resolves in datasets >=4;
    # the canonical parquet home is "Salesforce/wikitext".
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    train = _tokenize_and_block(ds["train"]["text"], tokenizer, seq_len, max_train_blocks)
    val = _tokenize_and_block(ds["validation"]["text"], tokenizer, seq_len, max_eval_blocks)
    test = _tokenize_and_block(ds["test"]["text"], tokenizer, seq_len, max_eval_blocks)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        _BlockDataset(train), batch_size=batch_size, shuffle=True, generator=g, drop_last=True
    )
    val_loader = DataLoader(_BlockDataset(val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(_BlockDataset(test), batch_size=batch_size, shuffle=False)
    return DataModule(
        "causal_lm",
        train_loader,
        val_loader,
        test_loader,
        {"vocab_size": tokenizer.vocab_size, "seq_len": seq_len,
         "n_train_blocks": len(train), "n_val_blocks": len(val)},
    )
