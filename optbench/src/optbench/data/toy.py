"""Tiny synthetic datasets used for fast smoke tests and, importantly, for
validating the cheap curvature proxy against the full Hessian (toy models small
enough for a full eigenspectrum).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from optbench.data.base import DataModule

# A small, repetitive corpus is enough for a char-level next-token smoke task.
_CORPUS = "the quick brown fox jumps over the lazy dog. " * 200


class _DictDataset(Dataset):
    """Yields dict samples; default_collate stacks them into a batched dict."""

    def __init__(self, tensors: dict):
        self.tensors = tensors
        self.n = len(next(iter(tensors.values())))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        return {k: v[i] for k, v in self.tensors.items()}


def _loaders(train, val, test, batch_size: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, generator=g, drop_last=True
    )
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def build_toy_regression(
    in_dim: int = 8, n: int = 900, batch_size: int = 32, seed: int = 0, noise: float = 0.1
) -> DataModule:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, in_dim)).astype("float32")
    w = rng.standard_normal((in_dim, 1)).astype("float32")
    y = (x @ w + noise * rng.standard_normal((n, 1))).astype("float32")
    x_t, y_t = torch.from_numpy(x), torch.from_numpy(y)

    n_tr, n_va = int(0.7 * n), int(0.15 * n)
    tr = _DictDataset({"x": x_t[:n_tr], "y": y_t[:n_tr]})
    va = _DictDataset({"x": x_t[n_tr : n_tr + n_va], "y": y_t[n_tr : n_tr + n_va]})
    te = _DictDataset({"x": x_t[n_tr + n_va :], "y": y_t[n_tr + n_va :]})
    tl, vl, tel = _loaders(tr, va, te, batch_size, seed)
    return DataModule("regression", tl, vl, tel, {"in_dim": in_dim, "out_dim": 1})


def build_toy_char_lm(seq_len: int = 64, batch_size: int = 16, seed: int = 0) -> DataModule:
    text = _CORPUS
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    vocab_size = len(chars)
    ids = np.array([stoi[c] for c in text], dtype="int64")

    # Non-overlapping windows of length seq_len+1; split into input / shifted label.
    seqs = [
        ids[i : i + seq_len + 1]
        for i in range(0, len(ids) - seq_len - 1, seq_len)
    ]
    seqs = np.stack(seqs)
    inp = torch.from_numpy(seqs[:, :-1])
    lab = torch.from_numpy(seqs[:, 1:])

    n = len(inp)
    perm = np.random.default_rng(seed).permutation(n)
    inp, lab = inp[perm], lab[perm]
    n_tr, n_va = int(0.7 * n), int(0.15 * n)

    def make(a, b):
        return _DictDataset({"input_ids": inp[a:b], "labels": lab[a:b]})

    tl, vl, tel = _loaders(make(0, n_tr), make(n_tr, n_tr + n_va), make(n_tr + n_va, n), batch_size, seed)
    return DataModule("causal_lm", tl, vl, tel, {"vocab_size": vocab_size})
