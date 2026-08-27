"""Tiny decoder-only (causal) char-level Transformer LM for the toy next-token task."""

from __future__ import annotations

import torch
import torch.nn as nn


class ToyCharTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        nhead: int = 2,
        num_layers: int = 2,
        dim_ff: int = 128,
        max_len: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def forward(self, idx):
        _, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None, :, :]
        # Causal mask: position i may not attend to j > i.
        mask = torch.triu(
            torch.full((t, t), float("-inf"), device=idx.device), diagonal=1
        )
        x = self.encoder(x, mask=mask)
        x = self.ln(x)
        return self.head(x)
