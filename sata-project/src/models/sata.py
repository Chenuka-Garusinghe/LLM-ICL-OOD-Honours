"""SATA (Shift-Aware Task Adapter) transformer architecture (Notebook 05).

SATA is meta-trained exclusively on synthetic tasks from src/data/generator.py
and never sees TableShift data during training. At evaluation time on real
data it receives standardised feature vectors (zero mean, unit variance
within each task's demo pool) — see sata_select.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SATA(nn.Module):
    """Shift-Aware Task Adapter.

    Input:
        demo_features:  (batch, n_demos, n_features) — candidate demonstrations
        demo_labels:    (batch, n_demos)              — demo labels (0 or 1)
        query_features: (batch, n_features)            — the query row

    Output:
        scores: (batch, n_demos) — relevance score per demo, sums to 1 via softmax
    """

    def __init__(self, n_features: int, d_model: int = 128, n_heads: int = 4, n_layers: int = 4):
        super().__init__()
        # Per-feature linear embedding
        self.feature_embed = nn.Linear(n_features, d_model)
        # Label embedding (2 classes)
        self.label_embed = nn.Embedding(2, d_model)
        # Learnable query token type embedding
        self.query_type_embed = nn.Parameter(torch.randn(1, 1, d_model))
        self.demo_type_embed = nn.Parameter(torch.randn(1, 1, d_model))

        # Standard transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Scoring head: one scalar per demo position
        self.score_head = nn.Linear(d_model, 1)

    def forward(
        self, demo_features: torch.Tensor, demo_labels: torch.Tensor, query_features: torch.Tensor
    ) -> torch.Tensor:
        batch_size, n_demos, n_feat = demo_features.shape

        # Embed demos: feature embedding + label embedding
        demo_emb = self.feature_embed(demo_features) + self.label_embed(demo_labels)
        demo_emb = demo_emb + self.demo_type_embed

        # Embed query: feature embedding only (no label)
        query_emb = self.feature_embed(query_features).unsqueeze(1)
        query_emb = query_emb + self.query_type_embed

        # Concatenate: [demo_1, demo_2, ..., demo_n, query]
        sequence = torch.cat([demo_emb, query_emb], dim=1)  # (batch, n_demos+1, d_model)

        # Self-attention (all tokens attend to all)
        encoded = self.transformer(sequence)

        # Extract demo positions only (not query)
        demo_encoded = encoded[:, :n_demos, :]  # (batch, n_demos, d_model)

        # Score each demo
        logits = self.score_head(demo_encoded).squeeze(-1)  # (batch, n_demos)
        scores = F.softmax(logits, dim=-1)

        return scores


class SATAQueryAgnostic(SATA):
    """Ablation: mask the query token so scores don't depend on query identity.

    Isolates whether per-query conditioning matters — the core SATA vs. ICR
    (in-context retrieval) differentiator referenced in Gate 2 / RQ4.
    """

    def forward(
        self, demo_features: torch.Tensor, demo_labels: torch.Tensor, query_features: torch.Tensor
    ) -> torch.Tensor:
        # Replace query with a zero vector — scores depend only on demo pool
        dummy_query = torch.zeros_like(query_features)
        return super().forward(demo_features, demo_labels, dummy_query)
