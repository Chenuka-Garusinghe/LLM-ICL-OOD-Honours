"""Similarity-k demo selection via cosine similarity on sentence embeddings.

Embeds serialised rows (demo pool + query) with all-MiniLM-L6-v2 and selects
the k demos with highest cosine similarity to the query.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def select(pool_texts: list[str], query_text: str, k: int, seed: int | None = None) -> list[int]:
    """Return indices (into pool_texts) of the k most similar demos to query_text."""
    model = _get_model()
    pool_emb = model.encode(pool_texts, normalize_embeddings=True)
    query_emb = model.encode([query_text], normalize_embeddings=True)[0]

    similarities = pool_emb @ query_emb
    top_k_idx = np.argsort(-similarities)[:k]
    return list(top_k_idx)
