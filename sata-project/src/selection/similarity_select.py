"""Similarity-k demo selection via cosine similarity on sentence embeddings.

Embeds serialised rows (demo pool + query) with all-MiniLM-L6-v2 and selects
the k demos with highest cosine similarity to the query.
"""

from __future__ import annotations

import numpy as np

_MODEL = None
_POOL_EMBED_CACHE: dict[int, np.ndarray] = {}
_POOL_CACHE_MAX = 64  # bounded so a long-running notebook can't leak memory


def _get_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer("all-MiniLM-L6-v2", device=_get_device())
    return _MODEL


def _pool_embeddings(model, pool_texts: list[str]) -> np.ndarray:
    """Encode `pool_texts` once and cache by content.

    Callers (Notebooks 02/03/06) invoke `select()` once per query, but the
    demo pool is identical across every query in a given (dataset, condition,
    seed) block -- without this cache, the same ~256-row pool was being
    re-embedded from scratch on every single query call.
    """
    cache_key = hash(tuple(pool_texts))
    cached = _POOL_EMBED_CACHE.get(cache_key)
    if cached is None:
        cached = model.encode(pool_texts, normalize_embeddings=True)
        if len(_POOL_EMBED_CACHE) >= _POOL_CACHE_MAX:
            _POOL_EMBED_CACHE.pop(next(iter(_POOL_EMBED_CACHE)))
        _POOL_EMBED_CACHE[cache_key] = cached
    return cached


def select(pool_texts: list[str], query_text: str, k: int, seed: int | None = None) -> list[int]:
    """Return indices (into pool_texts) of the k most similar demos to query_text.

    For a whole test set against one pool, prefer `select_batch` below --
    calling this once per query means one `model.encode()` forward pass per
    call instead of one batched pass for all of them.
    """
    model = _get_model()
    pool_emb = _pool_embeddings(model, pool_texts)
    query_emb = model.encode([query_text], normalize_embeddings=True)[0]

    similarities = pool_emb @ query_emb
    top_k_idx = np.argsort(-similarities)[:k]
    return list(top_k_idx)


def select_batch(pool_texts: list[str], query_texts: list[str], k: int) -> list[list[int]]:
    """Same result as calling `select()` once per entry of `query_texts`, but
    embedding every query in a single batched `model.encode()` call instead
    of one call per query -- the actual usage pattern in Notebooks 02/03/06
    (one fixed pool, hundreds of queries against it per condition)."""
    model = _get_model()
    pool_emb = _pool_embeddings(model, pool_texts)
    query_emb = model.encode(query_texts, normalize_embeddings=True)

    similarities = query_emb @ pool_emb.T  # (n_queries, n_pool)
    top_k_idx = np.argsort(-similarities, axis=1)[:, :k]
    return [list(row) for row in top_k_idx]
