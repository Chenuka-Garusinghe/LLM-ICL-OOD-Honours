"""SATA training loop (Notebook 05).

Trains on synthetic tasks only (src/data/generator.py), using KL divergence
against the ground-truth target distribution from sata_targets.py. Validated
via an XGBoost proxy (evaluate_sata_proxy) that doesn't require LLM calls —
this is the Gate 2 metric: does SATA's top-k selection beat the best protocol
and random selection on validation tasks?
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from xgboost import XGBClassifier

from src.models.sata_targets import compute_target_scores


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # CPU fallback: this training loop issues hundreds of thousands of tiny,
    # sequential forward/backward calls (n_train_tasks x 6 environments x
    # epochs). PyTorch's intra-op thread-pool dispatch overhead per call
    # outweighs any benefit from multiple threads at this batch size, so keep
    # it single-threaded here specifically -- unlike XGBoost/sklearn/numpy
    # elsewhere in this project, which do benefit from the multi-core
    # OMP_NUM_THREADS set in src/utils/config.py.
    torch.set_num_threads(1)
    return torch.device("cpu")


def _default_n_jobs() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 4


def _task_batch(task, env_type: str, n_demos: int, n_queries: int, seed: int | None = None):
    """Generate one (demos, queries) batch for a task/environment pair."""
    X, y, metadata = task.generate_environment(env_type, n_samples=n_demos + n_queries, seed=seed)
    demo_X, demo_y, demo_meta = X[:n_demos], y[:n_demos], metadata[:n_demos]
    query_X, query_y, query_meta = X[n_demos:], y[n_demos:], metadata[n_demos:]
    return (demo_X, demo_y, demo_meta), (query_X, query_y, query_meta)


def _fit_predict_one(demo_X_topk: np.ndarray, demo_y_topk: np.ndarray, query_row: np.ndarray, query_label) -> int:
    """Fit XGBoost on one query's top-k demos and check the prediction.

    `n_jobs=1` is deliberate: this is called from inside a ThreadPoolExecutor
    (see evaluate_sata_proxy / Notebook 05's evaluate_protocol_proxy), so each
    individual fit must stay single-threaded -- letting XGBoost also spawn its
    own OMP thread pool per call would oversubscribe the allocated cores
    (n_outer_threads x n_inner_threads) and run slower than either alone.
    """
    if len(np.unique(demo_y_topk)) < 2:
        # XGBoost needs >=2 classes to fit; an early/untrained model's top-k
        # can easily be single-class by chance. Fall back to that class.
        pred = demo_y_topk[0]
    else:
        clf = XGBClassifier(max_depth=4, n_estimators=100, verbosity=0, n_jobs=1)
        clf.fit(demo_X_topk, demo_y_topk)
        pred = clf.predict(query_row[None, :])[0]
    return int(pred == query_label)


def train_sata(model, train_tasks: list, val_tasks: list, config: Any, checkpoint_path=None) -> list[dict]:
    """Train SATA via KL-divergence against ground-truth target scores.

    `config` exposes the `sata` block of default.yaml (lr, epochs, max_demos,
    ...) plus `environments` from the `generator` block.

    Note on batching: queries within a task/environment share a demo pool, so
    each (task, env) pair yields one batch of size len(queries) with the demo
    pool broadcast across it, rather than looping per-query as in the
    pseudocode in the spec.

    If `checkpoint_path` is given, the state dict with the best (highest)
    val_proxy seen so far is written there after every epoch -- this is what
    makes models/sata_best.pt actually "best checkpoint by validation loss"
    rather than just whatever epoch training happened to stop on, and gives
    crash/interrupt resilience for a run that can take hours at full scale.
    """
    device = _resolve_device()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    log: list[dict] = []
    best_val_score = float("-inf")

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for task in tqdm(train_tasks, desc=f"Epoch {epoch}", leave=False):
            for env_type in config.environments:
                (demo_X, demo_y, demo_meta), (query_X, query_y, query_meta) = _task_batch(
                    task, env_type, n_demos=config.max_demos, n_queries=32
                )
                demo_X_t = torch.tensor(demo_X, dtype=torch.float32, device=device)
                demo_y_t = torch.tensor(demo_y, dtype=torch.long, device=device)

                targets = np.stack(
                    [compute_target_scores(qm, demo_meta) for qm in query_meta]
                )  # (n_queries, n_demos)
                target_tensor = torch.tensor(targets, dtype=torch.float32, device=device)

                n_queries = query_X.shape[0]
                demo_features_batch = demo_X_t.unsqueeze(0).expand(n_queries, -1, -1)
                demo_labels_batch = demo_y_t.unsqueeze(0).expand(n_queries, -1)
                query_features_batch = torch.tensor(query_X, dtype=torch.float32, device=device)

                pred_scores = model(demo_features_batch, demo_labels_batch, query_features_batch)

                loss = F.kl_div(
                    torch.log(pred_scores.clamp_min(1e-8)),
                    target_tensor,
                    reduction="batchmean",
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

        val_score = evaluate_sata_proxy(model, val_tasks, config)
        mean_loss = epoch_loss / max(n_batches, 1)
        log.append({"epoch": epoch, "loss": mean_loss, "val_proxy": val_score})
        print(f"Epoch {epoch}: loss={mean_loss:.4f}, val_proxy={val_score:.4f}")

        if checkpoint_path is not None and val_score > best_val_score:
            best_val_score = val_score
            torch.save(model.state_dict(), checkpoint_path)

    return log


def evaluate_sata_proxy(model, val_tasks: list, config: Any, k: int = 8, n_jobs: int | None = None) -> float:
    """Proxy validation metric (no LLM calls): fit XGBoost on SATA's top-k
    selected demos per query, evaluate accuracy on that query.

    Used for Gate 2 — compare against random/protocol selection baselines
    computed the same way in Notebook 05/06.
    """
    device = _resolve_device()
    model.to(device)
    model.eval()
    accs: list[int] = []

    if n_jobs is None:
        n_jobs = _default_n_jobs()

    with torch.no_grad(), ThreadPoolExecutor(max_workers=n_jobs) as pool:
        for task in val_tasks:
            (demo_X, demo_y, demo_meta), (query_X, query_y, query_meta) = _task_batch(
                task, "id", n_demos=config.max_demos, n_queries=32
            )
            n_queries = query_X.shape[0]
            demo_X_t = torch.tensor(demo_X, dtype=torch.float32, device=device).unsqueeze(0).expand(n_queries, -1, -1)
            demo_y_t = torch.tensor(demo_y, dtype=torch.long, device=device).unsqueeze(0).expand(n_queries, -1)
            query_t = torch.tensor(query_X, dtype=torch.float32, device=device)

            # One batched forward pass for every query in this task, instead of
            # n_queries separate single-row calls -- the demo pool is identical
            # across all of them, so this is the same computation, just batched
            # (mirrors train_sata's batching above).
            scores = model(demo_X_t, demo_y_t, query_t).cpu().numpy()  # (n_queries, n_demos)
            top_k_idx = np.argsort(-scores, axis=1)[:, :k]

            # XGBoost's C++ fit releases the GIL, so a thread pool gives real
            # parallelism across the allocated CPU cores for these otherwise-tiny,
            # otherwise-sequential per-query fits.
            futures = [
                pool.submit(_fit_predict_one, demo_X[top_k_idx[i]], demo_y[top_k_idx[i]], query_X[i], query_y[i])
                for i in range(n_queries)
            ]
            accs.extend(f.result() for f in futures)

    return float(np.mean(accs)) if accs else 0.0
