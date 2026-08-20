"""SATA training loop (Notebook 05).

Trains on synthetic tasks only (src/data/generator.py), using KL divergence
against the ground-truth target distribution from sata_targets.py. Validated
via an XGBoost proxy (evaluate_sata_proxy) that doesn't require LLM calls —
this is the Gate 2 metric: does SATA's top-k selection beat the best protocol
and random selection on validation tasks?
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from xgboost import XGBClassifier

from src.models.sata_targets import compute_target_scores


def _task_batch(task, env_type: str, n_demos: int, n_queries: int, seed: int | None = None):
    """Generate one (demos, queries) batch for a task/environment pair."""
    X, y, metadata = task.generate_environment(env_type, n_samples=n_demos + n_queries, seed=seed)
    demo_X, demo_y, demo_meta = X[:n_demos], y[:n_demos], metadata[:n_demos]
    query_X, query_y, query_meta = X[n_demos:], y[n_demos:], metadata[n_demos:]
    return (demo_X, demo_y, demo_meta), (query_X, query_y, query_meta)


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
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    log: list[dict] = []
    best_val_score = float("-inf")

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for task in train_tasks:
            for env_type in config.environments:
                (demo_X, demo_y, demo_meta), (query_X, query_y, query_meta) = _task_batch(
                    task, env_type, n_demos=config.max_demos, n_queries=32
                )
                demo_X_t = torch.tensor(demo_X, dtype=torch.float32)
                demo_y_t = torch.tensor(demo_y, dtype=torch.long)

                targets = np.stack(
                    [compute_target_scores(qm, demo_meta) for qm in query_meta]
                )  # (n_queries, n_demos)
                target_tensor = torch.tensor(targets, dtype=torch.float32)

                n_queries = query_X.shape[0]
                demo_features_batch = demo_X_t.unsqueeze(0).expand(n_queries, -1, -1)
                demo_labels_batch = demo_y_t.unsqueeze(0).expand(n_queries, -1)
                query_features_batch = torch.tensor(query_X, dtype=torch.float32)

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


def evaluate_sata_proxy(model, val_tasks: list, config: Any, k: int = 8) -> float:
    """Proxy validation metric (no LLM calls): fit XGBoost on SATA's top-k
    selected demos per query, evaluate accuracy on that query.

    Used for Gate 2 — compare against random/protocol selection baselines
    computed the same way in Notebook 05/06.
    """
    model.eval()
    accs = []

    with torch.no_grad():
        for task in val_tasks:
            (demo_X, demo_y, demo_meta), (query_X, query_y, query_meta) = _task_batch(
                task, "id", n_demos=config.max_demos, n_queries=32
            )
            demo_X_t = torch.tensor(demo_X, dtype=torch.float32).unsqueeze(0)
            demo_y_t = torch.tensor(demo_y, dtype=torch.long).unsqueeze(0)

            for i in range(query_X.shape[0]):
                query_t = torch.tensor(query_X[i], dtype=torch.float32).unsqueeze(0)
                scores = model(demo_X_t, demo_y_t, query_t).squeeze(0).numpy()
                top_k_idx = np.argsort(-scores)[:k]

                if len(np.unique(demo_y[top_k_idx])) < 2:
                    # XGBoost needs >=2 classes to fit; an early/untrained model's
                    # top-k can easily be single-class by chance. Fall back to
                    # that class as the prediction rather than crashing.
                    pred = demo_y[top_k_idx][0]
                else:
                    clf = XGBClassifier(max_depth=4, n_estimators=100, verbosity=0)
                    clf.fit(demo_X[top_k_idx], demo_y[top_k_idx])
                    pred = clf.predict(query_X[i : i + 1])[0]
                accs.append(int(pred == query_y[i]))

    return float(np.mean(accs)) if accs else 0.0
