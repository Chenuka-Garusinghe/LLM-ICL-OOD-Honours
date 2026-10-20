"""SATA training loop (Notebook 05).

Trains on synthetic tasks only (src/data/generator.py), using KL divergence
against the ground-truth target distribution from sata_targets.py. Validated
via an XGBoost proxy (evaluate_sata_proxy) that doesn't require LLM calls —
this is the Gate 2 metric: does SATA's top-k selection beat the best protocol
and random selection on validation tasks?
"""

from __future__ import annotations

import os
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.data.generator import sample_eval_tasks
from src.models.sata_targets import compute_target_scores, matched_target_scores
from src.models.standardise import standardise
from src.models.xgb_proxy import fit_predict_one
from src.selection.balanced_topk import balanced_sample, balanced_top_k
from src.utils.device import resolve_device


def _resolve_device(preference: str = "auto") -> torch.device:
    device = resolve_device(preference)
    if device.type == "cpu":
        # Tiny sequential batches: thread-pool dispatch overhead outweighs multi-threading.
        torch.set_num_threads(1)
    return device


def _default_n_jobs() -> int:
    if os.environ.get("SATA_N_JOBS"):
        # A container can report the host's cores rather than its allocation
        # (a Runpod pod: 192 reported, 24 allocated), so a pod run sets this.
        return int(os.environ["SATA_N_JOBS"])
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 4


def sata_config(config: Any, **overrides: Any) -> SimpleNamespace:
    """The namespace `train_sata` and `evaluate_sata_proxy` read: the `sata`
    block of default.yaml, the generator's training environments and queries
    per batch, and the evaluated environments the worst case is taken over."""
    out = SimpleNamespace(**vars(config.sata))
    out.environments = list(config.generator.environments)
    out.queries_per_env = config.generator.queries_per_env
    out.selection_environments = list(config.evaluation.environments)
    out.permute_columns = True
    for key, value in overrides.items():
        setattr(out, key, value)
    return out


def sample_sata_tasks(n_train: int, n_val: int, base_seed: int = 42, **kwargs) -> tuple[list, list]:
    """SATA's training and validation tasks (P5): the evaluation-task
    distribution (`sample_eval_tasks`) under the prefixes `sata_train` and
    `sata_val`, whose seed streams no evaluation or pilot task shares."""
    return (sample_eval_tasks(n_train, id_prefix="sata_train", base_seed=base_seed, **kwargs),
            sample_eval_tasks(n_val, id_prefix="sata_val", base_seed=base_seed, **kwargs))


def _task_batch(
    task,
    env_type: str,
    n_demos: int,
    n_queries: int,
    seed: int | None = None,
    pool_env: str | None = None,
    permutation: np.ndarray | None = None,
):
    """Generate one (demos, queries) batch for a task/environment pair.

    `pool_env` (default: same as `env_type`, i.e. v1's matched-pool
    behaviour -- one `generate_environment` call, split contiguously) lets
    demos be drawn from a *different* environment than the queries. Training
    with `pool_env="id"` means the demo pool is never oracle access to the
    query environment's shift -- the deployment-realistic setting that v1
    never trained or evaluated (REDESIGN_RATIONALE.md §4.4(d)/§5.3).
    Standardisation (src/models/standardise.py) is applied here so training
    sees the same z-scored features inference does (§4.5) -- v1 trained on
    raw features and only standardised at inference time.
    """
    if pool_env is None or pool_env == env_type:
        X, y, metadata = task.generate_environment(env_type, n_samples=n_demos + n_queries, seed=seed)
        demo_X, demo_y, demo_meta = X[:n_demos], y[:n_demos], metadata[:n_demos]
        query_X, query_y, query_meta = X[n_demos:], y[n_demos:], metadata[n_demos:]
    else:
        query_seed = None if seed is None else seed + 1
        demo_X, demo_y, demo_meta = task.generate_environment(pool_env, n_samples=n_demos, seed=seed)
        query_X, query_y, query_meta = task.generate_environment(env_type, n_samples=n_queries, seed=query_seed)

    demo_X, query_X = standardise(demo_X, query_X)
    if permutation is not None:
        # Column-role permutation (P5): evaluation tasks show their features in
        # a per-task random order, so the shortcut is not always column 8.
        # Metadata (regime, agreement, labels) does not depend on the order.
        demo_X, query_X = demo_X[:, permutation], query_X[:, permutation]
    return (demo_X, demo_y, demo_meta), (query_X, query_y, query_meta)


TARGETS = ("designed", "matched")
SELECTIONS = ("sample", "topk")


def rule_checkpoint_path(checkpoint_path, rule: str) -> Path:
    """Where the best checkpoint under a non-primary selection rule goes:
    models/sata_v2.pt -> models/sata_v2_topk.pt."""
    path = Path(checkpoint_path)
    return path.with_name(f"{path.stem}_{rule}{path.suffix}")


def select_demos(scores: np.ndarray, labels: np.ndarray, k: int, rule: str, rng=None) -> np.ndarray:
    """SATA's demonstrations under a selection rule: `topk` (k/2 highest per
    label) or `sample` (k/2 per label drawn in proportion to the scores)."""
    if rule == "topk":
        return balanced_top_k(scores, labels, k)
    if rule == "sample":
        return balanced_sample(scores, labels, k, rng)
    raise ValueError(f"unknown selection {rule!r}; expected one of {SELECTIONS}")


def batch_targets(task, demo_X: np.ndarray, demo_meta: list[dict], query_meta: list[dict],
                  permutation: np.ndarray | None, kind: str = "designed", k: int = 8) -> np.ndarray:
    """Target distributions for one batch, (n_queries, n_demos).

    `designed`: the structural targets (`compute_target_scores`), the
    planned ones with label-noise rows excluded. `matched` (P5, the user's
    decision; hiccups/21): `matched_target_scores`, which reads the
    shortcut's column, found through the batch's column-role permutation.
    """
    if kind == "designed":
        return np.stack([compute_target_scores(q, demo_meta) for q in query_meta])
    if kind == "matched":
        col = task.spurious_idx if permutation is None else int(np.flatnonzero(permutation == task.spurious_idx)[0])
        return np.stack([matched_target_scores(q, demo_meta, demo_X[:, col], k=k) for q in query_meta])
    raise ValueError(f"unknown targets {kind!r}; expected one of {TARGETS}")


def _val_seed(task, env_type: str) -> int:
    """Fixed data seed for one validation (task, environment), so every epoch's
    proxy is measured on the same pools and queries."""
    return int(np.random.SeedSequence([zlib.crc32(task.task_id.encode()), zlib.crc32(env_type.encode())])
               .generate_state(1)[0])


def train_sata(
    model,
    train_tasks: list,
    val_tasks: list,
    config: Any,
    checkpoint_path=None,
    resume_checkpoint_path=None,
) -> list[dict]:
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
    rather than just whatever epoch training happened to stop on.

    Early stopping: training stops once `config.patience` epochs pass with no
    improvement in val_proxy over the best seen so far (`getattr(config,
    "patience", 10)` -- 10 is a deliberately mild default: val_proxy is a
    noisy proxy metric that can move a point or two between consecutive
    epochs on its own, so a short patience risks stopping on noise rather
    than a genuine plateau, but it still bounds the wasted-epoch cost of a
    real plateau to a fraction of the full `config.epochs` budget). This
    was previously unconditional -- the loop always ran all `config.epochs`
    regardless of whether val_proxy had stopped improving many epochs
    earlier, which is exactly what happened during Notebook 05's first full
    run (val_proxy peaked at epoch 0 and never recovered through epoch 35+).

    Resume support: if `resume_checkpoint_path` is given, full training
    state (model + optimizer state, current epoch, best-so-far bookkeeping,
    and the log accumulated up to that point) is written there after every
    epoch, and automatically loaded from if the file already exists when
    this function is called -- so an interrupted multi-hour run (kernel
    killed, walltime hit, connection dropped) picks back up at the next
    epoch instead of restarting from epoch 0. This is a separate file from
    `checkpoint_path`, which stays a bare model state_dict (what Notebook 06
    and downstream code load directly via `SATA().load_state_dict(...)`) --
    resume state is a different, larger payload most callers don't want.
    Delete `resume_checkpoint_path`'s file to force a fresh run instead of
    resuming. Task/environment sampling itself uses `seed=None` throughout
    this loop already (see `_task_batch`), so training data is regenerated
    stochastically every epoch regardless of resume -- there is no RNG state
    to restore for exact reproducibility across a resume, only genuinely
    fresh (if statistically equivalent) synthetic batches.

    v3 (P5): every batch gets a fresh column-role permutation
    (`config.permute_columns`, default on), the task order is shuffled every
    epoch, and `config.queries_per_env` sets the queries per batch (32).
    `config.train_seed` makes every draw reproducible, including across a
    resume (the stream is fixed per epoch). `config.targets` picks the target
    distribution (`batch_targets`: "designed" or "matched"). Early stopping reads the worst of
    `config.selection_environments`; the log keeps every environment's proxy.
    """
    device = _resolve_device(getattr(config, "device", "auto"))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    patience = getattr(config, "patience", 10)

    # Selection rules validated every epoch (P5): the primary one picks
    # `checkpoint_path`; each other rule keeps its own best checkpoint
    # (`rule_checkpoint_path`), so the choice of rule stays reversible.
    primary = getattr(config, "selection", "topk")
    rules = (primary, *[r for r in SELECTIONS if r != primary])
    ckpt_of = {r: (checkpoint_path if r == primary or checkpoint_path is None
                   else rule_checkpoint_path(checkpoint_path, r)) for r in rules}
    best = {r: {"score": float("-inf"), "epoch": -1, "stale": 0} for r in rules}

    start_epoch = 0
    log: list[dict] = []

    if resume_checkpoint_path is not None and Path(resume_checkpoint_path).exists():
        resume_state = torch.load(resume_checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(resume_state["model_state"])
        optimizer.load_state_dict(resume_state["optimizer_state"])
        start_epoch = resume_state["epoch"] + 1
        best = resume_state["best"]
        log = resume_state["log"]
        print(
            f"Resumed from {resume_checkpoint_path}: starting at epoch {start_epoch}, "
            f"best val_proxy={best[primary]['score']:.4f} (epoch {best[primary]['epoch']})"
        )

    n_queries_per_env = getattr(config, "queries_per_env", 32)
    permute = getattr(config, "permute_columns", True)
    selection_envs = getattr(config, "selection_environments", None) or list(config.environments)
    train_seed = getattr(config, "train_seed", None)
    target_kind = getattr(config, "targets", "designed")
    for epoch in range(start_epoch, config.epochs):
        # With `config.train_seed`, every epoch's draws (task order, pool mode,
        # permutation, data) come from its own stream, so a run, and a resumed
        # run, is reproducible; without it they come from OS entropy, as on main.
        rng = np.random.default_rng(None if train_seed is None else [int(train_seed), epoch])
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        # sample_eval_tasks lists the linear tasks first; shuffle so every
        # stretch of an epoch mixes both families.
        for task in tqdm([train_tasks[i] for i in rng.permutation(len(train_tasks))], desc=f"Epoch {epoch}",
                         leave=False):
            for env_type in config.environments:
                # 75% id-pool (deployment-realistic: demos never have oracle
                # access to the query environment's shift) / 25% matched-pool
                # (v1 behaviour, kept so that secondary condition stays
                # in-training-distribution for SATA) -- REDESIGN_RATIONALE.md
                # §5.3. `mechanism` is always matched-pool: it changes the
                # causal->label MAPPING itself (a concept shift), not just
                # the input distribution or the spurious channel -- an
                # id-pool demo sharing the query's raw regime was generated
                # under id's mapping, which for 'tree' assigns that exact
                # regime the OPPOSITE label to mechanism's (leaf_labels are
                # deliberately complement-paired, and mechanism's full
                # causal-sign-flip maps every regime to its complement --
                # see generate_environment's 'mechanism' branch). Verified
                # empirically: mean same-regime id-demo/mechanism-query label
                # agreement was ~0.02 for 'tree', i.e. the same_regime target
                # bonus becomes actively adversarial there under an id pool.
                pool_env = env_type if env_type == "mechanism" else ("id" if rng.random() < 0.75 else env_type)
                perm = rng.permutation(_n_columns(task)) if permute else None
                (demo_X, demo_y, demo_meta), (query_X, query_y, query_meta) = _task_batch(
                    task, env_type, n_demos=config.max_demos, n_queries=n_queries_per_env,
                    seed=int(rng.integers(2**31 - 1)), pool_env=pool_env, permutation=perm,
                )
                demo_X_t = torch.tensor(demo_X, dtype=torch.float32, device=device)
                demo_y_t = torch.tensor(demo_y, dtype=torch.long, device=device)

                targets = batch_targets(task, demo_X, demo_meta, query_meta, perm, target_kind)  # (n_queries, n_demos)
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

        val = evaluate_sata_proxy(model, val_tasks, config, per_env=True, selection=rules)
        mean_loss = epoch_loss / max(n_batches, 1)
        row = {"epoch": epoch, "loss": mean_loss}
        for r in rules:
            worst = min(val[r][e] for e in selection_envs)
            row[f"val_proxy_{r}"] = worst
            row.update({f"val_{r}_{e}": a for e, a in val[r].items()})
            if worst > best[r]["score"]:
                best[r] = {"score": worst, "epoch": epoch, "stale": 0}
                if ckpt_of[r] is not None:
                    torch.save(model.state_dict(), ckpt_of[r])
            else:
                best[r]["stale"] += 1
        row["val_proxy"] = row[f"val_proxy_{primary}"]
        log.append(row)
        print(f"Epoch {epoch}: loss={mean_loss:.4f}, val_proxy={row['val_proxy']:.4f} ({primary}), "
              + ", ".join(f"{r} {row[f'val_proxy_{r}']:.4f}" for r in rules[1:]))

        if resume_checkpoint_path is not None:
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                 "best": best, "log": log},
                resume_checkpoint_path,
            )

        if all(b["stale"] >= patience for b in best.values()):
            print(
                f"Early stopping at epoch {epoch} (no val_proxy improvement in {patience} epochs under any rule) "
                f"-- best val_proxy={best[primary]['score']:.4f} at epoch {best[primary]['epoch']} ({primary})."
            )
            break

    # A resume file left on disk after a *normal* finish (ran out the epoch
    # budget, or stopped early) would otherwise silently hijack the next
    # intentional fresh run -- e.g. after changing lr/epochs -- since
    # train_sata auto-resumes whenever the file exists. Only an interrupted
    # run (killed/disconnected mid-loop, before reaching this line) should
    # leave one behind for next time.
    if resume_checkpoint_path is not None and Path(resume_checkpoint_path).exists():
        Path(resume_checkpoint_path).unlink()

    return log


def _n_columns(task) -> int:
    """Number of feature columns a task's rows have."""
    return int(task.n_features)


def evaluate_sata_proxy(model, val_tasks: list, config: Any, k: int = 8, n_jobs: int | None = None,
                        per_env: bool = False, selection: str | tuple[str, ...] | None = None):
    """Proxy validation metric (no LLM calls): fit XGBoost on SATA's top-k
    selected demos per query, evaluate accuracy on that query.

    Validates across **all** `config.environments` (v1 validated on `id`
    only, where the label-copy shortcut policy looked excellent -- Gate 2's
    proxy accuracy of 0.961 sat almost exactly at the spurious strength,
    REDESIGN_RATIONALE.md §4.2 Fact C) with `pool_env="id"` (matching
    training's deployment-realistic majority mode), and returns the
    **worst-environment** accuracy rather than a flat mean -- worst-case
    validation is the operational meaning of "shift-aware" (§5.3). Used for
    Gate 2/Gate S3a — compare against random/protocol selection baselines
    computed the same way in Notebook 05/06.

    v3 (P5): every (task, environment) uses a fixed data seed and column-role
    permutation (`_val_seed`), so the proxy moves only when the model does;
    on main it was redrawn every epoch. The worst environment is taken over
    `config.selection_environments` when given (P5: the evaluated id,
    covariate and spurious_reversal; the mechanism placebo is still scored and
    returned with `per_env=True`).

    `selection` is a rule (`select_demos`; default `config.selection`, else
    top-k) or a tuple of rules, which returns {rule: per-environment accuracy}.
    Draws under `sample` are seeded by (task, environment, query).
    """
    rules = selection if isinstance(selection, tuple) else (selection or getattr(config, "selection", "topk"),)
    device = _resolve_device(getattr(config, "device", "auto"))
    model.to(device)
    model.eval()

    if n_jobs is None:
        n_jobs = _default_n_jobs()

    env_accs: dict[str, dict[str, list[int]]] = {r: {env: [] for env in config.environments} for r in rules}

    with torch.no_grad(), ThreadPoolExecutor(max_workers=n_jobs) as pool:
        for task in val_tasks:
            for env_type in config.environments:
                # id-pool for every environment except 'mechanism' (always
                # matched-pool there) -- see train_sata's loop for why.
                pool_env = "mechanism" if env_type == "mechanism" else "id"
                seed = _val_seed(task, env_type)
                perm = (np.random.default_rng(seed).permutation(_n_columns(task))
                        if getattr(config, "permute_columns", True) else None)
                (demo_X, demo_y, demo_meta), (query_X, query_y, query_meta) = _task_batch(
                    task, env_type, n_demos=config.max_demos, n_queries=getattr(config, "queries_per_env", 32),
                    seed=seed, pool_env=pool_env, permutation=perm,
                )
                n_queries = query_X.shape[0]
                demo_X_t = (
                    torch.tensor(demo_X, dtype=torch.float32, device=device).unsqueeze(0).expand(n_queries, -1, -1)
                )
                demo_y_t = torch.tensor(demo_y, dtype=torch.long, device=device).unsqueeze(0).expand(n_queries, -1)
                query_t = torch.tensor(query_X, dtype=torch.float32, device=device)

                # One batched forward pass for every query in this task, instead
                # of n_queries separate single-row calls -- the demo pool is
                # identical across all of them, so this is the same computation,
                # just batched (mirrors train_sata's batching above).
                scores = model(demo_X_t, demo_y_t, query_t).cpu().numpy()  # (n_queries, n_demos)

                # XGBoost's C++ fit releases the GIL, so a thread pool gives real
                # parallelism across the allocated CPU cores for these otherwise-
                # tiny, otherwise-sequential per-query fits.
                for r in rules:
                    idx = [select_demos(scores[i], demo_y, k, r, np.random.default_rng([seed, i])) for i in range(n_queries)]
                    futures = [pool.submit(fit_predict_one, demo_X[idx[i]], demo_y[idx[i]], query_X[i], query_y[i])
                               for i in range(n_queries)]
                    env_accs[r][env_type].extend(f.result() for f in futures)

    selection_envs = getattr(config, "selection_environments", None) or list(config.environments)
    out = {}
    for r in rules:
        acc = {env: (float(np.mean(a)) if a else 0.0) for env, a in env_accs[r].items()}
        out[r] = acc if per_env else min(acc[e] for e in selection_envs)
    return out if isinstance(selection, tuple) else out[rules[0]]
