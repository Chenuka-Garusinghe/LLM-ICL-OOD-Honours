"""SATA-MI and its SATA-names control (P5, 5b; docs/research_plan.md, 5b implementation).

SATA-MI reads, besides each row's z-scored features x, Qwen's measured
zero-shot margin m0 for that row under the task's naming (centred on the
pool median, `PoolPrior.centred`); the query gets its own zero-shot margin,
centred by the same pool median. SATA-names reads instead s * x, where s is
the lexicon direction the displayed name suggests (+1, -1, 0 for weak and
abstract names), so it sees what names say without any LLM measurement.

Because m0 has to be measured, training uses fixed frames (suites
`sata_mi_train` and `sata_mi_val`, built like the evaluation frames), not
data drawn afresh every batch. Every epoch visits every (task, naming,
environment) once, with a fresh column order for SATA's view.

Targets (`mi_targets`): the matched targets (hiccups/21) plus the planned
conditional counter-prior term, q = normalise(pi + lambda * 1[C_t > 1/2] * u),
with u uniform over clean pool rows whose label contradicts the prior.

Validation uses the prior-augmented proxy (src/models/prior_proxy.py): the
AUROC of a per-naming reader score a1 * w.x_q + a2 * m0_q within each
(task, naming, environment) cell, worst environment after averaging over
tasks and namings, for drawn and top-k selection.
"""

from __future__ import annotations

import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.data.naming import assign_names, name_directions
from src.data.synthetic_bridge import FEATURE_NAMES, load_manifest, load_task_frame, pool_of, queries_of
from src.evaluation.metrics import auroc
from src.models.prior_proxy import Reader, demo_evidence
from src.models.sata_targets import matched_target_scores
from src.models.sata_train import SELECTIONS, _default_n_jobs, _resolve_device, rule_checkpoint_path, select_demos

MI_VARIANTS = ("sata_mi", "sata_names")
INPUT_WIDTH = {"sata_mi": 11, "sata_names": 20}
ENVS = ("id", "covariate", "spurious_reversal")


def mi_inputs(variant: str, Xz: np.ndarray, qz: np.ndarray, m0_pool: np.ndarray, m0_query: np.ndarray,
              directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """SATA's per-row input: [x, m0] (sata_mi) or [x, s * x] (sata_names)."""
    if variant == "sata_mi":
        return (np.column_stack([Xz, m0_pool]).astype(np.float32),
                np.column_stack([qz, np.asarray(m0_query).reshape(-1, 1)]).astype(np.float32))
    if variant == "sata_names":
        return (np.column_stack([Xz, Xz * directions]).astype(np.float32),
                np.column_stack([qz, qz * directions]).astype(np.float32))
    raise ValueError(f"unknown SATA-MI variant {variant!r}")


def mi_targets(query_meta: dict, demo_meta: list[dict], shortcut: np.ndarray, contra_clean: np.ndarray,
               conflict: float, lam: float, k: int = 8) -> np.ndarray:
    """Matched targets plus lambda * 1[C_t > 1/2] times a uniform distribution over clean rows
    whose label contradicts the prior; normalised."""
    q = matched_target_scores(query_meta, demo_meta, shortcut, k)
    if lam > 0 and conflict > 0.5 and contra_clean.any():
        q = q + lam * contra_clean / contra_clean.sum()
    return q / q.sum()


@dataclass
class Item:
    """One (task, naming): the pool and its queries per environment, with everything SATA-MI reads."""
    task_id: str
    naming: str
    X: np.ndarray                      # pool, z-scored, displayed order
    y: np.ndarray
    meta: list[dict]
    shortcut: np.ndarray               # pool shortcut values (z-scored)
    m0: np.ndarray                     # centred pool prior
    directions: np.ndarray             # lexicon direction per displayed feature
    contra_clean: np.ndarray
    conflict: float
    queries: dict = field(default_factory=dict)   # env -> (Xq, yq, meta_q, m0_q)


def _meta(df: pd.DataFrame) -> list[dict]:
    agree = df["agree_obs"].astype(bool).to_numpy()
    return [{"regime": int(r), "label": int(l), "y_clean": int(c), "agree_obs": bool(a),
             "is_counter_spurious": not bool(a), "spurious_consistent": bool(a)}
            for r, l, c, a in zip(df["regime"], df["label"], df["y_clean"], agree)]


def load_items(suite_dir: str | Path, pool_prior, query_margin: pd.Series, lexicons: dict, base_seed: int,
               namings=("abstract", "aligned", "flipped"), envs=ENVS) -> list[Item]:
    """Every (task, naming) of a suite whose pool prior and query margins were measured.

    `query_margin` holds raw zero-shot margins (logprob_1 - logprob_0) indexed by
    (task_id, naming, row_id); they are centred here by the pool's raw median.
    """
    manifest = load_manifest(suite_dir)
    table = pool_prior.table.assign(m=lambda t: t["logprob_1"] - t["logprob_0"])
    items = []
    for entry in manifest["tasks"]:
        frame = load_task_frame(suite_dir, entry["task_id"])
        pool = pool_of(frame)
        for naming in namings:
            t = table[(table["task_id"] == entry["task_id"]) & (table["naming"] == naming)]
            if t.empty:
                continue
            raw = t.set_index("row_id")["m"]
            med = float(raw.median())
            m0 = (raw - med).reindex(pool["row_id"]).to_numpy(float)
            names = assign_names(entry, naming, lexicons.get(entry["domain"]) if naming != "abstract" else None, base_seed)
            d = name_directions(names, lexicons.get(entry["domain"]) if naming != "abstract" else None)
            y = pool["label"].to_numpy(int)
            clean = y == pool["y_clean"].to_numpy(int)
            contra = (m0 > 0) != (y == 1)
            item = Item(entry["task_id"], naming, pool[FEATURE_NAMES].to_numpy(float), y, _meta(pool),
                        pool[entry["roles"]["spurious"]].to_numpy(float), m0,
                        np.array([d[c] for c in FEATURE_NAMES], dtype=float), contra & clean,
                        float(np.mean(contra)))
            for env in envs:
                q = queries_of(frame, env)
                keys = pd.MultiIndex.from_arrays([[entry["task_id"]] * len(q), [naming] * len(q), q["row_id"].to_numpy()])
                mq = query_margin.reindex(keys).to_numpy(float) - med
                if np.isnan(mq).any():
                    continue
                item.queries[env] = (q[FEATURE_NAMES].to_numpy(float), q["label"].to_numpy(int), _meta(q), mq)
            if item.queries:
                items.append(item)
    return items


def _batch(item: Item, env: str, variant: str, perm: np.ndarray | None):
    Xq, yq, mq_meta, m0q = item.queries[env]
    X, qX, d = item.X, Xq, item.directions
    if perm is not None:
        X, qX, d = X[:, perm], qX[:, perm], d[perm]
    demo_in, query_in = mi_inputs(variant, X, qX, item.m0, m0q, d)
    return demo_in, query_in, X, qX, yq, mq_meta, m0q


def train_mi(model, variant: str, train_items: list[Item], val_items: list[Item], readers: dict[str, Reader],
             config: Any, checkpoint_path=None, resume_checkpoint_path=None, lam: float = 0.5) -> list[dict]:
    """KL training on the MI targets with early stopping on the prior-augmented proxy
    (both selection rules logged, the primary one picks `checkpoint_path`)."""
    device = _resolve_device(getattr(config, "device", "auto"))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    patience = getattr(config, "patience", 10)
    primary = getattr(config, "selection", "sample")
    rules = (primary, *[r for r in SELECTIONS if r != primary])
    ckpt_of = {r: (checkpoint_path if r == primary or checkpoint_path is None
                   else rule_checkpoint_path(checkpoint_path, r)) for r in rules}
    best = {r: {"score": float("-inf"), "epoch": -1, "stale": 0} for r in rules}
    start, log = 0, []
    if resume_checkpoint_path is not None and Path(resume_checkpoint_path).exists():
        st = torch.load(resume_checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(st["model_state"]); optimizer.load_state_dict(st["optimizer_state"])
        start, best, log = st["epoch"] + 1, st["best"], st["log"]
    seed = getattr(config, "train_seed", 0)
    jobs = [(i, env) for i, it in enumerate(train_items) for env in it.queries]
    for epoch in range(start, config.epochs):
        rng = np.random.default_rng([int(seed), epoch])
        model.train()
        total, n = 0.0, 0
        for j in tqdm(rng.permutation(len(jobs)), desc=f"Epoch {epoch}", leave=False):
            i, env = jobs[j]
            item = train_items[i]
            perm = rng.permutation(item.X.shape[1])
            demo_in, query_in, X, qX, yq, q_meta, _ = _batch(item, env, variant, perm)
            targets = np.stack([mi_targets(qm, item.meta, item.shortcut, item.contra_clean, item.conflict, lam)
                                for qm in q_meta])
            nq = len(yq)
            pred = model(torch.tensor(demo_in, device=device).unsqueeze(0).expand(nq, -1, -1),
                         torch.tensor(item.y, dtype=torch.long, device=device).unsqueeze(0).expand(nq, -1),
                         torch.tensor(query_in, device=device))
            loss = F.kl_div(torch.log(pred.clamp_min(1e-8)), torch.tensor(targets, dtype=torch.float32, device=device),
                            reduction="batchmean")
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item(); n += 1
        val = evaluate_mi_proxy(model, variant, val_items, readers, rules=rules)
        row = {"epoch": epoch, "loss": total / max(n, 1)}
        for r in rules:
            worst = min(val[r].values())
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
        print(f"Epoch {epoch}: loss={row['loss']:.4f}, val_proxy={row['val_proxy']:.4f} ({primary}), "
              + ", ".join(f"{r} {row[f'val_proxy_{r}']:.4f}" for r in rules[1:]))
        if resume_checkpoint_path is not None:
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                        "best": best, "log": log}, resume_checkpoint_path)
        if all(b["stale"] >= patience for b in best.values()):
            print(f"Early stopping at epoch {epoch} -- best val_proxy={best[primary]['score']:.4f} "
                  f"at epoch {best[primary]['epoch']} ({primary}).")
            break
    if resume_checkpoint_path is not None and Path(resume_checkpoint_path).exists():
        Path(resume_checkpoint_path).unlink()
    return log


def reader_cells(sets: list[np.ndarray], item: Item, env: str, reader: Reader) -> float:
    """Reader AUROC of one (task, naming, environment) cell for per-query sets."""
    Xq, yq, _, m0q = item.queries[env]
    ev = np.array([demo_evidence(item.X[s], item.y[s], Xq[i:i + 1])[0] for i, s in enumerate(sets)])
    return float(auroc(yq, reader.score(ev, m0q)))


def evaluate_mi_proxy(model, variant: str, items: list[Item], readers: dict[str, Reader], k: int = 8,
                      rules: tuple[str, ...] = ("sample",), n_jobs: int | None = None) -> dict[str, dict[str, float]]:
    """Reader AUROC per environment (mean over tasks and namings) for each selection rule.
    Draws are seeded by (task, naming, environment, query)."""
    device = next(model.parameters()).device
    model.eval()
    acc = {r: {e: [] for e in ENVS} for r in rules}
    with torch.no_grad(), ThreadPoolExecutor(max_workers=n_jobs or _default_n_jobs()) as pool:
        futures = []
        for item in items:
            for env in item.queries:
                demo_in, query_in, *_ = _batch(item, env, variant, None)
                nq = len(query_in)
                sc = model(torch.tensor(demo_in, device=device).unsqueeze(0).expand(nq, -1, -1),
                           torch.tensor(item.y, dtype=torch.long, device=device).unsqueeze(0).expand(nq, -1),
                           torch.tensor(query_in, device=device)).cpu().numpy()
                base = zlib.crc32(f"{item.task_id}|{item.naming}|{env}".encode())
                for r in rules:
                    sets = [select_demos(sc[i], item.y, k, r, np.random.default_rng([base, i])) for i in range(nq)]
                    futures.append((r, env, pool.submit(reader_cells, sets, item, env, readers[item.naming])))
        for r, env, f in futures:
            acc[r][env].append(f.result())
    return {r: {e: float(np.mean(v)) for e, v in acc[r].items() if v} for r in rules}
