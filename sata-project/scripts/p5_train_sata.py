#!/usr/bin/env python3
"""Train SATA-v2, its query-agnostic ablation, or the planned-target ablation (P5, 5a).

Tasks come from the evaluation distribution (`sample_sata_tasks`: 1,000
training and 100 validation tasks with their own seed streams). The loop is
`train_sata`: KL to the variant's targets (`TARGETS_OF`: the neutral matched
targets for sata and sata_agnostic, the planned structural targets for
sata_designed; hiccups/21), Adam, early stopping on the
XGBoost proxy's worst evaluated environment. The best checkpoint goes to
the path in `sata.checkpoints` (configs/default.yaml), a resume file next to
it, and the per-epoch log to results/v3/synthetic/p5/train_<variant>.json.

`--benchmark` times training batches on the CPU and on the auto-detected
device instead (the plan's one-off check, since SATA's batches are small)
and writes p5/device_benchmark.json.

Usage:
  python scripts/p5_train_sata.py --benchmark
  python scripts/p5_train_sata.py --variant sata [--device mps] [--epochs 50]
  python scripts/p5_train_sata.py --variant sata_agnostic
  python scripts/p5_train_sata.py --variant sata_designed
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, resolve_path  # noqa: E402  (first: sets thread env vars)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.models.sata import TARGETS_OF, VARIANTS, build_sata  # noqa: E402
from src.models.sata_targets import compute_target_scores  # noqa: E402
from src.models.sata_train import _resolve_device, _task_batch, sample_sata_tasks, sata_config, train_sata  # noqa: E402
from src.utils.device import resolve_device  # noqa: E402


def _tasks(config, n_train: int, n_val: int):
    g = config.generator
    return sample_sata_tasks(n_train, n_val, base_seed=g.task_seed,
                             spurious_strength_range=tuple(g.spurious_strength_range),
                             label_noise=g.label_noise, n_features=g.n_features)


def benchmark(config, cfg, n_batches: int = 20) -> dict:
    """Seconds per training step (forward, backward, Adam) on each device."""
    train, _ = _tasks(config, 8, 0)
    rng = np.random.default_rng(0)
    batches = []
    for i in range(n_batches):
        task, env = train[i % len(train)], cfg.environments[i % len(cfg.environments)]
        (dX, dy, dm), (qX, qy, qm) = _task_batch(task, env, cfg.max_demos, cfg.queries_per_env,
                                                 seed=int(rng.integers(2**31 - 1)), pool_env="id" if env != "mechanism" else env,
                                                 permutation=rng.permutation(task.n_features))
        batches.append((dX, dy, np.stack([compute_target_scores(q, dm) for q in qm]), qX))
    out = {}
    for name in ("cpu", "auto"):
        device = _resolve_device("cpu") if name == "cpu" else resolve_device("auto")
        torch.manual_seed(0)
        model = build_sata("sata", config.generator.n_features, cfg.d_model, cfg.n_heads, cfg.n_layers).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        def step(b):
            dX, dy, tgt, qX = b
            n = qX.shape[0]
            pred = model(torch.tensor(dX, dtype=torch.float32, device=device).unsqueeze(0).expand(n, -1, -1),
                         torch.tensor(dy, dtype=torch.long, device=device).unsqueeze(0).expand(n, -1),
                         torch.tensor(qX, dtype=torch.float32, device=device))
            loss = F.kl_div(torch.log(pred.clamp_min(1e-8)), torch.tensor(tgt, dtype=torch.float32, device=device),
                            reduction="batchmean")
            opt.zero_grad()
            loss.backward()
            opt.step()
            return loss.item()

        step(batches[0])                                   # warm-up (kernel compilation)
        t0 = time.perf_counter()
        for b in batches:
            step(b)
        out[str(device)] = (time.perf_counter() - t0) / len(batches)
        print(f"{device}: {out[str(device)] * 1000:.1f} ms per training step")
    return {"seconds_per_step": out, "n_batches": n_batches, "batch": [cfg.queries_per_env, cfg.max_demos + 1],
            "torch": torch.__version__, "faster": min(out, key=out.get)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="sata")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--device", default=None, help="auto, cpu, mps or cuda (default: sata.device)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--n-val", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0, help="torch initialisation and training-data stream")
    parser.add_argument("--checkpoint", default=None, help="default: sata.checkpoints.<variant>")
    args = parser.parse_args()

    config = load_config()
    out_dir = resolve_path(config.paths.results) / "p5"
    out_dir.mkdir(parents=True, exist_ok=True)
    overrides = {k: v for k, v in {"device": args.device, "epochs": args.epochs}.items() if v is not None}
    cfg = sata_config(config, train_seed=args.seed, targets=TARGETS_OF[args.variant], **overrides)

    if args.benchmark:
        result = benchmark(config, cfg)
        (out_dir / "device_benchmark.json").write_text(json.dumps(result, indent=2))
        return

    g = config.generator
    train, val = _tasks(config, args.n_train or g.n_train_tasks, args.n_val or g.n_val_tasks)
    ckpt = resolve_path(args.checkpoint or getattr(config.sata.checkpoints, args.variant))
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    resume = ckpt.with_suffix(".resume.pt")
    torch.manual_seed(args.seed)
    model = build_sata(args.variant, g.n_features, cfg.d_model, cfg.n_heads, cfg.n_layers)
    print(f"{args.variant} ({cfg.targets} targets): {len(train)} training and {len(val)} validation tasks, device {cfg.device}, "
          f"up to {cfg.epochs} epochs (patience {cfg.patience}) -> {ckpt}")
    t0 = time.perf_counter()
    log = train_sata(model, train, val, cfg, checkpoint_path=ckpt, resume_checkpoint_path=resume)
    best = max(log, key=lambda r: r["val_proxy"])
    rules = [k[len("val_proxy_"):] for k in log[0] if k.startswith("val_proxy_")]
    best_by_rule = {r: max(log, key=lambda row: row[f"val_proxy_{r}"]) for r in rules}
    summary = {
        "variant": args.variant, "targets": cfg.targets, "checkpoint": str(ckpt.relative_to(resolve_path("."))), "seed": args.seed,
        "n_train_tasks": len(train), "n_val_tasks": len(val), "device": str(resolve_device(cfg.device)),
        "minutes": (time.perf_counter() - t0) / 60, "selection": cfg.selection,
        "best_epoch": best["epoch"], "best_val_proxy": best["val_proxy"],
        "best_by_selection": {r: {"epoch": b["epoch"], "val_proxy": b[f"val_proxy_{r}"]} for r, b in best_by_rule.items()},
        "config": {k: v for k, v in vars(cfg).items() if k != "checkpoints"}, "log": log,
    }
    path = out_dir / f"train_{args.variant}.json"
    path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"best val proxy {best['val_proxy']:.4f} at epoch {best['epoch']}; log -> {path}")


if __name__ == "__main__":
    main()
