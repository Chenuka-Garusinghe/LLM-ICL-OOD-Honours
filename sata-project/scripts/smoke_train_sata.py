#!/usr/bin/env python3
"""Stage 3f smoke test -- NOT a real checkpoint.

Runs a handful of epochs on a small local task set to confirm the changed
training loop (pool_env mixing, unified standardisation, balanced top-k,
all-environment worst-case proxy) runs end-to-end with sane loss/proxy
trends. The real Stage 3f retrain runs on the full production task suite,
which needs more wall-time than a local smoke test -- this only verifies
the code is correct, not that the model is any good yet.

Usage: python3 scripts/smoke_train_sata.py [--n-train 30] [--n-val 10] [--epochs 3]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.generator import generate_val_test_tasks  # noqa: E402
from src.models.sata import SATA  # noqa: E402
from src.models.sata_train import evaluate_sata_proxy, train_sata  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=30)
    parser.add_argument("--n-val", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    gen_config = SimpleNamespace(
        n_features=10,
        n_causal_range=[3, 5],
        n_val_tasks=args.n_val,
        n_test_tasks=args.n_train,
        rule_families=["linear", "threshold", "tree", "sparse_interaction"],
        heldout_family="sparse_interaction",
        spurious_strength_range=[0.80, 0.90],
        label_noise=0.02,
        coefficient_scale=1.5,
        environments=["id", "covariate", "spurious_reversal", "extrapolation", "missing_feature", "mechanism"],
    )
    val_tasks, train_tasks = generate_val_test_tasks(gen_config)
    # generate_val_test_tasks names them (val, test) -- reuse test_tasks as
    # "train" here since this smoke test doesn't need the real train/val
    # split semantics, just two non-overlapping small task sets.
    print(f"Smoke test: {len(train_tasks)} train tasks, {len(val_tasks)} val tasks, {args.epochs} epochs")

    train_config = SimpleNamespace(
        d_model=128, n_heads=4, n_layers=4, max_demos=64, max_features=16,
        lr=1.0e-4, epochs=args.epochs, batch_size=64, patience=args.epochs,
        environments=gen_config.environments,
    )

    model = SATA(n_features=gen_config.n_features, d_model=train_config.d_model,
                 n_heads=train_config.n_heads, n_layers=train_config.n_layers)

    log = train_sata(model, train_tasks, val_tasks, train_config)

    print("\nTraining log:")
    for row in log:
        print(f"  epoch {row['epoch']}: loss={row['loss']:.4f}, val_proxy(worst-env)={row['val_proxy']:.4f}")

    final_proxy = evaluate_sata_proxy(model, val_tasks, train_config)
    print(f"\nFinal worst-env proxy accuracy: {final_proxy:.4f}")
    print("\nSmoke test complete -- this is NOT a real checkpoint (too few tasks/epochs); "
          "confirms the training loop runs end-to-end with the Stage 3 fixes.")


if __name__ == "__main__":
    main()
