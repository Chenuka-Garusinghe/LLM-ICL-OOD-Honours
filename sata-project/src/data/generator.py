"""Synthetic task generator (Notebook 04) — SATA's training data source and
the ground-truth testbed for RQ2/RQ4.

Rule families:
  linear             logit = sum(coeff_i * X[:, causal_i]); y = sigmoid(logit) > 0.5
  threshold          y = (X[:, causal_0] > t0) AND/OR (X[:, causal_1] > t1)
  tree               depth 2-3 nested if/else on 2-3 causal features -> 4-8 regimes
  sparse_interaction y = (X[:, i] * X[:, j] > threshold), other causal feats are decoys

Environments (see class docstring for what each perturbs):
  id, covariate, spurious_reversal, extrapolation, missing_feature, mechanism

The generator is frozen after the XGBoost validation gate (see
xgboost_validation in Notebook 04); everything downstream uses frozen tasks.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

RULE_FAMILIES = ["linear", "threshold", "tree", "sparse_interaction"]
ENVIRONMENTS = [
    "id",
    "covariate",
    "spurious_reversal",
    "extrapolation",
    "missing_feature",
    "mechanism",
]


@dataclass
class SyntheticTask:
    """One synthetic tabular classification task with known ground truth."""

    task_id: str
    rule_family: str
    causal_features: list[int]
    coefficients: np.ndarray
    spurious_strength: float
    n_features: int = 10
    label_noise: float = 0.05
    spurious_idx: int = field(default=8, init=False)  # always feature 8
    noise_idx: int = field(default=9, init=False)      # always feature 9
    threshold: float = field(default=0.0)              # used by threshold/sparse_interaction

    def _base_features(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(size=(n_samples, self.n_features))

    def _apply_rule(self, X: np.ndarray) -> np.ndarray:
        """Compute the clean (pre-noise, pre-spurious) binary label from causal features."""
        causal = X[:, self.causal_features]

        if self.rule_family == "linear":
            logit = causal @ self.coefficients
            return (1 / (1 + np.exp(-logit)) > 0.5).astype(int)

        if self.rule_family == "threshold":
            c0, c1 = self.causal_features[0], self.causal_features[1]
            return ((X[:, c0] > self.threshold) | (X[:, c1] > self.threshold)).astype(int)

        if self.rule_family == "tree":
            c0 = self.causal_features[0]
            c1 = self.causal_features[min(1, len(self.causal_features) - 1)]
            branch_a = X[:, c0] > self.threshold
            branch_b = X[:, c1] > self.threshold
            return (branch_a & branch_b).astype(int)

        if self.rule_family == "sparse_interaction":
            i, j = self.causal_features[0], self.causal_features[min(1, len(self.causal_features) - 1)]
            return ((X[:, i] * X[:, j]) > self.threshold).astype(int)

        raise ValueError(f"Unknown rule_family: {self.rule_family}")

    def _regime(self, X: np.ndarray) -> np.ndarray:
        """Which decision-rule leaf/region each row falls in (used as demo metadata)."""
        causal = X[:, self.causal_features]
        signs = (causal > 0).astype(int)
        # Encode the sign pattern of causal features as an integer regime id.
        weights = 2 ** np.arange(signs.shape[1])
        return signs @ weights

    def generate_environment(
        self, env_type: str, n_samples: int, seed: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Generate (X, y, metadata) for a given environment type.

        metadata[i] contains: regime, is_counter_spurious, spurious_consistent, label.
        """
        rng = np.random.default_rng(seed)
        X = self._base_features(n_samples, rng)
        coeffs = self.coefficients
        spurious_strength = self.spurious_strength

        if env_type == "id":
            pass
        elif env_type == "covariate":
            shift_feats = rng.choice(self.n_features, size=min(3, self.n_features), replace=False)
            X[:, shift_feats] += rng.uniform(1.0, 2.0, size=len(shift_feats)) * rng.choice([-1, 1], size=len(shift_feats))
        elif env_type == "spurious_reversal":
            spurious_strength = 1.0 - spurious_strength
        elif env_type == "extrapolation":
            X *= rng.uniform(2.0, 3.0)
        elif env_type == "missing_feature":
            drop_idx = self.causal_features[0]
            X[:, drop_idx] = 0.0
        elif env_type == "mechanism":
            # Magnitude-only reweighting (+-50%, as in the spec pseudocode) barely
            # moves accuracy: a classifier's learned decision boundary is roughly
            # scale-invariant, so rescaling coefficients (or shifting threshold/tree's
            # implicit intercept) mostly just relabels points near the boundary
            # without breaking the feature-label relationship the classifier relies
            # on. To get a real "mechanism changed" shift, also flip the sign of
            # (at least) half the causal features' contribution to the rule -- this
            # reverses the relationship those features had with the label outright,
            # which a classifier trained on the original mechanism cannot compensate
            # for (flipping only one of several causal features leaves too much of
            # the original signal intact for "linear" in particular).
            coeffs = coeffs * rng.uniform(0.5, 1.5, size=coeffs.shape)
            n_flip = max(1, len(self.causal_features) // 2)
            flip_idx = rng.choice(self.causal_features, size=n_flip, replace=False)
        else:
            raise ValueError(f"Unknown env_type: {env_type}")

        rule_X = X
        if env_type == "mechanism":
            rule_X = X.copy()
            rule_X[:, flip_idx] *= -1

        original_coeffs = self.coefficients
        self.coefficients = coeffs
        y_clean = self._apply_rule(rule_X)
        self.coefficients = original_coeffs

        # Spurious feature: agrees with label w.p. spurious_strength.
        agree = rng.random(n_samples) < spurious_strength
        X[:, self.spurious_idx] = np.where(agree, y_clean, 1 - y_clean).astype(float) + rng.normal(
            scale=0.1, size=n_samples
        )
        # Pure noise feature.
        X[:, self.noise_idx] = rng.normal(size=n_samples)

        # Label noise.
        flip = rng.random(n_samples) < self.label_noise
        y = np.where(flip, 1 - y_clean, y_clean)

        regimes = self._regime(X)
        metadata = [
            {
                "regime": int(regimes[i]),
                "is_counter_spurious": not bool(agree[i]),
                "spurious_consistent": bool(agree[i]),
                "label": int(y[i]),
            }
            for i in range(n_samples)
        ]
        return X, y, metadata


def _sample_tasks(config: Any, n_tasks: int, id_prefix: str, family_choices: list[str]) -> list[SyntheticTask]:
    tasks: list[SyntheticTask] = []
    for i in range(n_tasks):
        family = random.choice(family_choices)
        n_causal = random.randint(*config.n_causal_range)
        causal_feats = sorted(random.sample(range(8), n_causal))
        coeffs = np.random.randn(n_causal) * 2  # scale for meaningful separation
        spur_strength = random.uniform(*config.spurious_strength_range)

        tasks.append(
            SyntheticTask(
                task_id=f"{id_prefix}_{i:04d}",
                rule_family=family,
                causal_features=causal_feats,
                coefficients=coeffs,
                spurious_strength=spur_strength,
                n_features=config.n_features,
                label_noise=config.label_noise,
            )
        )
    return tasks


def generate_task_suite(config: Any) -> list[SyntheticTask]:
    """Sample train_tasks + heldout_family_tasks per configs/default.yaml::generator.

    `config` is expected to expose the fields under the `generator` key of
    default.yaml (n_train_tasks, n_causal_range, rule_families, heldout_family, ...).
    """
    non_heldout_families = [f for f in config.rule_families if f != config.heldout_family]
    train_tasks = _sample_tasks(config, config.n_train_tasks, "train", non_heldout_families)
    heldout_tasks = _sample_tasks(config, config.n_heldout_family_tasks, "heldout", [config.heldout_family])
    return train_tasks + heldout_tasks


def generate_val_test_tasks(config: Any) -> tuple[list[SyntheticTask], list[SyntheticTask]]:
    """Sample separate n_val_tasks/n_test_tasks suites (non-heldout families,
    same distribution as generate_task_suite's train tasks but a disjoint
    task_id namespace) — used for the Notebook 04 gate and Notebook 05/06
    validation/held-out-test splits.
    """
    non_heldout_families = [f for f in config.rule_families if f != config.heldout_family]
    val_tasks = _sample_tasks(config, config.n_val_tasks, "val", non_heldout_families)
    test_tasks = _sample_tasks(config, config.n_test_tasks, "test", non_heldout_families)
    return val_tasks, test_tasks
