"""Synthetic task generator (Notebook 04) — SATA's training data source and
the ground-truth testbed for RQ2/RQ4.

Rule families:
  linear             p = sigmoid(sum(coeff_i * X[:, causal_i])); y ~ Bernoulli(p).
                     Coefficient scale is `generator.coefficient_scale`
                     (`_sample_tasks`) -- see `generate_environment`'s
                     concept-noise draw for why labels are sampled rather
                     than thresholded at p=0.5.
  threshold          majority vote over 3 independently-thresholded causal
                     features (>=2 of 3 conditions true) -- see `_apply_rule`
  tree               depth-3 nested if/else on up to 3 causal features -> up
                     to 8 fixed regimes, each leaf's label set at task-sample
                     time (`_sample_tasks`'s `_sample_tree_leaf_labels`,
                     which excludes degenerate "dictator"/"parity" leaf
                     functions -- see that function's docstring)
  sparse_interaction y = (X[:, i] * X[:, j] > threshold), other causal feats are decoys

Environments (see class docstring for what each perturbs):
  id, covariate, spurious_reversal, extrapolation, missing_feature, mechanism

`threshold` and `tree` were reworked (Notebook 04's validation gate, second
pass) from a flat 2-feature AND/OR and a flat 2-feature AND respectively --
both trivially easy for a shallow learner to recover exactly from 64 samples
regardless of spurious-feature strength, which is what made the original
XGBoost gate fail almost entirely on those two families (see Notebook 04's
gate markdown cell for the full empirical breakdown). The richer rules here
raise the sample complexity of the *true* rule enough that a proxy learner
has a genuine reason to lean on the spurious feature instead, which is the
failure mode this gate exists to detect in the first place.

The generator is frozen after Notebook 04's validation gate (see that
notebook for the gate itself and which classifier it validates against);
everything downstream uses frozen tasks.
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
    threshold: float = field(default=0.0)              # used by sparse_interaction
    # 'threshold'/'tree' families: set by `_sample_tasks`, or reconstructed
    # from generator_config.json's per-task metadata (list, not ndarray, once
    # round-tripped through JSON -- `_apply_rule` converts via np.asarray).
    thresholds3: Any = field(default=None)
    leaf_labels: Any = field(default=None)

    def _base_features(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(size=(n_samples, self.n_features))

    def _apply_rule(self, X: np.ndarray, u: np.ndarray | None = None) -> np.ndarray:
        """Compute the clean (pre-noise, pre-spurious) binary label from causal features.

        `u` is a per-row Uniform(0,1) draw used only by 'linear', which
        samples y ~ Bernoulli(p) rather than thresholding p at 0.5 -- a
        deterministic sign(logit) rule is scale-invariant, so it is either
        always more reliable than the spurious feature or never is,
        regardless of `coefficient_scale`, and a max-margin/logistic learner
        given both signals will never prefer the (rescalable, never-wrong)
        causal rule's shortcut-competitor to be anything but perfectly
        separable -- exactly the deterministic-vs-Bernoulli distinction
        Arjovsky et al. (2019)'s Colored MNIST construction relies on: a
        shortcut only gets learned when it is *more reliable in training*
        than the causal signal, which requires the causal signal to be
        genuinely noisy. `u=None` (e.g. diagnostics) falls back to the MAP
        label p>0.5. Other rule families ignore `u` entirely.
        """
        causal = X[:, self.causal_features]

        if self.rule_family == "linear":
            logit = causal @ self.coefficients
            p = 1 / (1 + np.exp(-logit))
            if u is None:
                return (p > 0.5).astype(int)
            return (u < p).astype(int)

        if self.rule_family == "threshold":
            # Majority vote over 3 independently-thresholded features rather
            # than an AND/OR of 2: naturally class-balanced (AND/OR of 2 skews
            # 25/75, letting a majority-class prior masquerade as a fit) and
            # harder to recover exactly from 64 samples, which is what gives
            # the spurious feature room to actually get exploited.
            thresholds3 = np.asarray(self.thresholds3)
            feats = self.causal_features[: len(thresholds3)]
            conds = np.stack(
                [X[:, f] > t for f, t in zip(feats, thresholds3)], axis=1
            )
            return (conds.sum(axis=1) >= 2).astype(int)

        if self.rule_family == "tree":
            # Depth-3 nested if/else over up to 3 causal features: each of
            # the 2**k sign-pattern "leaves" gets a fixed label from
            # `leaf_labels` (set in `_sample_tasks`, paired so a leaf and its
            # sign-flipped complement always carry opposite labels -- see
            # that function's comment for why the pairing matters for the
            # `mechanism` environment).
            leaf_labels = np.asarray(self.leaf_labels)
            k = int(np.log2(len(leaf_labels)))
            feats = self.causal_features[:k]
            signs = (X[:, feats] > 0).astype(int)
            weights = 2 ** np.arange(signs.shape[1])
            leaf_id = signs @ weights
            return leaf_labels[leaf_id]

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
            # moves accuracy on its own -- a classifier's learned decision boundary
            # is roughly scale-invariant, so rescaling coefficients (or shifting
            # threshold/tree's implicit intercept) mostly just relabels points near
            # the boundary without breaking the feature-label relationship the
            # classifier relies on. (It is no longer a total no-op for 'linear'
            # now that labels are sampled ~Bernoulli(sigmoid(logit)) rather than
            # thresholded -- rescaling shifts each row's flip probability -- but
            # the sign flip below is still what does the real work.) To get a
            # real "mechanism changed" shift, also flip the sign of ALL causal
            # features' contribution to the rule -- flipping only half left too
            # much of the original signal intact (confirmed empirically: the
            # id-vs-mechanism accuracy gap was too small on a majority of tasks
            # across all three families). For 'tree' specifically, a full sign
            # flip maps every leaf to its bit-complement leaf, which is why
            # `_sample_tree_leaf_labels` pairs each leaf with its complement and
            # forces them to opposite labels -- otherwise whether this flip
            # changes anything would depend on the luck of that task's random
            # leaf assignment.
            coeffs = coeffs * rng.uniform(0.5, 1.5, size=coeffs.shape)
            flip_idx = np.array(self.causal_features)
        else:
            raise ValueError(f"Unknown env_type: {env_type}")

        rule_X = X
        if env_type == "mechanism":
            rule_X = X.copy()
            rule_X[:, flip_idx] *= -1

        # Concept noise for 'linear' (see `_apply_rule`'s docstring) -- drawn
        # unconditionally for every family/environment so the rng stream a
        # given seed produces doesn't depend on rule_family, and so
        # `mechanism`'s clean-label counterfactual uses the same per-row
        # noise draw as `id` would have (only the rule/features differ).
        concept_u = rng.random(n_samples)

        original_coeffs = self.coefficients
        self.coefficients = coeffs
        y_clean = self._apply_rule(rule_X, concept_u)
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


def _is_degenerate_leaf_labels(leaf_labels: np.ndarray, k: int) -> bool:
    """True if a k-bit leaf->label lookup table is a "dictator" (equals the
    sign of a single feature, i.e. bit_i or its negation, for some i) or
    "parity" (equals the XOR of all k bits, or its negation).

    Both are degenerate for the validation gate's purposes even though
    they're each valid Boolean functions of the causal features: a dictator
    is exactly as easy to learn from 64 samples as a single-feature
    threshold rule, so a logistic-regression ERM never needs the spurious
    shortcut to fit it (empirically: passes the spurious-reversal criterion
    only ~43% of the time). Parity is the opposite failure -- it is *not*
    learnable from 64 samples by any shallow classifier (a k=3 XOR needs to
    see all 8 sign combinations to pin down), so even a causal-only oracle
    can't show a real id-vs-mechanism accuracy gap (empirically ~50% pass
    rate, oracle ID accuracy ~0.57). Excluding both leaves the "signed
    majority" functions, which are learnable-but-nontrivial for both roles.
    """
    leaves = np.arange(2 ** k)
    bits = (leaves[:, None] >> np.arange(k)) & 1  # (2**k, k); bit i matches
    # _apply_rule's `weights = 2 ** arange(k)` leaf-id encoding.
    for i in range(k):
        bit_i = bits[:, i]
        if np.array_equal(leaf_labels, bit_i) or np.array_equal(leaf_labels, 1 - bit_i):
            return True
    parity = bits.sum(axis=1) % 2
    return bool(np.array_equal(leaf_labels, parity) or np.array_equal(leaf_labels, 1 - parity))


def _sample_tree_leaf_labels(k: int, max_tries: int = 100) -> np.ndarray:
    """Sample a 'tree' family's fixed leaf->label lookup table (2**k entries).

    Pairs each leaf with its sign-flipped complement (leaf L with leaf
    2**k-1-L) and forces them to opposite labels -- this guarantees the
    `mechanism` environment's full causal-feature sign flip (which maps
    every row to its complement leaf) always inverts the label, rather than
    leaving it to chance whether a random per-leaf assignment happens to be
    complement-symmetric. Fixed empirically: plain-random leaf labels left
    `tree`'s id-vs-mechanism accuracy gap too small on most tasks.

    Rejects degenerate assignments (see `_is_degenerate_leaf_labels`) for
    k>=3. For k<2 every complement-antisymmetric function IS a dictator (a
    single bit and its complement are always assigned opposite labels by
    construction), so rejection would never terminate -- not an issue in
    practice since `configs/default.yaml`'s `n_causal_range` is [3, 5] and
    `tree` always uses k=min(3, n_causal)=3, but guarded here rather than
    silently hanging if that ever changes.
    """
    n_leaves = 2 ** k

    def _draw() -> np.ndarray:
        leaf_labels = np.empty(n_leaves, dtype=int)
        for leaf in range(n_leaves):
            complement = n_leaves - 1 - leaf
            if complement < leaf:
                continue
            bit = np.random.randint(0, 2)
            leaf_labels[leaf] = bit
            leaf_labels[complement] = 1 - bit
        return leaf_labels

    if k < 3:
        return _draw()

    for _ in range(max_tries):
        leaf_labels = _draw()
        if not _is_degenerate_leaf_labels(leaf_labels, k):
            return leaf_labels
    raise RuntimeError(
        f"Could not sample a non-degenerate {k}-bit tree leaf assignment in {max_tries} tries "
        "(expected ~2 tries for k=3 -- check _is_degenerate_leaf_labels/n_causal_range for drift)."
    )


def _sample_tasks(config: Any, n_tasks: int, id_prefix: str, family_choices: list[str]) -> list[SyntheticTask]:
    tasks: list[SyntheticTask] = []
    for i in range(n_tasks):
        family = random.choice(family_choices)
        n_causal = random.randint(*config.n_causal_range)
        causal_feats = sorted(random.sample(range(8), n_causal))
        # Only 'linear' reads coefficient magnitude (see _apply_rule) --
        # coefficients are still sampled for every family so the rng stream
        # each seed produces doesn't depend on which family got picked.
        coeffs = np.random.randn(n_causal) * getattr(config, "coefficient_scale", 2.0)
        spur_strength = random.uniform(*config.spurious_strength_range)

        task = SyntheticTask(
            task_id=f"{id_prefix}_{i:04d}",
            rule_family=family,
            causal_features=causal_feats,
            coefficients=coeffs,
            spurious_strength=spur_strength,
            n_features=config.n_features,
            label_noise=config.label_noise,
        )

        if family == "threshold":
            k = min(3, n_causal)
            task.thresholds3 = np.random.uniform(-0.3, 0.3, size=k)

        if family == "tree":
            k = min(3, n_causal)
            task.leaf_labels = _sample_tree_leaf_labels(k)

        tasks.append(task)
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
