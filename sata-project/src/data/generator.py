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

v3 (synthetic-experiments): evaluation and pilot tasks come from
`sample_eval_tasks` (linear + tree, exactly 3 load-bearing features, per-task
seeds); `_sample_tasks` is the older sampler, still used for SATA training
until P5 moves it to the evaluation distribution.
"""

from __future__ import annotations

import random
import warnings
import zlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.stats

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
    # v3 evaluation tasks (`sample_eval_tasks`): presentation domain for the
    # P3 naming factor, and the seed every data split of the task derives from.
    domain: str | None = field(default=None)
    data_seed: int | None = field(default=None)
    # Direction of the spurious feature: f8 = spurious_sign * ((2y - 1) d + e).
    # `sample_eval_tasks` gives half the tasks each sign, so a model prior such
    # as "bigger values mean 1" backs the shortcut in only half of them (P2).
    spurious_sign: int = field(default=1)
    # Details of the last `generate_environment` call (covariate shift draw,
    # probe base rates); the bridge copies them into the task manifest.
    last_env_info: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def load_bearing_features(self) -> list[int]:
        """The causal features the rule actually reads.

        `tree` and `threshold` read only the first 3 causal features, and
        `sparse_interaction` the first 2, so with more causal features the
        rest are decoys. v2 computed regimes over all causal features, which
        split each true tree leaf in two or four.
        """
        feats = list(self.causal_features)
        if self.rule_family == "tree":
            return feats[: int(np.log2(len(self.leaf_labels)))]
        if self.rule_family == "threshold":
            return feats[: len(self.thresholds3)]
        if self.rule_family == "sparse_interaction":
            return [feats[0], feats[min(1, len(feats) - 1)]]
        return feats

    def directions(self) -> list[int]:
        """Direction s_j in {+1, -1} of each load-bearing feature.

        Linear: sign of the coefficient (monotone for every value of the other
        features). Tree: the literal signs of the signed majority, read off the
        leaf table as the sign of the label difference between bit_j = 1 and
        bit_j = 0. Threshold rules are non-decreasing in every feature.
        `sparse_interaction` is not monotone, so it has no directions.
        """
        if self.rule_family == "linear":
            return [int(np.sign(c)) for c in np.asarray(self.coefficients)]
        if self.rule_family == "tree":
            leaf_labels = np.asarray(self.leaf_labels)
            k = int(np.log2(len(leaf_labels)))
            bits = (np.arange(2 ** k)[:, None] >> np.arange(k)) & 1
            return [
                int(np.sign(leaf_labels[bits[:, j] == 1].mean() - leaf_labels[bits[:, j] == 0].mean()))
                for j in range(k)
            ]
        if self.rule_family == "threshold":
            return [1] * len(self.thresholds3)
        raise ValueError(f"rule family {self.rule_family!r} is not monotone, so it has no directions")

    def to_meta(self) -> dict[str, Any]:
        """Every parameter needed to rebuild the task, as JSON-safe values."""
        def _list(v):
            return None if v is None else np.asarray(v).tolist()

        return {
            "task_id": self.task_id,
            "rule_family": self.rule_family,
            "causal_features": [int(f) for f in self.causal_features],
            "coefficients": _list(self.coefficients),
            "spurious_strength": float(self.spurious_strength),
            "n_features": int(self.n_features),
            "label_noise": float(self.label_noise),
            "threshold": float(self.threshold),
            "thresholds3": _list(self.thresholds3),
            "leaf_labels": _list(self.leaf_labels),
            "domain": self.domain,
            "data_seed": None if self.data_seed is None else int(self.data_seed),
            "spurious_sign": int(self.spurious_sign),
        }

    @classmethod
    def from_meta(cls, meta: dict[str, Any]) -> "SyntheticTask":
        """Inverse of `to_meta`."""
        return cls(
            task_id=meta["task_id"],
            rule_family=meta["rule_family"],
            causal_features=list(meta["causal_features"]),
            coefficients=np.asarray(meta["coefficients"], dtype=float),
            spurious_strength=float(meta["spurious_strength"]),
            n_features=int(meta.get("n_features", 10)),
            label_noise=float(meta.get("label_noise", 0.05)),
            threshold=float(meta.get("threshold", 0.0)),
            thresholds3=None if meta.get("thresholds3") is None else np.asarray(meta["thresholds3"], dtype=float),
            leaf_labels=None if meta.get("leaf_labels") is None else np.asarray(meta["leaf_labels"], dtype=int),
            domain=meta.get("domain"),
            data_seed=meta.get("data_seed"),
            spurious_sign=int(meta.get("spurious_sign", 1)),
        )

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
            logit = np.clip(causal @ self.coefficients, -500, 500)
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

    def _probe_base_rate(self, perturb, n_probe: int, rng: np.random.Generator) -> float:
        """Approximate P(y_clean=1) for a candidate perturbation of fresh base
        features, via an `n_probe`-sample dry run of `_apply_rule` (MAP labels,
        `u=None` -- exact stochastic agreement isn't needed for a calibration
        probe, just the base rate). `perturb(X)` returns the perturbed copy.
        """
        X_probe = self._base_features(n_probe, rng)
        X_probe = perturb(X_probe) #* apply or dont apply (for id) a shift to the ~N(0,1) distributed values
        return float(self._apply_rule(X_probe).mean())

    def _label_neutral_shift(self, rng: np.random.Generator):
        """Covariate shift that moves the rule's own inputs but keeps the label
        rate (generator_spec.pdf, covariate shift), for linear and tree tasks.

        Two of the load-bearing features are shifted, one towards label 1 and
        one towards label 0 by the same amount in label terms, and one
        distractor from features 0-7 by U(1, 2) x Rademacher. Returns
        (features, deltas, roles, push), or None when this task has no such
        shift (other families, or linear coefficients more than 2x apart).

        - Linear: feature j's push on the logit is c_j * delta_j. The pushes
          are +m and -m, so c . x keeps its distribution and the label rate is
          unchanged exactly. m ~ U(max|c|, 2 min|c|) puts both |delta| = m / |c|
          in [1, 2].
        - Tree (the rule reads signs only, so |c| = 1): both features move by
          m ~ U(1, 2), one towards its "1" side and one towards its "0" side.
          Their vote-1 probabilities are Phi(m) and 1 - Phi(m), and the
          label-1 rate of the majority, (p_1 + p_2) / 2, stays 1/2.

        v2 drew 3 features at random and rejection-sampled for a base rate
        within 0.03. A shift of one load-bearing feature never passes that
        test and a shift of distractors always does, so 21 of the 24
        evaluation tasks ended up shifting distractors only.
        """
        if self.rule_family not in ("linear", "tree"):
            return None
        lb = self.load_bearing_features()
        signs = self.directions()
        distractors = [f for f in range(8) if f not in self.causal_features]
        if len(lb) < 2 or not distractors:
            return None
        i, j = rng.choice(len(lb), size=2, replace=False)      # i moves towards label 1, j towards 0
        if self.rule_family == "linear":
            c = np.abs(np.asarray(self.coefficients, dtype=float))
            ci, cj = float(c[i]), float(c[j])
            if max(ci, cj) > 2 * min(ci, cj):
                return None
        else:
            ci = cj = 1.0
        push = float(rng.uniform(max(ci, cj), 2 * min(ci, cj)))
        d = int(rng.choice(distractors))
        feats = [lb[i], lb[j], d]
        deltas = [signs[i] * push / ci, -signs[j] * push / cj, float(rng.uniform(1.0, 2.0) * rng.choice([-1, 1]))]
        return feats, deltas, ["towards_1", "towards_0", "distractor"], push

    def _probe_shift(self, feats, deltas, n_probe: int, rng: np.random.Generator) -> tuple[float, float]:
        """MAP label-1 rate of the same fresh rows before and after a shift (a check, not a filter)."""
        X_probe = self._base_features(n_probe, rng)
        shifted = X_probe.copy()
        shifted[:, feats] += deltas
        return float(self._apply_rule(X_probe).mean()), float(self._apply_rule(shifted).mean())

    def _rejection_shift(self, rng: np.random.Generator):
        """v2's covariate shift, kept for families without a label-neutral
        shift: 3 of features 0-7 moved by U(1, 2) x Rademacher, redrawn (up to
        50 times) until the probed label-1 rate is within 0.03 of the id rate."""
        n_shift = min(3, 8)
        id_rate = self._probe_base_rate(lambda Xp: Xp, 2048, rng)
        cand_feats, cand_delta, shifted_rate = None, None, None
        accepted, n_draws = False, 0
        for _ in range(50):
            n_draws += 1
            cand_feats = rng.choice(8, size=n_shift, replace=False)
            cand_delta = rng.uniform(1.0, 2.0, size=n_shift) * rng.choice([-1, 1], size=n_shift)

            def _apply_shift(Xp, feats=cand_feats, delta=cand_delta):
                Xp = Xp.copy()
                Xp[:, feats] += delta
                return Xp

            shifted_rate = self._probe_base_rate(_apply_shift, 2048, rng)
            if abs(shifted_rate - id_rate) <= 0.03:
                accepted = True
                break
        else:
            warnings.warn(
                f"covariate shift rejection sampling for task {self.task_id} did not find a "
                f"base-rate-preserving shift in 50 tries (best delta={shifted_rate - id_rate:.3f}); "
                "using the last draw anyway."
            )
        return cand_feats, cand_delta, id_rate, shifted_rate, accepted, n_draws

    def _regime(self, X: np.ndarray) -> np.ndarray:
        """Sign pattern of the load-bearing features as an integer id, encoded
        like the tree leaf index (L = b1 + 2 b2 + 4 b3), so for tree tasks the
        regime is the leaf and for linear tasks the orthant."""
        signs = (X[:, self.load_bearing_features()] > 0).astype(int)
        weights = 2 ** np.arange(signs.shape[1])
        return signs @ weights

    def generate_environment(
        self, env_type: str, n_samples: int, seed: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Generate (X, y, metadata) for a given environment type.

        metadata[i] contains: regime, label, y_clean, agree_clean, agree_obs,
        is_counter_spurious and spurious_consistent. Agreement is the sign of
        the spurious feature, read in the task's direction (`spurious_sign`),
        against the label: `agree_clean` against the rule label, `agree_obs`
        against the observed (post-noise) label, which is what demonstrations
        show. `is_counter_spurious` / `spurious_consistent` follow the observed
        label (v2 used the clean one).
        """
        rng = np.random.default_rng(seed)
        X = self._base_features(n_samples, rng)
        coeffs = self.coefficients
        spurious_strength = self.spurious_strength
        self.last_env_info = {"env": env_type, "seed": seed}

        if env_type == "id":
            pass
        elif env_type == "covariate":
            # A pure P(x) shift: move the rule's inputs but not the label rate
            # (shift taxonomy, lit review §2.1; v1's shift moved the base rate
            # to 0.388 vs ~0.49 elsewhere). Features 8/9 are excluded: the
            # spurious and noise features are overwritten below, so shifting
            # them was a silent no-op (Fact 4.4(b)).
            shift = self._label_neutral_shift(rng)
            if shift is not None:
                cand_feats, cand_delta, roles, push = shift
                id_rate, shifted_rate = self._probe_shift(cand_feats, cand_delta, 20000, rng)
                accepted, n_draws, method = True, 1, "label_neutral"
            else:
                cand_feats, cand_delta, id_rate, shifted_rate, accepted, n_draws = self._rejection_shift(rng)
                roles, push, method = None, None, "rejection"
            X[:, cand_feats] += cand_delta
            self.last_env_info.update(
                method=method,
                shift_features=[int(f) for f in cand_feats],
                shift_delta=[float(d) for d in cand_delta],
                shift_roles=roles,
                label_push=None if push is None else float(push),
                probe_rate_id=float(id_rate),
                probe_rate_shifted=float(shifted_rate),
                probe_diff=float(shifted_rate - id_rate),
                accepted=accepted,
                n_draws=n_draws,
            )
        elif env_type == "spurious_reversal":
            spurious_strength = 1.0 - spurious_strength
        elif env_type == "covariate_scale":
            # Variance shift (exploratory, added after P4; hiccups/18). Every
            # covariate except the spurious feature is stretched by its own
            # factor in U(1.5, 3), so each row moves in proportion to its own
            # values: unlike the uniform `covariate` shift this can reorder rows
            # for a scorer whose weights differ from the rule's, and many values
            # leave the demonstrated range. The rule is applied to the stretched
            # values, so P(y | x) is unchanged and the label rate stays 1/2
            # (every feature is symmetric about 0). The spurious feature is
            # generated from the labels as usual, so the shortcut is intact;
            # the noise feature, redrawn below, is stretched there.
            scale_feats = [j for j in range(self.n_features) if j != self.spurious_idx]
            scale = rng.uniform(1.5, 3.0, size=len(scale_feats))
            X[:, scale_feats] *= scale
            self.last_env_info.update(scale_features=[int(j) for j in scale_feats],
                                      scale=[float(a) for a in scale])
        elif env_type == "extrapolation":
            # Per-feature range extension: push 2-3 causal features genuinely
            # outside a 64-row standard-normal pool's support (|x| in [2,4]),
            # rather than one global scale factor -- label-inert for
            # tree/sparse_interaction and erased by inference-time
            # standardisation (Fact 4.4(c)).
            # Drawn per-row (not just per-feature): a single fixed extreme
            # point for every row in the batch would collapse most rule
            # families' rows to one near-constant label instead of a genuine
            # out-of-support *distribution* -- each row keeps its own random
            # sign per extrapolated feature, only the magnitude is pushed
            # beyond the ~[-3,3] typical range of a 64-row standard-normal pool.
            n_extrap = min(int(rng.integers(2, 4)), len(self.causal_features))
            extrap_feats = rng.choice(self.causal_features, size=n_extrap, replace=False)
            magnitude = rng.uniform(2.0, 4.0, size=(n_samples, n_extrap))
            sign = rng.choice([-1.0, 1.0], size=(n_samples, n_extrap))
            X[:, extrap_feats] = sign * magnitude
        elif env_type == "missing_feature":
            # Independent marginal redraw -- an uninformative missing
            # measurement, not a constant that zeroes sparse_interaction's
            # product and forces y_clean == 0 for every row (Fact 4.4(a)).
            drop_idx = self.causal_features[0]
            X[:, drop_idx] = rng.normal(size=n_samples)
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

        # Spurious feature: continuous correlate of the label, reparameterised
        # so sign-agreement equals spurious_strength exactly (Fact 4.2's
        # enabling condition -- v1's discrete {0,1}+-0.1 construction was a
        # near-photocopy of the label at strength >=0.96). If y=1,
        # X8 ~ N(d, 1) with d = Phi^-1(strength), so P(X8>0) = Phi(d) =
        # strength. spurious_reversal's `spurious_strength = 1 - spurious_strength`
        # above flips d's sign, which is exactly the intended "reversed" correlate.
        # `spurious_sign` reflects the whole column, so with -1 the other columns,
        # |f8| and the agreement flags are exactly those of +1.
        d = scipy.stats.norm.ppf(spurious_strength)
        X[:, self.spurious_idx] = self.spurious_sign * ((2 * y_clean - 1) * d + rng.normal(size=n_samples))
        f8_task_direction = self.spurious_sign * X[:, self.spurious_idx]
        agree_clean = np.sign(f8_task_direction) == np.sign(2 * y_clean - 1)
        # Pure noise feature.
        X[:, self.noise_idx] = rng.normal(size=n_samples)
        if env_type == "covariate_scale":
            X[:, self.noise_idx] *= scale[scale_feats.index(self.noise_idx)]

        # Label noise, applied after f8 is generated from the clean label.
        flip = rng.random(n_samples) < self.label_noise
        y = np.where(flip, 1 - y_clean, y_clean)
        agree_obs = np.sign(f8_task_direction) == np.sign(2 * y - 1)

        regimes = self._regime(X)
        metadata = [
            {
                "regime": int(regimes[i]),
                "label": int(y[i]),
                "y_clean": int(y_clean[i]),
                "agree_clean": bool(agree_clean[i]),
                "agree_obs": bool(agree_obs[i]),
                "is_counter_spurious": not bool(agree_obs[i]),
                "spurious_consistent": bool(agree_obs[i]),
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

        if family == "sparse_interaction":
            # v1 left `threshold` at its dataclass default of 0.0 for every
            # task (never assigned here) -- combined with `missing_feature`
            # zeroing the product's first factor, that made the held-out
            # family's missing_feature cell ~98% constant-label (Fact 4.4(a)).
            # Calibrate threshold from a 10k-sample probe of the product of
            # two iid standard-normal features (X's columns are all N(0,1)
            # before any environment perturbation -- see _base_features) so
            # the id-environment base rate lands at a sampled target in
            # U(0.35, 0.65) instead of accidentally on the product's own
            # median-adjacent value.
            probe = np.random.randn(10000, 2)
            target_rate = random.uniform(0.35, 0.65)
            task.threshold = float(np.quantile(probe[:, 0] * probe[:, 1], 1 - target_rate))

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


def signed_majority_leaf_labels(signs: list[int] | np.ndarray) -> np.ndarray:
    """Leaf table of Maj(l1, l2, l3) with l_j = b_j if s_j = +1 else 1 - b_j,
    for leaf index L = b1 + 2 b2 + 4 b3. These 8 tables are exactly the ones
    `_sample_tree_leaf_labels` admits (generator_spec.pdf, tree theorem)."""
    signs = np.asarray(signs)
    k = len(signs)
    bits = (np.arange(2 ** k)[:, None] >> np.arange(k)) & 1
    literals = np.where(signs > 0, bits, 1 - bits)
    return (literals.sum(axis=1) * 2 > k).astype(int)


def _stable_int(text: str) -> int:
    """Seed component from a string that doesn't depend on Python's hash salt."""
    return zlib.crc32(text.encode())


def _spurious_signs(cells: list[tuple], rng: np.random.Generator) -> np.ndarray:
    """+1 or -1 per task, half of each (family, domain) cell in random order.

    An odd cell's extra sign alternates from one odd cell to the next, so a
    family whose two domain cells are odd is still balanced (the pilot suite).
    """
    signs = np.zeros(len(cells), dtype=int)
    extra = int(rng.choice([-1, 1]))
    for cell in sorted(set(cells), key=str):
        idx = [i for i, c in enumerate(cells) if c == cell]
        s = [1, -1] * (len(idx) // 2)
        if len(idx) % 2:
            s.append(extra)
            extra = -extra
        signs[idx] = rng.permutation(s)
    return signs


def sample_eval_tasks(
    n_tasks: int,
    id_prefix: str = "eval",
    base_seed: int = 42,
    spurious_strength_range: tuple[float, float] = (0.80, 0.90),
    label_noise: float = 0.02,
    n_features: int = 10,
    n_causal: int = 3,
    domains: tuple[str, ...] = ("loan", "medical"),
) -> list[SyntheticTask]:
    """The v3 evaluation-task distribution (generator_spec.pdf, task sampling).

    - Family is fixed by position: the first half of the tasks are linear, the
      rest tree. Domain alternates, so each family is split evenly by domain.
    - 3 load-bearing features drawn from indices 0-7 (8 is spurious, 9 noise).
    - Linear: c_j = u_j r_j with u_j ~ U(1, 2) and r_j Rademacher, so every
      load-bearing feature has a definite direction and a non-negligible effect.
    - Tree: uniform over the 8 admissible leaf tables (the signed majorities).
    - Spurious strength s ~ U(spurious_strength_range).
    - Spurious direction: +1 for half of each (family, domain) cell and -1 for
      the other half (`_spurious_signs`). In P2, Qwen's zero-shot margin rose
      with every feature, so with a fixed direction that prior backed the
      shortcut in every task.

    Each task draws its parameters from its own stream,
    SeedSequence([base_seed, prefix, i]), so they do not depend on the other
    tasks; only its family (first half linear) and its spurious direction
    (balanced over the suite, from a separate stream) depend on the suite size.
    The same stream supplies `data_seed`, from which every data split derives.
    """
    tasks: list[SyntheticTask] = []
    n_linear = n_tasks // 2
    families = ["linear" if i < n_linear else "tree" for i in range(n_tasks)]
    task_domains = [domains[i % len(domains)] if domains else None for i in range(n_tasks)]
    spurious_signs = _spurious_signs(
        list(zip(families, task_domains)),
        np.random.default_rng(np.random.SeedSequence([base_seed, _stable_int(id_prefix), _stable_int("spurious_sign")])),
    )
    for i in range(n_tasks):
        rng = np.random.default_rng(np.random.SeedSequence([base_seed, _stable_int(id_prefix), i]))
        family = families[i]
        causal = sorted(int(f) for f in rng.choice(8, size=n_causal, replace=False))
        signs = rng.choice([-1, 1], size=n_causal)
        if family == "linear":
            coefficients = rng.uniform(1.0, 2.0, size=n_causal) * signs
            leaf_labels = None
        else:
            # The tree rule ignores coefficient magnitudes; storing the literal
            # signs keeps `coefficients` meaningful and the stream identical.
            coefficients = signs.astype(float)
            leaf_labels = signed_majority_leaf_labels(signs)
            assert not _is_degenerate_leaf_labels(leaf_labels, n_causal)
            assert np.array_equal(leaf_labels[::-1], 1 - leaf_labels), "complement antisymmetry"
        tasks.append(SyntheticTask(
            task_id=f"{id_prefix}_{i:04d}",
            rule_family=family,
            causal_features=causal,
            coefficients=np.asarray(coefficients, dtype=float),
            spurious_strength=float(rng.uniform(*spurious_strength_range)),
            n_features=n_features,
            label_noise=label_noise,
            leaf_labels=leaf_labels,
            domain=task_domains[i],
            data_seed=int(rng.integers(2**31 - 1)),
            spurious_sign=int(spurious_signs[i]),
        ))
    return tasks


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
