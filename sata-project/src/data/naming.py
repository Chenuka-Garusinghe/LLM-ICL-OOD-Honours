"""Naming conditions (generator_spec.pdf, naming protocol).

Every task is shown under three namings:

- abstract: every column keeps its displayed name, f0 to f9;
- aligned: each load-bearing column is named by one member of an antonym
  pair, the member whose real-world association with label 1 agrees with the
  rule's direction s_j;
- flipped: the other member of the same pair, so the name's association
  contradicts the rule.

The 7 other columns (distractors, spurious, noise) get weak-prior names that
are identical in aligned and flipped. Pairs and weak names are drawn per task
from a stream that does not depend on the naming, so the aligned and flipped
prompts of a task differ only in the three load-bearing names
(Proposition "Identification"). Lexicons live in configs/lexicons.yaml; the
grid uses only the `final` names, which passed the P3 screening
(src/inference/priors.py).
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from src.data.synthetic_bridge import FEATURE_NAMES

NAMINGS = ("abstract", "aligned", "flipped")
LEXICON_PATH = Path(__file__).resolve().parents[2] / "configs" / "lexicons.yaml"
N_PAIRS_PER_TASK = 3


@dataclass(frozen=True)
class Lexicon:
    domain: str
    pairs: tuple[tuple[str, str], ...]   # (raises P(label 1), lowers it)
    weak: tuple[str, ...]

    def direction(self, name: str) -> int:
        """+1 for a pair's first member, -1 for its second, 0 for a weak name."""
        for pos, neg in self.pairs:
            if name == pos:
                return 1
            if name == neg:
                return -1
        if name in self.weak:
            return 0
        raise KeyError(f"{name!r} is not in the {self.domain} lexicon")

    @property
    def names(self) -> list[str]:
        return [n for pair in self.pairs for n in pair] + list(self.weak)


def load_lexicons(stage: str = "final", path: str | Path = LEXICON_PATH) -> dict[str, Lexicon]:
    """Lexicons per domain; `stage` is "final" (screened) or "candidates"."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    out = {}
    for domain, spec in raw["domains"].items():
        block = spec.get(stage)
        if block is None:
            continue
        out[domain] = Lexicon(domain, tuple(tuple(p) for p in block["pairs"]), tuple(block["weak"]))
    return out


def screening_rules(path: str | Path = LEXICON_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)["screening"]


def naming_seed(base_seed: int, task_id: str) -> int:
    """One stream per task, shared by the aligned and flipped namings."""
    ss = np.random.SeedSequence([int(base_seed), zlib.crc32(task_id.encode()), zlib.crc32(b"naming")])
    return int(ss.generate_state(1)[0])


def assign_names(entry: dict, naming: str, lexicon: Lexicon | None, base_seed: int) -> dict[str, str]:
    """Display name of every displayed column of a task (manifest `entry`)."""
    if naming == "abstract":
        return {c: c for c in FEATURE_NAMES}
    if naming not in NAMINGS:
        raise ValueError(f"unknown naming {naming!r}; expected one of {NAMINGS}")
    if lexicon is None:
        raise ValueError(f"naming {naming!r} needs the {entry['domain']!r} lexicon")
    load_bearing = list(entry["roles"]["load_bearing"])
    others = [c for c in FEATURE_NAMES if c not in load_bearing]
    if len(lexicon.pairs) < len(load_bearing) or len(lexicon.weak) < len(others):
        raise ValueError(f"the {lexicon.domain} lexicon needs {len(load_bearing)} pairs and "
                         f"{len(others)} weak names; it has {len(lexicon.pairs)} and {len(lexicon.weak)}")
    rng = np.random.default_rng(naming_seed(base_seed, entry["task_id"]))
    pair_idx = rng.choice(len(lexicon.pairs), size=len(load_bearing), replace=False)
    weak_idx = rng.permutation(len(lexicon.weak))[: len(others)]
    names = {c: lexicon.weak[w] for c, w in zip(others, weak_idx)}
    for col, m in zip(load_bearing, pair_idx):
        pos, neg = lexicon.pairs[m]
        agrees = entry["directions"][col] > 0
        if naming == "aligned":
            names[col] = pos if agrees else neg
        else:
            names[col] = neg if agrees else pos
    return {c: names[c] for c in FEATURE_NAMES}


def name_directions(names: dict[str, str], lexicon: Lexicon | None) -> dict[str, int]:
    """The direction each column's name suggests (+1, -1, or 0 for weak and abstract names)."""
    if lexicon is None:
        return {c: 0 for c in names}
    return {c: lexicon.direction(n) for c, n in names.items()}


def codebook(names: dict[str, str]) -> dict:
    """Codebook for `serialise_row` (display names by column)."""
    return {c: {"name_extended": n, "type": "continuous"} for c, n in names.items()}


def naming_table(entries: list[dict], lexicons: dict[str, Lexicon], base_seed: int,
                 namings: tuple[str, ...] = NAMINGS) -> dict:
    """Names of every task under every naming, for the run record."""
    return {e["task_id"]: {n: assign_names(e, n, lexicons.get(e["domain"]), base_seed) for n in namings}
            for e in entries}
