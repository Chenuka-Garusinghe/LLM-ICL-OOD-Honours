"""Uniform demo ordering (Stage 1d).

v1 rendered demos into the prompt in selector-return order. For SATA and
similarity that is *descending* score -- the best demo placed first, i.e.
farthest from the query, where recency-sensitive attention values it least
(Lu et al. 2022, "Fantastically Ordered Prompts"). Random and the
hand-designed protocols returned effectively shuffled order already, so the
ordering nuisance hit precisely the retrieval-based methods asymmetrically.
A single seeded shuffle applied uniformly to every method's selection turns
ordering into a controlled nuisance identical across conditions, since the
hypothesis under test is about selection, not ordering. See
REDESIGN_RATIONALE.md §4.3/§5.1.
"""

from __future__ import annotations

import numpy as np


def shuffle_order(demo_ids: list[int], seed: int) -> list[int]:
    """Return `demo_ids` in a seeded random order.

    The seed should be recorded (see `order_seed` in
    src/utils/results_schema.py::RESULTS_SCHEMA_V2) so the exact ordering is
    reproducible.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(demo_ids))
    return [demo_ids[i] for i in order]
