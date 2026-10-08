"""Features hidden from the demonstrations (notebook 03.1; generator_spec.pdf,
demonstrations without the shortcut; hiccups/20).

The demo-mask experiment removes the shortcut feature, or the shortcut and the
noise feature, from every demonstration line, while queries keep all ten
features. Which displayed feature is which comes from the ground truth fixed
before the column-role permutation: the generator always puts the shortcut in
its column 8 and the noise feature in column 9 (`SyntheticTask.spurious_idx`,
`noise_idx`), and the manifest records where the permutation displays every
generator column. Nothing is estimated from the data.

Selection must not see a hidden feature either, so the grid restricts the
selection context to `visible_features`, and the strategies that select on the
shortcut itself are not defined under a mask that hides it.
"""

from __future__ import annotations

from dataclasses import fields

from src.data.generator import SyntheticTask
from src.data.synthetic_bridge import FEATURE_NAMES

DEMO_MASKS = {
    "none": (),
    "spurious": ("spurious",),
    "spurious_noise": ("spurious", "noise"),
}

# Generator column of each maskable role, read from the generator itself.
_DEFAULTS = {f.name: f.default for f in fields(SyntheticTask)}
GENERATOR_COLUMNS = {"spurious": _DEFAULTS["spurious_idx"], "noise": _DEFAULTS["noise_idx"]}

# These strategies choose rows by the shortcut's agreement with the label.
NEEDS_SHORTCUT = frozenset({"counter_spurious", "counter_prior_matched"})


def hidden_features(entry: dict, mask: str) -> tuple[str, ...]:
    """Displayed names of the features a mask hides, for a manifest entry.

    Each name is the displayed position of the role's generator column under
    the task's recorded permutation (perm[g] = displayed position of generator
    column g), checked against the roles the manifest stores.
    """
    if mask not in DEMO_MASKS:
        raise ValueError(f"unknown demo mask {mask!r}; expected one of {tuple(DEMO_MASKS)}")
    perm = entry["permutation"]
    names = []
    for role in DEMO_MASKS[mask]:
        name = f"f{int(perm[GENERATOR_COLUMNS[role]])}"
        if entry["roles"][role] != name:
            raise ValueError(f"{entry['task_id']}: the manifest's {role} feature {entry['roles'][role]!r} "
                             f"is not generator column {GENERATOR_COLUMNS[role]} ({name!r})")
        names.append(name)
    return tuple(names)


def visible_features(entry: dict, mask: str) -> list[str]:
    """The displayed features a demonstration shows under a mask, in display order."""
    hidden = set(hidden_features(entry, mask))
    return [f for f in FEATURE_NAMES if f not in hidden]


def check_strategies(strategies, mask: str) -> None:
    """Refuse strategies that select on the shortcut when the mask hides it."""
    if "spurious" in DEMO_MASKS.get(mask, ()):
        bad = sorted(set(strategies) & NEEDS_SHORTCUT)
        if bad:
            raise ValueError(f"{bad} select on the shortcut feature, which the {mask!r} mask hides")
