"""System instruction and prompt templates for the classifier LLM."""

from __future__ import annotations

SYSTEM_TEMPLATE = (
    "You are a classifier. Given the features of an individual, predict {task_description}.\n"
    "Respond with exactly one word: {label_0} or {label_1}."
)

SYSTEM_TEMPLATE_WITH_MEANINGS = (
    "You are a classifier. Given the features of an individual, predict {task_description}.\n"
    "Respond with exactly one word, {label_0} or {label_1}, where "
    "{label_0} = {label_0_meaning} and {label_1} = {label_1_meaning}."
)

FEATURE_RANKING_TEMPLATE = (
    "You are analyzing a {task_description} prediction task.\n"
    "Given the following features: {feature_list}\n"
    "Rank these features from most important to least important for predicting {label_description}.\n"
    "Respond with a comma-separated list of feature names, most important first."
)


def build_classification_prompt(
    task_description: str,
    label_tokens: tuple[str, str],
    demo_lines: list[str],
    query_line: str,
    label_meanings: tuple[str, str] | None = None,
) -> str:
    """Assemble the full classification prompt: system instruction + demos + query.

    `demo_lines` are already-serialised demo rows (see src/data/serialisation.py);
    `query_line` is the serialised query row ending in "->".

    `label_meanings` (real arm only): a ("meaning of label_0", "meaning of
    label_1") pair, e.g. from tableshift_loader.TASK_DESCRIPTIONS. When given,
    the instruction spells out what each label means. When omitted (synthetic
    arm), the original bare "one word: 0 or 1" instruction is used unchanged.
    """
    if label_meanings is None:
        system = SYSTEM_TEMPLATE.format(
            task_description=task_description, label_0=label_tokens[0], label_1=label_tokens[1]
        )
    else:
        system = SYSTEM_TEMPLATE_WITH_MEANINGS.format(
            task_description=task_description,
            label_0=label_tokens[0], label_1=label_tokens[1],
            label_0_meaning=label_meanings[0], label_1_meaning=label_meanings[1],
        )
    body = "\n".join(demo_lines)
    return f"{system}\n\n{body}\n\n{query_line}"


def build_feature_ranking_prompt(
    task_description: str, feature_list: list[str], label_description: str
) -> str:
    return FEATURE_RANKING_TEMPLATE.format(
        task_description=task_description,
        feature_list=", ".join(feature_list),
        label_description=label_description,
    )
