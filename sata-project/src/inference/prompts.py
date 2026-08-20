"""System instruction and prompt templates for the classifier LLM."""

from __future__ import annotations

SYSTEM_TEMPLATE = (
    "You are a classifier. Given the features of an individual, predict {task_description}.\n"
    "Respond with exactly one word: {label_0} or {label_1}."
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
) -> str:
    """Assemble the full classification prompt: system instruction + demos + query.

    `demo_lines` are already-serialised demo rows (see src/data/serialisation.py);
    `query_line` is the serialised query row ending in "->".
    """
    system = SYSTEM_TEMPLATE.format(
        task_description=task_description, label_0=label_tokens[0], label_1=label_tokens[1]
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
