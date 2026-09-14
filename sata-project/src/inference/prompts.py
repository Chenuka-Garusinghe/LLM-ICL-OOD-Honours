"""System instruction and prompt templates for the classifier LLM."""

from __future__ import annotations

# v1's synthetic task_description ("the label of a synthetic binary
# classification task", set in Notebook 06) told the model nothing about
# *what kind* of task this is -- there is no statement that a rule over the
# features exists and should be induced from the labelled examples. Under
# the Bayesian view of ICL (lit review ref [14]), the prompt is the evidence
# the model uses to infer the latent task; v1's prompt didn't identify the
# task as rule induction at all. This description states that plainly while
# deliberately keeping abstract `feature_N` names (no real-world priors
# injected into the synthetic arm). See REDESIGN_RATIONALE.md §4.1/§5.1.
SYNTHETIC_TASK_DESCRIPTION = (
    "each example lists 10 numeric measurements and its category; the category "
    "is determined by an unknown rule over some of the measurements; infer the "
    "rule from the labelled examples and classify the final one"
)

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


def build_chat_messages(
    task_description: str,
    label_tokens: tuple[str, str],
    demo_lines: list[str],
    query_line: str,
    label_meanings: tuple[str, str] | None = None,
) -> list[dict[str, str]]:
    """Same content as `build_classification_prompt`, split into a
    (system, user) message pair for src/inference/chat.py::ChatFormatter --
    the v2 prompt path (Stage 1a). `build_classification_prompt` (the v1
    raw-string builder) is kept unchanged for the `prompt_version`
    comparability column.
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
    user = f"{body}\n\n{query_line}" if body else query_line
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_feature_ranking_prompt(
    task_description: str, feature_list: list[str], label_description: str
) -> str:
    return FEATURE_RANKING_TEMPLATE.format(
        task_description=task_description,
        feature_list=", ".join(feature_list),
        label_description=label_description,
    )
