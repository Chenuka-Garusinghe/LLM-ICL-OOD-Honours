"""Prompt templates (generator_spec.pdf, prompt templates).

One template serves every condition. The system message states the task as
rule induction over standardised measurements; under naming it adds what the
two labels mean in the task's domain. The aligned and flipped namings share
a domain, so their system messages are byte-identical, and the label tokens
are `0`/`1` everywhere.

The user message is the newline-joined demonstrations followed by the query
line (serialisation.py); a zero-shot prompt holds only the query.

- Instruct models: the (system, user) pair goes through the tokenizer's chat
  template with the assistant generation prompt, so the next token is the
  answer.
- Base model: a raw completion prompt, the system text, a blank line, the
  demonstrations, then the query ending in "-> " with a trailing space.
  Demonstrations end in "-> 1", which tokenises as "->", " ", "1" (" 1" is
  not a single token), so the trailing space makes the answer token "0" or
  "1", as in the demonstrations.

v2's template ("Given the features of an individual, predict Classify the
input ...") was ungrammatical and a leftover from the real-data arm.
"""

from __future__ import annotations

SYSTEM_TEMPLATE = (
    "You are a classifier. {subject} All measurements are standardised "
    "(0 = average, 1 = one standard deviation above average). The label follows "
    "an unknown rule over some of the measurements. Infer the rule from the "
    "labelled examples, then classify the final example. Respond with exactly "
    "one character: 0 or 1."
)

ABSTRACT_SUBJECT = "Each example lists {n} numeric measurements and a label (0 or 1)."

# Named conditions (P3). The sentence depends only on the domain, never on the naming.
DOMAIN_SUBJECTS = {
    "loan": (
        "Each example describes a loan applicant with {n} measurements and a label: "
        "1 means the loan was approved, 0 means it was denied."
    ),
    "medical": (
        "Each example describes a patient with {n} measurements and a label: "
        "1 means the patient is at high risk, 0 means low risk."
    ),
}

# RQ3 self-report (pi_self): the same framing without the answer-format line,
# then the demonstrations and a ranking request instead of a query.
RANKING_SYSTEM_TEMPLATE = (
    "You are a classifier. {subject} All measurements are standardised "
    "(0 = average, 1 = one standard deviation above average). The label follows "
    "an unknown rule over some of the measurements. Infer the rule from the "
    "labelled examples."
)
RANKING_REQUEST = (
    "Rank all {n} measurements from most to least important for predicting the label. "
    "Respond with the measurement names only, separated by commas, most important first."
)

FEATURE_RANKING_TEMPLATE = (
    "You are analyzing a {task_description} prediction task.\n"
    "Given the following features: {feature_list}\n"
    "Rank these features from most important to least important for predicting {label_description}.\n"
    "Respond with a comma-separated list of feature names, most important first."
)


def _subject(domain: str | None, n_features: int | None) -> str:
    """The sentence describing an example. `n_features=None` drops the count, for
    prompts whose demonstrations and query show different numbers of
    measurements (the demo-mask experiment, notebook 03.1)."""
    subject = ABSTRACT_SUBJECT if domain is None else DOMAIN_SUBJECTS[domain]
    if n_features is None:
        return subject.replace("{n} ", "")
    return subject.format(n=n_features)


def system_message(domain: str | None = None, n_features: int | None = 10) -> str:
    """System message; `domain=None` is the abstract condition."""
    return SYSTEM_TEMPLATE.format(subject=_subject(domain, n_features))


def ranking_system_message(domain: str | None = None, n_features: int | None = 10) -> str:
    """System message of the RQ3 feature-ranking prompt; `domain=None` is abstract."""
    return RANKING_SYSTEM_TEMPLATE.format(subject=_subject(domain, n_features))


def ranking_request(n_features: int = 10) -> str:
    """The ranking request, placed after a blank line below the demonstrations."""
    return "\n" + RANKING_REQUEST.format(n=n_features)


def user_message(demo_lines: list[str], query_line: str) -> str:
    return "\n".join([*demo_lines, query_line])


def completion_prompt(system: str, demo_lines: list[str], query_line: str) -> str:
    """Raw completion prompt for base models (no chat template)."""
    return f"{system}\n\n" + "\n".join([*demo_lines, f"{query_line} "])


def render_prompt(
    tokenizer,
    system: str,
    demo_lines: list[str],
    query_line: str,
    use_chat_template: bool = True,
) -> str:
    """The full prompt string the model scores."""
    if not use_chat_template:
        return completion_prompt(system, demo_lines, query_line)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message(demo_lines, query_line)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_feature_ranking_prompt(
    task_description: str, feature_list: list[str], label_description: str
) -> str:
    return FEATURE_RANKING_TEMPLATE.format(
        task_description=task_description,
        feature_list=", ".join(feature_list),
        label_description=label_description,
    )
