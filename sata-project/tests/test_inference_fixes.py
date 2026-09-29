"""Unit tests for Stage 1's inference-layer fixes (REDESIGN_RATIONALE.md §5.1).

These need no live model -- ChatFormatter is exercised against a hand-written
stub whose `apply_chat_template` mimics the documented HF signature.
"""

from __future__ import annotations

import numpy as np

from src.data.serialisation import _format_number
from src.inference.calibration import calibrate
from src.inference.chat import ChatFormatter
from src.selection.ordering import shuffle_order


class _StubTokenizer:
    """Records the call it received so the test can assert on it, and
    returns a deterministic rendering that includes both messages -- close
    enough to a real chat template's shape without needing `transformers`.
    """

    def __init__(self):
        self.last_call = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.last_call = {
            "messages": messages,
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
        }
        system = next(m["content"] for m in messages if m["role"] == "system")
        user = next(m["content"] for m in messages if m["role"] == "user")
        return f"<system>{system}</system><user>{user}</user><assistant>"


def test_chat_formatter_includes_both_messages_and_generation_prompt():
    tokenizer = _StubTokenizer()
    formatter = ChatFormatter(tokenizer)
    rendered = formatter.render(system="You are a classifier.", user="feature_0: 1.00 ->")

    assert "You are a classifier." in rendered
    assert "feature_0: 1.00 ->" in rendered
    assert rendered.endswith("<assistant>")
    assert tokenizer.last_call["tokenize"] is False
    assert tokenizer.last_call["add_generation_prompt"] is True
    assert len(tokenizer.last_call["messages"]) == 2


def test_format_number_is_fixed_width():
    assert _format_number(0, 2) == "0.00"
    assert _format_number(0.5, 2) == "0.50"
    assert _format_number(-1.236, 2) == "-1.24"
    assert _format_number(3, 2) == "3.00"


def test_calibrate_collapses_to_uniform_when_raw_equals_content_free():
    # If the raw prediction carries no evidence beyond the model's inherent
    # output bias (raw == content-free), calibration should remove all of
    # it, landing exactly at (0.5, 0.5).
    lp0, lp1 = np.log(0.9), np.log(0.1)
    p0, p1 = calibrate(lp0, lp1, lp0, lp1)
    assert abs(p0 - 0.5) < 1e-6
    assert abs(p1 - 0.5) < 1e-6


def test_calibrate_amplifies_evidence_beyond_the_content_free_prior():
    # Raw is MORE confident in class 0 than the content-free baseline
    # already is -- there is genuine evidence beyond the prior bias, so
    # calibration should push further toward class 0, not toward 0.5.
    p0, p1 = calibrate(np.log(0.99), np.log(0.01), np.log(0.9), np.log(0.1))
    assert p0 > 0.9
    assert p0 + p1 == 1.0 or abs(p0 + p1 - 1.0) < 1e-9


def test_shuffle_order_is_a_permutation_and_deterministic_per_seed():
    demo_ids = [10, 11, 12, 13, 14, 15, 16, 17]
    order_a = shuffle_order(demo_ids, seed=42)
    order_b = shuffle_order(demo_ids, seed=42)
    order_c = shuffle_order(demo_ids, seed=7)

    assert sorted(order_a) == sorted(demo_ids)
    assert order_a == order_b
    assert order_a != order_c
