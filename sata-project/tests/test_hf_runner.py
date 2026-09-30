"""Gate G1, model side: prefix caching, tokenisation and prompt snapshots.

The runner tests use the tiny cached `hf-internal-testing/tiny-random-gpt2`,
so they run in seconds. Tokenizer and snapshot tests load the real Qwen2.5
and Llama-3.1 tokenizers from the local Hugging Face cache (no weights) and
are skipped when a tokenizer is not cached.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.serialisation import serialise_row
from src.data.suites import suite_tasks
from src.data.synthetic_bridge import FEATURE_NAMES, build_task_frame, pool_of, queries_of
from src.inference.hf_runner import PROJECT_ROOT, HFRunner, label_log_probs
from src.inference.prompts import completion_prompt, render_prompt, system_message
from src.utils.config import load_config

TINY = "hf-internal-testing/tiny-random-gpt2"
QWEN = "Qwen/Qwen2.5-7B-Instruct"
LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
SNAPSHOTS = Path(__file__).parent / "snapshots"


@pytest.fixture(scope="module")
def tiny():
    try:
        return HFRunner(TINY, device="cpu", dtype="float32", use_chat_template=False, local_files_only=True)
    except OSError:
        pytest.skip(f"{TINY} is not in the local Hugging Face cache")


def _tokenizer(path):
    from dotenv import load_dotenv
    from transformers import AutoTokenizer

    load_dotenv(PROJECT_ROOT / ".env")
    try:
        return AutoTokenizer.from_pretrained(path, local_files_only=True)
    except OSError:
        pytest.skip(f"{path} tokenizer is not in the local Hugging Face cache")


PREFIX = "Rule task.\n\nf0: 0.30; f1: -1.04 -> 1\nf0: -0.20; f1: 0.51 -> 0\n"
SUFFIXES = ["f0: 0.12; f1: 1.10 -> ", "f0: -1.50; f1: -0.02 -> ", "f0: N/A; f1: N/A -> "]


def test_prefix_cached_scoring_matches_full_forward(tiny):
    """G1: label log-probs from the prefix cache equal a full forward pass within 1e-4."""
    cached = tiny.score(PREFIX, SUFFIXES)
    full = tiny.batch_predict([PREFIX + s for s in SUFFIXES])
    for a, b in zip(cached, full):
        assert abs(a.logprob_0 - b.logprob_0) < 1e-4
        assert abs(a.logprob_1 - b.logprob_1) < 1e-4
    assert tiny.n_prefix_fallbacks == 0
    # Scoring twice gives the same answer: the cache is cropped back after each suffix.
    again = tiny.score(PREFIX, SUFFIXES[::-1])[::-1]
    assert all(abs(a.logprob_1 - b.logprob_1) < 1e-5 for a, b in zip(cached, again))


def test_label_log_probs_match_the_models_own_output(tiny):
    """The fp32 label-logit path (decoder + two rows of the output layer) equals
    the model's full forward pass when both run in fp32."""
    import torch

    prompt = PREFIX + SUFFIXES[0]
    with torch.inference_mode():
        ids = torch.tensor([tiny.encode(prompt)])
        logprobs = torch.log_softmax(tiny.model(input_ids=ids).logits[0, -1].float(), dim=-1)
    i0, i1 = tiny.label_ids()
    res = tiny.batch_predict([prompt])[0]
    assert res.logprob_0 == pytest.approx(float(logprobs[i0]), abs=1e-5)
    assert res.logprob_1 == pytest.approx(float(logprobs[i1]), abs=1e-5)


def test_label_probabilities_never_sum_above_one_in_bf16():
    """With a bf16 output layer, the label probabilities stay a sub-distribution
    and the margin equals the fp32 margin (the normaliser uses the fp32 label logits)."""
    import torch

    torch.manual_seed(0)
    head = torch.nn.Linear(64, 500, bias=False).to(torch.bfloat16)
    with torch.no_grad():
        head.weight[:2] *= 40                  # label logits near 20, where bf16 steps are 0.125
    for _ in range(20):
        h = torch.randn(64).to(torch.bfloat16)
        lp0, lp1 = label_log_probs(h, head, [0, 1])
        assert np.exp(lp0) + np.exp(lp1) <= 1 + 1e-6
        exact = (head.weight[:2].float() @ h.float()).tolist()
        assert lp1 - lp0 == pytest.approx(exact[1] - exact[0], abs=1e-4)


def test_broken_token_boundary_falls_back_to_a_full_pass(tiny):
    text = "measurements standardised above average"
    split = next((i for i in range(1, len(text))
                  if tiny.encode(text)[: len(tiny.encode(text[:i]))] != tiny.encode(text[:i])), None)
    if split is None:
        pytest.skip("no BPE boundary break found in the probe text")
    before = tiny.n_prefix_fallbacks
    res = tiny.score(text[:split], [text[split:]])
    assert tiny.n_prefix_fallbacks == before + 1
    assert abs(res[0].logprob_1 - tiny.batch_predict([text])[0].logprob_1) < 1e-5


def test_label_tokens_are_single_pieces(tiny):
    for path in (QWEN, LLAMA):
        tok = _tokenizer(path)
        assert [len(tok.encode(t, add_special_tokens=False)) for t in ("0", "1")] == [1, 1]
    assert len(tiny.label_ids()) == 2


def test_llama_prompt_has_exactly_one_bos():
    """G1: templated prompts are tokenised without adding special tokens (v2 had two BOS)."""
    tok = _tokenizer(LLAMA)
    prompt = render_prompt(tok, system_message(), ["f0: 0.10 -> 1"], "f0: 0.20 ->")
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    assert ids[0] == tok.bos_token_id
    assert ids.count(tok.bos_token_id) == 1
    assert tok(prompt)["input_ids"].count(tok.bos_token_id) == 2   # the v2 bug, for the record


@pytest.mark.parametrize("path", [QWEN, LLAMA])
def test_base_prompt_answer_token_matches_the_demonstrations(path):
    """The base-model prompt ends in "-> "; appending the label reproduces the
    demonstrations' tokens, so the answer token is exactly "0" or "1"."""
    tok = _tokenizer(path)
    prompt = completion_prompt(system_message(), ["f0: 0.10; f1: -0.40 -> 1"], "f0: 0.20; f1: 1.30 ->")
    assert prompt.endswith("-> ")
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    one = tok.encode("1", add_special_tokens=False)
    assert tok(prompt + "1", add_special_tokens=False)["input_ids"] == ids + one


def _snapshot_prompt(tokenizer=None):
    """System message, the first two pool rows of the eval suite's eval_0000 as
    demonstrations and its first ID query (the prompt NB01 and the spec show)."""
    frame, _ = build_task_frame(suite_tasks("eval", load_config())[0])
    pool, query = pool_of(frame), queries_of(frame, "id").iloc[0]
    demos = [serialise_row({f: r[f] for f in FEATURE_NAMES}, label=str(int(r["label"]))) for _, r in pool.head(2).iterrows()]
    q = serialise_row({f: query[f] for f in FEATURE_NAMES})
    if tokenizer is None:
        return completion_prompt(system_message(), demos, q)
    return render_prompt(tokenizer, system_message(), demos, q)


@pytest.mark.parametrize("name,path", [("qwen_instruct", QWEN), ("llama_instruct", LLAMA), ("base", None)])
def test_prompt_snapshot(name, path):
    """G1: the rendered prompt is frozen; the spec quotes the Qwen snapshot.
    After a deliberate prompt change, delete the snapshot file and rerun to refresh it."""
    rendered = _snapshot_prompt(_tokenizer(path) if path else None)
    snap = SNAPSHOTS / f"prompt_{name}.txt"
    if not snap.exists():
        SNAPSHOTS.mkdir(exist_ok=True)
        snap.write_text(rendered)
        pytest.skip(f"wrote new snapshot {snap.name}")
    assert rendered == snap.read_text()
