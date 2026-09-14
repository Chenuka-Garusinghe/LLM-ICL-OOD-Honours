"""Local MLX inference backend (Stage 1g -- dev-loop productivity on Apple
Silicon, branch: local-testing-w-mlx-models).

Mirrors VLLMRunner's interface (`batch_predict`, `get_confidence` semantics)
so notebooks/scripts can swap backends via a config flag / INFERENCE_BACKEND
env var without per-call-site branching. Chat-templates every prompt via
src/inference/chat.py::ChatFormatter (Stage 1a) before generation -- this is
the actual instrument fix (REDESIGN_RATIONALE.md §4.1/§5.1), not something
`MLXRunner` does differently from what the HPC/vLLM path should also do once
Stage 4 wires the chat-template call site into the notebooks.

Constrained two-way classification is done directly: one forward pass per
prompt via the loaded MLX model, logits at the final position, softmax
restricted to the two label-token ids -- mirrors
src/inference/llm_runner.py::get_confidence's math exactly, no sampling
params needed since decoding is greedy/constrained by construction (we only
ever read the two label-token logits, never actually sample).
"""

from __future__ import annotations

import numpy as np

from src.inference.chat import ChatFormatter
from src.inference.llm_runner import INVALID, PredictionResult, get_confidence


def resolve_label_token_ids(tokenizer, label_tokens: tuple[str, str]) -> list[int] | None:
    """Same contract as src/inference/llm_runner.py's function of the same
    name: None if a label token isn't a single vocab piece for this
    tokenizer (caller should treat that prompt as unconstrained/INVALID).
    """
    ids: list[int] = []
    for tok in label_tokens:
        try:
            enc = tokenizer.encode(tok, add_special_tokens=False)
        except TypeError:
            enc = tokenizer.encode(tok)
        if len(enc) != 1:
            return None
        ids.append(int(enc[0]))
    return ids


class MLXRunner:
    """Loads one MLX model (mlx-community bf16 or quantised checkpoint) and
    serves constrained single-token classification, exactly like
    VLLMRunner.batch_predict but via a per-prompt forward pass instead of
    vLLM's batched generate().
    """

    def __init__(self, model_path: str):
        from mlx_lm import load

        self.model, self.tokenizer = load(model_path)

    def chat_formatter(self) -> ChatFormatter:
        return ChatFormatter(self.tokenizer)

    def _next_token_logprobs(self, prompt: str) -> dict[str, float]:
        """Run one forward pass, return {decoded_token: logprob} for the
        full vocab at the final position -- mirrors what
        src/inference/llm_runner.py's batch_predict extracts from vLLM's
        per-token Logprob objects (decoded, stripped).
        """
        import mlx.core as mx

        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = self.model(input_ids)[0, -1, :].astype(mx.float32)  # (vocab,)
        log_probs = logits - mx.logsumexp(logits)
        # NumPy has no bfloat16 dtype, so an mx.array still in the model's
        # native bfloat16 fails np.array()'s buffer-protocol conversion
        # ("Item size 2 ... does not match ... item size 1") -- the
        # .astype(mx.float32) above must happen before crossing over.
        log_probs = np.array(log_probs)

        # Only the two label-token ids are ever consulted downstream (via
        # get_confidence), so there's no need to materialise/decode the
        # entire vocab -- callers pass label_tokens through
        # resolve_label_token_ids and index directly.
        return log_probs

    def batch_predict(self, prompts: list[str], label_tokens: tuple[str, str]) -> list[PredictionResult]:
        allowed_ids = resolve_label_token_ids(self.tokenizer, label_tokens)
        if allowed_ids is None:
            return [
                PredictionResult(prediction=INVALID, confidence=0.0, p0=0.0, p1=0.0, logprob_0=-100.0, logprob_1=-100.0)
                for _ in prompts
            ]

        results = []
        for prompt in prompts:
            log_probs = self._next_token_logprobs(prompt)
            logprobs_dict = {
                label_tokens[0]: float(log_probs[allowed_ids[0]]),
                label_tokens[1]: float(log_probs[allowed_ids[1]]),
            }
            results.append(get_confidence(logprobs_dict, label_tokens))
        return results
