"""vLLM interface: prompt -> prediction + logprobs.

Uses SamplingParams(logprobs=True, max_tokens=1, temperature=0) and extracts
logprobs for the two label tokens via a constrained softmax, so that
"confidence" is comparable across conditions regardless of the full
vocabulary distribution.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# vLLM's V1 engine normally runs its worker in a *subprocess*
# (VLLM_ENABLE_V1_MULTIPROCESSING=1, the default), forked by default. That's
# broken when the parent process already has an initialized CUDA context --
# which is exactly the case here, since VLLMRunner is constructed from
# inside a live Jupyter kernel: "Cannot re-initialize CUDA in forked
# subprocess. To use CUDA with multiprocessing, you must use the 'spawn'
# start method."
#
# Switching just the start method to "spawn" does NOT fix this in a Jupyter
# kernel (verified) -- it trades that crash for a second one, because
# multiprocessing's spawn re-imports `__main__` in the child to reconstruct
# state, and `__main__` here is ipykernel's own launcher, which has side
# effects on import (RuntimeError: "An attempt has been made to start a new
# process before the current process has finished its bootstrapping phase").
#
# The fix that actually works: disable the V1 engine's subprocess entirely
# so it runs in-process (InprocClient) -- no fork, no spawn, no re-import.
# Confirmed working end-to-end (model load + real generation) on a gpuvolta
# V100 node.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

# Separately: V100 (compute capability 7.0) hits an unrelated bug in
# flashinfer's CUDA-arch JIT check (`minor.isdigit()` called on an int,
# flashinfer's own bug) when vLLM tries to use its optimized sampling
# kernel, which it does by default whenever flashinfer is importable.
# Disabling it falls back to vLLM's plain PyTorch top-k/top-p sampler --
# functionally identical, just without the JIT-compiled kernel. Confirmed
# needed on V100; harmless to force off on other GPUs too.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

INVALID = "INVALID"


@dataclass
class PredictionResult:
    prediction: str
    confidence: float
    p0: float
    p1: float
    logprob_0: float
    logprob_1: float


def get_confidence(logprobs_dict: dict[str, float], label_tokens: tuple[str, str]) -> PredictionResult:
    lp0 = logprobs_dict.get(label_tokens[0], -100)
    lp1 = logprobs_dict.get(label_tokens[1], -100)
    p0 = np.exp(lp0) / (np.exp(lp0) + np.exp(lp1))
    p1 = 1 - p0
    pred = label_tokens[0] if p0 >= p1 else label_tokens[1]
    confidence = max(p0, p1)
    return PredictionResult(prediction=pred, confidence=confidence, p0=p0, p1=p1, logprob_0=lp0, logprob_1=lp1)


class VLLMRunner:
    """Thin wrapper around vllm.LLM for constrained single-token classification."""

    def __init__(self, model_path: str, tensor_parallel: int = 1, gpu_memory_utilisation: float = 0.9, max_model_len: int = 4096):
        from vllm import LLM

        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel,
            gpu_memory_utilization=gpu_memory_utilisation,
            max_model_len=max_model_len,
        )

    def generate_text(self, prompts: list[str], max_tokens: int = 128) -> list[str]:
        """Unconstrained free-text generation (e.g. Notebook 03's feature-ranking
        prompt), as opposed to batch_predict's constrained single-token classification.
        """
        from vllm import SamplingParams

        sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0)
        outputs = self.llm.generate(prompts, sampling_params)
        return [out.outputs[0].text.strip() for out in outputs]

    def batch_predict(self, prompts: list[str], label_tokens: tuple[str, str]) -> list[PredictionResult]:
        """Run a batch of prompts through vLLM and return per-prompt predictions.

        Top token not in the label set (rare with constrained decoding) is
        logged as INVALID: excluded from accuracy but included in the count
        by the caller.
        """
        from vllm import SamplingParams

        sampling_params = SamplingParams(logprobs=20, max_tokens=1, temperature=0)
        outputs = self.llm.generate(prompts, sampling_params)

        results = []
        for out in outputs:
            token_logprobs = out.outputs[0].logprobs[0]  # dict-like: token -> Logprob
            logprobs_dict = {
                lp.decoded_token.strip(): lp.logprob for lp in token_logprobs.values()
            }
            if label_tokens[0] not in logprobs_dict and label_tokens[1] not in logprobs_dict:
                results.append(
                    PredictionResult(
                        prediction=INVALID, confidence=0.0, p0=0.0, p1=0.0, logprob_0=-100.0, logprob_1=-100.0
                    )
                )
                continue
            results.append(get_confidence(logprobs_dict, label_tokens))
        return results
