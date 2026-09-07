"""vLLM interface: prompt -> prediction + logprobs.

Uses SamplingParams(logprobs=True, max_tokens=1, temperature=0) and extracts
logprobs for the two label tokens via a constrained softmax, so that
"confidence" is comparable across conditions regardless of the full
vocabulary distribution.
"""

from __future__ import annotations

import gc
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

    def shutdown(self) -> None:
        """Best-effort GPU memory release -- NOT reliable, do not depend on it.

        VLLM_ENABLE_V1_MULTIPROCESSING=0 (see module docstring) makes vLLM
        run its engine in-process rather than in an isolated subprocess, so
        just dropping/reassigning the `runner` variable does not return its
        CUDA memory to the driver. This method's del + gc.collect +
        torch.cuda.empty_cache + destroy_model_parallel/destroy_distributed_
        environment was an attempt to force that release, but confirmed
        NOT sufficient in practice: a second VLLMRunner still failed to
        init with "Free memory ... is less than desired GPU memory
        utilization" after calling this on the first. vLLM's V1 engine has
        no reliable in-process model-unload. Use VLLMWorkerRunner (same
        interface, runs the engine in a real OS subprocess) for any loop
        that constructs more than one runner in the same kernel/process --
        e.g. looping over config.base_llms.
        """
        try:
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )
            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception:
            pass

        del self.llm
        gc.collect()

        try:
            import torch
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass

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


class VLLMWorkerRunner:
    """Same interface as VLLMRunner, but runs vLLM in a genuinely separate
    OS subprocess (src/inference/vllm_worker.py) instead of in-process.

    Use this, not VLLMRunner, in any loop that constructs more than one
    runner in the same kernel/process (e.g. looping over config.base_llms)
    -- see VLLMRunner.shutdown's docstring for why in-process teardown
    between models does not reliably free GPU memory. Exiting the
    subprocess on shutdown() forces the CUDA driver to reclaim it.
    """

    def __init__(self, model_path: str, tensor_parallel: int = 1, gpu_memory_utilisation: float = 0.9, max_model_len: int = 4096, ready_timeout: float = 1800):
        import subprocess
        import sys
        import uuid
        from pathlib import Path

        project_root = Path(__file__).resolve().parents[2]
        self._socket_path = f"/tmp/vllm_worker_{uuid.uuid4().hex}.sock"
        self._proc = subprocess.Popen(
            [
                sys.executable, "-m", "src.inference.vllm_worker",
                self._socket_path, model_path,
                str(tensor_parallel), str(gpu_memory_utilisation), str(max_model_len),
            ],
            cwd=str(project_root),
        )
        self._conn = self._connect(ready_timeout)

    def _connect(self, timeout: float):
        import time
        from multiprocessing.connection import Client

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"vLLM worker process exited early (code {self._proc.returncode}) "
                    "before it finished loading the model -- check the notebook's "
                    "stderr output above for the worker's own traceback."
                )
            try:
                return Client(self._socket_path, family="AF_UNIX")
            except (FileNotFoundError, ConnectionRefusedError):
                time.sleep(1)
        raise TimeoutError(f"vLLM worker did not become ready within {timeout}s")

    def generate_text(self, prompts: list[str], max_tokens: int = 128) -> list[str]:
        self._conn.send({"op": "generate_text", "prompts": prompts, "max_tokens": max_tokens})
        return self._conn.recv()

    def batch_predict(self, prompts: list[str], label_tokens: tuple[str, str]) -> list[PredictionResult]:
        self._conn.send({"op": "batch_predict", "prompts": prompts, "label_tokens": list(label_tokens)})
        return self._conn.recv()

    def shutdown(self) -> None:
        """Tell the worker to exit and wait for it -- this is what actually
        releases the GPU memory (the OS reclaiming a terminated process's
        CUDA context), unlike VLLMRunner.shutdown's in-process best-effort.
        """
        import contextlib
        import os as _os

        try:
            self._conn.send({"op": "shutdown"})
            self._conn.close()
        except Exception:
            pass

        try:
            self._proc.wait(timeout=60)
        except Exception:
            self._proc.kill()
            self._proc.wait(timeout=30)

        with contextlib.suppress(FileNotFoundError):
            _os.remove(self._socket_path)
