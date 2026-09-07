"""Standalone subprocess entrypoint for running a single vLLM model.

Launched by VLLMWorkerRunner (src/inference/llm_runner.py) as a genuinely
separate OS process (subprocess.Popen), not merely a new Python object in
the Jupyter kernel. This is a deliberate step up from the in-process
VLLMRunner: vLLM's V1 engine does not reliably return its GPU memory to
the driver after explicit teardown (del + gc.collect + torch.cuda.empty_cache
+ destroy_model_parallel/destroy_distributed_environment) -- confirmed in
practice here, where a second model's LLM() init still failed with "Free
memory ... is less than desired GPU memory utilization" after calling
VLLMRunner.shutdown() on the first. Exiting the OS process is the only
reliable way to force the CUDA driver to reclaim that memory.

Talks to the parent over a Unix-domain socket (multiprocessing.connection),
not stdout/stdin -- vLLM and its dependencies write unstructured logs and
progress bars to stdout, which would corrupt a line-based JSON/text
protocol there.
"""

from __future__ import annotations

import os
import sys

# Same two workarounds as VLLMRunner (see its module docstring for why):
# V1 multiprocessing forks/spawns badly under a process that's already
# touched CUDA -- irrelevant to correctness here since this is a fresh
# process, but harmless to keep, and flashinfer's V100 sampler bug fix
# is still needed regardless of process layout.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


def main() -> None:
    socket_path, model_path, tensor_parallel, gpu_memory_utilisation, max_model_len = sys.argv[1:6]

    from multiprocessing.connection import Listener
    from vllm import LLM, SamplingParams

    from src.inference.llm_runner import INVALID, PredictionResult, get_confidence

    llm = LLM(
        model=model_path,
        tensor_parallel_size=int(tensor_parallel),
        gpu_memory_utilization=float(gpu_memory_utilisation),
        max_model_len=int(max_model_len),
    )

    listener = Listener(socket_path, family="AF_UNIX")
    try:
        conn = listener.accept()
    finally:
        # Only this one connection is ever accepted; stop listening for
        # more so the socket file doesn't linger as accept-able.
        listener.close()

    try:
        while True:
            request = conn.recv()
            op = request["op"]

            if op == "shutdown":
                break

            if op == "generate_text":
                sampling_params = SamplingParams(max_tokens=request["max_tokens"], temperature=0)
                outputs = llm.generate(request["prompts"], sampling_params)
                conn.send([out.outputs[0].text.strip() for out in outputs])

            elif op == "batch_predict":
                label_tokens = tuple(request["label_tokens"])
                sampling_params = SamplingParams(logprobs=20, max_tokens=1, temperature=0)
                outputs = llm.generate(request["prompts"], sampling_params)

                results = []
                for out in outputs:
                    token_logprobs = out.outputs[0].logprobs[0]
                    logprobs_dict = {
                        lp.decoded_token.strip(): lp.logprob for lp in token_logprobs.values()
                    }
                    if label_tokens[0] not in logprobs_dict and label_tokens[1] not in logprobs_dict:
                        results.append(
                            PredictionResult(
                                prediction=INVALID, confidence=0.0, p0=0.0, p1=0.0,
                                logprob_0=-100.0, logprob_1=-100.0,
                            )
                        )
                        continue
                    results.append(get_confidence(logprobs_dict, label_tokens))
                conn.send(results)

            else:
                conn.send({"error": f"unknown op {op!r}"})
    finally:
        conn.close()


if __name__ == "__main__":
    main()
