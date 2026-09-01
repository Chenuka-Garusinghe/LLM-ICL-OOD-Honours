"""LLM interface: prompt -> prediction + logprobs.

Backed by plain HuggingFace `transformers`, not vLLM. The primary workload
here is *single-token classification*: score the next token after a prompt
ending in "->" and compare the logprobs of the two label tokens. That is one
forward pass with no generation loop, no KV-cache reuse across steps and no
sampling, so vLLM's continuous batching / PagedAttention buy almost nothing,
while its torch/CUDA version pinning is a recurring source of environment
breakage (see scripts/deploy_to_colab.sh).

Two correctness advantages over the previous vLLM implementation, beyond the
simpler dependency:

* Exact label logprobs. The vLLM path requested `logprobs=20` and matched on
  decoded token *strings*, so a label token outside the top 20 silently became
  INVALID or a -100 floor. Here the label token ids are indexed directly in the
  final-position log-softmax, so their logprobs are always exact and INVALID
  effectively disappears -- which matters because R-AUC is computed from these
  confidences.
* Deliberate label-token resolution. `serialise_row` ends a query line with
  "->" and writes demos as "... -> 1.0", so the token the model actually
  predicts next carries a *leading space*. The vLLM path only matched it by
  accident, via `.strip()` on the decoded token. `_resolve_label_token_ids`
  below resolves ids from " " + label explicitly and asserts the two labels are
  distinguishable by their first token.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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


class HFRunner:
    """Constrained single-token classification over a causal LM.

    Callers construct this as `HFRunner(path, **vars(config.inference))`.
    """

    def __init__(
        self,
        model_path: str,
        max_model_len: int = 4096,
        batch_size: int = 16,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_model_len = max_model_len
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Decoder-only + batching requires LEFT padding: with right padding the
        # final position of a short sequence is a pad token, so logits[:, -1]
        # would score padding instead of the prompt's last real token -- wrong
        # numbers, no error raised.
        self.tokenizer.padding_side = "left"
        # Truncate from the left too: prompts are system instruction + demos +
        # query, and the query is at the END. Right truncation would cut off the
        # thing being classified.
        self.tokenizer.truncation_side = "left"
        if self.tokenizer.pad_token_id is None:
            # Llama/Qwen ship without a pad token; reusing EOS is standard and
            # harmless here because attention_mask excludes the padding anyway.
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            device_map = "auto"
        else:
            # CPU fallback keeps the pilot runnable locally for debugging; fp16
            # on CPU is both slow and numerically poor, so use fp32.
            dtype = torch.float32
            device_map = None

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype, device_map=device_map
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def _resolve_label_token_ids(self, label_tokens: tuple[str, str]) -> tuple[int, int]:
        """First-token id of each label as it appears in context (leading space).

        Labels may be multi-token ("0.0" -> " 0", ".", "0"); only the first token
        is needed, and it is sufficient *provided* the two labels differ there.
        """
        ids = []
        for label in label_tokens:
            encoded = self.tokenizer.encode(f" {label}", add_special_tokens=False)
            if not encoded:
                raise ValueError(f"label {label!r} encoded to an empty token sequence")
            ids.append(encoded[0])

        if ids[0] == ids[1]:
            raise ValueError(
                f"label tokens {label_tokens!r} share a first token "
                f"(id {ids[0]}, {self.tokenizer.decode([ids[0]])!r}), so single-token "
                "scoring cannot distinguish them. Use label strings that differ in "
                "their first token."
            )
        return ids[0], ids[1]

    def batch_predict(self, prompts: list[str], label_tokens: tuple[str, str]) -> list[PredictionResult]:
        """Score each prompt's next token against the two label tokens."""
        import torch

        id_0, id_1 = self._resolve_label_token_ids(label_tokens)

        results: list[PredictionResult] = []
        for start in range(0, len(prompts), self.batch_size):
            chunk = prompts[start:start + self.batch_size]
            encoded = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_model_len,
            ).to(self.device)

            # Left padding fixes *attention* via attention_mask, but a plain
            # forward pass still derives position ids from the raw sequence
            # index, so real tokens sitting behind N pads would be encoded at
            # positions N, N+1, ... instead of 0, 1, ... . That changes RoPE's
            # rotation (Llama/Qwen) and absolute position embeddings alike, so
            # the same prompt scores differently depending on how much padding
            # its batch-mates forced onto it -- silently, with no error. Derive
            # position ids from the mask instead; this is the same idiom
            # transformers' own generation utils use for left-padded batches.
            attention_mask = encoded["attention_mask"]
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

            with torch.no_grad():
                logits = self.model(**encoded, position_ids=position_ids).logits

            # Left padding means the last position is the real final prompt token
            # for every row in the batch.
            final_logprobs = torch.log_softmax(logits[:, -1, :].float(), dim=-1)
            lp0_batch = final_logprobs[:, id_0].tolist()
            lp1_batch = final_logprobs[:, id_1].tolist()

            for lp0, lp1 in zip(lp0_batch, lp1_batch):
                results.append(
                    get_confidence({label_tokens[0]: lp0, label_tokens[1]: lp1}, label_tokens)
                )
        return results

    def generate_text(self, prompts: list[str], max_tokens: int = 128) -> list[str]:
        """Unconstrained greedy generation (e.g. Notebook 03's feature-ranking
        prompt), as opposed to batch_predict's constrained single-token scoring.
        """
        import torch

        outputs: list[str] = []
        for start in range(0, len(prompts), self.batch_size):
            chunk = prompts[start:start + self.batch_size]
            encoded = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_model_len,
            ).to(self.device)

            with torch.no_grad():
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Strip the prompt: with left padding every row's prompt occupies the
            # same leading width, so slice at the input length.
            prompt_len = encoded["input_ids"].shape[1]
            for row in generated[:, prompt_len:]:
                outputs.append(self.tokenizer.decode(row, skip_special_tokens=True).strip())
        return outputs
