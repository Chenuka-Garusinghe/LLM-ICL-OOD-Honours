"""PyTorch runner for the frozen language models (Hugging Face transformers).

The only inference backend. The model runs on the auto-detected device
(src/utils/device.py: CUDA, then Apple Metal/MPS, then CPU) in bf16 where the
device supports it.

Scoring is constrained to the two label tokens: one decoder pass per prompt,
the output layer applied at the last position only, and the two label
log-probabilities become a `PredictionResult` (llm_runner.py's two-way
softmax). The label logits are computed in fp32 from the final hidden state
(`label_log_probs`), so margins are not rounded to bf16.

`score(prefix, suffixes)` is the fast path. Queries that share a
demonstration set share the prompt up to the query line, so the prefix runs
once and each query suffix is scored against its cached keys and values,
with the cache cropped back to the prefix afterwards. For a causal decoder
this equals a full forward pass (generator_spec.pdf, prefix caching)
provided the tokens of prefix + suffix start with the tokens of the prefix;
every suffix is checked, and a suffix that breaks that condition is scored
with a full forward pass instead.

Templated prompts are tokenised with `add_special_tokens=False`: the chat
template already inserts Llama's BOS, and adding it again gave v2 prompts two.
"""

from __future__ import annotations

import gc
import os
import time
import warnings
from pathlib import Path

from src.utils.device import device_name, resolve_device, resolve_dtype  # sets the MPS fallback before torch loads

import torch
from dotenv import load_dotenv

from src.inference.chat import ChatFormatter
from src.inference.llm_runner import PredictionResult, get_confidence, resolve_label_token_ids
from src.inference.prompts import render_prompt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABEL_TOKENS = ("0", "1")


class HFRunner:
    """One causal LM, loaded once, serving constrained two-token classification."""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        dtype: str = "bfloat16",
        use_chat_template: bool = True,
        attn_implementation: str = "sdpa",
        env_file: str | Path | None = PROJECT_ROOT / ".env",
        local_files_only: bool = False,
    ):
        # The gated Llama repo needs HF_TOKEN even when every file is cached:
        # transformers probes it for an optional tokenizer file and gets a 401.
        if env_file is not None and Path(env_file).exists():
            load_dotenv(env_file)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_path = model_path
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(self.device, dtype)
        self.device_name = device_name(self.device)
        self.use_chat_template = use_chat_template

        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=self.dtype, attn_implementation=attn_implementation,
            local_files_only=local_files_only,
        )
        self.model = model.to(self.device).eval()
        self.load_seconds = time.perf_counter() - t0
        self.n_prefix_fallbacks = 0
        self._label_ids: dict[tuple[str, str], list[int]] = {}

    # ------------------------------------------------------------------ #
    # prompts and tokens
    # ------------------------------------------------------------------ #
    def chat_formatter(self) -> ChatFormatter:
        return ChatFormatter(self.tokenizer)

    def render(self, system: str, demo_lines: list[str], query_line: str) -> str:
        return render_prompt(self.tokenizer, system, demo_lines, query_line, self.use_chat_template)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def label_ids(self, label_tokens: tuple[str, str] = LABEL_TOKENS) -> list[int]:
        if label_tokens not in self._label_ids:
            ids = resolve_label_token_ids(self.tokenizer, label_tokens)
            if ids is None:
                raise ValueError(f"label tokens {label_tokens} are not single tokens for {self.model_path}")
            self._label_ids[label_tokens] = ids
        return self._label_ids[label_tokens]

    # ------------------------------------------------------------------ #
    # scoring
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def _forward(self, input_ids: list[int], cache=None):
        """Run the decoder (without the output layer); returns its final hidden states and cache."""
        ids = torch.tensor([input_ids], device=self.device)
        return self.model.base_model(input_ids=ids, past_key_values=cache, use_cache=True)

    @torch.inference_mode()
    def _result(self, out, label_tokens: tuple[str, str]) -> PredictionResult:
        """Label log-probabilities at the last position (`label_log_probs`)."""
        lp0, lp1 = label_log_probs(out.last_hidden_state[0, -1], self.model.get_output_embeddings(),
                                   self.label_ids(label_tokens))
        return get_confidence({label_tokens[0]: lp0, label_tokens[1]: lp1}, label_tokens)

    def batch_predict(self, prompts: list[str], label_tokens: tuple[str, str] = LABEL_TOKENS) -> list[PredictionResult]:
        """One full forward pass per prompt (no cache reuse)."""
        return [self._result(self._forward(self.encode(p)), label_tokens) for p in prompts]

    def score(
        self,
        prefix: str,
        suffixes: list[str],
        label_tokens: tuple[str, str] = LABEL_TOKENS,
    ) -> list[PredictionResult]:
        """Score prefix + suffix for every suffix, running the prefix once."""
        prefix_ids = self.encode(prefix)
        n = len(prefix_ids)
        cache = self._forward(prefix_ids).past_key_values
        results = []
        self.last_token_counts = []
        for suffix in suffixes:
            full_ids = self.encode(prefix + suffix)
            self.last_token_counts.append(len(full_ids))
            if full_ids[:n] != prefix_ids or len(full_ids) == n:
                self.n_prefix_fallbacks += 1
                if self.n_prefix_fallbacks == 1:
                    warnings.warn("prompt tokens do not start with the prefix tokens; "
                                  "scoring that suffix with a full forward pass")
                results.append(self._result(self._forward(full_ids), label_tokens))
                continue
            results.append(self._result(self._forward(full_ids[n:], cache=cache), label_tokens))
            # Drop the suffix again (negative crop removes that many tokens;
            # a positive crop length is deprecated in transformers 5.17).
            cache.crop(n - cache.get_seq_length())
        return results

    @torch.inference_mode()
    def generate_text(self, prompts: list[str], max_new_tokens: int = 128) -> list[str]:
        """Greedy free-text generation for already-rendered prompts (RQ3 feature rankings)."""
        outputs = []
        for p in prompts:
            ids = torch.tensor([self.encode(p)], device=self.device)
            gen = self.model.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
            outputs.append(self.tokenizer.decode(gen[0, ids.shape[1]:], skip_special_tokens=True).strip())
        return outputs

    def close(self) -> None:
        """Release the model so the next one fits in memory."""
        del self.model
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.empty_cache()


@torch.inference_mode()
def label_log_probs(h: torch.Tensor, head: torch.nn.Module, ids: list[int]) -> tuple[float, float]:
    """Full-vocabulary log-probabilities of the two label tokens from the
    final (normed) hidden state `h`.

    The two label logits are computed in fp32. The model's own bf16 output
    layer rounds its logits to bf16, which at logit sizes around 20 is a step
    of 0.125: margins then fall on a 1/16 grid, many queries tie exactly, and
    cached and full passes flip near-ties (Llama agreed on only 96% of
    predictions). The normaliser takes the other tokens' logits from the
    output layer and the two label logits in fp32, so the label probabilities
    never sum to more than 1 (with the bf16 values in the normaliser, Qwen's
    reached about 1.1; the margin was unaffected).
    """
    logits = head.weight[ids].float() @ h.float()
    if getattr(head, "bias", None) is not None:
        logits = logits + head.bias[ids].float()
    vocab = head(h).float()
    vocab[ids] = logits
    lp0, lp1 = (logits - torch.logsumexp(vocab, dim=-1)).tolist()
    return lp0, lp1


def runner_from_config(model_cfg, inference_cfg, **kwargs) -> HFRunner:
    """Build a runner from a `models` entry and the `inference` block of the config."""
    return HFRunner(
        model_cfg.path,
        device=os.environ.get("SATA_DEVICE", inference_cfg.device),
        dtype=inference_cfg.dtype,
        use_chat_template=model_cfg.chat_template,
        **kwargs,
    )
