"""Mechanistic probes (P6): logit lens, residuals and steering on the grid's prompts.

`ProbeRunner` wraps an `HFRunner` and scores exactly as `HFRunner.score`
does, prefix cache included, but asks the decoder for its hidden states on
every query pass. For each query it keeps

- the logit-lens margin at every layer (generator_spec.pdf, logit-lens
  margin): m_l = w . norm(h_l) for l < L and m_L = w . h_L, where h_l is the
  last-position state after layer l (h_0 the embeddings), norm the model's
  final normalisation and w the difference of the two label rows of the
  unembedding, all in fp32. In transformers 5.17 the last hidden state has
  already been through the final norm, so m_L is the runner's margin (the P6
  gate);
- optionally the residuals h_0..h_L at the last position (fp16), for the
  linear probes and the steering vectors;
- optionally a steering hook that adds alpha * v to the output of one decoder
  layer while the query passes run (the cached demonstration prefix is not
  steered).

`ProbeGridRunner` is a `GridRunner` whose units also return these records,
so demonstration sets and prompts are exactly the P4/P5 ones.
"""

from __future__ import annotations

import contextlib
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.experiments.synth_grid import GridRunner
from src.inference.hf_runner import LABEL_TOKENS


def decoder_parts(model) -> tuple[torch.nn.ModuleList, torch.nn.Module]:
    """The decoder layers and the final normalisation of a causal LM (Qwen/Llama or GPT-2 naming)."""
    base = model.base_model
    layers = getattr(base, "layers", None) or getattr(base, "h", None)
    norm = getattr(base, "norm", None) or getattr(base, "ln_f", None)
    if layers is None or norm is None:
        raise ValueError(f"cannot find the decoder layers and final norm of {type(model).__name__}")
    return layers, norm


@torch.inference_mode()
def lens_margins(hidden_states, norm: torch.nn.Module, w: torch.Tensor) -> np.ndarray:
    """Logit-lens margins m_0..m_L at the last position; the last state is already normed."""
    last = [h[0, -1] for h in hidden_states]
    normed = [norm(h) for h in last[:-1]] + [last[-1]]
    return torch.stack([h.float() @ w for h in normed]).cpu().numpy()


class ProbeRunner:
    """`HFRunner.score` with per-layer margins, optional residuals and optional steering."""

    def __init__(self, runner, keep_residuals: bool = False):
        self.runner = runner
        self.keep_residuals = keep_residuals
        self.layers, self.norm = decoder_parts(runner.model)
        self.n_layers = len(self.layers)
        self.records: list[list[dict]] = []      # one list per score() call, one dict per suffix
        self.last_token_counts: list[int] = []
        self._steer: tuple[int, torch.Tensor] | None = None

    def __getattr__(self, name):                # everything else is the wrapped runner's
        return getattr(self.runner, name)

    # ------------------------------------------------------------------ #
    def label_direction(self, label_tokens=LABEL_TOKENS) -> torch.Tensor:
        ids = self.runner.label_ids(label_tokens)
        head = self.runner.model.get_output_embeddings()
        return (head.weight[ids[1]].float() - head.weight[ids[0]].float())

    def set_steering(self, layer: int | None, vector=None, alpha: float = 0.0) -> None:
        """Add alpha * vector to hidden state h_layer (the output of decoder layer layer - 1)
        during query passes; layer None (or alpha 0) turns steering off."""
        if layer is None or alpha == 0:
            self._steer = None
            return
        if not 1 <= layer <= self.n_layers:
            raise ValueError(f"steering layer {layer} outside 1..{self.n_layers}")
        v = torch.as_tensor(np.asarray(vector), dtype=self.runner.dtype, device=self.runner.device)
        self._steer = (layer, alpha * v)

    @contextlib.contextmanager
    def _steering(self):
        if self._steer is None:
            yield
            return
        layer, delta = self._steer

        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                return (output[0] + delta,) + tuple(output[1:])
            return output + delta

        handle = self.layers[layer - 1].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    @torch.inference_mode()
    def _forward(self, ids: list[int], cache=None):
        t = torch.tensor([ids], device=self.runner.device)
        return self.runner.model.base_model(input_ids=t, past_key_values=cache, use_cache=True,
                                            output_hidden_states=True)

    @torch.inference_mode()
    def _record(self, out, w, label_tokens, fallback: bool) -> tuple:
        result = self.runner._result(out, label_tokens)
        rec = {"lens": lens_margins(out.hidden_states, self.norm, w), "fallback": fallback}
        if self.keep_residuals:
            rec["resid"] = torch.stack([h[0, -1] for h in out.hidden_states]).to(torch.float16).cpu().numpy()
        return result, rec

    def score(self, prefix: str, suffixes: list[str], label_tokens=LABEL_TOKENS):
        """As `HFRunner.score`; the per-suffix records go to `self.records[-1]`."""
        w = self.label_direction(label_tokens)
        prefix_ids = self.runner.encode(prefix)
        n = len(prefix_ids)
        cache = self.runner._forward(prefix_ids).past_key_values      # the prefix is never steered
        results, recs = [], []
        self.last_token_counts = []
        for suffix in suffixes:
            full_ids = self.runner.encode(prefix + suffix)
            self.last_token_counts.append(len(full_ids))
            with self._steering():
                if full_ids[:n] != prefix_ids or len(full_ids) == n:
                    self.runner.n_prefix_fallbacks += 1
                    if self.runner.n_prefix_fallbacks == 1:
                        warnings.warn("prompt tokens do not start with the prefix tokens; "
                                      "scoring that suffix with a full forward pass")
                    r, rec = self._record(self._forward(full_ids), w, label_tokens, True)
                else:
                    r, rec = self._record(self._forward(full_ids[n:], cache=cache), w, label_tokens, False)
                    cache.crop(n - cache.get_seq_length())
            results.append(r)
            recs.append(rec)
        self.records.append(recs)
        return results


class ProbeGridRunner(GridRunner):
    """A grid runner whose units also write per-layer margins (and residuals) for every query."""

    def __init__(self, probe: ProbeRunner, *args, probe_dir: str | Path, condition: str = "none",
                 shard_tag: str = "all", **kwargs):
        super().__init__(probe, *args, **kwargs)
        self.probe = probe
        self.probe_dir = Path(probe_dir)
        self.condition = condition
        self.shard_tag = shard_tag
        (self.probe_dir / "resid").mkdir(parents=True, exist_ok=True)

    def run_unit(self, unit):
        self.probe.records = []
        rows = super().run_unit(unit)
        calibrated = unit.strategy != "zero_shot"          # the content-free query is each group's last suffix
        recs = [r for group in self.probe.records for r in (group[:-1] if calibrated else group)]
        if len(recs) != len(rows):
            raise AssertionError(f"{unit.key}: {len(recs)} probe records for {len(rows)} rows")
        lens = np.stack([r["lens"] for r in recs])
        tag = hashlib.sha1(f"{self.condition}|{unit.key}".encode()).hexdigest()[:16]
        out = pd.DataFrame({
            "condition": self.condition, "unit_key": unit.key, "naming": unit.naming, "task_id": unit.task_id,
            "strategy": unit.strategy, "seed": unit.seed, "env": rows["env"].to_numpy(),
            "row_id": rows["row_id"].to_numpy(), "label": rows["label"].astype(int).to_numpy(),
            "y_clean": rows["y_clean"].to_numpy(),
            "margin": (rows["logprob_1"] - rows["logprob_0"]).to_numpy(),
            "lens": list(lens.astype(np.float32)), "fallback": [r["fallback"] for r in recs],
            "resid_file": f"resid/{tag}.npy" if self.probe.keep_residuals else None,
        })
        if self.probe.keep_residuals:
            np.save(self.probe_dir / "resid" / f"{tag}.npy", np.stack([r["resid"] for r in recs]))
        path = self.probe_dir / f"lens.{self.shard_tag}.parquet"
        if path.exists():
            out = pd.concat([pd.read_parquet(path), out], ignore_index=True)
        out.to_parquet(path, index=False)
        return rows
