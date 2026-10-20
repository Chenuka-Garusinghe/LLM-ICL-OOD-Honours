"""P6 probes: the logit lens reproduces the runner, caching changes nothing, steering is local.

Uses the tiny cached `hf-internal-testing/tiny-random-gpt2` in fp32, like the
runner tests; the P6 gate itself (|m_L - m| <= 1e-3 for Qwen in bf16) is
checked on the real run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.generator import sample_eval_tasks
from src.data.synthetic_bridge import load_manifest, write_suite
from src.experiments.synth_grid import enumerate_units
from src.inference.hf_runner import HFRunner
from src.inference.probes import ProbeGridRunner, ProbeRunner
from tests.test_synth_pipeline import StubRunner

TINY = "hf-internal-testing/tiny-random-gpt2"
PREFIX = "Rule task.\n\nf0: 0.30; f1: -1.04 -> 1\nf0: -0.20; f1: 0.51 -> 0\n"
SUFFIXES = ["f0: 0.12; f1: 1.10 -> ", "f0: -1.50; f1: -0.02 -> ", "f0: N/A; f1: N/A -> "]


@pytest.fixture(scope="module")
def tiny():
    try:
        return HFRunner(TINY, device="cpu", dtype="float32", use_chat_template=False, local_files_only=True)
    except OSError:
        pytest.skip(f"{TINY} is not in the local Hugging Face cache")


def _margins(results):
    return np.array([r.logprob_1 - r.logprob_0 for r in results])


def test_final_lens_margin_is_the_runner_margin(tiny):
    """The P6 gate on the tiny model: m_L equals the runner's margin, and scores are unchanged."""
    probe = ProbeRunner(tiny)
    got = probe.score(PREFIX, SUFFIXES)
    want = tiny.score(PREFIX, SUFFIXES)
    np.testing.assert_allclose(_margins(got), _margins(want), atol=1e-6)
    lens = np.stack([r["lens"] for r in probe.records[-1]])
    assert lens.shape == (len(SUFFIXES), probe.n_layers + 1)
    np.testing.assert_allclose(lens[:, -1], _margins(want), atol=1e-4)


def test_cached_lens_equals_full_forward(tiny):
    """Every layer's margin and residual from the prefix cache equals a full forward pass."""
    probe = ProbeRunner(tiny, keep_residuals=True)
    probe.score(PREFIX, SUFFIXES)
    cached = probe.records[-1]
    w = probe.label_direction()
    for suffix, rec in zip(SUFFIXES, cached):
        _, full = probe._record(probe._forward(tiny.encode(PREFIX + suffix)), w, ("0", "1"), True)
        np.testing.assert_allclose(rec["lens"], full["lens"], atol=1e-4)
        np.testing.assert_allclose(rec["resid"].astype(float), full["resid"].astype(float), atol=2e-3)
        assert rec["resid"].shape == (probe.n_layers + 1, tiny.model.config.hidden_size)


def test_steering_is_local_and_removed(tiny):
    """Steering at layer l changes margins from layer l on, not before, and leaves no hook behind."""
    probe = ProbeRunner(tiny)
    n_hooks = [len(m._forward_hooks) for m in probe.layers]      # transformers keeps its own hooks
    probe.score(PREFIX, SUFFIXES)
    base = np.stack([r["lens"] for r in probe.records[-1]])
    layer = 2
    v = np.random.default_rng(0).normal(size=tiny.model.config.hidden_size)
    probe.set_steering(layer, v, alpha=3.0)
    steered_results = probe.score(PREFIX, SUFFIXES)
    steered = np.stack([r["lens"] for r in probe.records[-1]])
    np.testing.assert_allclose(steered[:, :layer], base[:, :layer], atol=1e-5)
    assert np.abs(steered[:, layer:] - base[:, layer:]).max() > 1e-3
    np.testing.assert_allclose(steered[:, -1], _margins(steered_results), atol=1e-4)
    probe.set_steering(None)
    probe.score(PREFIX, SUFFIXES)
    np.testing.assert_allclose(np.stack([r["lens"] for r in probe.records[-1]]), base, atol=1e-5)
    assert [len(m._forward_hooks) for m in probe.layers] == n_hooks
    probe.set_steering(layer, v, alpha=0.0)           # alpha 0 is no steering
    assert probe._steer is None


class StubProbe:
    """A fake ProbeRunner: StubRunner's scores, with a per-layer 'lens' whose last entry is the margin.
    (The tiny model's 1,000-token vocabulary cannot encode a grid prompt.)"""

    model_path, device, device_name, dtype = "stub", "cpu", "stub", "float32"
    keep_residuals, n_layers = True, 3

    def __init__(self):
        self.stub, self.records = StubRunner(), []

    def render(self, system, demo_lines, query_line):
        return self.stub.render(system, demo_lines, query_line)

    def score(self, prefix, suffixes, label_tokens=("0", "1")):
        results = self.stub.score(prefix, suffixes, label_tokens)
        self.last_token_counts = self.stub.last_token_counts
        self.records.append([{"lens": np.array([0.0, 1.0, 2.0, r.logprob_1 - r.logprob_0], dtype=np.float32),
                              "resid": np.zeros((4, 2), dtype=np.float16), "fallback": False} for r in results])
        return results


def test_probe_grid_writes_one_record_per_row(tmp_path):
    """The probe grid's records line up with the grid's rows; the content-free query is dropped."""
    suite = tmp_path / "suite"
    write_suite(sample_eval_tasks(1, id_prefix="t"), suite, suite="t")
    manifest = load_manifest(suite)
    probe = StubProbe()
    grid = ProbeGridRunner(probe, "stub", suite, manifest, base_seed=42, envs=["id", "spurious_reversal"],
                           queries_per_env=4, run_name="p6test", probe_dir=tmp_path / "probe", condition="none")
    units = enumerate_units(["stub"], ["abstract"], [manifest["tasks"][0]["task_id"]],
                            ["zero_shot", "random", "similarity"], ["gold"], [8], 1)
    out = tmp_path / "grid.parquet"
    grid.run(units, out, log=lambda *_: None)
    rows = pd.read_parquet(out)
    lens = pd.read_parquet(tmp_path / "probe" / "lens.all.parquet")
    assert len(lens) == len(rows) == 3 * 8
    m = rows.set_index(["unit_key", "row_id"])
    for _, r in lens.iterrows():
        want = m.loc[(r.unit_key, r.row_id)]
        assert abs(r.lens[-1] - (want.logprob_1 - want.logprob_0)) < 1e-5
        assert abs(r.margin - (want.logprob_1 - want.logprob_0)) < 1e-9
        assert r.y_clean == want.y_clean and r.env == want.env
    resid = np.load(tmp_path / "probe" / lens.resid_file.iloc[0])
    assert resid.shape[0] == (lens.resid_file == lens.resid_file.iloc[0]).sum()
