# 07 — The bf16 output layer rounded the scores

**When:** 28 September 2026, the first P2 benchmark · **Status:** resolved (the normaliser follow-up was also fixed on 28 September)

## What happened

A prompt's queries can be scored in two ways:
- **cached:** the demonstration prefix is run once and each query is scored against its cache;
- **full:** a full forward pass for every query.

Llama-3.1-8B-Instruct's predictions agreed between the two on only 96% of queries.

## How we found it

The margins (logit of "1" minus logit of "0") took only a few distinct values:
- The model's output layer runs in bf16. At logit sizes around 20, bf16 values are spaced 0.125 apart.
- So every margin fell on a 1/16 grid, and 19% of the p1 values within a demonstration set were exactly tied.
- A tiny numerical difference between the two paths then flipped those ties. The largest margin difference was 0.25.

## Why it matters

- **Ties:** a coarse, tied score makes AUROC ties arbitrary.
- **False alarm:** it made a correct cache look faulty.

## What we did

- **fp32 label logits.** The runner computes the two label logits in fp32 from the final (normed) hidden state, using the two rows of the output layer, while the rest of the model stays in bf16 (`src/inference/hf_runner.py`, `_result`).
- **A test.** On the tiny test model, the fp32 path equals the model's own output when both run in fp32.
- **Restart.** P2 was restarted with the fixed runner.

## Result

The ties disappeared.
- Agreement rose to 99.2% for Qwen and 97.6% for Llama, and Llama's largest cached − full margin difference fell from 0.25 to 0.096.
- The remaining Llama shortfall is a different story ([08](08-cache-agreement-rule-near-ties.md), [10](10-llama-answer-format.md)).

## Follow-up (fixed, minor)

The full-vocabulary normaliser still comes from the bf16 output layer. So the *absolute* label log-probabilities can be slightly inconsistent with the fp32 label logits:
- Qwen's label mass P("0") + P("1") reaches about 1.1 at the 95th percentile.
- Margins, p1, AUROC and calibration are unaffected, because the normaliser cancels.

- **Fixed on 28 September.** `label_log_probs` puts the two fp32 label logits into the vocabulary logits before the log-sum-exp.
- **Test.** `test_label_probabilities_never_sum_above_one_in_bf16` uses a bf16 output layer with logits near 20. The old normaliser sums above 1 in 8 of 20 draws (up to 1.062); the new one never does.
- **Already-saved outputs.** The P2 outputs were computed before this fix; only their absolute label log-probabilities are affected, by at most about 0.1.

## Evidence

- Spec, Scoring section.
- Known issues row 17.
- `tests/test_hf_runner.py::test_label_log_probs_match_the_models_own_output`.

## How to tell it

> Scores are read from the probabilities of the two label tokens. In bf16, the output layer rounds logits to steps of 0.125, which put every margin on a coarse grid and tied a fifth of the scores within a demonstration set. We compute the two label logits in fp32 from the final hidden state, which removes the rounding at negligible cost.

- **Examiner question:** "Does bf16 affect your results?"
  Only through the hidden states, which are the same for every condition. The label scores themselves are exact in fp32.
