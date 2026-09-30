# 10 — Llama was not answering in the expected format

**When:** 28 September 2026, after the P2 rerun's gate · **Status:** resolved for now: Llama is paused (the user's decision), and a label-mass validity rule now guards the gate

## What happened

Llama-3.1-8B-Instruct failed the learnability gate at every k:
- Δ = 0.026, 0.042 and 0.045 at k = 8, 16 and 32;
- every 95% interval includes 0.

## How we found it

The scores are the log-probabilities of the tokens "0" and "1". The share of next-token probability on those two tokens (the label mass) differs completely between the models:

| Median label mass, P("0") + P("1") | Qwen2.5-7B-Instruct | Llama-3.1-8B-Instruct |
| --- | --- | --- |
| zero-shot | 1.00 | 0.0000 |
| k = 8 | 1.00 | 0.02 |
| k = 16 | 0.99 | 0.01 |
| k = 32 | 1.00 | 0.11 |

Llama almost never intends to answer with a digit. Two other symptoms point the same way:
- Its margins barely vary between queries: within a demonstration set their sd is 0.03–0.06, against Qwen's 1.2–2.8.
- This explains its near-ties in the cache check ([08](08-cache-agreement-rule-near-ties.md)).

## Why it matters

The two-token score compares two tokens Llama was not going to produce, so its failed gate may measure the answer format rather than whether Llama learns from the labels. As it stands, the result cannot be interpreted.

## Diagnosis (16:26)

`scripts/p2_answer_format.py` on pilot_0000, with the same prompts both models see:

| | Most likely first token | Greedy reply |
| --- | --- | --- |
| Llama, zero-shot | "To" (0.81–0.83), then "I", "There" | "To infer the rule, I'll analyze the given examples. However, I need at least two examples…" |
| Llama, k = 8 | "To" (0.48–0.50), "Based" (0.27–0.29), "After" (0.11–0.14); "0" and "1" about 0.01 each | "To infer the rule, I will analyze the given examples. After examining the examples, I notice that the label is 1 when…" |
| Qwen, zero-shot and k = 8 | "1" or "0" (label mass 1.00) | "1" |

- **Why Llama does this.** The system message asks the model to "Infer the rule from the labelled examples, then classify the final example", and then to "Respond with exactly one character: 0 or 1." Qwen follows the second sentence; Llama answers the first by explaining its reasoning. The two-token score then reads the leftovers of a reply Llama never meant to give.
- **Tokenisation.** In both tokenisers a space before a digit is its own token, so a reply prefilled with `Answer: ` (ending in a space) makes the digit the natural next token: "Answer: 1" is "Answer", ":", " ", "1".

## What we are doing

1. **Diagnose (done).** `scripts/p2_answer_format.py` lists the 10 most likely next tokens and a short greedy continuation for a few pilot prompts, zero-shot and k = 8, for both models. Output: `results/v3/synthetic/p2/answer_format_<model>.json`.
2. **Likely fix.** Prefill the assistant's reply for every instruct model, for example with `Answer: `, so the next token is the answer. The candidates (none, `Answer: `, `Label: `) are being measured on the pilot for label mass, score spread and AUROC.
3. **New validity rule.** A model's median label mass on the pilot must be at least 0.9 before its gate result counts.
4. **Rerun.** Llama's P2 grid (about 55 minutes), and Qwen's (about 40 minutes) if the shared template changes.

### Prefill test (16:50)

The test used 4 pilot tasks × 20 queries, zero-shot and k = 8 (gold, seed 0), with three versions of the reply's start. The script is scratch, and its outputs are copied below.

| Llama-3.1-8B-Instruct | label mass (zero-shot / k = 8) | margin sd within task (k = 8) | AUROC (k = 8) | share predicted 1 (k = 8) | greedy reply |
| --- | --- | --- | --- | --- | --- |
| no prefill | 0.0000 / 0.020 | 0.086 | 0.574 | 0.54 | "To infer the rule, I will analyze…" |
| `Answer: ` | 0.988 / 0.995 | 0.112 | 0.639 | 1.00 | "1" |
| `Label: ` | 0.998 / 0.985 | 0.207 | 0.706 | 0.79 | "1" |

- **Qwen:** label mass 1.00 in every version; k = 8 AUROC 0.701 / 0.749 / 0.730.
- **Caveats:** the AUROCs are noisy (4 tasks, one seed). The four test tasks (pilot_0000, 0003, 0006, 0009) all have σ_t = +1, so their absolute AUROCs include the prior effect ([09](09-numeric-prior-fixed-spurious-direction.md)). The comparison between prefills is unaffected.
- **Conclusion:** a prefill makes both models answer with a label token (label mass at least 0.98).

## Decision (28 September)

- **Qwen only, for now** (the user's decision). Llama-3.1-8B-Instruct is paused rather than fixed.
  - Qwen already answers with the label, so its prompt stays unchanged and its P2 results stand. There is no rerun.
  - If Llama is resumed, the fix is known: prefill its reply with `Label: ` and rerun its P2 grid.
- **Validity rule adopted.**
  - The gate table records each model's median label mass, and a result counts only at 0.9 or above (`LABEL_MASS_THRESHOLD`, `valid` in `learnability_gate`).
  - Qwen: 0.996, valid. Llama: 0.009–0.11, not valid.
  - Tested in `test_learnability_gate_needs_the_label_tokens`.
- **Question from the user:** "Are we using instruct models?" Yes: Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct are chat-tuned and get the chat template. Qwen2.5-7B base (planned for the reduced grid) gets a plain completion prompt ending in `-> `, so it cannot start explaining the way Llama did.

## Evidence

- `results/v3/synthetic/p2_learnability.parquet`: full-vocabulary `logprob_0` / `logprob_1`, so the label mass can be recomputed.
- `results/v3/synthetic/p2/learnability_gate.json`.

## How to tell it

The story will depend on the fix. The general point is worth making either way:

> Constrained scoring assumes the model intends to answer with one of the label tokens. We check that assumption directly: a model must put most of its next-token probability on the labels before its results are interpreted.

- **Examiner question:** "Why did Llama fail?"
  Answer with the label-mass evidence before any claim about learnability.
