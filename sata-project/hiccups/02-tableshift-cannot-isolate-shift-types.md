# 02 — TableShift names three shift types but isolates none

**When:** 18 September 2026 (audit commit `1c28a1a` on `main`) · **Status:** resolved: experiments became synthetic

## What happened

The research questions compare shift *types* (covariate, spurious, concept). The real-data arm used TableShift, which was assumed to provide them.

## How we found it

An audit of the TableShift paper (Gardner, Popović & Schmidt, NeurIPS 2023 Datasets & Benchmarks, §2 and Appendix E.2) found three problems:
- **Its shifts are mixtures.** The paper says real shifts are "composed of an unknown mixture of all three forms", and that disentangling them is out of scope.
- **Its OOD splits are cut on domain variables** (geography, year, demographics), chosen because they produce a performance gap, not a particular type of shift.
- **Its "concept shift" metric (Δ_y|x) never uses y.** It is a Fréchet distance between classifier activations on the ID and OOD splits, so it is covariate shift measured in representation space, not a change in p(y | x).

## Why it matters

Any claim of the form "strategy A helps under shift type B" would be unsupported on TableShift, because no split is known to be of type B.

## What we did

- The synthetic generator became the core: each environment changes exactly one thing, by construction.
- Corrections fitted on the four TableShift tasks (calibration choices, verbalisers, scale escalation) were not carried over. Behaviour is measured first on the new data (18 September decision).
- The approved v3 plan (26 September) is synthetic only.
- *To fill in: why WHYSHIFT, scoped earlier as a real-data option, was also left out.*

## Evidence

- `TABLESHIFT_AUDIT.md` on `main`.
- `tables/v2/tableshift_shift_type_audit.csv` on `main`.

## How to tell it

> Comparing shift types requires data in which the type of shift is known. TableShift's own paper states that its shifts are unknown mixtures, and its concept-shift metric does not involve the label at all, so it cannot support shift-type claims. We therefore built synthetic tasks in which each environment changes exactly one factor, and we treat real benchmarks as motivation rather than as evidence for type-specific claims.

- **Figure idea:** a 2 × 3 grid, TableShift vs the synthetic generator × (covariate, spurious, concept), with "isolated?" ticks.
- **Examiner question:** "Why not real data?"
  Because the question is about shift types, and no public tabular benchmark isolates them. Synthetic data trades realism for identifiability, which is the property this question needs.
