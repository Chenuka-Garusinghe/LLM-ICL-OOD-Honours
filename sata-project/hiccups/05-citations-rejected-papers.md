# 05 — Attribution: citing borrowed ideas, and two rejected papers

**When:** 27 September 2026 · **Status:** resolved

## What happened

Several ideas in the specification came from the reading list, the literature review or other sources, but were not credited.

## How we found it

The user asked for the specification to be compared against the reading list and the literature review (`Lit-review.pdf`), and for the sources to be searched online.
Each reference was checked against the original paper: title, authors, venue, year, and whether the paper actually shows the claim attributed to it.
Publication status was checked on DBLP and OpenReview.

## What we did

- **122 references verified** and added to `docs/references.bib`, with an attribution table in the specification (§6). 119 remain, after removing the two rejected papers below and the MLX reference when MLX was dropped.
- **Removed:** two papers whose peer review ended in rejection: Bigelow et al. 2025 (arXiv:2511.00617) and Jiao et al. 2026 (arXiv:2603.04464). This was the user's rule: do not cite rejected papers.
- **Kept as arXiv references,** because their reviews had not concluded (the user's decision): Wei et al. 2023, Kato et al. 2025, ActAdd, STaDS and LMPriors. The DBLP check found no later published version of any of the 13 arXiv-only references.
- **Problems found in the literature review itself** (worth fixing if it is reused in the thesis):
  - Harutyunyan et al. is now in TMLR (2026), not only the ICML workshop version.
  - Zhu et al. 2026 is image-only (diffusion inpainting). For tabular marginal-sampling removal, cite Covert, Lundberg & Lee (JMLR 2021).
  - Chen et al. 2026: "spurious correlations carried in context" is a loose paraphrase; only their mismatched-context condition supports it.
  - Quiñonero-Candela et al. is originally MIT Press 2009; the review cites the 2022 reprint.

## Why it matters

Academic integrity, and credibility with examiners: every design choice that came from prior work is credited to that work, and nothing rests on a paper that failed peer review.

## Evidence

- `docs/references.bib` (each entry has a source comment).
- `docs/generator_spec.pdf` §6 (attribution table).

## How to tell it

In the thesis this is not a story to tell. It is a standard to meet quietly. Two practical points:
- use the corrected versions of the references above;
- if a rejected paper was ever the *source* of an idea the design uses, keep the credit and discuss it rather than dropping the citation.
