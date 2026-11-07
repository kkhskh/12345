# Simulated review 2 — causal inference and experimental design

Internal simulation, not a conference review.

## Main summary

The estimands are mostly well defined: paired \(\Delta Y\) under a specified residual intervention, with clustering on recipient or triplet. Calibration is treated as part of the estimand, which is correct. The COST protocol deviation is disclosed. Confirmation is unused. Those are strengths.

## Strongest positive point

Separating equal L2 from equal JS, refusing to compare a 7/48 gradient baseline at the full-sample mean, and refusing to treat STAGE_ONLY’s mean share as a causal decomposition.

## Major weaknesses

1. **No local Level 1 recomputation.** Headline 48-pair intervals are session records. A causal paper cannot ship that as verified.
2. **Multiple looks at the same 48 SPEC pairs.** Fixed-JS is not an independent sample. The bootstrap seed 82 is reused. Multiplicity is discussed but not quantified.
3. **Information confounding in NST.** The learned map sees \(b_t\), including legality. Wrong-target controls also see a board, so the contrast is target identity given the same feature class—not “no privileged information versus the model.”
4. **Off-manifold interventions.** \(h_E+t\delta\) is an unconstrained add in a 388-dimensional subspace. There is no overlap, energy, or cycle check. The causal effect of that intervention is well defined; the interpretation that it is a *state* is not.
5. **FRESH identical-state overlap (24 prefixes).** Game-level clustering does not fix state leakage into training.

## Minor weaknesses

Pair construction conditions on nonempty legal-set differences (selection on the outcome’s support). Hamming A/B is simulator geometry, not model difficulty. VNF equal-L2 factorial is a different intervention than JS-calibrated NST, so a null there does not refute NST.

## Potentially fatal objection

“Do the reported comparisons identify the causal effect the authors claim?”

They identify the effect of *these constructed edits* on \(Y\). They do not identify the effect of a naturally occurring board-state variable, and they do not identify a unique kernel coordinate. If the abstract is read as the latter, the design fails. If it is read as the former, the design is adequate except for the missing array replay.

## Required changes

Mount Drive and recompute. Report a sensitivity that clusters by board-state hash as well as game ID. Pre-declare that NST’s information set includes the legal mask in the main methods paragraph (the draft does this; keep it).

## Evidence supporting the criticism

NST preregistration: symbolic \(b\) includes the 64-square legal mask. COST forensic letter exists. FRESH census reported 24 identical board+player prefixes.

## Would the current paper be defensible?

Not as a verified causal study until Level 1 passes. The *design* is defensible for a probe-relative intervention effect. The *package* is not, because the logits are elsewhere.
