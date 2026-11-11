# Open questions (this research line only)

1. **Which operation differs?** Is the P-count R1 discount a property of the frozen layer-0–1 encoding of "5/5", for example
   an unconditional "perfect group" feature that cannot be read conditionally on the other groups' votes? Or is it a property
   of the transfer-trained layer 2? Candidate test: train layer 2 of a P-count parent on targets that only reward R1 credit,
   and see whether the frozen representation supports it (a probe of *readability*, not decodability). Alternatively, align
   residual spaces across arms for a transplant.
2. **Pathway decomposition.** Q/K/V/O, LayerNorm and MLP contributions to the R1 update, inside layer 2 and at the
   readout, are untested.
3. **Collateral effects.** Does ablating the site count subspace damage non-boundary decisions? This is required before any
   Outcome-B claim.
4. **Implementation diversity in T1.** Parents 72 and 74 route the boundary information outside the decision-site positions.
   Where? Do they share a pathway with each other?
5. **Objective vs curriculum.** A narrowly targeted objective intervention (mandate §12) was not run. Two examples: P-count
   pretraining with an added term that distinguishes "5/5 because copying" from "5/5 because reliable", or T1 pretraining
   with the count-only objective. Either would separate count acquisition from evidence-update learning.
6. **Long-transfer limit.** From 2000 to 4000 steps the gap is flat to diverging. Whether it closes with far longer
   transfer, or with unfrozen layers 0–1, is unknown.
7. **Other boundaries.** Only 4→5 in this generator was tested. Are there analogous interpretation differences at other
   support changes, for example 5→4 (losing perfect agreement) or two simultaneous perfect groups (|H| = 2)? These regimes
   had too few natural pairs here.
8. **Mathematics.**
   - An analytic proof of T1(c), the sign of the Bayes update in R2/R3 and R1, for all X_q.
   - Proposition 3′ under mixed gates.
   - A tight lower bound on κ in the corrected Proposition 1.
9. **Pooled-statistic instability.** The pooled B45 effect differed between development (d ≈ 0.3) and confirmation
   (d ≈ 1.2), mainly through R3. Why R3 excesses vary across cohorts is not understood.

Relation to the broader map: these questions refine the earlier finding that representing a relationship differs from
interpreting its decision significance. Item 1 is the direct continuation. Items 2–4 are prerequisites for any mechanism
claim.
