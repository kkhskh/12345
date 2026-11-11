# Open questions (this research line only)

> **Post-review.** Items 1, 2, 4 and 8 are updated after reviews R1, R4 and R5. Item 10 is added. Pre-review text: commit
> 9389496.

1. **Which operation differs?** There are three candidate loci:
   - the frozen layer-0–1 encoding of "5/5", for example a saturated categorical feature that is hard to read conditionally
     on the other groups' votes;
   - the transfer-trained layer 2, learning an over-general "5/5 → discount" rule;
   - the inherited readout WA/LN_f.

   After review neither of the first two is favoured (R4-6). Candidate tests:
   - train a fresh layer 2, or a probe from frozen resid_pre2 at all positions, to produce the exact-Bayes R1 margin
     (a test of *readability*, not decodability);
   - test whether the count code at the readout token is categorical or graded;
   - re-initialise WA in both arms;
   - align residual spaces across arms for a transplant.
2. **Pathway decomposition.** An exploratory 4-model split shows the P-count discount needs layer-2 attention and the MLP
   jointly. Per-head Q/K/V/O and LayerNorm contributions are untested, although the hooks exist (R4-5).
3. **Collateral effects.** Does any intervention that removes the discount preserve non-boundary decisions? This is
   required before any Outcome-B claim.
4. **Where T1's R1 credit lives.** T1's R1 credit flows mostly outside the frozen "site" (question q's source tokens). At
   the readout token it is positive in 8/8 parents. A symmetric, preregistered localisation is needed.
5. **Objective vs curriculum.** A narrowly targeted objective intervention (mandate §12) was not run. Two examples: P-count
   pretraining with an added term that separates "5/5 because copying" from "5/5 because reliable", or T1 pretraining with
   the count-only objective. Either would separate count acquisition from evidence-update learning.
6. **Unfrozen and longer transfer.** From 2000 to 4000 steps the gap is flat to diverging. Whether it closes with far longer
   transfer, or with unfrozen layers 0–1, is unknown. The claim is scoped to frozen-feature transfer.
7. **Other boundaries.** Only 4→5 in this generator was tested. Are there analogous differences at other support changes,
   for example 5→4, or two simultaneous perfect groups (|H| = 2)? These regimes had too few natural pairs here.
8. **Mathematics.**
   - An exact enumeration or analytic proof of T1(c) for |H| = 1. It is currently sample-based.
   - A tight lower bound on κ in the corrected Proposition 1.
   - Uniform-approximation versions of Proposition 3′: the hard-gate branch is a non-uniform limit.
9. **Pooled-statistic instability.** The pooled B45 effect differed between development (d ≈ 0.3) and confirmation
   (d ≈ 1.2), mainly through R3 and the two weak T1 parents. Why R3 excesses vary across cohorts is not understood.
10. **Matched mechanism controls.** Two are needed: an agreement-moving, count-preserving control for specificity, and norm-
    and dimension-matched random subspace controls. Both should be frozen before use (R3-3, R3-4, R3-6).

**Relation to the broader map.** These questions refine the earlier finding that representing a relationship differs from
interpreting its decision significance. This investigation shows a curriculum-associated *decision-response* difference,
but cannot yet say whether it sits in representation or in use. Item 1 is the direct continuation. Items 2–4 and 10 are
prerequisites for any mechanism claim.
