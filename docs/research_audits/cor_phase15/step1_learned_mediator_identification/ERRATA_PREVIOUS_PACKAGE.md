# Corrections to `step1_mechanism_identification_continuation` (mandate §H)

The historical package is **not edited**. This file supersedes the claims listed below.

1. **A1\* is narrowed to the MLP branch contribution.**
   - A1\* proves `sign q = sign(ab)` for the MLP output `q`.
   - It equals the **total** margin only in the construction `agreement_family`, where nothing else writes to the readout direction and there is no final LN. In that case the margin equals `q` to within 8.9e−16 (M7).
   - With other residual contributions or a final LN, A1\* says nothing about the total margin without additional conditions.
   - This package's T-MED is stated on the total margin.
2. **T-RF′ is scoped to the monotone single-index additive class.**
   - Zero order reversals is a necessary condition for `h(f(a) + g(b))` with h *strictly increasing*.
   - The test therefore rejects only that class. It does **not** distinguish relational computation from an aggregate-only `h(z)` with non-monotone h.
   - The AGG-NL models (`m ≈ h(a + b)`, aggregate-only) show 423–450 reversals out of 465 row pairs (`logs/aggnl_reversals.json`).
   - The previous wording, "graded outputs can certify a cross-source interaction", is withdrawn. Reversals certify only that the margin is not monotone-additive.
3. **T-MS is a representation-specific diagnostic.**
   - The four-quadrant sign test checks one particular signed encoding of the gate feature.
   - An affine shift or an invertible recoding can represent the same relation and still fail it.
   - It is neither necessary nor universal. This package replaces it with an encoding-invariant content test: an additive-surrogate replacement inside an output-free, aggregate-matched subspace (`PROTOCOL.md` §4.7).
4. **Interval-certification language.**
   - The C′ "certified" regions rest on hand-written outward rounding.
   - The bound relies on assumptions about the accuracy of library `exp`, `sqrt` and `tanh`, which are covered by relative padding but not formally established.
   - The hardened version was not independently re-verified for soundness.
   - The correct wording is "**numerically certified under stated floating-point assumptions**", not proved.
5. **Results added after the previous review round** were re-reviewed in this round (`reviews/`).
