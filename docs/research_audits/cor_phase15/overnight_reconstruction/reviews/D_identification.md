# Review D. Identification

Same cloud environment. Not an external referee.

**Objection.** Computing `ΔX = LN(h+δ)-LN(h)` is already a piece of the forward pass, so budget B1 cheats.

**Resolution.** The budget list says the first LayerNorm is allowed and the margin is not. On the injective ensemble that computation returns a nonzero `ΔX` and the kernel test abstains. Abstaining is not evaluating `R`. If the budget is shrunk to B0, Theorem U says there is no finite centered bound at all. Either way the generic block does not gain a non-constant certificate.

**Objection.** The read-token branch Jacobian singular value on seed 0 is `0.015`, but several other seeds have values near `10^{-6}`. "Generic full rank" is an overclaim.

**Resolution.** The register does not say every seed has a large singular value of that read-token map. It says the stacked `QKV` map is injective on seeds 0–11, with smallest singular value above `0.5`, and that one analytic point of the read-token branch map has singular value `0.015`, so identical vanishing is false. Small singular values are ill-conditioning of one factor, not a kernel.

**Status.** Identification of a non-constant family on the Gaussian ensemble remains unresolved only for exotic cancellations outside the factor kernel. The factor-kernel route is closed there.
