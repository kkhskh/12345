# 10. What is certified, and what is only checked

Theorem S3 certifies `f + g ≤ 0` for every real step, both sites, under the special-gain hypotheses. The argument is the case split in note 5. It does not sample the segment. The float tests sample the scalar formula and the block to check that the implementation is the function named in the proof. A sample inside the bound is not the inclusion proof.

Theorem S2 certifies the strict comparison with the marginal sum for every positive gain and every base on the line. It does not certify a uniform numerical gap size. At base coordinate `2`, with unit gains and `ρ = 1`, the attention marginal is only `1.964174864088619e-6` and the joint branch value at the endpoint is `1.550181267351292e-6`. The inequality remains strict and the scale is the saturated LayerNorm slope.

No zonotope transformer, interval-bound propagator, or Taylor model was implemented for the nonlinear block. Those methods are the right general language for a dependence-preserving enclosure, and they already exist. Building a loose copy would not strengthen the exact line result. For the linear ball, Theorem S4 is already the exact support, so an outer approximation is unnecessary.

Arithmetic: the proof uses exact real operations and the integer comparisons in `special_gain_constants`. The block code uses float64. Agreement between the vectorized block, the loop block, and the scalar formula is at absolute error below `10^{-10}` on the tested steps. That is an implementation check. It is not a certified rounding bound.

The special-gain number `0` is not claimed for a Euclidean ball, for a moving attention pattern, or for a final-normalized readout.
