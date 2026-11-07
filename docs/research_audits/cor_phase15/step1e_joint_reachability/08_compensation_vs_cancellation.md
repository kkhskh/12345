# 8. Compensation and cancellation

The line phenomenon has a specific definition. Whenever the step increases, the read-token hidden coordinate increases, and `y ↦ s_gain tanh(-α λ(y))` decreases. The MLP readout therefore moves against the attention readout for every positive step. The definition uses the weights and `λ`. It does not use a task label, and it does not say "the output got smaller."

It is not an error-correcting code. There is no external target. At step `1` the raw response is `0.818715171479619`, not a small residual. The branch sum at that point is `-0.181284828520381`. The attention piece alone is `0.7070997102248`. The coupling removes the positive branch corner. It does not pin the raw response at zero.

Several nearby patterns are different, and the package does not merge them.

- **Product corner.** With `s_gain = -1` and the other special gains, both readouts are positive at step `1`: attention `0.7070997102248`, MLP `0.888384538745181`, sum `1.595484248969981`. The image is still a curve, but the support gap is zero. Opposite orientation is doing the work, not the mere fact that one map follows the other.
- **Base-point accident.** `R'(0)` and `R(1)` have opposite signs. A cancellation read off the base Jacobian does not survive.
- **Skip surrogate.** At step `0.2`, the skip is `0.2` and the raw response is `0.018548133077188`. The surrogate is an order of magnitude larger than the response. At step `0.01` the skip is positive and the raw response is `-0.232881400055210`.
- **Small weights.** The special gains are `1/2` and `1`. The strong-gain numerical check uses `μ = 2` and `s_gain = 2`. The strict inequality of Theorem S2 does not require small weights. The exact value `0` does require the special gain; that is a stronger evaluation, not the only strict case.
- **Readout accident.** If the readout is the normalized all-ones vector, both branch increments on the line are orthogonal to it and the branch gap disappears. The orientation is relative to `w`.

The stable fact is the opposite monotonicity along this family. Calling it compensation is fair only in that narrow sense.
