# 4. Information versus permitted computation

## The conflation

If a protocol is given the weights, the base state, the intervention, the edited token, and the structural equations, then `R` is a determined real number. Forbidding the protocol from evaluating the perturbed forward pass does not remove that information. It forbids an operation. The overnight budgets B0, B1, and B2 mixed these two restrictions. B1 was described as if the response were absent from the data. On a deterministic block it is not absent. It is uncomputed.

## Regimes

Operation counts are algebraic, not measured wall times. No timing experiment was run.

**Regime 0 — architecture only.** The block diagram, `ε>0`, the intervention type, and the raw-versus-normalized choice. No numeric weight. Permitted work is symbolic. Cost: none beyond reading the diagram. Theorem U says this regime has no finite uniform bound on nonconstant raw responses over unbounded branch weights.

**Regime 1 — frozen weights, no perturbed forward and no derivative of `m`.** All weights, gains, biases, `H`, `t`, and `δ` are readable. Permitted primitives: arithmetic, the Euclidean spectral norm of each factor, the closed-form LayerNorm radius and Lipschitz constant, and the matrix-vector products `W_O w` and `W2.T w`. Not permitted: evaluating `m` at `H+E`, a Jacobian-vector product of `m`, or a Hessian. Precomputation is one spectral norm per factor (`W_Q`, `W_K`, `W_V` per head, `W_O`, `W1`, `W2`: eight decompositions on this block). Each new intervention then costs one inner product `w·δ` and one norm. The resulting number is uniform in `H`. It is a bound, not `R`.

**Regime 2 — local derivatives.** Regime 1, plus Jacobian-vector products of `m` or of a named path at named states. One JVP at the base state returns `∇m · δ`. That is the linear term. It is not a finite certificate. The sum of the attention and MLP JVPs is exactly the branch part of `∇m`, by linearity of differentiation on `H2 = H + A + M`. Computing the paths separately and adding them does not create information beyond the total derivative.

**Regime 3 — higher derivatives or a proved remainder.** Regime 2, plus a bound on the derivative of `∇m` along the segment, or an integral remainder. The scalar `1/s` in the LayerNorm bracket has derivative `O(ε^{-3/2})` in the variance near a constant token. At `ε = 10^{-5}` the factor `ε^{-3/2}` equals `10^{7.5} ≈ 3.162×10^7` before it is multiplied by gains and by the branch maps. A global Taylor remainder at `||δ|| = 1` built from that factor is not a competitor to a sharp local estimate. This note does not claim a fully expanded Hessian constant; the order is enough to show that "add a remainder" is not automatically a tighter finite bound than Theorem S.

**Regime 4 — direct evaluation.** One additional forward pass of the block returns `R` exactly. Cost per intervention: one forward. There is no reusable inequality unless a Regime 1–3 certificate is computed as well.

## What reuse is worth

Theorem S, once the spectral norms are cached, applies to every base state and every finite `δ`. A gradient certificate must be recomputed at each new `H`, and still needs a remainder before it bounds `R`. Direct evaluation must be repeated for each pair `(H, δ)`. Those are different products: a uniform bound, a local linear prediction, and an exact value. A smaller number is not a better method when the guarantees differ.

## Consequence for the audit formula

The structural bound in `06` is a Regime 1 algorithm. It is not evidence that Regime 1 lacks the information to determine `R`. A Regime 4 algorithm on the same inputs returns `R`. The gap between them is computational permission and uniformity, not a missing variable.
