# 12 — Path relations and interactions

The four-term interaction of two token edits `E_1` and `E_2`, from a common `H`, is

\[
I(E_1,E_2)
= R(E_1+E_2) - R(E_1) - R(E_2),
\]

where `R(E)=m(H+E)-m(H)` and the sum of edits is addition in the residual, followed by one recomputation. This is not deletion, and it is not the biased-sum necessity of `03`.

## What is zero

Constant shifts at two different tokens add exactly in `H_2`, each on its own token, for any `ε≥0`. If the readout reads only token 2, a constant shift at token 1 has `R=0` whether or not token 2 is also shifted, and `I=0`. The maps are nonlinear. The interventions lie in the symmetry.

Value-only edits with every pattern clamped are affine in `V` before the second norm, because `b_O` does not depend on `V`. They are not affine after `LN_2` and `tanh`. A zero interaction at the attention output does not imply a zero interaction at `m`. The tests do not claim that stronger statement.

## What is not zero

On seed 4, token-1 edit `(0.3,-0.2,0.4,0.1)` and token-2 edit `(-0.2,0.5,0.1,-0.4)` have `I≈-0.062808`. Token 1 is a legal key and value for token 2, and the MLP is nonlinear. Those two facts also hold for the constant shifts on this block, and those shifts have `I=0`. They do not force a nonzero interaction, and this number does not split the softmax contribution from the MLP contribution. What it shows is that this pair of recomputed edits is not additive.

## Necessity is not this number

The biased sum `b+c_1+c_2` has `I=0` and both deletions necessary. The block can show the same split in miniature. A constant shift at the read token has response `a 1^T w`, which is additive in `a` and has zero interaction with another constant shift, while whether that response preserves a sign depends on the margin's baseline and on `1^T w`. Interaction does not answer the sign question.

## Cancellation of path derivatives

On seed 0 the direction `u` has `Dm=0` and `R(0.2 u)≈-0.629`. First-order cancellation of skip, attention, and MLP, in whatever proportion they occur, does not cancel the finite response. The path relation "the directional pieces sum to zero" is the statement `Dm=0`. It predicts the linear term. It does not predict `R`.

No additional scalar, computed from `H` without `w` and without the weights, is shown here to predict `I` or the remainder. The quantities that do predict them are the block and the edits.
