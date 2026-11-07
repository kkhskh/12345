# 6. The coupling that was proved, and the ones that were not

The relation used by Theorem S2 is: on the stated family, the attention readout is strictly increasing in the step and the MLP readout is strictly decreasing in the same step. Both statements are derivatives of the scalar reduction. They can be checked from `μ`, `α`, `s_gain`, `ε`, and the axis alignment. They do not take `R` as an input.

That relation implies the corner of the marginal box is empty, and Theorem S1 turns the empty corner into a strict support gap. The implication is sufficient on this family. It is not necessary for every conceivable gap. A negative `μ` can still separate the argmaxima; the proof above simply does not cover it.

## Inner-product form

On the line, `a_r` and `b_r` are parallel to `n` and point opposite ways for every positive step. Their inner product is negative whenever both are nonzero. This is the same monotonicity, written in the residual. It is not an independent axiom.

The inner-product bound does not extend to every edit. For `δ = (1, 1.2)` at the read token, from the zero state and the special weights, the attention readout is `-0.7067534927402195` and the MLP readout is `0.8883814182149048`. They are not the line orientation, and their sum is `0.181627925474685`, which is outside the special-gain bound `0`.

## What was not established

A uniform anti-alignment `w · J_ψ(h)^T w ≤ 0` for every state and a generic MLP is false as a claim and was not assumed. The rank-one opposing MLP makes the scalar derivative negative on the line; a width-2 MLP with no sign hypothesis is used only for the constant-fiber lemma.

Splitting `||W_O^T (w + v)|| ≤ ||W_O^T w|| + σ(W_O) ||v||` before applying an attention Lipschitz constant returns a separate aligned bound of the Step 1D kind. The triangle inequality erases the joint improvement. No spectral certificate of that split form is claimed to beat Theorem Align.

A global Lipschitz comparison that uses only `||J_A||` and `||∇(w·ψ)||`, and then replaces `||w + ∇(w·ψ)||` by `sqrt(||w||^2 + β^2)`, is a valid joint bound when the inner product is nonpositive. On this block with `ε = 10^{-5}` that bound is dominated by `1/√ε` and is much larger than the exact marginal sum on the segment. It is not the result.
