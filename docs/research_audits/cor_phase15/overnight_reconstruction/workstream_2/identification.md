# Workstream 2. Decision-state identification

No unique intrinsic germ is declared. A germ in the sense of the audited definitions is `s = P φ`. The equation does not say which `P` is the decision.

## Attacks

**P = 0.** Then `s = 0`, the coboundary is zero, and both `O` and `D` are zero for every input, including a negative readout. The zero cochain is an algebraic section because the kernel of a linear coboundary is a linear subspace and therefore contains `0`. Vanishing discrepancy is not a correct decision. Checked on the two-node identity edge.

**Invertible P.** An invertible change of stalk coordinates can be absorbed into `(ρ, M, W)`. If those are not transformed with `P`, Euclidean `O` moves while the block and `m` do not. Invertibility removes the collapse `P=0` and does not produce an invariant decision state. LayerNorm is not this change of coordinates. It is a function inside the block. A raw dot `w·h` and a dot `w·LN(h)` are different scalars. The constant `a 1` at the read token changes the raw margin by `a 1^T w` and changes `w·LN(H2)` by `0`, because the final normalization kills the skip that the raw readout keeps. Test: `test_final_layernorm_readout_kills_the_constant_the_raw_readout_keeps`.

**Coordinates chosen after R.** The seed-16 pair is separated by coordinate `H_{0,3}` and is not separated without it. Keeping that coordinate because the responses differed uses `R` in the definition of `P`. The resulting sufficiency is circular. A projection fixed before the intervention can still fail, as that same deletion shows.

**Label encoded in P.** If a coordinate of `s` is defined to be the margin, then a function of `s` can recover the margin. The state was not identified. It was copied from the outcome.

**Train-only agreement.** A rank-one `ρ` built from one pair `(s_u, s_v)` satisfies `ρ s_u = s_v` and has residual zero on that pair. Replacing the child by another vector from the same state leaves a nonzero residual equal, in the worked example, to a missing-parent vector up to sign. The training residual does not order margins. Statistical agreement on the fitting pairs is not an intervention identity. Sutter, Minder, Hofmann, and Pimentel, NeurIPS 2025, Theorem 1, is the stronger fact: under their Assumptions 1–5, an unrestricted alignment represents any algorithm in any network. A train-only linear fit is a restriction, and it still does not identify a causal state.

## Conditions that would identify a state, and their status

1. **Sufficiency for a named family, proved from the equations without evaluating R on that family.** Status: met by the constant-shift family for every `P`, vacuously, and by Theorem K when the kernel hypothesis is met. Not met by coordinate deletion. Not met by `O` or `D`.
2. **The same P fixed without the evaluation responses.** Status: required, and not implied by the definition of `P`. Choosing the seed-16 coordinate after the sign violates it.
3. **Invariance under reparameterizations that preserve `m`.** Status: false for Euclidean `O` under stalk rescaling, and false for a raw-readout identity under a final LayerNorm.
4. **Uniqueness.** Status: false. `P=0` and an invertible `P` are both legal, and they do not define the same germ. Superposition, in the sense already audited, is an additional non-uniqueness and was not repaired.

No conjunction of these four holds for a single `P` that is not already the full state, the derivative of `m`, or the constant-shift symmetry.
