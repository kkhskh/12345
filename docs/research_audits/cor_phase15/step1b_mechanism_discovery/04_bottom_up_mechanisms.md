# 04 — Bottom-up, from the architecture

Start from the block, not from a score.

**Residual addition.** For a linear readout of `h_out = h + a + m`, path ablation of `a` changes `L` by `-w·a`. The inner product is attribution along that addend. Vector agreement of `a` and `m` is neither necessary nor sufficient for that inner product (`test_linear_redundancy_is_not_agreement_of_the_vectors`). Cancellation can hide large addends (`test_cancellation_hides_large_paths_from_the_linear_readout`).

**Shared nonlinearity.** If the next map is `σ(h+a+m)`, mixed derivatives generally do not vanish. Architecture predicts `I ≠ 0` for parallel addends that meet in `σ`, and `I = 0` for a linear readout of a sum. Both predictions are the Hessian of the next map. They do not use a cochain residual of the unintervened state.

**Attention.** Weights depend on the input. A shared linear parent map that matches one pattern fails on another. Relational "agreement" of two tokens under a fixed `ρ` is therefore not the attention equation.

**LayerNorm.** Euclidean residuals and inner products are not invariant (`test_layernorm_changes_euclidean_geometry`). A score that treats the residual stream as Euclidean is representation-dependent in a way the architecture itself already introduces, even before an analyst rescales coordinates.

**Readout.** Later layers can read directions a probe threw away. That is the factorization condition. Bottom-up, it is the statement that `P` is not the identity.

None of these primitives yields a scalar inconsistency that signs `R_c` without `w` or `σ`. Additivity of frozen addends when `L` is linear in the sum is a toy identity, not a prediction of the next LN/MLP. Softmax with supplied logits is not QKV attention. LN on a 3-vector is not residual-stream geometry at model width. The surviving negative is still right: a constant linear `ρ` is not the block. The surviving positive identities are ordinary calculus on the toys.
