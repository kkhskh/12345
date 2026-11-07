# 10 — Family D: multi-path reconciliation

The right structural equation at a node with parents `p_1,…,p_k` is

```text
h_v = f_v(h_{p1}, …, h_{pk}).
```

Pairwise `h_v ≈ ρ_i h_{pi}` is a different equation. It is false of addition (`test_pairwise_parent_equality_is_the_wrong_constraint`) and of softmax mixing.

**Joint residual of addition.** `u+w-v` is zero on recomputed additive traces. Useful variation across decisions then cannot live in that residual. It can live in `L(v)`.

**Wrong aggregate with agreeing pairwise maps.** If both parents equal 1 and `v=1`, pairwise residuals against `ρ=I` vanish, while an external target 0 is missed. Agreement of surrogates is not correctness (`test_compatible_parents_wrong_aggregate_needs_an_external_target`).

**Candidate objects that are not norms.** Rank of the matrix whose columns are path vectors; the projection of those paths onto `w`; the mixed Jacobian of `f_v`. The first is a representation property. The second is attribution. The third is local sensitivity of the aggregator. None is `||d s||`.

**Graph refinement.** Duplicating an additive edge changes a pairwise residual score and does not change `f_v`. The global object of a chosen analysis graph is not automatically a property of the computation. Step 1A CE-12 already showed this; it remains the right warning.

**Status.** Multi-path structure matters. The quantity it recommends is `f_v` or its Jacobian, not a coboundary of pairwise maps.
