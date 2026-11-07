# Research journal

The previous step had left a precise gap: a non-constant family, a quantity that does not use `R` or `Dm`, and a bound tighter than the constant-shift identity and Cauchy–Schwarz. The first pass looked for an exact symmetry larger than the constants. The finite level set of LayerNorm at `ε>0` is still the constants when at most one gain is zero. The proof is the scale identity `ε = λ^2 ε`, checked in SymPy independently of the numeric map. Two zero gains are a genuine enlargement; the algebraic pair was written down and evaluated.

With the level set known, a centered edit produces a nonzero `ΔX` at the zero state. Setting the pattern to uniform and scaling `W_O` makes the read-token response `α ||w||^2 / T`. The recomputed block matches at `α = 1, 10, 100`. That closes budget B0. There is no finite weight-free certificate.

The Lipschitz route closes for a different reason. Differentiability says every valid Lipschitz constant is at least `||g||`, so it cannot undercut Cauchy–Schwarz. An explicit majorant was derived anyway, using the bracket norm `1/s` and the `1/2` Lipschitz constant of softmax from Popoviciu's variance bound. On seed 0 it is `2894603.0377523545` against a gradient norm `3.548311334245126`. Stress checks stayed inside it. It is a bound and not a competitor.

The remaining positive construction puts one finite `ΔX` into the branch kernels by projecting the weights. On `δ = (0, 0.45, -0.25, 0.15)` at the read token, with `w` along the first axis, the recomputed response is `0` and the branch tensor is numerically zero, while `g·δ = 0.1066527641320657` and the Cauchy number is `0.4554966651252395`. Scales other than `1` leave the family. Seeds 0–11 have injective stacked `QKV` maps, so the construction adds nothing there. A Jacobian version of the same projection matches the linear skip and misses the finite response; the gap grows from step `0.8` to step `4`.

A search for an exotic zero on seed 0 did not find one. That search is not a proof. It is recorded as open and is not used as a candidate.

The `O`/`D` algebra, the deletion pair, the value-only sign, the remainder, the feature-3 fiber, and the final-LN transfer were rechecked so the package does not depend on the reader's memory of the earlier audits. They came out as before. No favorable score was computed. The historical AUC and Spearman comparisons were not refit.

Reviews A–H were then written against these quantifiers in the same environment. The objections that stuck were looseness of `Λ`, the manufactured character of the kernel point, and the open exotic cell. None of them restored a generic centered bound or a COR successor.

The verdict is the boundary, not a theory.
