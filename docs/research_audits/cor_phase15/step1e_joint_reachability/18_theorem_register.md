# 18. Theorem register

| Name | Statement | Hypotheses | Status |
| --- | --- | --- | --- |
| S1 | `sup(f+g) = sup f + sup g` iff one point attains both marginal maxima, when those maxima exist | attainment, no structure on the block | proved |
| Graph | With a fixed causal pattern and value map `μ I`, `μ ≠ 0`, the MLP increment is a function of the attention increment | single-token edit, MLP after LayerNorm, any widths of that form | proved; constant-fiber test uses a non-rank-one MLP |
| S2 | On the invariant line, `max(f+g) < max f + max g` for every forward segment | `μ, α, s_gain, ε > 0`, readout and MLP along `n`, `W_Q = W_K = 0`, isotropic value map | proved for both edit sites |
| S3 | That joint maximum equals `0`, and the gap equals `f(ρ)` | S2 plus `μ = 1/2`, `α = s_gain = 1`, `ε = 10^{-5}`, zero base | proved |
| S4 | Linear ball gap `ρ(\|\|p\|\| + \|\|q\|\| - \|\|p+q\|\|)` | Euclidean ball, `a = Pδ`, `b = Qδ` | proved; equality iff nonnegative parallel |
| Raw full response | `R(t) ≤ ρ < ρ + f(ρ)` on the special-gain read-token segment | S3 and `w = n` | proved as an upper bound; the endpoint `0.8187...` is the observed maximum, not the certificate |
| Final LN transfer | none | — | not claimed; note 14 is a counterexample |
| Spectral joint split | none that beats the separate aligned bound | triangle inequality on `w+v` | shown to collapse; not claimed |

The strong-gain grid values in notes 12 and 13 are numerical. They illustrate S2. They are not a second exact-maximum theorem.

Step 1D's Theorems U, L, S, Align, Kernel-out, and the product-ball ceiling are unchanged and are not restated as new results.
