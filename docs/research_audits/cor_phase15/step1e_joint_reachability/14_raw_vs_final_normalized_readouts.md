# 14. Final normalization

The raw response on the zero-base line is the coordinate of `H2`, because `H2` stays on `span{n}` and `w = n`. Final LayerNorm with `γ = 1` and `β = 0` sends that coordinate `y` to `λ(y)`. The objective is no longer `w · (δ + a + b)`.

| step | raw `R` | final-LN response |
| --- | --- | --- |
| `0.01` | `-0.232881400055210` | `-1.413952871819700` |
| `0.20` | `0.018548133077188` | `1.374816323880916` |
| `1.00` | `0.818715171479619` | `1.414192464454970` |

At step `0.01` the normalized response is more negative than the raw response by about `1.18`. A raw bound does not bound it. At step `0.20` the raw response nearly vanishes and the normalized response is `1.375`. The near-cancellation is an artifact of the raw inner product. At step `1` the normalized response has saturated near `√2`.

The joint set of `(a, b)` is still defined. The support problem for `m_norm` is the support of `λ` of the sum, not the support of the sum. Theorems S2 and S3 are not restated for `m_norm`. The normalized column above is a counterexample to a silent transfer, computed from the same block the raw theorem uses.
