# 15. Where the information comes from

| Ingredient | Source | Uses the target response? |
| --- | --- | --- |
| `μ > 0`, `α > 0`, `s_gain > 0` | frozen weights | no |
| `W_Q = W_K = 0`, value map `μ I`, MLP row and column along `w` | frozen weights | no |
| base coordinates on the line | the base activation, before any edit | no |
| special gain and zero base | weights and that activation | no |
| `f(ρ) = μ λ(ρ)` | closed form of one LayerNorm | no; this is the attention marginal, not `f+g` |
| joint bound `0` | Theorem S3 | no |
| exact `max(f+g)` for a non-special gain | one-dimensional maximization | yes, if performed by evaluating the joint readout |

The sign hypotheses are not the sentence "the joint response is smaller than the product." The product gap is derived after the signs.

Evaluating the scalar formula on a grid evaluates the target family. That grid is a check and, for the strong gain, the source of the endpoint-maximum observation. It is not an input to Theorem S3.

## Cost, as operation counts

Checking the weight hypotheses is a constant number of matrix comparisons of the recorded shapes: two heads, width 2, one MLP row. It does not grow with a sampled intervention set. The bound `0` is then a constant. Reusing it on every `ρ` does not require another pass.

The baseline number `f(ρ)` is one evaluation of `λ`. It does not depend on token count beyond the structural hypothesis that the causal weight times the value map equals `μ`.

A generic Euclidean ball in width `d`, with a moving pattern and a wide MLP, does not inherit this constant cost. The exact joint support on that ball is an optimization over the block. This package does not claim a reduction of that optimization. If a certificate costs a global nonconvex maximization of the same response, it has not saved the prediction; it has renamed it.

No wall-clock timings are reported. Pytest's own summary line is a test log, not a scientific cost.
