# Counterexample catalog

All witnesses use the seed-0 block unless a modification is named. Responses are recomputed by the vectorized forward. The loop forward agrees on the checks in `tests/test_step1d.py`.

## CE-L-scope

`α = 10^{-6}` on `W_O` and `W2`. Unit `δ ⊥ w`, nonconstant. `σ_min(Q,K,V) = 0.7886607004096067`.

```text
B_dir = 0.004051326443971923
||∇m|| ||δ|| = 0.8764350281115764
R = -1.326797428591320e-07
```

Valid against the claim that every weight-informed finite bound is at least the Cauchy number. The branch increments are `2.57e-7` and `4.93e-7`. Relevance as a model of a trained block is a different question and is not granted.

## CE-vacuous

Same construction at `α = 1`.

```text
B_dir = 1629250.228383555
B_align = 873666.373048331
|R| = 0.010805881111744242
||ΔA||_F = 0.256558
||ΔM||_F = 0.097857
```

The certificate holds and does not rank this edit.

## CE-kernel

`W_O w = 0`, `W2.T w = 0`, QKV singular value unchanged.

Raw response equals `w·δ` within `10^{-15}` at steps `0.01, 0.2, 1, 3` on the orthogonal direction and on `e_0`. `||ΔA||_F = 0.2536775090589375` and `||ΔM||_F = 0.10766150630303412` on the unit orthogonal edit. `σ(W_O) = 1.2823531777637214`, `σ(W2) = 0.9960563055080783`.

## CE-cancel

Rank-one update of `W2` from the base hidden derivative along the unit orthogonal `δ`. After the update the path derivatives are `-0.013476113870395795` and `+0.013476113870395797`. At step `2`, `R = -0.16901883579206622`.

## CE-flip

Left reflection of `W2` through `w`. `B_dir` and `B_align` agree with the unreflected model to `10^{-9}`. Responses on the unit orthogonal edit: `-0.010805881111744242` and `-0.05639898778472055`.

## CE-final-orth

Zero `W_O` and `W2`, unit orthogonal `δ`, final LayerNorm with `γ = 1`, `β = 0`, `ε = 10^{-5}`. Raw `R = 0`. Normalized `R = -0.32024254867419905`. Diameter bound `3.505739880594811`.

## CE-final-kernel

Output-kernel weights, same outer LayerNorm, unit orthogonal `δ`. Raw `R = 0`. Normalized `R = -0.13186230895395856`.

## CE-trace

Seed-0 weights. Base margin `0.4644721976523223`. Read token shifted by `-1.158811460185589 w`, margin `-0.4644721976523225`. Both structural residuals are `(0, 0)`.

## CE-const-final

Constant edit `δ = 1` on the output-kernel model. Raw `R = 1.5010589548912336 = w·1`. Final-LayerNorm `R` is numerically `0`.
