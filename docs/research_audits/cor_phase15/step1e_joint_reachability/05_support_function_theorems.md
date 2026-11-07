# 5. Support theorems

## Theorem S1. When the product is exact

Let `f` and `g` be real functions on a nonempty set `U`. Then

```text
sup_U (f + g) ≤ sup_U f + sup_U g.
```

Suppose the three suprema are attained. Equality holds if and only if some point of `U` attains both marginal suprema.

The inequality is pointwise. If one point attains both marginals, its joint value equals the sum, so the joint supremum matches and that point attains it. Conversely, a joint maximizer whose value equals the sum of the marginal suprema must meet each supremum, because neither function can exceed its own supremum.

If `U` is compact and `f`, `g` are continuous, the maxima exist. On the segments below, `U = [0, ρ]` and both readouts are `C^∞`.

This criterion is the support function of a subset of a product. It is the right equality test. It is not a Transformer decision theorem.

## Scalar reduction

For `h = y n` and `ε > 0`,

```text
λ(y) = y / sqrt(y^2 / 2 + ε),    λ'(y) = ε / (y^2 / 2 + ε)^(3/2) > 0.
```

SymPy reduces `λ'(y) - ε / (y^2/2+ε)^{3/2}` to `0`. Thus `λ` is strictly increasing, odd, and bounded by `√2`. On the structural block, a step `t` along `n` at one token changes the attention readout by

```text
f(t) = μ (λ(y0 + t) - λ(y0)).
```

The read-token hidden coordinate moves by `t + f(t)` if the edit is at the read token, and by `f(t)` if the edit is at an earlier token. The MLP readout is

```text
g(t) = s_gain ( tanh(-α λ(y1(t))) - tanh(-α λ(y1(0))) ).
```

## Theorem S2. Opposite monotonicity

Assume `μ > 0`, `α > 0`, `s_gain > 0`, and `ε > 0`. For either intervention site, and for every real base coordinate,

```text
f'(t) = μ λ'(y0 + t) > 0,
g'(t) = s_gain sech^2(-α λ(y1(t))) (-α) λ'(y1(t)) y1'(t) < 0.
```

The hidden slope `y1'` is `1 + μ λ' > 0` at the read token and `μ λ' > 0` at an earlier token. So `f` is strictly increasing and `g` is strictly decreasing on every real interval. On `[0, ρ]`,

```text
max f = f(ρ),    max g = 0,
```

and those maxima are attained at different endpoints when `ρ > 0`. Theorem S1 gives

```text
max (f + g) < f(ρ) = max f + max g.
```

The same split applies to the full raw response at the read token, `R(t) = t + f(t) + g(t)` when `w = n`, against the three-marginal sum `ρ + f(ρ) + 0`. The skip and the attention readout are both maximized at `ρ`, where `g(ρ) < 0`, so that corner is also missed. At an earlier token the skip at the readout is `0`, and `R = f + g`.

The hypotheses are signs of weights. They do not mention the value of `R`.

## Theorem S3. Exact joint maximum for one gain

Add `μ = 1/2`, `α = 1`, `s_gain = 1`, `ε = 10^{-5}`, and base coordinates `y0 = y1 = 0`. Then for every `t ≥ 0` and for both intervention sites,

```text
f(t) + g(t) ≤ 0,
```

with equality only at `t = 0`. Therefore on every forward segment the joint branch maximum is `0`, and the gap against the marginal sum is exactly

```text
f(ρ) = (1/2) λ(ρ).
```

At `ρ = 1`, `λ(1) = 1.4141994204496` and `f(1) = 0.7070997102248`. The endpoint values are `g(1) = -0.888384538745181` at the read token and `g(1) = -0.888379599988688` at token `0`. Both are strictly negative, as the theorem requires. The full read-token response is `R(1) = 0.818715171479619`, which is positive: the branch bound `0` is not a claim that the raw response is nonpositive.

### Proof of the sign

Let `u = λ(t) ∈ [0, √2)`. Inverting `λ` gives `t = u √ε / sqrt(1 - u^2/2)`. The earlier-token hidden coordinate is `u/2`, which is the smaller of the two sites. The read-token coordinate is `t + u/2 ≥ u/2`. Since `λ` and `tanh` are increasing on `[0, ∞)`, it is enough to prove

```text
tanh(λ(u/2)) ≥ u/2
```

and the read-token inequality follows. Equality holds only at `u = 0`.

The integer facts checked by `proofs/certificate.py` are: a degree-8 exponential partial sum already forces `tanh(1) > √2/2`; `(1/100)^2 > 2ε`, so `λ(1/100) > 1`; and `16 · 50000 > 9 · 12`, which is `4k > 3` for `k = sqrt(50000) / (2 sqrt(3))`.

**Case `u ≥ 1/50`.** Then `u/2 ≥ 1/100 > sqrt(2ε)`, so `λ(u/2) > 1` and `tanh(λ(u/2)) > tanh(1) > √2/2 > u/2`.

**Case `0 < u ≤ 1/50`.** Then `s(u/2) ≤ sqrt(3/50000)`, so `λ(u/2) ≥ k u` with the `k` above. If `λ(u/2) > 1`, the previous tanh bound is larger than `u/2 ≤ 1/100`. If `λ(u/2) ≤ 1`, then `k u ≤ 1` and

```text
tanh(z) > z - z^3/3
```

on `(0, 1]`. The derivative of `tanh(z) - z + z^3/3` is `z^2 - tanh(z)^2`. SymPy reduces that derivative identity to zero, and `tanh(z) < z` for `z > 0` makes the derivative positive. The cubic `q(z) = z - z^3/3` is increasing on `[0, 1]`, so `q(λ(u/2)) ≥ q(k u)`. The worst ratio on this case is at `k u = 1`, where `q(ku)/u = 2k/3 > 1/2`. Thus `tanh(λ(u/2)) > u/2`.

## Theorem S4. Linear coupled maps

Let `a = P δ`, `b = Q δ`, and let `U` be the Euclidean ball of radius `ρ`. For a raw linear readout,

```text
max_S w·(a+b) = ρ ||(P+Q)^T w||,
max w·a + max w·b = ρ (||P^T w|| + ||Q^T w||).
```

The gap is zero if and only if `P^T w` and `Q^T w` are nonnegative multiples, including the case where either is zero. The identity

```text
(||p|| + ||q||)^2 - ||p+q||^2 = 2(||p|| ||q|| - p·q)
```

is the Cauchy defect. SymPy checks it on `R^2`. It holds for every `ρ > 0`. It is the support function of the joint linear image, which is the Euclidean-ball form of the fact affine arithmetic keeps by sharing a noise symbol. It is not a finite-response bound for the nonlinear block.

An explicit pair with both images nonzero: `P = diag(2, 0)`, `Q = diag(-1, 3)`, `w = (1, 1)`, `ρ = 2`. The joint support is `6.324555320336759` and the marginal sum is `10.32455532033676`. The gradients `(2, 0)` and `(-1, 3)` are not nonnegative multiples.

The block's derivative at a point has this shape, with `P = J_A` and `Q = J_ψ (I + J_A)` at the read token. Using that derivative as a finite bound is a different claim and is false: see `07_finite_path_operator_analysis.md`.
