# Workstream 1. Finite response on one pre-norm block

The intervention and the response are fixed before any bound.

Let `H` be a residual state of shape `(T, d)` with `T=3`, `d=4`. The block is

```text
X = LN_ε(H)
H1 = H + Attn(X)
H2 = H1 + MLP(LN_ε(H1))
m(H) = w · H2[r]
```

with `r = T-1`, causal scaled dot-product attention, two heads, and a `tanh` MLP. This is the block of step 1C-R. `Attn` and `MLP` are the structural maps in `proofs/block.py`.

```text
do(H_t <- H_t + δ)   replaces token t and recomputes X, H1, and H2.
R(H, t, δ) = m(do(H_t <- H_t + δ)) - m(H).
```

`Dm · E` is the directional derivative of `m` at `H` along that replacement. It is not `R`. A value-only derivative clamps the attention pattern and is a different linear map. On seed 16 and the recorded direction, the total derivative is `+0.02019538349844552` and the value-only derivative is `-0.008580691798845158`.

The constant-shift identity is used as a lemma, not re-proved as a mechanism. For every residual edit `E` and every per-token constant `C`,

```text
H2(H+E+C) = H2(H+E) + C,
R(H, E+C) = R(H, E) + κ(C),
```

where `κ` is `a 1^T w` if that token is `r` and `0` otherwise. The four-term interaction of `C` with `E` is zero. The centered part of a non-constant edit is not given by `w`. On seed 7 a centered edit at the read token disagrees with the same edit after the branch weights are set to zero. That disagreement is rechecked in `test_constant_component_factors_and_centered_part_uses_the_branch`.

## Families

Four non-equivalent questions are kept apart. An arrow that is not proved below is marked unsupported.

### A. Response sufficiency

A statistic `P` is sufficient for a family `F` of interventions when `P(H)=P(H')` implies `R(H, t, δ)=R(H', t, δ)` for every member of `F`. A relation of several values of `P` is a function of the tuple. It cannot separate two states the tuple does not separate.

On seed 16, `H` and the state with `H_{0,3}` increased by `2` agree after coordinate 3 is dropped, and the intervention `δ=(0.4,-1.2,0.3,0.5)` at token 1 has responses of opposite sign. No function of the dropped-coordinate tuple returns both signs. This is the fiber obstruction already recorded; it is rechecked, not replaced by a new projection theorem.

On the constant-shift family every function of `H` is sufficient, because `R` does not depend on `H`. That sufficiency is vacuous.

The finite kernel test below is a property of `(ΔX, ΔU, weights)`, not of a coordinate projection of `H`. When the stacked map `(W_Q, W_K, W_V)` is injective and every gain is nonzero, the test accepts a non-constant single-token edit only if it demands a nonzero vector in a trivial kernel, which never happens. Injectivity on seeds 0 through 11 is a measurement: each smallest singular value is greater than `0.5`. It is not a theorem that every weight matrix is injective.

### B. Multi-path coordination

These four are not the same predicate.

- **Information redundancy.** Two statistics determine the same responses on a named family. Dropping coordinate 3 destroys that, on the seed-16 pair.
- **Causal redundancy.** A deletion leaves the sign of a margin. For `m=c1+c2` with `c1=1`, `c2=-0.5`, the sum is positive and deleting `c2` leaves `1`. The deletion of `c2` is not necessary. If the sum of two contributions is positive, at most one deletion-to-zero is necessary.
- **Additive necessity.** Both deletions can be necessary while the four-term interaction is zero: baseline `-1.5`, contributions `1` and `1`. Each deletion leaves `-0.5`. The interaction of an affine map is identically zero.
- **Nonlinear synergy.** A nonzero four-term interaction. `(u+v)^2` has interaction `2uv`. `u^2+v^2` is nonlinear and has interaction `0`. On the block, headwise leave-one-out finite responses need not sum to the joint response; that remains the seed-4 existence fact from step 1C-R and is not re-estimated as a rate.

Softmax reweighting is compensation inside one attention equation. It is not, by itself, evidence of a second repair mechanism.

### C. Computational transport

On a recomputed trace the structural assignments `H1 = H + Attn(LN(H))` and `H2 = H1 + MLP(LN(H1))` have residual zero. Seed 0 has margin about `+0.4644721976523223` and seed 2 has margin about `-1.3564763847549894`. The residual does not order them.

A rank-one map solved on one pair zeros that pair and need not zero a second pair (`test_same_residual_two_sources` separates a missing parent from a wrong `ρ` only after the parent vector is retained; the scalar norm does not). A clamp that sets `H1` to a value other than `H+Attn(LN(H))` is a different intervention from the one that defines `R`. Its nonzero structural residual is off-manifold stitching. The same numeric residual is produced by a wrong linear `ρ` on a recomputed trace. The scalar does not say which.

No stitched counterfactual in this note is claimed to be an on-manifold intervention.

### D. Finite-response constraints

The competitors are the gradient, the Hessian, a Lipschitz bound, a path-specific (value-only) derivative, and direct evaluation of `R`.

Direct evaluation equals `R` because it is `R`. The gradient equals the linear term and, on seed 0 along the unit vector orthogonal to the token-1 gradient, the linear term is about `0` while the finite response at step `0.2` is about `-0.629`. The Hessian-scale ratio `R/step^2` at step `10^{-3}` on that ray is about `-14.05`. Computing it spends finite evaluations of `m` or a second derivative of `m`. The value-only path can have the opposite sign from the total derivative. Cauchy–Schwarz bounds the linear term,

```text
|Dm · E| ≤ ||g_t|| ||δ||,
```

and does not bound the remainder.

## The quantity that does not use R or Dm

**Budget B0**, weight-free. Architecture, `ε`, gains, biases of both LayerNorms, readout `w`, read index, base `H`, edit `(t, δ)`. Excludes `W_Q, W_K, W_V, W_O, b_O, W1, b1, W2, b2`, excludes `R`, excludes `Dm`.

**Budget B1**, weights allowed, response forbidden. B0 plus the branch matrices. Still excludes `R` and `Dm`. LayerNorm values of the edited token are allowed: they are the first structural map, not the margin.

**Budget B2**, forbidden for a reduction. Any value of `m`, `Dm`, or a Hessian-vector product of `m`.

## Theorem F. Finite level set of LayerNorm

Let `ε>0` and let `γ` have at most one zero coordinate. Then `LN_ε(h+δ)=LN_ε(h)` if and only if `δ` is constant.

The constant direction is the shift invariance already proved for every `ε≥0`. The converse is finite, not infinitesimal. Write `z = C h`, `s = sqrt(||z||^2/d + ε)`, and `φ = z/s`. Equality of LayerNorm on every nonzero-gain coordinate says `φ` agrees on a coordinate set `S` with `|S| ≥ d-1`. Restriction from the centered subspace `1^⊥` onto those coordinates is injective: a centered vector supported on at most one index is zero. So the centered vectors of `h` and `h+δ` satisfy `z' = λ z` with `λ = s'/s > 0` whenever they are not both zero, and both zero is the constant case. Then `λ^2 v + ε = λ^2 (v+ε)`, hence `ε(1-λ^2)=0`, hence `λ=1`. SymPy factors that residual as `eps (1-lam^2)` and returns the only positive solution `lam=1`. That check does not call the numeric LayerNorm.

If two or more gains are zero the restriction is not injective. For `γ=(1,1,0,0)`, `ε=1`, `h=(1,-1,0,0)`, `b=√2`, `a=√(1+b^2/2)=√2`, the displacement from `h` to `(a,-a,b,-b)` is not constant and the LayerNorm outputs agree. The same displacement is visible when every gain is `1`.

The Jacobian rank statement is not this theorem. For `ε>0` the bracket has kernel `span{1}` and, when every gain is nonzero, rank `d-1`. That is infinitesimal. Theorem F is the finite level set, and it needs the gain count.

## Theorem U. No finite weight-free bound on a non-constant edit

Fix `ε>0`, every gain nonzero, and a non-constant `δ`. At the zero state, `ΔX = LN(δ)-LN(0)` is nonzero by Theorem F. Set `W_Q=W_K=0`, every MLP weight and bias to zero, `W_V` to the split identity, and

```text
W_O = α outer(ΔX, w) / ||ΔX||^2.
```

Legal scores are identically zero, so each causal row is uniform and does not depend on `X`. The MLP difference is zero. The read-token attention difference is `(1/T) α w`. Therefore

```text
R = α ||w||^2 / T
```

when the edited token is not the readout, plus `w·δ` when it is. With `w=(0.2,-0.1,0,0.3)` and `T=3`, the coefficient is `0.0466...`. The recomputed block matches the closed form at `α ∈ {1,10,100}`.

So `|R|` is unbounded as `α` varies, while every object in budget B0 is fixed. The only function of B0 that upper-bounds `|R|` for all branch weights is `+∞`, once `δ` is not constant.

The constant family is exactly the family whose finite response does not depend on the branch weights. That maximality now has both directions: constants by the shift identity, and every non-constant by Theorem F plus this scaling.

## Theorem L. A Lipschitz certificate cannot undercut Cauchy–Schwarz

Suppose `|m(x)-m(y)| ≤ Λ ||x-y||_2` on a convex neighborhood of `H`, and `m` is differentiable at `H`. For a unit vector `u` and a small scalar `t`,

```text
|m(H+tu)-m(H)| / |t| ≤ Λ.
```

The limit is `|∇m(H)·u| ≤ Λ`, so `||∇m(H)||_2 ≤ Λ`. Hence

```text
Λ ||δ|| ≥ ||g_t|| ||δ|| ≥ |g_t · δ|.
```

Any valid Lipschitz upper bound on the finite response is at least the Cauchy–Schwarz number of the linear term. It is not a stricter certificate.

Softmax contributes a sharp constant in the explicit majorant below. For `p` on a simplex and `||v||_2=1`, the quadratic form of `diag(p)-pp^T` is the variance of the coordinate random variable `v_I`. Popoviciu's inequality bounds a variance on an interval `[m,M]` by `(M-m)^2/4`. A Euclidean unit vector has coordinate range at most `√2`, because `(v_i-v_j)^2 ≤ 2(v_i^2+v_j^2) ≤ 2`. Therefore the variance is at most `1/2`, and softmax is globally `1/2`-Lipschitz in the Euclidean norm. Equality is attained at `p=(1/2,1/2)` and `v=±1/√2`.

LayerNorm's bracket, for `ε>0`, has operator norm `1/s = 1/sqrt(v+ε) ≤ 1/sqrt(ε)`. The eigenvalues are `0`, `ε/s^3` along `z` when `z≠0`, and `1/s` on the orthogonal complement of `span{1,z}`. The numeric SVD matches this spectrum. Thus

```text
Lip(LN) ≤ ||γ||_∞ / sqrt(ε)
```

on all of `R^d`. The constant is approached at constant vectors. It is not an artifact of the product `||C/s|| + ||zz^T||`, which overcounts because those terms cancel along `z`.

Composing the global bounds gives a finite `Λ` that depends only on the weights, the gains, the biases, `ε`, and the architecture. The attention estimate expands `Q'K'^T - QK^T` into three terms and absorbs `||ΔX||^2` by the diameter of the LayerNorm image, which is bounded by `||γ||_∞ √d + ||β||`. The formula is `lipschitz_constant` in `proofs/response_bounds.py`. On seed 0 it equals `2894603.0377523545`. The token-1 gradient has norm `3.548311334245126`. The ratio is about `8.16×10^5`. Random finite moves on seeds 0 through 5, at scales up to 20, stayed under this `Λ` by many orders (the largest observed ratio of `|Δm|` to `Λ||ΔH||` was about `7×10^{-7}`). The certificate is valid and much weaker than Cauchy–Schwarz.

A state-dependent majorant that replaced `1/sqrt(ε)` by `1/sqrt(v+ε)` at the base token would still be a Lipschitz constant only after the minimum of `v` on the whole segment is used. Even the base-point value on seed 0, token 1, is about `26.8` rather than `316`, and the resulting product remains far above `||g||`. It cannot cross below `||g||` if it is valid. No such tightening was needed for the negative statement, and none was promoted to a predictor.

## Theorem K. A conditional exact family, and why it is not generic

Assume the edited token's finite increment `ΔX = LN1(h+δ)-LN1(h)` lies in `ker W_Q[h] ∩ ker W_K[h] ∩ ker W_V[h]` for every head, and the induced `ΔU = LN2(h1+δ)-LN2(h1)` lies in `ker W1`. Then `Q,K,V` do not change, the attention output does not change, the MLP preactivation does not change, and

```text
H2(H+E) = H2(H) + E,    R = 1_{t=r} w·δ.
```

The prediction uses `w` and the read index. It does not use `R` or `Dm`. The hypothesis is checked by matrix products against `ΔX` and `ΔU`.

This family always contains the constant line: those increments are zero, and zero lies in every kernel. That case is the shift identity.

It contains a non-constant vector on a constructed weight setting. Seed-1 weights are projected so that they kill

```text
δ = (0, 0.45, -0.25, 0.15)
```

at the read token, with `w=(1,0,0,0)`. Then `w·δ=0`. The recomputed response is `0`. The branch tensor differs by less than `10^{-15}`. The token gradient satisfies `g·δ = 0.1066527641320657` and `||g|| ||δ|| = 0.4554966651252395`. The exact certificate `0` is a valid upper bound on `|R|` and is strictly smaller than the Cauchy–Schwarz number. The gradient's own inner product is not the finite response.

The same vector at scales `0.25, 0.5, 1.5, 2` has nonzero branch increment. A draw of 400 other centered edits on those weights had branch norm at least `0.053`. The extra solution is not an open set of edits.

On the Gaussian ensemble the extension is empty. For seeds 0 through 11 the stacked map `R^4 → R^{12}` given by all `W_Q, W_K, W_V` has smallest singular value greater than `0.5`, so its kernel is `{0}`. Theorem F then forces `δ` constant. The conditional equality adds no non-constant edit on that open set of weights. One explicit full-rank point is enough to know the degeneracy is not the whole weight space. Real-analyticity of the singular-value function in the weights puts the rank drop in a measure-zero set; the samples are the concrete side of that statement, not a proof about every matrix.

## Theorem J. The Jacobian kernel is not the finite kernel

There is a weight setting and a unit centered direction `u` at the read token such that `J_LN(h) u` lies in every `Q,K,V` kernel and `J_LN2(H1) u` lies in `ker W1`. The branch directional derivative is then zero, so `Dm·u` equals the skip derivative. At step `0.8` the finite LayerNorm increment is not in `span{J_LN u}` (distance `0.0939`), the skip prediction is `-0.6853485283715309`, and the recomputed `R` is `-0.6779358472770767`. The absolute gap increases at steps `0.8, 1.5, 2.5, 4`. A first-order structural test and the finite test are different predicates. The chain rule does not supply the finite identity, and the finite identity's hypothesis is not the chain rule.

## What was not obtained

No inequality of the form `|R| ≤ f(B0)` with `f` finite on non-constant `δ` and valid for all branch weights exists, by Theorem U.

No Lipschitz certificate is strictly tighter than `||g_t|| ||δ||`, by Theorem L. The explicit certificate is vastly looser.

On weights whose `Q,K,V` stack is injective, the factor-kernel test adds nothing to the constant-shift identity.

A search for an exotic non-constant zero of the branch, on seed 0, at radii `0.05, 0.2, 0.5, 1, 2`, with 12 random restarts and projected steps, reached minima `0.037, 0.086, 0.272, 0.478, 0.617`. That is not a proof that no exotic zero exists. It is recorded as an open cell, not as a theorem.

Direct evaluation remains the only examined procedure that returns `sign(R)` for a general `δ` on a general weight. The derivative returns the linear sign. The constant-shift evaluation returns the finite value on constants. The kernel test returns the finite value on a non-generic set and otherwise abstains.

## Classification

| object | class |
|---|---|
| Shift identity | Retained as pre-norm plus addition. Not a new mechanism. |
| Theorem F | Retained as the finite companion of the known kernel of `LN`. |
| Theorem U | Retained as a budget boundary. |
| Theorem L | Retained. It is the mean-value inequality. |
| Explicit `Λ` | Retained as a valid, non-competitive certificate. |
| Theorem K on injective `QKV` | Reduces to the shift identity. |
| Theorem K on the constructed weights | Conditional existence. Unused-direction calculation. Not a COR successor. |
| Theorem J | Retained as a separation between orders. |
| `O`, `D`, a cohomology class of `ds` | Unsupported as constraints on `R`. |
| A relation among insufficient coordinates | Cannot add the missing sign. Circular if the kept coordinate is chosen after seeing `R`. |

## Dependency arrows for this stream

```text
structural equations of the block
    --exact equation--> H2 and m
do(H_t <- H_t+δ)
    --exact equation--> R
shift invariance of LN
    --theorem--> R(H,E+C)=R(H,E)+κ(C)          [cited lemma]
Theorem F
    --theorem--> non-constant δ moves LN when at most one gain is zero
Theorem F + explicit scale family
    --theorem--> |R| unbounded outside budget B0
differentiability of m
    --theorem--> Λ ≥ ||∇m||
kernel membership of (ΔX, ΔU)
    --checkable assumption--> hypothesis of Theorem K
Theorem K
    --conditional result--> R = w·δ_read
injectivity of stacked QKV
    --measurement--> seeds 0..11; --conditional result--> Theorem K adds no non-constant edit
Jacobian membership
    --theorem--> linear skip match; --counterexample--> not the finite match
exotic branch zero at generic weights
    --conjecture--> not excluded by the local rank or by the search
O or D or [ds]
    --unsupported--> sign(R) or a tighter finite bound
```
