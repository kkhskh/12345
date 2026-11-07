# 09 — Why a squared distance does not determine a sign

## 1. The one-dimensional fact, and the reason it is general

Identity edge, `M=I`, `D(s_1,s_2) = (s_1-s_2)^2 / 2`. This depends on the difference through its square. In particular `D(s)=D(-s)` while a linear readout of the first coordinate changes sign (`test_squared_distance_discards_sign`).

**Theorem Sign.** Let `r` be any `M`-normal vector and `g ∈ ker d`. Then `D(g+r)=D(g-r)`. Let `L(s)=w·s`. If `w·r ≠ 0`, then `L(g+r)` and `L(g-r)` have first-order displacements of opposite sign about `g`, and they are not equal. Therefore the level sets of `D` do not determine the sign of the inconsistent component of a linear readout.

**Proof.** `D(g+r)=||r||_M^2=||−r||_M^2=D(g-r)` because `g` is the projection of both points when `r ⊥_M ker d`. Linearity gives `L(g+r)-L(g)=w·r=-(L(g-r)-L(g))`. QED.

The square is what discards the sign. Replacing `D` by the signed vector `ds` does not escape Theorem NFL: CE-01 holds `ds` fixed and flips `w`. The signed residual determines a direction in the cochain. It does not determine `w`. Predicting `L` from `ds` requires a further map `ψ(ds)`. If `ψ` is identified without `L`, it is some fixed map, and CE-01's two heads still disagree with it. If `ψ` is identified from `L`, the predictor contains the readout it was supposed to predict.

## 2. Gradient alignment is the sign, and it belongs to attribution

At `s=(1,0)` with `O=(s_1-s_2)^2`,

```text
∇O = (2, -2),    ∇L = (1, 0),    ∇L·∇O = 2 > 0.
```

The steepest descent step `δ = -∇O = (-2, 2)` has `∇L·δ = -2 < 0`. The attribution step `∇L` has positive derivative `1` and is not parallel to `-∇O` (`test_ce17_obstruction_descent_can_oppose_the_logit_gradient`).

**Theorem Descent.** Let `∇` denote the Euclidean gradient. The directional derivative of a differentiable `L` along the vector `-∇D` equals `-∇L·∇D`. The directional derivative along `∇L` equals `||∇L||^2` and does not depend on `D`.

This is a first-order statement. A finite step along `-∇D` can decrease `L` even when the directional derivative is positive. Anti-alignment means a negative inner product, not that the two gradients are parallel.

When `M` is not a multiple of `I`, the direction from `s` to `Π_M s` need not be the Euclidean steepest-descent direction `-∇D`. The derivative along the repair `-(s-Π s)` equals `-∇L·(s-Π s)`. The sign is still a pairing with `∇L`. CE-17 is the Euclidean case `M=I`, where the two directions agree, and there the descent step has the wrong sign for `L`.

Phase 1's bridge table said that alignment of `∇O` with `∇m` would make obstruction reduction increase the margin. For the steepest-descent direction that statement has the inequality backwards. Anti-alignment is the condition that makes `-∇D` increase `L`. Either wording has the same scientific consequence: the sign is the inner product with `∇L`. Computing that inner product requires the output gradient. A method that needs `∇L` in order to know whether to descend on `D` is using attribution to supervise obstruction. It is not an obstruction prediction that replaces attribution.

When the two gradients are anti-aligned, `-∇D` and `∇L` agree in sign on the directional derivative and still need not be the same direction. The component of `-∇D` orthogonal to `∇L` changes `D` and does not change `L` at first order. That orthogonal component is exactly the part of the obstruction geometry that the logit does not see. It does not improve the first-order prediction beyond `∇L`.

## 3. Magnitude versus ranking versus probability of reversal

| Question | Does `D` answer it? | What would |
|---|---|---|
| Sign of `R` along a declared `δ` | No | `∇L·δ` |
| Absolute bound on `|L(s)-L(Πs)|` | Yes, given a Lipschitz constant | Theorem Lip |
| Which of two `δ` changes `L` more | No, unless both are ranked by `∇L` | Attribution |
| Probability of a sign flip under a noise model | Only after a noise model and a margin are added | Update Proposition 2, in which `O` is slack |

## 4. Update Proposition 2 in this light

The proof in file 05 shows sign stability from `|m|` clearing a Jacobian budget. The squared obstruction term is not the source of the sign. It cannot be, by Theorem Sign, unless a further assumption restores a direction. The printed proposition does not restore one. It adds `τ O` as an unsigned penalty.

## WHERE DOES THE CAUSAL CONTENT ENTER?

In `w`, equivalently `∇L`. The squared consistency score does not contain it.
