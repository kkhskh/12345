# 08 — Representation and metric invariance

## 1. Algebraic transformation law

Let each stalk transform by an invertible linear map, `s'_v = A_v s_v`, collected into a block-diagonal invertible `A`. Let `A_head` act on a residual by `(A_head r)_e = A_{head(e)} r_e`. Set

```text
ρ'_{u→v} = A_v ρ_{u→v} A_u^{-1},
M' = A^{-T} M A^{-1},
W' = A_head^{-T} W A_head^{-1}.
```

The weight law is the full congruence. Transforming only the diagonal blocks and dropping cross-edge entries does not preserve `O`. Then `d' s' = A_head (d s)`, `||d' s'||_{W'}^2 = ||d s||_W^2`, and `||s'-g'||_{M'}^2 = ||s-g||_M^2` for `g' = A g`. So `O` and `D` are unchanged, and `ker d` transforms onto `ker d'`. For a joint parent sum, each parent map is conjugated separately.

**Proof.** Substitute `s_u = A_u^{-1} s'_u` into `ρ s_u - s_v` and multiply by `A_v`. The metric identity is `r^T W r = (A_head r)^T W' (A_head r)` by the congruence above. QED.

This is invariance of the whole tuple `(s, ρ, M, W)`. It is not invariance of the Euclidean formula used when `M` and `W` are left as identities.

## 2. Euclidean failure

`ρ=1`, `s=(1,0)`, Euclidean `O=1`. Rescale the child by `A_v=2`, set `ρ'=2`, and leave `W=1`. Then `O'=4`. The transformed weight `W'=1/4` returns `O` to `1` (`test_ce10_reparameterization_changes_euclidean_obstruction`). A code path that freezes `W=I` after the rescaling does not compute the invariant.

## 3. Rankings

The two matrices in CE-07 do lie on one stalkwise congruence orbit: `A = diag(√10, √10, 1/√10, 1/√10)` carries `diag(10,10,1,1)` to `diag(1,1,10,10)`. The invariance law would also replace the cochains by `A s` and would preserve each distance. CE-07 keeps the numerical cochains fixed and changes `M`. That is a different operation, and it reverses the ranking. Invariance still does not choose a canonical `M`. It says that if one changes coordinates and transforms `M` and `s` together, the number is stable.

## 4. Estimators

A correctly reweighted residual has a transformed minimizer set, `ρ ↦ A_v ρ A_u^{-1}`. The Euclidean minimum-norm selection inside that set does not commute with the change of coordinates unless the tie-break is rewritten as the transformed Frobenius norm. Unweighted least squares with an untransformed ridge does not recover `ρ'`. One scalar counterexample is a single sample `s_u=(1,0)`, `s_v=1`, minimum-norm `ρ=(1,0)`, and a shear of the parent coordinate that produces a different minimum-norm solution of the same value constraint. The algebraic invariance of a population residual is not an invariance of the fitting procedure.

A quantity that is invariant only after a correct transformation of objects the estimator does not transform should be reported as coordinate-dependent. The Euclidean residual used in the experimental code is coordinate-dependent. A coordinate-dependent residual can still be computed. It is not an intrinsic geometric property of the network.

## 5. Downstream readout

If `L` is defined on activations before `P`, a change of stalk coordinates does not change `L`. It can change Euclidean `O`. The prediction "low `O` means high `L`" is then not invariant under a transformation that leaves the network and the decision alone (Phase 1 Attack J: same decisions, different obstruction). CE-10 is the calculation.

## WHERE DOES THE CAUSAL CONTENT ENTER?

Not in the transformation law. The law says the score is stable when every auxiliary object moves with the coordinates. The decision is stable when the auxiliary objects do not move at all. Those are different invariances, and the score has only the first.
