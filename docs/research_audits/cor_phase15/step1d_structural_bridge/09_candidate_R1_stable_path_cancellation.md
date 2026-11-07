# 9. Candidate R1 — stable path cancellation

## The relation, defined without `R`

Let `α(δ) = w · ΔA_r` and `μ(δ) = w · ΔM_r`, so `R = 1_{t=r} w·δ + α(δ) + μ(δ)`. Write `J_A` and `J_M` for their derivatives at the base state along a fixed `δ`. The proposed relation is

```text
Q_1(δ) = J_A + J_M.
```

`Q_1` is a Regime 2 scalar. It does not use the finite response. By the chain rule on `H2 = H + A + M`,

```text
Q_1(δ) = ∇m · δ - 1_{t=r} w·δ.
```

So `Q_1` is the branch part of the total derivative. It is independently computable, and it is not independent of `∇m`.

## First-order gap versus separate norms

`|J_A + J_M| ≤ |J_A| + |J_M|`, strict when the two scalars are nonzero and have opposite signs. On seed 0 along the unit orthogonal `δ`,

```text
J_A = -0.013476113870395795,   J_M = 0.0065497667689125945,
Q_1 = -0.0069263471014832005.
```

The separate sum of absolute values is `0.02002588063930839`. The gap is the triangle inequality on two scalars. It bounds the linear term. It does not bound `R`.

## Finite cancellation does not follow

A rank-one update of `W2`, built from the base-point hidden derivative so that the new `J_M` equals `-J_A`, uses no finite response. After the update,

```text
J_A = -0.013476113870395795,   J_M = +0.013476113870395797,
Q_1 = 1.73×10^{-18},   w·δ = 0.
```

Both paths remain nonzero. The finite responses along `t δ` are

| step `t` | `R` | linear term `Q_1 t` |
|---|---|---|
| 0.001 | `6.98×10^{-9}` | `~10^{-21}` |
| 0.05 | `1.82×10^{-5}` | `~10^{-19}` |
| 0.2 | `3.31×10^{-4}` | `~10^{-19}` |
| 1 | `0.01143755544615821` | `~10^{-18}` |
| 2 | `-0.16901883579206622` | `~10^{-18}` |

Pointwise cancellation at `t = 0` does not control the integral

```text
α(δ) + μ(δ) = ∫_0^1 (J_A(H + tδ) + J_M(H + tδ)) dt.
```

The integrand leaves zero as soon as the state leaves the base point. At step `2` the finite branch response has absolute value `0.169` while the first-order prediction is numerically zero.

## When finite cancellation is reliable

Two cases were proved, and both reduce.

1. If `W_O w = 0` and `W2.T w = 0`, then `α = μ = 0` at every finite step, so the integral vanishes. Each path is unread by `w`. The separate aligned bound is already zero. This is Theorem Kernel-out, not a cancellation between two live readout contributions.

2. If a bound `|J_A + J_M| ≤ β` is known on the whole segment, the integral is at most `β`. Certifying `β` requires derivative information along the segment (Regime 2 or 3) or a Lipschitz bound on the joint derivative. The joint derivative is `∇m` minus the skip. A uniform bound on it is an isotropic statement about `∇m` and is at least as hard as Theorem L's constant. A sampled integral of JVPs converges to `R` and costs more derivative evaluations than one forward. Neither version is a new structural relation, and the global second-derivative scale `ε^{-3/2} ≈ 3.16×10^7` prevents a cheap Regime 3 remainder from beating Theorem S.

## Separation that the separate norms cannot see

Reflect `W2` on the left by the Householder map `I - 2 ww^T/||w||^2`. Singular values are invariant because the reflector is orthogonal, and `||W2.T w||` is invariant because the reflector sends that vector to its negative. `W_O` is untouched. Therefore `B_dir` and `B_align` agree on the original weights and the reflected weights, to `10^{-9}` or better. The finite responses on the unit orthogonal edit do not:

```text
original R = -0.010805881111744242,   Q_1 = -0.0069263471014832005,
reflected R = -0.05639898778472055,   Q_1 = -0.020025880639308387.
```

The separate Regime 1 certificates do not distinguish the pair. The quantity that does is `Q_1`, which is the branch derivative. The example separates "norm of each path" from "sum of the path derivatives." It does not separate "sum of the path derivatives" from "the gradient."

## Verdict on R1

`Q_1` is real, measurable without `R`, and strictly tighter than `|J_A| + |J_M|` as a bound on the linear term when the paths oppose each other. As a finite-response constraint it is the fundamental theorem of calculus applied to `∇m`. The step-`2` counterexample shows that the base-point relation does not imply a small finite response. No additional smoothness hypothesis that was both checkable in Regime 1 and strong enough to control the integral was found. R1 does not clear Level 3.
