# 08 — Exact response and its linearization

Three quantities are kept apart.

| level | object | what it uses |
|---|---|---|
| A | `R(H,t,δ)`, the recomputed block | every weight and the finite `δ` |
| B | `Dm(H)·E_{t,δ}` | the chain rule at `H` |
| C | the remainder `R - Dm·E` | whatever the first derivative does not see |

Level B matches level A at step `10^{-6}` on the checked direction. It does not match at finite steps in a direction the derivative kills.

## A direction with derivative zero and finite response not zero

On seed 0, with NumPy 1.26.4, the gradient `g ∈ R^4` of `m` with respect to token 1's residual is computed by four directional derivatives:

\[
g ≈ (1.93178219,\ -2.10943103,\ -1.39327748,\ 1.57092632),
\qquad \|g\| ≈ 3.5483.
\]

The unit vector `u ∝ (g_1, -g_0, 0, 0)` satisfies `g·u = 0`. The directional derivative along `u` is about `-1.1×10^{-16}`. The finite responses are

| step | `R` | `R / step²` |
|---|---|---|
| `10^{-3}` | `-1.40474795×10^{-5}` | `-14.0474795` |
| `0.2` | `-0.629265931` | `-15.7316483` |
| `1` | `-0.811890576` | `-0.811890576` |

The ratio `R/step²` at step `10^{-3}` is `-14.0474795`. That ratio still contains higher-order terms, so it is not the pure degree-2 Taylor coefficient. It is enough that the ratio stays order one while the first derivative is zero: the response is not accounted for by `Dm`. At step `1` the same quadratic extrapolation of this ratio would be about `-14`, and the actual response is `-0.811890576`. The sign of the finite intervention is negative. The first-order prediction is zero, which has no sign. Reporting a linear sign on this direction does not report the intervention.

This is one direction at one state. It does not say every kernel vector of `g` has a large finite response. The constant vector `1` may or may not lie in the kernel of `g`. When the readout is on token 2 and the edit is a constant on token 1, both the finite response and the derivative are exactly zero, by the shift theorem. Kernel of the derivative is not one phenomenon.

## When the linear sign is the wrong guide

- The direction is orthogonal to `g` and the quadratic term dominates, as above.
- The direction is not small. Along `u`, `R/step²` moves from `-14.0474795` at step `10^{-3}` to `-15.7316483` at step `0.2`, and at step `1` the response is `-0.811890576`.
- The value-only derivative has the opposite sign from `Dm`, so even a correct linear sign for the wrong intervention is the wrong sign. That pair is seed 16, in `06`. The value-only figure is still first-order. The error is the omitted path, not the remainder.

## Cancellation

`dH_2 = dH + dA + dM` at the level of the directional derivative. These three terms can have readout components of both signs. Their sum is `Dm`. If the sum is zero, the first-order response cancels. The seed-0 vector `u` is a case: whatever the three components are, they add to zero in the readout, and the finite response does not cancel.

There is no further identity, in this block, that predicts the finite remainder from a relation that does not already determine the second derivative of `m`. Computing that second derivative is a Hessian-vector calculation. It is available. It is not a consistency score.

## Large representation change, small response

The constant shift on a token the readout does not read is an arbitrary large change of `H` with response exactly zero. The representation moves by `a` in every coordinate of that token. The decision margin does not move. Distance in residual space is not a proxy for `|R|`.

A small `δ` can still have a large derivative if `g` is large. On seed 0, `||g||` is greater than `1`. A unit step along `g` has first-order size `||g||`. "Small perturbation, large response" is the size of `g`, not a separate mechanism.
