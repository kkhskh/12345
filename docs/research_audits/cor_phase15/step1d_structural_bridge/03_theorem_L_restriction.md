# 3. Theorem L, restricted

## Statement

Let `m: R^n → R` be differentiable at `x`, and suppose `|m(x) - m(y)| ≤ Λ ||x - y||_2` for every `y` in a convex neighborhood of `x`. Then `Λ ≥ ||∇m(x)||_2`.

Proof: for a unit vector `u` and a small scalar `t`, the difference quotient is at most `Λ`. The limit is `|∇m(x) · u|`. Taking the supremum over `u` yields the gradient norm.

## What the inequality quantifies

`Λ` is a single number that has to dominate every direction. The lower bound `Λ ≥ ||∇m||` is the cost of that uniformity. A certificate `B(x, δ)` that is allowed to depend on `δ` is a different object. Theorem L gives no lower bound of the form `B(x, δ) ≥ ||∇m|| ||δ||`.

Directions in which a structural decomposition is available include: orthogonality of `δ` to a linear skip, a factor whose Lipschitz constant is small, an output of a branch lying in `ker(w)`, and a contractive composition. None of these produces an isotropic `Λ` below `||∇m||`. All of them can produce a number `B(x, δ)` below `||∇m|| ||δ||` for some `δ`. The seed-0 calculation in `05` is an instance, not an exception to Theorem L.

## First-order Cauchy–Schwarz is not a finite bound

`|∇m · δ| ≤ ||∇m|| ||δ||` bounds the derivative. On the unit orthogonal edit at output scale `10^{-6}`,

```text
|∇m · δ| = 5.192333216467657e-08,
|R|      = 1.326797428591320e-07.
```

The finite response is larger than the absolute linear term. Using `||∇m|| ||δ|| = 0.8764350281115764` as though it were a finite-response competitor confuses the isotropic product with a certificate. That product is an upper bound on the linear term only.

## Where a directional bound does beat a valid isotropic bound

Let `Λ_iso = ||w|| (1 + L_A L_1)(1 + L_M L_2)` be the isotropic constant proved in `06`, and let `B_dir` be the directional certificate there. For every `δ`,

```text
Λ_iso ||δ|| - B_dir(δ) = ||w|| ||δ|| - |1_{t=r} w·δ| ≥ 0,
```

with equality if and only if the skip inner product saturates Cauchy–Schwarz, or the token is not the readout and one is comparing a pure branch bound with an isotropic bound that still pays for a skip at a different token. When the edit is at the readout and `δ` is not parallel to `w`, `B_dir < Λ_iso ||δ||`. The gap is exactly the slack in Cauchy–Schwarz on the skip. It does not depend on the branch weights being small. At unit scale the two bounds are `1629250.228383555` and `1629251.104818525`. The directional bound wins by `||w|| = 0.8764349701487028` and both numbers are far above `|R| = 0.010805881111744242`. The win is real and uninformative at that scale.

Beating `Λ_iso ||δ||` is the comparison Theorem L actually leaves open. Beating `||∇m|| ||δ||` is a different comparison and holds only when the branch rate is small enough; see `08`.
