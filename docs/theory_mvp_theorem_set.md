# MVP Theorem Set

The minimum viable paper requires three mathematical results: the energy–distance equivalence,
the minimum-energy repair rule, and the cohomology correction. This document states each one
precisely, proves it, and records exactly what it does and does not license. Every claim here
is checked numerically by `run_08_theory_numerical_checks.py`.

## 0. Setup and notation

Fix an input `x` and a candidate contrast `c = (a, b)`. Let `K = (V, E)` be the computation
complex and let each vertex carry a finite-dimensional stalk `F(v) = R^{r_v}`. Write

- `C0 = prod_{v in V} F(v)`, with a positive-definite inner product `<s, s'>_M = s^T M s'`;
- `C1 = prod_{e in E} F(head(e))`, with edge weights collected in `W = diag(w_e I)`, `w_e > 0`;
- `d : C0 -> C1` the coboundary, `(d s)(e : u -> v) = rho_{u->v} s(u) - s(v)`;
- `L = d^T W d`, the weighted sheaf Laplacian;
- `H0 = ker d`, the space of compatible global sections.

Two candidate definitions of obstruction are in play:

    D(s) = min_{g in H0} || s - g ||_M^2          (distance to global compatibility)
    O(s) = || d s ||_W^2 = s^T L s                (computable Laplacian residual)

`D` is the definition we want conceptually; `O` is the quantity we can compute. Theorem 1 is
what makes using `O` in place of `D` legitimate.

## 1. Theorem 1: Energy–distance equivalence

**Statement.** Let `P` be the `M`-orthogonal projection onto `H0 = ker d`, and let
`V_perp = (ker d)^{perp_M}` be the `M`-orthogonal complement. Then there exist constants
`0 < lambda_min_plus <= lambda_max < infinity`, namely the extreme nonzero generalized
eigenvalues of the pencil `(L, M)`, such that for every `s in C0`:

    lambda_min_plus * || (I - P) s ||_M^2  <=  s^T L s  <=  lambda_max * || (I - P) s ||_M^2

Equivalently, `lambda_min_plus * D(s) <= O(s) <= lambda_max * D(s)`.

**Proof.**

*Step 1: `ker L = ker d`.* Since `W` is positive definite, `s^T L s = || d s ||_W^2 = 0` holds
if and only if `d s = 0`. As `L` is symmetric positive semidefinite, `s^T L s = 0` is equivalent
to `L s = 0`. Hence `ker L = ker d = H0`.

*Step 2: `D(s) = || (I - P) s ||_M^2`.* `H0` is a linear subspace of the finite-dimensional
inner-product space `(C0, M)`, so the minimizer of `|| s - g ||_M^2` over `g in H0` is the
`M`-orthogonal projection `P s`, and the minimum value is `|| s - P s ||_M^2`. This is the
standard projection theorem; finite dimension guarantees the minimum is attained.

*Step 3: the energy only sees the complement.* Write `s = P s + (I - P) s`. Because
`P s in ker d`, we have `d s = d (I - P) s`, and therefore

    O(s) = || d (I - P) s ||_W^2 = ((I - P) s)^T L ((I - P) s).

So both `O(s)` and `D(s)` depend on `s` only through `v = (I - P) s in V_perp`.

*Step 4: Rayleigh bounds on `V_perp`.* Consider the generalized Rayleigh quotient
`R(v) = (v^T L v) / (v^T M v)` for nonzero `v in V_perp`. The set
`{ v in V_perp : v^T M v = 1 }` is compact and `R` is continuous on it, so `R` attains a
minimum `lambda_min_plus` and a maximum `lambda_max`. Positivity of the minimum is the key
point: if `R(v) = 0` for some `v in V_perp` then `v in ker L = ker d = H0` by Step 1, so
`v in H0 intersect V_perp = {0}`, contradicting `v != 0`. Hence `lambda_min_plus > 0`. By
construction these are the extreme eigenvalues of `(L, M)` restricted to `V_perp`, i.e. the
extreme nonzero generalized eigenvalues of the pencil.

*Step 5: combine.* For `v = (I - P) s`, Step 4 gives
`lambda_min_plus * v^T M v <= v^T L v <= lambda_max * v^T M v`. Substituting Step 3 for the
middle term and Step 2 for the outer terms yields the claim. Both bounds are tight, attained at
the corresponding generalized eigenvectors. QED

**What this licenses.** `O(s) = 0` if and only if `D(s) = 0`, and the two quantities are
equivalent up to the condition number `kappa = lambda_max / lambda_min_plus`. Reporting the
computable Laplacian residual is therefore a faithful proxy for distance to global
compatibility.

**What this does not license.** The bound is a two-sided *scaling* equivalence, not equality.
`O` and `D` are not in general monotonically related across different complexes, different
transports, or different edge weights, because `lambda_min_plus` and `lambda_max` depend on
`rho`, `W`, and `M`. Comparing obstruction values across sheaf specifications requires either
normalizing by the spectrum or holding the specification fixed. When `kappa` is large the
proxy is loose, so `kappa` should be reported alongside any obstruction number.

## 2. Proposition 1: Minimum-energy repair under local response

**Statement.** Fix a site `u` with activation `h_u`, and consider an intervention
`h_u -> h_u + delta`. Assume on a neighbourhood of `h_u`:

1. `O_c` is locally quadratic: `O_c(h_u + delta) = O_c(h_u) + g^T delta + (1/2) delta^T H_O delta`,
   with `g = grad_{h_u} O_c` and `H_O` symmetric;
2. `m_c` is locally linear: `m_c(h_u + delta) = m_c(h_u) + J delta`, with `J = J^m_{c,u}`;
3. the off-target cost is `(mu / 2) delta^T F delta` with `F` symmetric positive semidefinite,
   plus a ridge `(lambda / 2) || delta ||^2`.

Let `A = H_O + mu F + lambda I`. If `A` is positive definite, the objective

    Phi(delta) = O_c(h_u + delta) - beta m_c(h_u + delta) + (mu/2) delta^T F delta + (lambda/2) ||delta||^2

has the unique global minimizer

    delta_star = - A^{-1} ( g - beta J^T ).

**Proof.** Substituting the local models and dropping terms independent of `delta`:

    Phi(delta) = const + (g - beta J^T)^T delta + (1/2) delta^T A delta.

This is a quadratic form. Its gradient is `grad Phi = (g - beta J^T) + A delta`, and its Hessian
is `A`. If `A > 0` then `Phi` is strictly convex and coercive, so the unique stationary point is
the unique global minimizer; setting `grad Phi = 0` gives `delta_star = -A^{-1}(g - beta J^T)`. QED

**The positive-definiteness condition is not automatic.** `F >= 0` and `lambda > 0` give
`mu F + lambda I > 0`, but `H_O` is the Hessian of the obstruction and need not be positive
semidefinite. `A > 0` holds if and only if

    lambda > max(0, - lambda_min(H_O + mu F)).

If this fails, `delta_star` is a saddle or a maximizer rather than a repair, and the formula must
not be used. In practice this means the ridge `lambda` has to be chosen against the measured
spectrum of `H_O`, not set to an arbitrary small constant. The numerical check exercises both the
well-conditioned and the indefinite regime and confirms the closed form matches direct numerical
minimization only in the former.

**Scope.** Assumptions 1 and 2 are local. Proposition 6 of the theory document bounds the
approximation error as `o(||delta||^2)` inside a neighbourhood where the response is twice
differentiable, so `delta_star` is meaningful only for interventions small enough to stay in one
piecewise-linear activation regime. This is why the toy-world and factual experiments sweep an
intervention radius rather than applying a single unbounded edit.

## 3. The cohomology correction

**The tempting but invalid construction.** One might try to define the obstruction of a local
evidence state `s` as the cohomology class `[d s] in H^1(K; F)`.

**Statement.** Let `s in C0` be a globally defined 0-cochain and let
`H^1 = ker d^1 / im d^0` with `d^0 = d`. Then `[d s] = 0` in `H^1`, for every `s`. The class
therefore carries no information about `s`.

**Proof.** By definition `d s in im d^0`. The cochain complex satisfies `d^1 d^0 = 0`, so
`d s in ker d^1` and the class `[d s]` is well defined. But `H^1` is the quotient of `ker d^1`
by exactly `im d^0`, and `d s` lies in the subgroup being quotiented out. Hence `[d s] = 0`. QED

**Consequence.** Obstruction cannot be a cohomology class of the coboundary of a global
0-cochain. It must be defined *metrically*, as distance to the space of compatible global
sections, which is the definition `D(s)` above. Theorem 1 then makes that metric definition
computable. This is the correction the theory document applies in its section 5.7, and it is the
reason the framework reports `O(s)` rather than a cohomological invariant.

**Where genuine cohomological obstruction does live.** The vanishing above depends on `s` being
defined on all of `K`. For a partial section `s_U` on a subcomplex `U subset K`, the obstruction
to *extending* `s_U` to a global section is a class in relative cohomology
`H^1(K, U; F)`, which is generally nonzero. The relative setting is the correct home for a
cohomological statement; the absolute setting is not.

**Numerically.** `im d^0` is a proper subspace of `C1` whenever `d` is not surjective, so the
statement `[d s] = 0` is a real constraint on coboundaries rather than a triviality about `C1`:
a random 1-cochain has a nonzero residual against `im d^0`, while `d s` has residual zero to
machine precision for every `s`. The check script verifies both halves.

## 4. What the MVP still owes

- Theorem 1 and Proposition 1 are proved here and verified numerically. The verification is a
  check of the algebra, not evidence that transformer obstruction is well conditioned; measured
  `kappa` on real models should be reported per experiment.
- Proposition 2 (obstruction–margin stability), Proposition 3 (non-identifiability), and
  Proposition 4 (local decision boundary) are stated in the theory document but are not part of
  the minimum viable theorem set and are not proved here.
- Proposition 7 (superposition non-uniqueness) is the theoretical counterpart of the Experiment
  0B finding that a probe-identified subspace can be destroyed without changing behaviour. That
  empirical result is documented in `docs/experiment_0b_modular_addition_toy_world.md`.
