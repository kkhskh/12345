# 03 — Obstruction equivalence and spectral analysis

Finite-dimensional real vector spaces throughout. Numerical checks: `tests/test_cor_phase1_math.py` and prior `run_08_theory_numerical_checks.py`.

## 1. Assumptions

Let `d: C^0 → C^1` be linear, `M` an inner-product matrix on `C^0`, `W` on `C^1`.

| Assumption | Used for |
|---|---|
| `M ≻ 0` | Existence and uniqueness of the `M`-orthogonal projection onto `ker d`; formula with `M^{-1}` |
| `W ≻ 0` | `O=0 ⇔ ds=0`; `ker(d^T W d)=ker d` |
| `M ⪰ 0` only | Minimizer of `D` may be a whole affine space |
| `W ⪰ 0` only | `O=0` can hide residuals in `ker W` |
| `d` rank-deficient | Pseudoinverse required; formula still valid on `im d` |
| Empty edge set | `d` has 0 rows, `O≡0`, `ker d=C^0`, `D≡0` |
| Disconnected graph | `ker d` is a product of component kernels; not a topological obstruction |
| Zero-dimensional stalks | Vacuous constraints |

Do not invert `d d^T` when it is singular. Use the Moore–Penrose inverse.

## 2. Exact global obstruction formula

**Claim.** If `M ≻ 0`, then the minimizer of `D(s)=min_{dg=0} ‖s-g‖_M^2` exists, is unique, and

```text
D(s) = (d s)^T (d M^{-1} d^T)^+ (d s).
```

Status: PROVEN UNDER STATED ASSUMPTIONS.

### Existence and uniqueness

`(C^0, ⟨·,·⟩_M)` is a finite-dimensional Hilbert space. `ker d` is a closed linear subspace. The projection theorem supplies a unique `g_*=P_M s ∈ ker d` with `s-g_* ⊥_M ker d`, and `D(s)=‖s-g_*‖_M^2`.

### Optimality equations

Lagrange function `L(g,λ)= (s-g)^T M (s-g) + 2 λ^T d g`. Stationarity:

```text
M(s-g) = d^T λ,    d g = 0.
```

Hence `s-g = M^{-1} d^T λ` and `d(s-g)=ds = (d M^{-1} d^T) λ`.

### Pseudoinverse

`ds ∈ im d` always. Also `im(d M^{-1} d^T)=im d` because `M^{-1}≻0`. Therefore `λ = (d M^{-1} d^T)^+ ds + ker(d M^{-1} d^T)` does not affect the quadratic form. Substituting:

```text
D(s) = (s-g)^T M (s-g) = λ^T d M^{-1} d^T λ = (ds)^T (d M^{-1} d^T)^+ (ds).
```

Gauge in `λ` along `ker(d M^{-1} d^T)` is irrelevant. Gauge in `g` along `ker d` is fixed by `M`-orthogonality when `M≻0`.

Verified: `test_weighted_projection_formula_matches_direct_minimization`, `test_singular_d_pseudoinverse_still_matches_projection`.

## 3. Local vs global obstruction

If `W ≻ 0`, then `O(s)=0 ⇔ ds=0`.
If `M ≻ 0`, then `D(s)=0 ⇔ ds=0`.

Hence, under those hypotheses,

```text
O(s)=0  ⇔  D(s)=0  ⇔  s ∈ ker d.
```

Status: PROVEN UNDER STATED ASSUMPTIONS.

**Critical question:** are `O` and `D` merely different quadratic weightings of the same residual `ds`?

Yes. Both are quadratic forms of `ds`:

```text
O(s) = (ds)^T W (ds)
D(s) = (ds)^T (d M^{-1} d^T)^+ (ds)
```

Choosing `W = (d M^{-1} d^T)^+` on `C^1` makes `O(s)=D(s)` identically. Verified: `test_choosing_W_star_makes_O_eq_D`.

The word “global” does not add a property beyond **distance to a linear subspace** (the homogeneous solution space of the linear constraints). It is not a cohomological invariant of a 0-cochain (see file 05).

They can still **rank** interventions differently, because different SPD quadratics on the same vector `ds` need not be monotone with respect to each other except through the spectral sandwich. MVP Theorem 1 already says this: comparing `O` across different `(ρ,W,M)` is illegitimate without holding the specification fixed or normalizing by the spectrum.

## 4. Spectral interpretation

MVP Theorem 1: on `(ker d)^{⊥_M}`,

```text
λ_min^+ D(s) ≤ O(s) ≤ λ_max D(s)
```

where `λ_min^+, λ_max` are the extreme positive generalized eigenvalues of `(L,M)` with `L=d^T W d`. Proof in `docs/theory_mvp_theorem_set.md` is valid under `M≻0`, `W≻0`. Numerical check previously reported median `κ=λ_max/λ_min^+ ≈ 378` and maximum `κ≈4.8e6` on random sheaves. A large `κ` means `O` can move a lot with little change in `D`.

The sandwich does **not** establish a causal relationship to logits. It is ordinary Rayleigh-quotient comparison of two quadratics with the same kernel.

## 5. Metric arbitrariness

Changing `M` or `W` can swap candidate rankings. Explicit example in `test_attack_c_metric_ranking_swap`: two disconnected identity edges; candidate 1 has residual on edge 1, candidate 2 on edge 2; swapping edge weights or node weights reverses both `O` and `D` order.

There is no independent principle in Definitions 3–4 that fixes `M` or `W`. Euclidean coordinates of whatever basis the probe/PCA produced are a convention, not an identification theorem. A metric chosen because it correlates with response is outcome leakage.

## 6. What Theorem 1 does and does not license

Licenses: using `O` as a **spectrally comparable proxy** for `D` for a **fixed** `(d,M,W)`.

Does not license:

- treating `O` as a distinct global causal quantity;
- comparing obstruction across sheaf specifications;
- inferring decisions, repair, or stability;
- ignoring `κ`.

The original PDF’s phrase “equivalent finite-dimensional residual energy” is stronger than the theorem, which is a sandwich, not equality. Status of that phrasing: CONDITIONAL RESULT, equality only after a special choice of `W`.
