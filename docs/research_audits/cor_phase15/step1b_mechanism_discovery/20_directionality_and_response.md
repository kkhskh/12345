# 20 — Directionality

A nonnegative scalar discards sign (Theorem Sign; `test_vector_residual_keeps_the_sign_a_scalar_discards`). Extensions:

| Proposed richer object | What it keeps | What it still needs to sign `R` |
|---|---|---|
| Vector residual `ds` | A direction in cochain space | A map to `L`; without it, CE-01 still flips the head |
| Operator / Jacobian | `∇L` or `J` | That is the baseline |
| Subspace | Which directions are live | Sufficiency, not a residual |
| Interaction operator | Mixed derivatives | Curvature of `L`, not `D` |
| Conditional sign given context | A context variable that correlates with `∇L` | Identification of that variable without `L` |

A constraint on the set of possible responses, without selecting one, is possible: `|ΔL| ≤ K ||δ||`. That is generic. A nontrivial tightening needs a checkable subset of perturbations with a strictly smaller `K`, identified without `∇L`. Not obtained from `O`.

If signed prediction requires `∇L`, the remaining consistency term's first-order contribution orthogonal to `∇L` is idle. Step 1A Theorem Descent, as corrected, still says that.

**Status.** Vector-valued residuals do not solve directionality. Operators that do are sensitivity operators.
