# 19 — Exact computation, four cases

**Case A — Full-state structural equation.** `c=f(p)` holds as maps. Recomputed traces satisfy it. A residual of that equation is 0 on Domains A and B. Useful variation of decisions is not in that residual. `PROVED` (Theorem Z, scoped to a fixed `f`).

**Case B — Projected relation.** `P_v f` need not equal `ρ P_u`. Leftover is dropped coordinates, kinks, or affine-versus-Jacobian mismatch. Affine mismatch: `J h ≠ f(h)` even when `J = Df(h)` (`test_affine_offset_is_not_the_jacobian_times_the_state`). Nonzero projected residual does not prove a candidate conflict. `PROVED` as the leftover's algebra. `UNIDENTIFIED` as a separator from "real conflict."

**Case C — Joint imposed constraints.** Several candidate-specific pairwise equations can be mutually unsatisfiable while the true `f_v` holds. That incompatibility is a property of the imposed surrogates (wrong parent set, wrong `ρ`). It is scientifically meaningful only if those surrogates were independently the causal variables. Otherwise it is an inconsistent description. `PROVED` that pairwise addition constraints can contradict a true sum; `UNIDENTIFIED` as a network conflict.

**Case D — Stitched or jointly clamped states.** A stitch can violate `f`. A truncated `do` on several nodes is not a stitch: it is a well-defined SCM operation. Violation of the untruncated `f` is expected under truncation (the child is no longer `f(parent)`). That violation can coincide with a large `ΔL` or with zero `ΔL`, depending on the readout. The residual of the untruncated `f` on a truncated world is not a predictor of `R`; it is the definition of the clamp.

Do not say that every useful consistency measure vanishes under exact computation. Say that every residual of an exact full-state equation does, on recomputed traces of that equation. Do not say that a nonzero leftover is a conflict. Say that it is a failure of the chosen projected transport.
