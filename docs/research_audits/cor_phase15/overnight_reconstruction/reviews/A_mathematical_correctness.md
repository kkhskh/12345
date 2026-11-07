# Review A. Mathematical correctness

Same cloud environment as the proofs. Not an external referee.

**Objection.** The eigenvalue claim `||bracket||_2 = 1/s` drops the rank-one term. The triangle inequality gives a larger number, so the sharp claim needs the eigenspaces.

**Resolution.** On `span{1}` the bracket is zero. On `span{z}` it acts by `ε/s^3`. On the orthogonal complement of both it acts by `1/s`. The largest is `1/s`. SVD on a random 4-vector at `ε=10^{-5}` matches the closed-form list. The triangle inequality is an upper bound, not the norm.

**Objection.** Theorem F's restriction argument fails if the nonzero coordinates of `z` on `S` do not determine `λ`.

**Resolution.** If `z_S=0`, injectivity of restriction from `1^⊥` forces `z=0`, then `φ_S=0`, then `z'_S=0`, then `z'=0`. If some `z_i≠0`, every coordinate in `S` shares the ratio `s'/s`, including the zeros. Two zero gains are exactly the case where restriction stops being injective, and the algebraic pair is the counterexample.

**Objection.** The three-term attention estimate might undercount, so `Λ` might be false.

**Resolution.** The stress check on seeds 0–5 and scales up to 20 had `|Δm| / (Λ ||ΔH||)` at most about `7×10^{-7}`. The comparison with Cauchy–Schwarz does not need sharpness. Theorem L does not use the explicit constant at all: it is the definition of the derivative.

**Status.** The proved rows survive this pass. The explicit `Λ` remains a loose majorant.
