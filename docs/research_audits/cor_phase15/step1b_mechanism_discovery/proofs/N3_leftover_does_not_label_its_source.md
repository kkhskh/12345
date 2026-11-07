# Proof N3 — A leftover does not name its source

Classification: `PROVED` for linear leftovers and linear readouts.

**Statement.** Let `r(h) = P_v f(h) - ρ P_u h`. Fix a point `h` with a given leftover `r(h)`. There exist two linear readouts `L_+` and `L_-` such that the first-order responses to a kernel displacement of `P_u` that does not change `r` at first order (when `f` and `ρ` are linear and the displacement lies in `ker P_u`) have opposite signs, while `r` is unchanged.

**Simpler working form, used in the tests.** Two systems can share an additive residual of `0` at the unintervened point and have different four-term interactions (`I=0` versus `I=2`). So `I` is not a function of that residual. Two linear heads can share `Q` and have opposite `sign(ΔL)` (N1). So a leftover vector does not determine the sign of `ΔL`.

**What this does not prove.** It does not classify every nonlinear source of leftover (kink, omitted parent, affine offset) into a unique diagnostic. A complete taxonomy of leftover sources is `UNRESOLVED`. The proved fragment is already enough to forbid reading `r` as a candidate conflict.
