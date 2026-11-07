# 23 — Synthetic search

Harness: `tests/test_mechanisms.py`, library `proofs/mechanism_math.py`.

**Reproduced.** CE-01.

**New exact checks.** Affine offset, factorization gap, Blind kernel, joint sufficiency, family-relative sufficiency, path inner products, vector-versus-readout redundancy, cancellation, gating versus correction, majority vote, pairwise versus joint parents, truncation `I`, linear `I=0`, square `I=2`, residual-versus-`I`, LayerNorm, attention mix, residual block nonlinearity, vector residual sign, probe-preserving kernel, description versus mediation.

**Randomized.** Linear `I=0`, 50 draws, seed `20261107`. Blind kernel signs, 30 draws, seed `20261108`, both signs observed. Dimensions: 3-dimensional maps for `I`; 8-dimensional activations with rank-3 `P` for Blind.

**Minimization.** Failures reduce to N1, to `w·p_i`, or to the Hessian of the next map. No extra mechanism appeared in the draws.

**Not claimed.** The absence of a random counterexample to a restricted identity is not a theorem beyond that identity. No pretrained weights were loaded.
