# Workstream 5. Distinctive prediction

A coherent formula is not a prediction. The kernel equality is written out here so the rejection is about a specific object.

## Examined specification, rejected

- **Model class.** One pre-norm block, `T=3`, `d=4`, two heads, `tanh` MLP, `ε=1e-5`, causal mask, raw residual readout on the last token. Not a stack. Not a final LayerNorm.
- **Outcome.** The finite response `R` of `do(H_t <- H_t+δ)`.
- **Measured quantity.** Membership of the finite increments `ΔX` and `ΔU` in the kernels of `(W_Q, W_K, W_V)` and `W1`. Budget B1. Not `R`, not `Dm`.
- **Intervention family.** Those `δ` that pass the membership test, together with the constant line, which always passes.
- **Assumptions.** The block equations. Positive `ε`. The kernel products are computed exactly.
- **Predicted relation.** `R = 1_{t=r} w·δ`.
- **Falsification.** A `δ` that passes the products and whose recomputed `R` differs from the skip dot. The constructed example does not falsify it. A scale other than the designed one does not pass the products; that is a limit, not a falsification of the conditional statement.
- **Strongest competitor.** Direct evaluation, which returns `R` with no hypothesis. Next, `Dm·δ`, which on the example equals `0.10665...` while `R=0`. Cauchy–Schwarz returns `0.455...` against the exact `0`. The constant-shift identity does not apply to this non-constant `δ`. A Hessian bound spends second derivatives. Causal abstraction already counts an unread direction as a variable the mechanism does not use (Geiger et al., JMLR 2025, interchange and exact transformation). The value-only path is a different intervention.

The equality is exact on its hypothesis and strictly tighter than Cauchy–Schwarz on the example. It is rejected as a successor because the hypothesis is a kernel calculation on the structural weights, it is empty of non-constant edits whenever the `QKV` stack is injective, and the non-constant witness was obtained by building the weights around one `δ`. That is not an independently identified relation that constrains a fixed generic block beyond the shift identity.

No second specification survived. In particular, `O` and `D` remain quadratic forms of `ds`. They do not predict `sign(R)`. A Lipschitz constant predicts an unsigned magnitude and cannot beat `||g|| ||δ||`.

## What would have counted

A family of non-constant `δ` defined on an open set of branch weights, a quantity in budget B0 or B1, and a finite bound strictly smaller than both the shift evaluation's gap on centered edits and the Cauchy number, valid without setting the weights to satisfy the bound. Theorems U and L say the two obvious routes, weight-free bounds and Lipschitz certificates, do not do this. Theorem K does it only off that open set.

This file does not authorize an experiment. Phase 2 stays closed. The known comparisons are constraints, not fits: attribution repair AUC `0.979` against the best obstruction-using score `0.955`; obstruction-only Spearman with `Δm` about `0.095` against attribution about `0.905`. The pair `0.936` versus `0.141` is not used.
