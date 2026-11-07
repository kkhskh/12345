# 13 — Where the directional information enters

The sign of the first-order response in a direction `δ` is the sign of `Dm(H)·E_{t,δ}`, when that number is nonzero. The real values of `Dm·E_{t,δ}` on a basis determine the gradient at token `t`. The signs alone do not. Signs on a basis fix an orthant, and more than one direction shares an orthant. A rule that returns only a sign, and that is unchanged when `δ` is replaced by `-δ`, returns the same answer for two directions whose derivatives are opposite. Whenever `Dm·δ≠0`, that rule cannot be correct for both. The value-only derivative is a different linear form. On seed 16 it has the opposite sign from `Dm` for one explicit `δ`. A notation that keeps only that piece is an omission, not a rewriting.

## Budgets

| predictor | weights required | natural `H` | the intervention `δ` | `R` on the evaluation point |
|---|---|---|---|---|
| recompute `R` | all | yes | yes | the predictor is `R` |
| `Dm·E` | all, through the chain rule | yes | yes | not the finite `R`; the linear prediction |
| value-only derivative | all except that patterns are clamped, so `W_Q` and `W_K` are used only to build the frozen `P` | yes | yes | not `R`, and not always the sign of `Dm` |
| constant-shift response | none, if the readout token is known; the sum `1^T w` if the edited token is the readout token | not the values, only the index | the constant `a` | the exact `R` for this family only |
| a projection `P(H)` | none inside `P` | the kept coordinates | yes | feature 3 does not determine the seed-16 sign; not every `P` fails |
| structural residual of the block | none beyond the check | the activations | — | identically zero, no constraint |

The constant-shift identity is stronger than a fact about pure constants. For an arbitrary finite edit `E` and a constant `C` on one token, `R(H,E+C)=R(H,E)+κ`, with `κ` equal to `a 1^T w` when that token is read and `0` otherwise. The proof is R6 at the base point `H+E`, written out in `25`. The interaction of a constant with every other edit is exactly zero. A centered direction does not inherit this. It does not predict the seed-16 sign, and it does not predict the remainder along `u`. Those still require either an evaluation of `m` or the derivatives. Recomputation determines the signs and is not a reduction. The derivatives determine the linear signs and are not this identity.

A relational residual of projected activations is not in the table as a successful predictor. On the recomputed family it is zero when the relation is the true equation. The feature-3 projection fails the fiber test in `10` for one non-constant intervention. It is not the case that every proper projection fails: the constant-shift family makes every projection sufficient, and a projection that keeps the coordinate `H_{0,3}` still separates the seed-16 pair. Either way, none of these residuals improves on `Dm` for a general direction.

## Hessian

Predicting the quadratic term along `u` requires second derivatives, or enough finite evaluations to estimate them. That is a larger budget than `Dm`, not a smaller one. A consistency score that reproduced the Hessian-vector product would have consumed the Hessian. It would not be a reduction.

## Circularity

Assuming "the margin moves with the proposed score" puts the sign in the hypothesis. None of the identities above uses that assumption. The identities that determine a sign are recomputation, the chain rule, and the shift evaluation of `1^T w`. Each of those is the response, a derivative of the response, or an exact evaluation of the response on a symmetry. They are not independent of `m`. There is no demonstrated route to the sign of a general `R` that avoids `m`'s derivatives or `m`'s values.
