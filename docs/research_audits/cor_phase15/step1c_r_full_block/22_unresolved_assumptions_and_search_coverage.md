# 22 — What was not settled

## Architectures not differentiated

One pre-norm block was differentiated. Not differentiated: a stack, post-norm, a parallel residual, RMSNorm in place of LayerNorm, rotary positions, multi-query attention, and Q/K/V biases inside the numerical Jacobian. The shift identity survives an additive bias after `XW`, because that bias does not change the fact that a constant shift leaves `X` fixed. The numerical JVPs do not include those biases. A post-norm shift is a different calculation and was not claimed.

## Families not searched

The opposite-sign pairs are existence proofs on seeds 0, 1, 4, and 16. They are not a measure of how often a value-only sign fails. No sweep over widths, depths, or initializations was used to estimate a rate. A rate would not change the existence statements, and it would not create a mechanism.

Projections were not searched. Feature 3 was chosen because two states differ there and the responses differ. The result is "this projection fails," not "every projection fails." The constant-shift family is the case where every projection succeeds, for the vacuous reason in R15.

## Scores not ruled out

R21 does not say that every function of the weights and the state fails to predict `sign(R)`. Recomputing `m` predicts it. The constant-shift evaluation predicts it on that family. `Dm` predicts the linear sign and, by R13, does not predict every finite sign. The claim inside the examined list is narrower. A proper projection need not determine `sign(R)`: feature 3 is one failure, and the constant-shift family is a success for every projection. The structural residual of the true equations does not determine `sign(m)` or `sign(R)`. A one-pair transport residual can move while `m` does not. A path sum that is the chain rule determines the linear sign and is the derivative. A different relation, fixed without `R` and not equal to those, was not exhibited and was not exhaustively excluded.

## Smoothness

`tanh` is smooth. A ReLU kink is not analyzed on this block. Step 1C's statement about a single linear piece remains in Step 1C. It is not re-proved here.

## What a later argument would have to add

A non-constant family of `δ`, a quantity computed without evaluating `R` or `Dm`, and a theorem in which that quantity bounds `R` on that family more tightly than the constant-shift identity and the Cauchy bound. The quantity's inputs have to be listed the way `13` lists them. If the list contains the branch weights and the state, the comparison is with the chain rule. If it does not, seed 16 is the shape of the counterexample to try first.

That argument is not supplied here. It is not a reason to open a reconstruction, a Step 2 protocol, or Phase 2.
