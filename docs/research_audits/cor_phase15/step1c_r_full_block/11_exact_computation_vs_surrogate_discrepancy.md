# 11 — Exact equations and surrogate residuals

On every input the implementation uses, the block assigns

\[
H_1 = H + A(\operatorname{LN}_ε(H)), \qquad H_2 = H_1 + M(\operatorname{LN}_ε(H_1)).
\]

The residual of either equation is zero. Recomputing `A` from the stored normalized `X` reproduces the stored attention output to `10^{-12}`. A discrepancy defined as violation of these equations is identically zero on the intervention family of `05`. It cannot order the responses. Seed 0 has a finite response about `-0.629` in a direction whose first derivative vanishes, and the structural residual is zero there too.

A projected relation `P_v H_1 - ρ P_u H` can be nonzero while the exact equation holds, because `P` drops coordinates or because `ρ` is not `A ∘ LN` on the kept coordinates. That number is the error of the surrogate `(P, ρ)`. It is not, by itself, a conflict between pathways inside the block. The pathways inside the block satisfy their equations.

## Clamps

If `H` and `H_1` are both replaced by external values, the residual `H_1 - H - A(LN(H))` measures the inconsistency of the pair of clamps with the attention equation. The intervention that defines `R` does not do this. Using the clamp residual as a property of `R` moves to a different intervention. Step 1C already separated these rows. The block makes the separation concrete: the recomputed residual is a floating-point zero; a nonzero residual appears only after an overwrite.

## Fit

A map `ρ` solved so that `ρ P_u H = P_v H_1` at one state has residual zero at that state because it was solved. The block weights already determine `H_1` from `H`. The solved `ρ` is not those weights unless it was derived from them. Matching one state does not match the seed-16 pair: those two states share a projection that deletes feature 3 and do not share `R`. A transport fit to one of them is not a transport of the intervention response.

Where the causal information would have to enter is the identification of `ρ` with the actual branch, or a theorem that the surrogate error controls `R`. The first is the definition of the block, after which the residual is zero and the prediction is recomputation. The second is not proved for any proper projection in this directory. The feature-3 example is a counterexample to "the projected states agree, so the responses agree."
