# 14. Identifiability and novelty

## What can be checked on a frozen block without fitting `R`

`ε > 0`; the pre-norm residual equations; `σ(W_Q), σ(W_K), σ(W_V), σ(W_O), σ(W1), σ(W2)`; `||W_O w||` and `||W2.T w||`; and `w·δ` for a chosen edit. All of these are Regime 1. None is fit to an evaluation response. Historical Direction 1 scores are not recomputed and not retuned.

Checking `||W_O w|| = 0` and `||W2.T w|| = 0` is an ordinary null-space calculation. If both vanish, Theorem Kernel-out predicts the raw response exactly. That prediction can be falsified by one intervention. It does not require a decision germ.

## Novelty against the sources named in the mandate

No theorem number is cited unless the overnight primary notes or the source's own abstract, as recorded there, states it. This note does not add new numbers.

| statement proved here | Geiger et al., JMLR 2025 | Sutter et al., NeurIPS 2025 | Pîslar et al., CLeaR 2025 | Olivieri and Pérez Rodríguez, arXiv:2605.25225 (preprint) | Vaswani 2017; Ba, Kiros, Hinton 2016; Xiong et al. 2020; Elhage et al. 2021 |
|---|---|---|---|---|---|
| Theorem U, raw readout, unbounded branch scale | A hard intervention can replace a mechanism by a constant; U is the opposite, a mechanism scaled without bound | Not their representation dilemma | Not a combine operation | Their sensitivity is first-order; U is finite and weight-free | Pre-norm (Xiong) lets a constant miss the branch while a centered edit hits it |
| Theorem L | Not their subject | Not their subject | Not their subject | A sensitivity field is a derivative; L is the mean-value inequality | Not their subject |
| Theorem S | Not an interchange abstraction | Not an alignment | Not a partition | A derivative bound would sit in Regime 2; S does not use `∇m` | Scaled dot-product attention, LayerNorm, pre-norm, and the OV path with a moving pattern. S is the Lipschitz composition of those maps |
| Theorem Kernel-out | The special case of a mechanism that ignores a direction, here at the output inner product rather than the input kernel | An unrestricted map can manufacture agreement; this restriction is the factor `W_O w = 0` | Not a combine operation | Not a field identity | With the pattern free, the output of attention is still `concat W_O`. The null condition is that factorization |
| R3 bound `|R_m - R_s| ≤ B_branch` | The numerical skeleton of an approximate implementation, not their Definition 33 | Not their Theorem 1 | Not their strength/accuracy tradeoff | Not a Green function | The residual split `x + F(LN(x))` |

Nothing in the right-hand columns is claimed as a theorem number invented for this note. The overnight primary notes record Geiger et al. Definitions 9, 11, 25, 33, 44, and 46, and Theorems 20 and 21 as modularity results inside that paper; Sutter et al. Theorem 1 as the input-restricted distributed abstraction under their Assumptions 1–5; Pîslar et al. Definitions 1 and 2 as the combine operation and its strength. This package uses those citations only to say that S, Align, and Kernel-out are not those theorems.

Elhage et al. 2021 identify the frozen-pattern OV path as linear. Theorem S does not freeze the pattern. The softmax Lipschitz constant is part of the estimate precisely because the pattern moves. A value-only derivative remains a different map, as in the overnight block.

## What is not novel

Theorem S is the mean value inequality applied to LayerNorm, softmax, the value map, and a tanh MLP, plus the triangle inequality. Theorem L is the same inequality read in the other direction. Theorem Align is Cauchy–Schwarz on `W_O w` and `W2.T w`. The path-sum identity is linearity of the derivative. R3 is an approximation lemma. Calling any of these a new causal mechanism would be a naming error.

## What remains after that reduction

A corrected scope for U and L. An explicit constant for this block, checked by a second forward. A scale threshold `α_*` separating the regime where the directional bound undercuts the Cauchy number from the regime where it does not. A proof that the examined joint upgrades do not beat the separate aligned bound. Those are boundary results. They are the content that survives after the reduction to existing calculus.
