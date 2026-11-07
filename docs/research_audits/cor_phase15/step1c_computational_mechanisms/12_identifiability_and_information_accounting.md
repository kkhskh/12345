# 12 — Identifiability and information accounting

## What each candidate consumes

| candidate | architecture | weights | natural activations | labels | logits | gradients | new interventions |
|---|---|---|---|---|---|---|---|
| recomputation of `R` | yes | yes | the intervened node | no | only if the margin is defined from them | no | the evaluation intervention itself |
| fiber check for a proposed `P` | yes, to know the family | yes if `R` is computed from the block | pairs with the same `P` | no | the margin on those pairs | no | enough interventions to see a fiber |
| N1 obstruction to sign | the statistic `Q` and the response form | no, beyond what `Q` and `L` contain | one `h` | no | no | no | a kernel displacement and its opposite |
| clamped-pattern linearity | the attention equation and the clamp | `A` and `W_O` if the number, not only the zero interaction, is required | the pattern at the natural state | no | the readout `w` if the margin is required | the Jacobian is the closed form | value edits, or none if only `I=0` is claimed |
| four-term interaction | the map being probed | inside the map | the baseline | no | if the map is the margin | no | the four states |
| `O` or `D` | a chosen graph | no | the projected cochain | no | no | no | none |
| a projected residual | chosen `P` and `\rho` | no, unless `\rho` is a Jacobian of the block | the two ends | no | no | only if `\rho` is a Jacobian | none on the natural trace |

Labels do not enter the mathematical claims of this step. They would enter a correctness claim. No correctness claim is made.

## Assumption classes

**Architectural.** Pre-norm residual, `Q,K,V` linear in the normalized residual, row-wise softmax, causal mask, MLP of the stated shape, population LayerNorm with optional `\gamma,\beta`, logits from a final LN and a matrix. Used as the definitions in `04`. Not estimated.

**Representation.** Which coordinates a structural map reads is an architectural fact about that map. Fiber constancy is a property of the response `R`. They are the same fact only when `R` is a function of exactly those coordinates. Calling a subspace "the coordinates the family reads" does not establish that.

**Identification.** `ρ` and `W` and `M` are fixed without using `R`. The rank-one `\rho=z_v z_u^\top/(z_u^\top z_u)` satisfies `\rho z_u=z_v` whenever `z_u\neq 0`. Its residual is also zero on every pair `(t z_u, t z_v)` and every pair `(z_u+u_\perp, z_v)` with `u_\perp` orthogonal to `z_u`. It is not zero only on the single fitting pair, and it is not evidence that `\rho` is the block. A `\rho` set equal to a Jacobian uses the Jacobian.

**Causal.** The intervention row in `05` is the one named by the theorem. A clamp of `A` is not a residual edit. A clamp of a descendant is not recomputation.

**Outcome.** A sentence that puts `sign(R)`, or a decrease in loss, on the right-hand side of an implication whose left-hand side is `O` or `D`. Every such sentence in the original theory remains an outcome assumption. None has been derived from the architectural assumptions alone.

## Where the information enters

If the derivation assumes that the margin is a monotone function of the residual, the sign enters through that assumption. The derivation is then a restatement. This is the status of the original signed bridge, unchanged.

If the derivation evaluates `F_θ` on the intervened state, the information enters through the computation. The prediction is recomputation. A relational summary is not required.

If the derivation uses `\nabla m`, the information enters through the full first-order map, or through the partial map consistent with the clamps. The value-only formula is this case. The relational summary does not appear.

If the derivation uses only `P(h)` and a `\rho` that was not fit to `R`, then by the fiber theorem it predicts `R` only when `R` is constant on those fibers. In the attention counterexample it is not. The information in the dropped key coordinate enters the block and does not enter `P`.

## Baselines

The budgets are not the same. A comparison that ignores the budget is not a comparison.

| baseline | information it spends | what it can say | what the relational residual adds in the cases checked |
|---|---|---|---|
| 1. Constant | nothing about `h` | a fixed number | a varying residual is a different function. It is not automatically a better predictor of `R` |
| 2. Geometry of activations | `h` or `P(h)`, a norm or a cosine | descriptive distances | `O` is one such distance when `W` is a metric. Error 2: it is not `D` |
| 3. Margin and confidence | the logits | the observed decision, not the effect of an intervention that has not been run | nothing about `R` unless the logits determine the intervention, which they do not in N1 |
| 4. First-order sensitivity | Jacobian-vector product | the linear response, and an exact response when the map is linear | no tighter value |
| 5. Second-order sensitivity | Hessian or the four states | the local or exact interaction | no identification of the source of a leftover |
| 6. Path-specific analysis | the clamps that define the path | the path effect under those clamps | no further constraint |
| 7. Causal abstraction | an alignment plus the interchange family | whatever the alignment preserves, and nothing it does not | an arbitrary `\rho` is an alignment that has not been shown to preserve `R` |
| 8. The relational residual | `P`, `\rho`, `W` or `M`, and the cochain | `O` or `D` or `ds` | this is the candidate. It does not dominate 4–7 on the families where 4–7 apply, and it does not determine sign where N1 applies |

A relational theory would be informative if it constrained `R` from the architecture and from natural activations, without spending the evaluation intervention and without assuming the sign. The clamped-pattern law "interaction is zero" meets the budget and is the attention equation. It does not use a residual. No residual examined here meets the budget.

## Roles

| quantity | role it actually has here | role it does not have |
|---|---|---|
| `O`, `D`, `ds` | descriptive statistics of a chosen cochain and a chosen quadratic | a causal effect, a mediator, a sufficient representation for `R` |
| `Q(h)` in N1 | a measurement that defines fibers | a variable the block is shown to compute as its decision rule |
| `A V W_O` | a computationally implemented map | a consistency score |
| fiber constancy | the definition of exact sufficiency | an empirical finding about a trained probe |
| Jacobian bound | a mathematical bound | a mechanism distinctive to COR |
| zero structural residual on recomputed traces | a check that the equation was applied | a predictor of `R` |

Reducing `O` is not an intervention on the model unless the intervention is defined on activations or weights and the change in `O` is a consequence. The audits do not contain such an intervention that is identified independently of the margin.
