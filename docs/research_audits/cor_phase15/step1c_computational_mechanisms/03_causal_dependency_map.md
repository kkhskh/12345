# 03 — Causal dependency map

The map runs from the block's equations to a response. Each arrow is one of: `STRUCTURAL EQUATION`, `PROVED IMPLICATION`, `CONDITIONAL IMPLICATION`, `STATISTICAL HYPOTHESIS`, `MEASUREMENT RELATION`, `UNESTABLISHED CONNECTION`.

An unresolved arrow is written as unresolved. There is no arrow from a consistency score to decision quality.

## From the residual state to the block output

These are the equations of a pre-norm block, derived in `04`. They hold by the definition of that block, not by a fit.

| from | to | kind |
|---|---|---|
| residual `X` | `LN(X)` | `STRUCTURAL EQUATION` |
| `LN(X)` | `Q,K,V` | `STRUCTURAL EQUATION` (three learned linear maps) |
| `Q,K` and the mask | scores, then the row-stochastic pattern `A` | `STRUCTURAL EQUATION` |
| `A` and `V` | head output `A V W_O` | `STRUCTURAL EQUATION` |
| head output | residual, by addition | `STRUCTURAL EQUATION` |
| updated residual | next `LN`, then MLP | `STRUCTURAL EQUATION` |
| final residual | `LN`, then unembedding | `STRUCTURAL EQUATION` of a standard decoder |

`A` clamped and `Y=AVW_O`: the output is linear in the value edit. `PROVED IMPLICATION`. The clamp is part of the hypothesis. Zero interaction of an arbitrary map of `V` does not imply this formula.

`W_Q=0` on the bias-free map `Q=XW_Q`, with no LayerNorm in front of `X`: residual edits of that map are additive. `PROVED IMPLICATION` for that map only. It is not the pre-norm block. A nonzero query bias leaves the scores dependent on `K`. A value path through per-token LayerNorm is additive across tokens when the pattern is constant, and need not be additive across coordinates inside one token.

## From states to researcher measurements

| from | to | kind |
|---|---|---|
| residual `h` | `z=P(h)` | `MEASUREMENT RELATION`. `P` is chosen by the analyst unless a separate argument identifies it |
| `z_u,z_v` | `ds=\rho z_u-z_v` | `MEASUREMENT RELATION`. `ρ` is part of the measurement |
| `ds`, `W` | `O` | `MEASUREMENT RELATION`. Does not read `M` |
| `s`, `d`, `M` | `D` | `MEASUREMENT RELATION` |
| full residual and the block | response `R` | `STRUCTURAL EQUATION` followed by a readout. This is recomputation |

## From measurements to responses

| claim | kind |
|---|---|
| equal projections imply equal responses, for every pair of states in the named set and every intervention in the named family | `CONDITIONAL IMPLICATION` in both directions. This is exact sufficiency (`proofs/fiber_and_trace.md`). The converse quantifier, equal responses imply equal projections, is a different statement and is not sufficiency |
| a coordinate projection that drops a key feature determines the attention output | `UNESTABLISHED CONNECTION`, and false for the two-token counterexample |
| a quantity that was defined to be a function of `(z_u,z_v)` | definitional identity. Substitution adds no constraint on a response that was not already a function of the tuple |
| a scalar `ψ(z_u,z_v)` determines every response the tuple determines | `UNESTABLISHED CONNECTION`, and false for `ψ=z_v-z_u` on the pairs `(0,0)` and `(1,1)` |
| `Q` constant on `\{\Delta,-\Delta\}` determines `sign(L(\Delta))` for every such linear `L` | `UNESTABLISHED CONNECTION`, and false (N1) |
| `\delta\in\ker P_u` leaves `r=P_v A h-\rho P_u h` invariant | `UNESTABLISHED CONNECTION`, and false unless `δ` is also in the kernel of the full measurement |
| recomputed structural residual predicts the response | `UNESTABLISHED CONNECTION`, and impossible on that family, because the residual is identically zero while the response need not be constant |
| nonlinearity of the next map implies nonzero four-term interaction at the intervention of interest | `UNESTABLISHED CONNECTION`, and false (`u^2+v^2`, and `(u+v)^3` at `(1,-1)`) |
| `C^2` and mixed partial identically zero on a rectangle imply identically zero interaction | `PROVED IMPLICATION` |
| `O` or `D` determines the sign of a decision margin | `UNESTABLISHED CONNECTION`. The universal form is disproved in Phase 1 |
| changing `M` changes `O` at fixed `d,W,s` | false. Not an arrow |
| small Euclidean residual distance implies a small logit change | `UNESTABLISHED CONNECTION`. LayerNorm does not preserve Euclidean norms or fixed-readout signs, and the logit need not factor through the residual that was measured |
| value-only response equals the Jacobian-vector product | `PROVED IMPLICATION` when `A` is clamped, because the map is exactly linear. This is the Jacobian, not a second mechanism |
| a consistency score tightens that identity | `UNESTABLISHED CONNECTION`. The identity does not take the score as an input |

## Where a consistency-like quantity could sit

The only place a residual of a structural equation appears in the block itself is as a check that recomputation was carried out. On recomputed traces it is zero. A projected residual `P_v h_v-\rho P_u h_u` can be nonzero while the full equation holds, because `P` discards coordinates or because `\rho` is the wrong transport. That number is a measurement of the pair `(P,\rho,h)`, not a variable the block multiplies.

No `PROVED IMPLICATION` runs from that number to `sign(R)` in the families examined here. The arrow is an `UNESTABLISHED CONNECTION`.

## Interventions are not extra arrows in the forward map

`do(h_v \leftarrow h_v+\delta)` replaces the structural equation at `v` and recomputes descendants. `do` on an ancestor and a descendant at once overrides the descendant. Those are semantics for reading the map, recorded in `05`. They are not new computational nodes.

## Statistical hypotheses

No arrow in this step is a `STATISTICAL HYPOTHESIS`. No sample of model activations was used. A later empirical claim that a particular projection is approximately sufficient on a particular intervention family would be a statistical hypothesis and would need its own design. It is not licensed by this map.
