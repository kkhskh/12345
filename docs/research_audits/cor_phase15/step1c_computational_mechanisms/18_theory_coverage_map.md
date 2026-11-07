# 18 — Theory coverage map

Coverage is the set of classes for which a theorem or a targeted counterexample was written. It is not a sample of random maps, and it is not a claim that other classes behave the same way.

| question | class examined | what was varied | result | label |
|---|---|---|---|---|
| Does nonlinearity imply nonzero `I` at a named pair? | polynomials, `C^2` rectangles, ReLU pieces | separability, pair, order of the first nonzero derivative | no | `FORMALLY RULED OUT` |
| When is `I` identically zero? | `C^2` on an open rectangle; polynomials | mixed partial | iff separable | `CONDITIONAL THEOREM` |
| Does `M` enter `O`? | the Phase 1 formulas | `M` alone, then `W` alone | no | `FORMALLY RULED OUT` that it does |
| Does a fixed readout survive LN? | population LN, with and without `\gamma` | `h`, `w`, `\gamma` | not in general | `VALID COUNTEREXAMPLE` |
| Does plain RMSNorm preserve linear signs? | the closed form, no block | `\gamma` absent or present | yes without `\gamma` | `CONDITIONAL THEOREM` |
| Does `ker P_u` preserve a linear residual? | linear `A`, `\rho` | which operator is tested | only `ker(P_v A-\rho P_u)` | `FORMALLY RULED OUT` for `P_u` alone |
| When does `P` determine `R`? | arbitrary `P`, plus one attention pair | fibers | iff fiber constancy | `CONDITIONAL THEOREM` |
| Can two states share `P` and differ in attention? | two-token causal QKV | the key coordinate | yes | `VALID COUNTEREXAMPLE` |
| Does a relation beat the tuple? | any scalar of the pair | a difference versus the second coordinate | no | `FORMALLY RULED OUT` |
| Value edits at fixed `A` | the attention formula | the clamp | exact linearity, `I=0` | `GENERIC EXISTING RESULT` |
| Residual edits of QKV | scalar causal head | `W_Q` zero or not; one numerical baseline | both zero and nonzero occur | `VALID COUNTEREXAMPLE` to universality; existence is not a trained-model fact |
| Recomputed residual as a predictor | any structural equation | the response | identically zero, hence blind | `FORMALLY RULED OUT` |
| Arbitrary `\rho` | one pair | whether `\rho` sees `z_v` | manufactures agreement | `VALID COUNTEREXAMPLE` |
| Relational tightening of a Jacobian bound | value-only and projected residuals | information budget | not obtained | `NOT INVESTIGATED` outside these families; inside them, no |
| Original signed bridge | unrestricted `O`, `D` | already varied in Phase 1 | false | `FORMALLY RULED OUT` |
| Multi-layer stacks, parallel blocks, post-norm, rotary bias, trained compensation | not examined | — | — | `NOT INVESTIGATED` or `ARCHITECTURAL RELEVANCE UNRESOLVED` |

## Equivalences inside the covered classes

Separability and identically zero interaction, for `C^2` rectangles. One pair with zero interaction is not separability. Clamped `A` and `Y=AVW_O` imply zero value-interaction. The converse is false. Exact sufficiency and fiber constancy, with the quantifiers "equal projections imply equal responses." Invariance of the linear residual and membership in the kernel of `P_v A-\rho P_u`.

## Deliberately not sampled

Random high-dimensional maps, random projections, and random interventions were not used as evidence. A random draw that failed to find a counterexample would not prove a universal claim. The universal claims that failed did so by one explicit map each. The universal claims that survived are proved for the class in the hypothesis column, not estimated from a sample.

## Trained Transformers

No class in the table is "weights of Othello-GPT" or "weights of any other frozen model." Architectural relevance of the pre-norm equations is the reason the equations were written. It is not evidence that a trained head sits at a particular operating point, that its `W_Q` is nonzero in a given subspace, or that a particular projection is the one the readout uses. Those are `ARCHITECTURAL RELEVANCE UNRESOLVED` as empirical claims and `NOT INVESTIGATED` as experiments.
