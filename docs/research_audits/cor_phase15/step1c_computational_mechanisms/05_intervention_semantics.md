# 05 — Intervention semantics

The frozen computation is a structural system `F_θ`. An intervention replaces one or more equations and evaluates the rest. The symbol `do` is that replacement. It is not a gradient, and it is not a change of weights, unless the intervention is stated to be a weight intervention.

## Activation intervention

For a node `h_v` with parents `pa(v)`,

\[
\mathrm{do}(h_v \leftarrow h_v+\delta)
\]

replaces the equation `h_v=f_v(h_{pa(v)})` by the constant equation `h_v=h_v^{natural}+\delta`, then evaluates every descendant with the new value. Non-descendants stay at their natural values. In a causal decoder, tokens before the edited position are non-descendants of that position's residual.

The response of a margin `m_c` is

\[
R_c(x,v,\delta)=m_c(F_θ(x;\mathrm{do}(h_v\leftarrow h_v+\delta)))-m_c(F_θ(x)).
\]

`R_c` depends on `v`, on `\delta`, and on which descendants are recomputed. Two interventions with the same projected statistic can have different `R_c`. That is N1, restated for this definition of `R_c`.

## Six operations that are not interchangeable

| operation | what is replaced | what is recomputed |
|---|---|---|
| activation patch | one activation node is set to a stated value | its descendants |
| weight intervention | a parameter inside `f_v` | every node that reads that parameter |
| path-specific intervention | a named edge is set, other edges out of the same node keep their natural values | only descendants along the edited path |
| simultaneous hard interventions | several nodes are set to imposed values | every node that was not clamped and that descends from a clamped node. A clamped node is not recomputed from its parents |
| sequential interventions | one intervention is evaluated, then another is applied to the resulting state | depends on the order |
| counterfactual stitch | a node is set to the value it takes under a different input | descendants, as in an activation patch |

Simultaneous intervention on an ancestor and a descendant is well-defined. The imposed descendant value overrides the descendant's equation. The ancestor's change does not pass through that descendant: the intervention screens the path off. Reporting the output as the total effect of the ancestor would credit a path that is not in the intervened system. It does not double-count the path. The path is absent. Step 1B's two-node truncation calculator is this override. It is not ordinary recomputation of a Transformer.

The one-dimensional `tanh` witness later in the tests is a scalar branch. For attention, the residual is a sequence and the branch is not a function of one position.

## Three maps that a residual edit can mean

Let `branch` be the attention or MLP map, and let the residual update be `h+branch(h)`.

| name | value after the intervention |
|---|---|
| total effect | `(h+\delta)+branch(h+\delta)` |
| skip only, branch clamped | `(h+\delta)+branch(h)` |
| descendant clamped | a constant chosen by the analyst |

`test_three_intervention_semantics_differ` uses `branch=\tanh` and shows the three numbers differ. Theorems in this step name which row they use.

The value-only attention theorem uses a path-specific clamp of the pattern: `A` stays at its natural value, `V` is replaced. The QKV existence result uses the total effect on `X`, so `Q`, `K`, and `V` all move.

## What a theorem must state

Every general claim below is tied to one of these rows. A proof about clamped `A` is not a proof about `do(X\leftarrow X+\delta)`. A proof about recomputed children is not a proof about a child clamp. A proof about a linear margin is not a proof about a margin after LayerNorm unless LayerNorm is inside the function being differentiated.
