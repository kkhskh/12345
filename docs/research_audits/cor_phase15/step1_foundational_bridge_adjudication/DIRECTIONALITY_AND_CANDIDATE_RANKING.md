# Directionality and candidate ranking

## Where the sign comes from (mandate §8)

| Result | Sign of the response is | Information source | Consistency's share |
|---|---|---|---|
| CI.3 (added source from zero weight; not realizable by substitution) | `sign(v'_j − M)` | DLA of the source against DLA of the head | the relation is readout-defined |
| CI.3* (layer-0 substitution) | `sign(w_j(v'_j−v_j) + (w'_j−w_j)(v'_j−M))` | own change, plus weight change × DLA deviation | only through the weight change |
| NC.2 (null edit) | `−sign(a)·sign(2h̃^T v + ‖v‖²)` | `sign(a)` from the readout and state; the change in the residual norm | none: the norm of a sum is not a relation (SS) |
| RB.3 | under relation-blindness, `sign(f(a)−f(a'))`, the source alone | — | none: relation-blindness means consistency cannot set direction |
| `O`, `D` (Phase 1, 1A) | not determined (`D(g+r) = D(g−r)`) | — | — |
| Gradient baseline | `sign(∇m · δ)` | downstream sensitivity | — |

For CI, a gradient in the attention-weight coordinate gives the same sign, `∂M/∂w_j = (v_j − M)/Z`. So CI's sign is not *more* information than weight-space sensitivity. It is the same information, identified as a relation. Per mandate §8, this is disclosed: the directional information is downstream-free here, because it lives inside the head, and it coincides with a weight-space derivative.

A squared or nonnegative discrepancy cannot supply a sign (1A Theorem Sign; RB.3). Every signed result above uses a signed relation: a deviation from consensus, or the sign of a norm change times `sign(a)`.

## Candidate ranking (mandate §9)

**Comparability.** Logits share one scale up to a candidate-independent shift, and pairwise margins remove the shift. Candidate identity is the vocabulary index. No per-candidate projector, rank, metric, or normalization is used. The Phase 1 ranking problems (Attack C: metric swap; projector rank; transport complexity) therefore have no object to act on.

**What is proved about ranking.**

1. **NC.3–5, a ranking theorem (norm channel, not consistency).** On decision-null edits, all `|C|` logits (relative to a reference) are affine in one common `τ`. The reachable rankings are the orders of a family of lines over `τ ∈ (0, 1/σ_min]`. Each pair crosses once in `σ`. It flips at most once along σ-monotone paths, and only if `aκ < 0` and `σ* > σ_min`. Increasing norm moves every pair toward the `κ` order. The theorem is restricted to `|C| < d` and to order within `C`. Under RMSNorm (`κ ≡ 0`) no pair can flip.
2. **CI.4, a unanimity theorem (direct term).** If all sources attended by a head prefer `c` over `c'` (`v_j > 0`), the head's direct contribution prefers `c`. If they conflict, the sign depends on weights.
3. **RB on margins.** A relation-blind pair margin admits `f + g` scores, so additive evidence scores rank the cells. Under relation-sensitivity, no such scores exist (RB.1).

**What is not proved.** Full-vocabulary ranking from any consistency relation. Ranking under general (non-null) edits through deep paths. These are weaker than the original decision-selection ambition, and are classified that way: a restricted ranking theorem, not a candidate-selection theory.

**Metric and representation reversals.** RB.2 rules out reversal by the choice of decision metric. NC and CI scalars are invariant under the architecture's function-preserving symmetries. The remaining reversals come from substantive changes, such as a different candidate set `C`, which changes `N_C`, or different slots. These are design choices, frozen in advance.
