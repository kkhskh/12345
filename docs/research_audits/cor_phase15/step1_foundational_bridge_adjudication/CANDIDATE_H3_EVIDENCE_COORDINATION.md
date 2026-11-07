# Candidate H3: candidate-specific evidence coordination

**Question.** Can the way the model aggregates independent candidate-related pathways imply a measurable relation between evidence coherence and candidate preference?

**1. Candidate evidence without final logits as a construction target.** Evidence sources are input slots whose tokens are chosen by task semantics (support `c`, neutral, oppose). The variable is the token, not a fitted feature. At canonical sites, a source's evidence for the pair is `v_j = ũ^T(V_j W_O)`. This is a frozen-weight projection, not a fit to `m`.

**2. Redundancy, opposition, suppression, context.** Each is a cell type in the factorial: agree-agree, agree-oppose, a neutral-dominated context. RB classifies the decision function over all of them at once, without a metric.

**3. Preference rather than path magnitude.** The decision variable is the margin order over cells, which is preference. NC gives ranking-level statements for norm-channel edits.

**4. Signed evidence versus nonnegative discrepancy.** Sources have signed levels. RB.3 shows that direction-dependence is a crossover. A nonnegative discrepancy (`|e_A − e_B|`, `O`, `D`) is symmetric under exchanging the sources and cannot by itself carry a direction (1A Theorem Sign).

**5. Coherent evidence for a wrong answer.** Unanimous wrong evidence: by CI.4 the head prefers the wrong candidate. Coherence is not correctness (CORR).

**6. Inconsistent evidence, correct answer.** Conflict where the stronger side is right. The head's sign follows the weighted consensus.

**7. Architectural aggregation rule.** Residual sum at cuts (SS): relation-blind for a linear readout. Softmax convex combination within heads (CI): relation-sensitive. LN normalization at the end (NC, LN): relation-sensitive through the norm. MLP: generic.

**8. Does the rule reduce to attribution?** The residual-sum part, yes: additive attribution is exact there. The softmax and LN parts, no: the sign of a source's effect depends on the other sources (D, LN).

**9. Invariance.** RB.2: invariant to the decision metric and to level relabeling. CI and NC scalars: invariant under function-preserving symmetries.

**10. Conditional prediction.** RB-forced held-out orders (RB.6) separate strictly monotone saturation of a summed evidence score, which is relation-blind, from relation-sensitive aggregation. But RB is invariant to relabeling levels, so a violation says nothing about *agreement* (review C, FATAL-1), and random tables violate it about 86% of the time. The earlier dilution-sign prediction is withdrawn (see H1). No consistency-specific prediction follows.

**Positive model.** Construction D (crossover, relation-sensitive), with designed weights. **Nearby negative model.** The same block with the value map replaced by a residual-sum write (no softmax normalization), plus a linear readout: relation-blind (SS). **Separating assumption.** Whether aggregation is normalized: a convex combination or a norm, as opposed to a sum. **Identifiable?** Yes, from the architecture and from `w`, `v`. **Plausible?** Softmax heads are ubiquitous. Whether *evidence* aggregation runs through one head is an empirical question. **Restates the outcome?** No. **Generalizes to?** Every softmax head (direct term); every LN-final model (norm channel). **Frozen Transformer?** Satisfies CI and NC exactly; A4 unknown. **Strongest alternative.** Weight-space sensitivity, which gives the same sign.

**Status.** RB: `PROVED` (known mathematics: conjoint measurement; robust ordinal regression for RB.6). It is retained as a **necessary condition**: if `m` is relation-blind, no relation between the sources can set the direction of either source's effect. As a consistency mechanism, H3 is a `REJECTED_CANDIDATE`. Its only architecture-native realizations are softmax reallocation (H1) and the final-norm channel, and neither carries agreement-specific information.
