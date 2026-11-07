# Main research result (corrected after the seven-role review)

## The integrated result (§14)

**In an engineered pre-norm Transformer (C′), the use of an independently defined source relation in a candidate decision can be established through a complete, noncircular chain.** The chain has three parts:

1. Corrected, certified operating conditions: C′1*.
2. An on-manifold matched contrast showing the gate pathway is necessary: X4.
3. A mediator-level sign test showing the gate feature is a candidate-symmetric relation rather than an additive gate: T-MS, S1 and S2.

**The key limit is that behaviour at the decision level cannot establish relation use.**

- For fixed-prior deferral rules, an additive-gate twin and a fully relation-free twin reproduce C′'s decisions on [−3, 3]² (T-RF).
- Both twins also pass X4.
- Graded outputs *can* certify that some cross-source interaction exists (T-RF′: C′ shows 357/435 order reversals, both twins 0). They cannot certify that the interaction is agreement.

**In small learned models, a cross-source interaction appears where the objective forces one.**

- It appears in T1, which is XNOR. It also appears in T4, where it is not forced, since the T4 rule is additively representable.
- It is absent when the objective does not need it: T2 and T5a, on the training range.
- T3 and T5b are solved by exact relation-free threshold logic.
- **The implementation of the learned interaction is not identified.** The preregistered unit criterion was non-specific, and final-site interchange measures readout alignment.

## Arrow-by-arrow record (engineered C′)

| arrow | proof | evidence | assumptions | limits | not established |
|---|---|---|---|---|---|
| 1. Source semantics fixed | generator | §4 catalogue | the generator is the ground truth | — | sources in any pretrained model |
| 2. Representations identified | closed form: R carries `(x, y)` on e1, e2 | closed form vs forward pass to 1e−14 | engineered basis | — | — |
| 3. Relation feature characterised | A1*: `sign q_R = sign(ab)` | 0/120,000; derived cones | (N1)–(N4) | (A0) false; LN couples sources (N2) | — |
| 4. Downstream operation identified | query gain `G = λq̂/√2` | C2: `q_R` alone sufficient (1.000) in 3 families; values alone sufficient for a-flips (0.91) | single gate head | shares not additive; redundant copies break single-site necessity | unique mediator |
| 5a. The pathway is necessary (on-manifold) | — | X4: equal relation-blind margin, opposite decisions | reachability (numeric for C′) | strong agreement has no partner; **also passed by additive twins** | — |
| 5b. The mediator is a relation | T-MS | S1: sign pattern 500/500 (twin 0/500). S2: (−,−) interchange routes C′ to sources, the twin to the prior (1.000 each; off-manifold) | — | S2 is off-manifold | an on-manifold version of S2 |
| 6. Candidate-specific response | C′1* | certified rectangles on [τ, 3]; witness that the decision is not a function of G | positive parameters; prior votes c′ | dead bands come from the design, not from continuity (T-CO corrected) | — |
| 7. Controls and baselines | T-SS, T-B-TM (no final LN), T-C4 | sum-only reader Δm = 0; attacks map out failures | C4 class | twins reproduce decisions; B-MLP and B-signs tie | **any predictive advantage** |

## Positive claims that survive review

1. **C′1*, A1* and B-TM (without final LN).** Decision-level theorems with signed, inspectable conditions. C′ has certified operating rectangles and witnessed dependence on both G and value mass.
2. **The engineered relation-identification chain**, X4 + T-MS (S1/S2) + C2. It separates a relation gate from an additive gate that has identical decisions, an identical relation-blind pathway and identical X4 and C2 results.
3. **T-RF′.** The graded single-cancellation test is a behavioural certificate that a cross-source interaction exists. It is applied here to engineered circuits and learned models.
4. **The learned null.** No relation emerges under integration objectives (T2, T5a). This is robust across seeds and transfer variants, within the training range.
5. **B's confidence modulation** (R7). At equal uniform-routing margin, agreement roughly doubles the margin (+5.19 vs +2.40). B-TM predicts its direction from frozen weights (3,000/3,000), within its class. It is cancelled by a competing writer.
6. **The preregistered held-out sign prediction**: T1/TF1, 33/35. It is limited by the non-specific unit criterion.

## Negative results that survive review

1. **Decision-level behaviour never shows relation use for fixed-prior deferral rules** (T-RF). This is known threshold logic, applied here. It does **not** cover all candidate-directed rules: the K = 3 rule "else the unpicked candidate" needs an interaction.
2. **X4-type necessity tests do not identify a relation.** A mediator-level sign test is required.
3. **For K ≥ 3, a bilinear score on linearly encoded evidence cannot implement argmax agreement** (T-B-IMP). With a per-source nonlinearity first, it can.
4. **Learned-model identification at unit or direction level failed.** The criterion is non-specific, and final-site interchange measures readout alignment.
5. **"No unit reads two sources" is not "relation-free"** in a pre-norm Transformer, because shared LN couples the sources (N2).
6. **No predictive advantage** over relation-free or unstructured baselines (F29).

## Withdrawn after review

- "Every gated rule needs a dead band"
- "The historical C′1 had an impossible form"
- B-TM with final LN
- "Only candidate-symmetric rules need interaction"
- "C2: relation effect entirely through the query route"
- "Values carry 1.8%" as a general claim
- All interchange-based use and non-use claims in the learned models
- "T4 learned models use no relation"
- "T1 relation computed redundantly via attention"
- Unit-level identification in learned models
- The claim that the residual rotation was inapplicable

This document does not propose closing Step 1. `OPEN_RESEARCH_OBLIGATIONS.md` lists the work that remains.
