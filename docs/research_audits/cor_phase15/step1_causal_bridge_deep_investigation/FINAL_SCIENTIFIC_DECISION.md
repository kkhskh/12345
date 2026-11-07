# Final scientific decision: Phase 1.5, Step 1

**Primary verdict: `STEP1_ONLY_KNOWN_NONLINEAR_MECHANISMS_SURVIVE`.**

The candidate mechanisms work. Three architecturally distinct families of Transformer circuits compute a relation between evidence sources and use it in the candidate decision, with proved class theorems. They are fully explained by ordinary nonlinear interaction (an XNOR unit), attention composition (Q/K-composition), and gating.

The conditions of `STEP1_RESTRICTED_IMPLEMENTED_MECHANISM_PROVED` are also met. The more specific label is chosen because distinctiveness was tested and came out negative (`BASELINES_AND_NOVELTY.md`; R4).

**Preserved.** O/D has no established universal decision implication. Direction 1's Othello results do not validate obstruction. `OBSTRUCTION_READY = NO`. `PHASE_2 = NO GO`. No pretrained model was run, and no sealed data was used. Historical audits are unmodified (`logs/historical_files_unchanged.txt`).

## 1. Strongest defensible positive claims

1. **Existence, in three families.**
   - **A.** A one-layer pre-norm block with a 4-unit GELU MLP makes the sign of the margin equal the sign agreement of two sources (`±3.2142216080`). Three independent implementations reproduce it.
   - **B.** A two-layer block with moving attention gates one source's vote by a bilinear query–key product with another source.
   - **C / C′.** Two-block circuits amplify net evidence under agreement (C), or route to the sources under agreement and to a prior under conflict (C′).
2. **Class theorems with proved boundaries:** A1 with (A0), B1 with (B4), C1, C′1, C′2. Each hypothesis is removed in turn with a witness.
3. **Depth theorem (C′2).** No single pre-norm attention layer can make its allocation between two positions depend on the relation of either to a third. Relation-dependent routing needs Q- or K-composition.
4. **Identification results.**
   - The binary positive control cannot distinguish relation computation from a nonlinearity of net evidence (A2, A3).
   - A contrast matched at the reader's post-normalization input does distinguish them, on reachable inputs (A4).
   - The agreement mediator is necessary and sufficient by interchange in C and C′.

## 2. Strongest defensible negative claims

1. **No bridge survives as explanation.**
   - Every directional decision consequence is supplied by sign hypotheses on the weights.
   - Mirrored circuits implement the opposite rule in the same architecture: for example, amplifying under disagreement (3.185 against 0.246).
   - So the architecture carries no implication that consistency stabilizes or favors decisions, which is the original thesis.
2. **Not identified.**
   - Source polarity is fixed only through the outcome pathway.
   - Per-head and per-unit quantities, raw scores and patterns, preactivations, and summed writes all change under exact function-preserving maps.
   - What is identifiable: function-level effects, score differences, readout-normalized reader even parts, and `q̂` up to scale.
3. **Candidate-relativity is limited to a binary pair with scalar evidence.** Beyond that, the gate's relation (an inner product of features) is not agreement about the preferred candidate.
4. **Head- or unit-level claims do not determine decisions when downstream re-readers or parallel writers exist** (PB9; R1 S3; C with a third block gives `m = −8.02`).
5. **Every result reduces to established mechanisms.**

## 3. Answers to the mandate's 25 questions

1. **Can a Transformer explicitly compute a relation between candidate-relevant evidence sources and use it to select a candidate?** Yes, by construction. In A, `sign m = sign(ab)`. In C′, agreement decides "follow the sources" against "defer to the prior".
2. **Independently defined consistency relation, or post hoc score?** The relation, sign agreement of the coded inputs, is defined from the input semantics before evaluation. Its internal identification is not independent: polarity is fixed by the outcome path, and the pairing is fixed by the reader weights.
3. **What operator implements it?**
   - A: paired even units, `h(x+y) − h(x−y)`.
   - B: a bilinear query–key score followed by softmax.
   - C: ReLU gating on `(Ŝ, q̂)`.
   - C′: a query reading the MLP agreement feature (Q-composition).
4. **What changes in the margin when the relation changes?**
   - A: the sign flips, `m(a,b) = −m(a,−b)`.
   - B: the agreeing source's vote gains weight. Agreement implies the shared sign (unanimity), and conflict has a reversal band.
   - C: `|m|` grows with agreement at fixed `Ŝ` (4.588 against 0.270).
   - C′: in conflict the decision switches to the prior.
5. **Which parts come from source strength rather than the relationship?**
   - A: the sign is pure relation; the magnitude is product-like for weak evidence and min-like for strong.
   - C: the sign is pure net evidence; agreement only scales the margin.
   - C′: in conflict, strength has no effect (−2.631 constant).
   - B: the vote's sign is the source's own; its weight is the relation.
6. **Does the effect survive controls and ablations?** Yes, within the classes: ablations remove it, interchange transfers it, negative controls lack it. It does not survive perturbation for C's sign law, nor parallel writers or re-readers for B. One prediction-level control failed: raw-sum matching (PA9). The A4 contrast at the reader repaired it post hoc.
7. **Does it occur in a nonlinear MLP, moving Q/K attention, or a deep rereading circuit?** All three were constructed and tested: A, B, and C/C′.
8. **A restricted theorem, not just a favorable example?** Yes: A1, B1, C1, C′1, C′2, with proved boundaries. They are implemented-mechanism existence theorems.
9. **What determines the direction of the decision response?** The signs of the gate weights and readout: `λ`, `γ`, `k`, the unit output signs, `Δu^T v`, and the key signs. Consistency itself does not.
10. **Are those assumptions independently identifiable?** Only modulo the gauge maps, and polarity is fixed through the readout. Partially at best.
11. **Can they be satisfied without encoding the desired logit change?** No. The sign hypotheses are the directional content (R2; check 5 fails).
12. **Does the result distinguish candidates, or only predict a generic output change?**
   - A does not distinguish candidates: `m(a,b) = m(−a,−b)`.
   - B, C and C′ do, for a binary pair with scalar evidence.
   - With 3 or more candidates, B's relation is not candidate agreement.
13. **Can the same consistency value coexist with opposite candidate decisions?** Yes in B, C and C′: two agreeing pairs supporting opposite candidates have the same agreement and opposite decisions. In A, no: equal agreement gives equal decisions.
14. **What information resolves the ambiguity?** Candidate-directed evidence: the sign of the source votes, or of net evidence. It is not contained in the relation.
15. **Is the relation invariant under function-preserving transformations?**
   - `x̂ŷ` is invariant as a score difference.
   - `sign(xy)` is invariant only after readout normalization, and only up to a joint flip.
   - `q̂` is invariant up to scale.
   - "The circuit reads `x − y`" is not invariant.
16. **If not, what is identifiable?** Function-level effects of coded inputs, score differences, reader even parts grouped by direction, and the A4 contrast at an identified reader.
17. **Does the mechanism require O/D?** No. O/D appears nowhere.
18. **Is it a standard nonlinear interaction, conjunction, attention router, or causal abstraction?** Yes: an XNOR unit, Q/K-composition, gating, and exact abstractions of them.
19. **What distinctive content remains?** No distinctive theory. One minor distinction survives, expressible in standard terms: a candidate-symmetric agreement detector against agreement-gated evidence. The depth theorem C′2 is an elementary property of single attention layers.
20. **Which of the ten failures are resolved?** F6 (diagnostic), F7 (procedure demonstrated here), and F30 (conditionally). F3 is resolved conditionally, for pairwise preference inside designed classes. F25 is sharpened: a sign predictor using the relation alone is impossible in these classes.
21. **Which remain unresolved, and why?**
   - F1: existence only; no trained-model theory.
   - F2: direction set by gate signs; both signs implementable.
   - F4: no distinctive prediction.
   - F5: identification only modulo the gauges; polarity via the outcome path.
   - F29: no advantage.
22. **What would falsify the strongest surviving candidate in a frozen trained Transformer?** For C′-type deferral: in a frozen model, preregister sources and a prior token. If interchanging the identified agreement feature at the read position leaves the routing score differences toward the sources unchanged, the candidate is falsified for that model. The same holds if the decision in conflict cells tracks the stronger source. (Protocol in §5.)
23. **Enough justification to propose a later empirical identification study?** Mathematically, only as a test of whether trained models contain such gates. That is a mechanistic-interpretability question, not a test of COR. A study is justified only if framed that way and separately authorized.
24. **What would justify closing Phase 1.5 Step 1?** This result. Every family named in the mandate has been examined. The bridge is rejected as explanation, and the surviving mechanisms are known ones. Recommendation: close Step 1 with this verdict.
25. **What would justify abandoning the original consistency-to-decision hypothesis?** The combination established here. The architecture implements consistency-to-decision rules of either sign with equal ease, so no architectural law ties consistency to good or stable decisions. The rules that exist are explained by ordinary mechanisms. The original O/D score plays no role. **Recommendation: abandon the hypothesis as a distinctive theory.** Keep the relation-sensitive circuits only as ordinary mechanistic-interpretability objects.

## 4. Coverage and its limits

**Examined.**
- MLP agreement: designed, continuous, five activations, widths, encodings, noise, perturbations, final LN, readouts, vector evidence.
- Moving Q/K attention: dissection, query-only and key-only substitutions, masking, competition, three candidates, perturbations.
- Multi-block: MLP rereading C, with exact matching and robustness; attention rereading C′, with the depth theorem and mediation.
- Identifiability: ten function-preserving map types, plus the external audit's checks.

**Not examined.**
- Stacks deeper than two blocks, except reviewer counterexamples.
- RMSNorm and rotary architectures.
- Trained weights: not authorized.
- Exact maps between attention and MLP computation (a conjecture).

None of these is a central class that the mandate names.

## 5. Prospective identification specification (for later review only)

This is for a separately authorized study. It is not authorized here.

1. **Freeze**
   - the model checkpoint;
   - the task template, with two input evidence slots and a prior token;
   - token lists per level, giving polarity by task semantics;
   - the candidate pair and read position;
   - the decision scalar;
   - the matched statistic and the site where it is matched (the reader's input, never the raw residual);
   - the reader-identification rule (below);
   - the head and unit search space;
   - the falsification criteria.
2. **Reader identification without the response.** Find components whose activations change under single-source substitution. Group the even parts by direction. Compute score differences for heads whose query composes with earlier MLP outputs. Use activations only, never `Δm`.
3. **Tests.** Run these in the stated order. Report all of them, with no revision.
   - Matched-reader-input contrasts (A4).
   - Interchange of the candidate agreement feature (C′3).
   - Unequal-level factorials, because B is invisible to 2×2 designs.
   - Path patching for downstream re-readers.
   - The same tests on the gauge-invariant quantities only.
4. **Interpretation.** A positive finding would show that a trained model contains a relation gate. It would not validate COR, and its sign would be an empirical fact, not a prediction of any consistency theory.

## 6. Next justified decision

Close Phase 1.5 Step 1 with `STEP1_ONLY_KNOWN_NONLINEAR_MECHANISMS_SURVIVE`. Retire O/D and the obstruction framing. Do not open Step 2 or Phase 2 on a successor drawn from this work. If trained-model gating is of interest, propose it as an ordinary interpretability study using §5, under a separate authorization.
