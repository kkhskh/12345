# Final research synthesis: answers to the 22 questions (§23), corrected after review

Each answer points to the document that carries its evidence. Status labels follow `CONDITIONAL_CAUSAL_THEOREMS.md`.

**1. What is the strongest corrected agreement-sensitive decision theorem?**

The strongest theorem is **C′1***, the source-vs-prior selection rule for C′.

- **Condition (S+):** `G ≥ 0` and `v_A + v_B + v_R > Π·e^{−Gμ/σ_P}`.
- **Condition (S−):** `G ≤ 0` and `Π·e^{|G|μ/σ_P} > v⁺`.
- **Status:** proved for every positive parameter setting. 0 violations in 39,887 extreme-parameter evaluations. Operating rectangles certified on [τ, 3].

Companion results:

- A1*, the sign law under shared LN2;
- B-TM, the total-margin consequence of a gate, without final LN;
- C4, bypass dominance.

**2. What exact evidence-strength and routing conditions does C′ require?**

The exact boundary is `F = Σ_j e^{s_j} v_j = 0`.

Certified operating floors on [τ, 3]:

- |a|, |b| ≥ 0.09 in agreement;
- ≥ 0.02 in conflict.

The ratio region `|weak| ≥ 0.05·|strong|` is certified in both conflict quadrants.

Beyond 3 the conflict band grows to about 0.039 in absolute terms. The dead bands come from C′'s prior-plus-gate design: `m(a, 0) ≠ 0`. They are not forced by continuity, since `min(a, b)` has no dead band (corrected T-CO).

**3. Can those conditions be identified without evaluating the target decision response?**

Yes. G, `v_j` and Π come from frozen weights and forward states. The thresholds are derived from the circuit, then validated on new points.

**4. Can the MLP mechanism be proved under its actual shared LayerNorm dependence?**

Yes (**A1***).

- It needs only sign preservation under a common positive scale.
- Magnitude separability (A0) is false.
- The derived cones are corrected after review: the ReLU unequal-weight cone has two bounds, and mixed-sign cross-talk has two cones.
- N2: shared LN itself couples the sources.

**5. How does an independently specified source relation appear in the computation?**

| setting | how it appears |
|---|---|
| engineered A and C′ | as `sign(uv)` on LN-coupled coordinates. In C′ it is confirmed as a candidate-symmetric relation by T-MS |
| B | as a bilinear score, which for K ≥ 3 is not agreement |
| learned T1 and T4 | a cross-source interaction exists (T-RF′), but its locus is unidentified |
| learned T2, T5a and T3 | essentially no interaction |

**6. Does the relation-sensitive mechanism survive changes in source magnitude, context and candidate set?**

| change | outcome |
|---|---|
| magnitude | yes, within derived bands; scoped to [−3, 3] for the C′ rectangles |
| distractor content | yes under (N1)–(N4) |
| encoding drift | degrades (0.895 at mix 0.6) |
| candidate set | the bilinear gate fails for K = 3 |

**7. Does a small learned Transformer develop a comparable mechanism without hand-programmed weights?**

It develops *a* cross-source interaction in T1 (XNOR) and T4. It does not develop one under integration objectives (T2, T5a). Whether that interaction is *comparable* to A's or C′'s is not identified.

**8. Is its internal implementation actually identified, or only its input/output behaviour?**

Only its existence is identified, through graded behaviour (T-RF′). Unit- and direction-level identification failed:

- the unit criterion is non-specific;
- final-site interchange measures readout alignment.

**9. Which causal intervention establishes that the identified feature is used?**

| setting | interventions |
|---|---|
| engineered | X4 (necessity, on-manifold), plus T-MS S2 (mediator interchange from agreement-for-c′ into conflict, off-manifold), plus S1 (the mediator's sign pattern) |
| learned | none that is specific |

**10. Can the mechanism's effect be separated from ordinary source strength and aggregate evidence?**

Only relative to a declared aggregate *and* a declared function class.

- Matched relation-blind pairs exist only in the reachable set (T-R).
- Matching all reader coordinates also matches the relation (X3).
- Because `ab = ((a+b)² − a² − b²)/2`, "relational" is relative to the class `h(f(a) + g(b))`.

**11. Can it be described invariantly under relevant function-preserving transformations?**

- **Invariant:** the function; graded-test results; and, in engineered circuits, T-MS (the sign pattern of a mediator).
- **Not invariant:** coordinate-level and unit-level descriptions.
- **Residual rotation:** applicable and exact, per the reviewer.

**12. Does the mechanism explain a candidate margin, a preference reversal, or only a generic output change?**

| mechanism | what it explains |
|---|---|
| C′ | a preference reversal on matched pairs |
| A | a decision sign |
| B | a confidence change, with direction predicted by B-TM |
| R6 | a net-evidence reversal under a stated rule |

**13. Does its directional prediction come from independently inspectable computation or outcome-dependent tuning?**

- **Engineered:** inspectable signed weights.
- **Learned:** the preregistered frozen-weight sign prediction scored 33/35 on held-out T1/TF1, but it rests on a non-specific unit set.

**14. Does the result add anything beyond ordinary nonlinear interaction, gating or causal abstraction?**

No new mechanism and no new mathematics. What it adds:

- the precise placement of the question, through T-RF, T-RF′, T-MS and N2;
- certified operating conditions;
- a demonstration that necessity tests (X4-type) cannot identify a relation without a mediator sign test.

**15. What new information is needed to distinguish agreement-sensitive computation from a functionally equivalent alternative?**

At the decision level this is impossible for fixed-prior deferral rules. It requires either:

- **graded outputs** (T-RF′), which certify an interaction but not agreement; or
- **internal sign information about the mediator** across all four quadrants, including agreement for c′ (T-MS).

**16. What happens with three or more candidates?**

- The inner product differs from argmax agreement.
- A bilinear score on linear encodings cannot implement argmax agreement (T-B-IMP); with per-source one-hot features it can.
- Learned T5b is solved by threshold logic: B-signs scores 1.000.
- Some candidate-directed rules ("else the unpicked candidate") need an interaction.

**17. What do additional writers, bypasses and readers do to the causal conclusion?**

| addition | effect |
|---|---|
| competing writer | cancels or reverses the conclusion |
| rereader | reverses it |
| bypass | dominates when `\|m_bp\| > max \|v_j\|` |
| final LN | can flip the sign of a routing effect (B-TM restricted) |
| redundant copies | break single-site necessity |
| equal-source value changes | can carry a source change without the relation (C2 a-flip: values 0.91) |

**18. What are the strongest surviving positive claims?**

1. C′1*, A1* and B-TM.
2. The engineered identification chain: X4, T-MS and C2.
3. T-RF′ as a behavioural interaction certificate.
4. The learned integration null.
5. B's confidence modulation, within B-TM's class.

**19. What are the strongest new negative results?**

1. T-RF, scoped to fixed-prior deferral rules.
2. X4-type necessity does not identify a relation.
3. Learned unit and interchange identification failed.
4. N2: shared LN couples sources.
5. Several reviewed claims were withdrawn; see `ADJUDICATION.md`.

**20. Which original fatal problems have gained a genuine mathematical or identification result?**

| failure | rating |
|---|---|
| F1 | engineered progress |
| F4 | graded and internal partial |
| F5 | partial |
| F6, F7 | procedural progress |
| F2, F3, F25, F29 | sharpened |

**21. Which remain unresolved?**

- F1 for learned and pretrained models;
- F4 at the decision level, which is impossible for the deferral family;
- F29;
- F3 for K ≥ 3 learned representations.

**22. What exact research problems should be investigated next within Phase 1.5 Step 1?**

See `OPEN_RESEARCH_OBLIGATIONS.md`. In order:

1. a fresh review of the post-review mechanisms;
2. a preregistered confirmation;
3. an on-manifold version of S2;
4. localising the learned interaction with specificity controls;
5. strength-dependent rules;
6. T5b and TF2-AO;
7. continuous T-RF;
8. extrapolation;
9. an LN-aware "relation-free" definition;
10. learned reachable contrasts.
