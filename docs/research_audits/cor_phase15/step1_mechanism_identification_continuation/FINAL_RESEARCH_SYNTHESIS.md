# Final research synthesis: answers to the 22 questions (§23)

Each answer points to the document that carries its evidence. Status labels follow `CONDITIONAL_CAUSAL_THEOREMS.md`.

**1. What is the strongest corrected agreement-sensitive decision theorem?**

The strongest theorem is **C′1***. It gives source-vs-prior selection for conditional rereading in terms of two quantities, checked against each other:

- the routing gain `G = λq̂/√2`, where `sign G = sign(ab)`;
- the log ratio of value masses.

It has three sufficient conditions (S+, S−, S−0). It is proved for every positive parameter setting, with 0 violations on fresh points and fresh parameters. Its operating rectangles are certified by interval arithmetic.

Companion results:

- **A1***, the sign law under shared LN2;
- **B-TM**, the total-margin consequence of a gate;
- **C4**, bypass dominance and downstream preservation.

**2. What exact evidence-strength and routing conditions does C′ require?**

The exact boundary is `F = Σ_j e^{s_j} v_j = 0`, an explicit formula.

Sufficient conditions:

- for following the sources: `v_A + v_B + v_R > Π·e^{−Gμ/σ_P}`;
- for deferring to the prior: `Π·e^{|G|μ/σ_P} > v⁺`.

Certified operating floors:

- |a|, |b| ≥ 0.09 in agreement;
- |a|, |b| ≥ 0.02 in conflict;
- conflict is also certified whenever `|weak| ≥ 0.05·|strong|` with strong ∈ [0.2, 3].

The dead bands are unavoidable (T-CO). Their widths trade off against λ and π. See `CPRIME_CORRECTED_OPERATING_REGIONS.md`.

**3. Can those conditions be identified without evaluating the target decision response?**

Yes. G, `v_j` and Π are computed from frozen weights and the forward state, with no use of the margin's sign. The certification runs on the closed form. The thresholds are derived, not fitted, and they were validated on new points afterwards.

**4. Can the MLP mechanism be proved under its actual shared LayerNorm dependence?**

Yes (**A1***).

- Proved: sign preservation under a common positive scale (P3), and exact ratio preservation (P5).
- Not true: magnitude separability (P4). The historical assumption A0 was false.
- Normalization creates interactions (N1).
- Derived cones cover the perturbed cases. See `MLP_NORMALIZATION_THEOREMS.md`.

**5. How does an independently specified source relation appear in the computation?**

It depends on the mechanism:

| mechanism | how the relation appears |
|---|---|
| A, and C′ block 1 | `sign(uv)` of LN-coupled coordinates, read by an even nonlinearity pair |
| B | a bilinear score `⟨z̃_A, z̃_B⟩`. For K ≥ 3 this is **not** candidate agreement (T-B-IMP) |
| learned T1 models | units in the MLP population, plus attention-side features (`sign(ab)` is decodable at 0.998 before the MLP); TF2-AO computes it with attention alone |
| learned T2–T5 models | it does not appear as a used feature |

**6. Does the relation-sensitive mechanism survive changes in source magnitude, context and candidate set?**

| change | outcome |
|---|---|
| magnitude | yes outside the derived dead bands. ReLU cones are scale-free; GELU cones move with the LN scale |
| context (distractor content) | yes under (N1)–(N4) |
| encoding drift | degrades gracefully (0.895 at mix 0.6) |
| candidate set | **no**: the bilinear gate tracks the inner product, not agreement, for K = 3 |

**7. Does a small learned Transformer develop a comparable mechanism without hand-programmed weights?**

Only when the objective is candidate-symmetric agreement (T1). It does not when the objective is integration (T2, T5a) or reconciliation (T3, T4, T5b). Those tasks are solved additively, and a relation-free baseline matches them. See `LEARNED_SYNTHETIC_MECHANISMS.md`.

**8. Is its internal implementation actually identified, or only its input/output behaviour?**

**Identified**, relative to the generator labels:

- the agreement direction in the residual (probe selected without the margin; interchange transfers 100%, against 0% for a random direction);
- a population of relation units with a gauge-invariant set-level causal effect.

**Not identified:**

- a unique unit-level implementation (indices and counts change; redundancy between MLP and attention);
- equivalence to the designed A pairing.

**9. Which causal intervention establishes that the identified feature is used?**

| setting | intervention |
|---|---|
| engineered | the on-manifold matched relation-blind contrast (X4), plus path-specific interchange (C2: `q_R` 100%, `q_B` 0%) |
| learned | interchange along the agreement probe, restricted to matched pairs with differing decisions, against a random-direction control; plus population ablation, which drops accuracy to about 0.75 in 7 of 10 T1 models with MLPs |

**10. Can the mechanism's effect be separated from ordinary source strength and aggregate evidence?**

Yes, but only relative to a declared aggregate, and only inside the reachable set.

- T-R: matched-aggregate conflict partners exist only for `|Ŝ| < d/sqrt(d+ρ²)`.
- X3: matching *all* reader coordinates also matches the relation.
- Raw-sum matching is invalid under LN (N1, X1).

**11. Can it be described invariantly under relevant function-preserving transformations?**

| level | invariant? |
|---|---|
| function | yes |
| residual-stream probes and interchange effects | yes |
| set-level unit-population effects | yes, under QK, VO, head permutation, unit permutation and unit split |
| unit indices, counts, Q/K and V/O coordinates | no |

Residual rotation is untested in the learned models. See `REPARAMETERIZATION_AND_BASELINES.md` §1.

**12. Does the mechanism explain a candidate margin, a preference reversal, or only a generic output change?**

- **C′:** a preference reversal on matched pairs (X4: +1.890 vs −2.630).
- **A:** a decision sign (A1*).
- **B:** in the tested pairs only a confidence (magnitude) change (X6). It changes a decision only under T-B-TM's sign condition.
- **R6:** it reverses the net-evidence decision iff `sign q ≠ sign S` and `|q| > |S|` (0 violations in 3,000).

**13. Does its directional prediction come from independently inspectable computation or outcome-dependent tuning?**

It comes from inspectable computation:

- engineered: signs of `λ`, `wv`, `π` and the readout;
- learned: the step-6 prediction from `W_out·(γ_f ⊙ ΔW_U)` times the generator-conditional activation difference, which is 86–94% correct for TF1 on T1.

For T3 the same prediction is at chance, which correctly shows that those units are not the mechanism.

**14. Does the result add anything beyond ordinary nonlinear interaction, gating or causal abstraction?**

Not as a mechanism. The added content is placement:

- **T-RF** shows where "consistency" has computational content: candidate-symmetric relations only.
- **T-CO** shows which form a gated rule must take.
- **T-B-IMP** shows a limit of bilinear gating for K ≥ 3.
- **X4** is a reachable identification criterion.

All of these are expressible in standard terms.

**15. What new information is needed to distinguish agreement-sensitive computation from a functionally equivalent alternative?**

Internal interventions. By T-RF, behaviour cannot do it for candidate-directed rules. The minimal sufficient evidence is:

1. a declared relation-blind pathway, i.e. the circuit with the relation feature ablated;
2. reachable matched pairs equalising that pathway;
3. a decision difference on those pairs.

The relation-free twin of C′ fails step 1, because it has no relation feature to ablate.

**16. What happens with three or more candidates?**

- Inner-product agreement is not argmax agreement: P(dot < 0 | agree) = 0.080, and P(dot > 0 | disagree) = 0.436.
- A single attention score cannot implement argmax agreement exactly (T-B-IMP). The best bilinear classifier reaches 0.821, against 0.987 for an MLP.
- The single bilinear gate suppresses routing for 8.4% of agreeing pairs.
- Learned K = 3 reconciliation (T5b) is solved additively; B-add reaches 0.991.

**17. What do additional writers, bypasses and readers do to the causal conclusion?**

| addition | effect |
|---|---|
| competing writer | cancels the agreement contrast (B5: kC = 3) or reverses it beyond |
| rereader | reverses it (B5, C3 γ = 3) |
| bypass | dominates when `|m_bp| > max |v_j|` (C4, verified) |
| reversed relation reader (extra head w = 1.5, competing feature 1.5) | inverts the gate |
| redundant copies | break single-site necessity (200/200 decisions preserved when one copy is ablated) |
| monotone downstream map | preserves the decision |

**18. What are the strongest surviving positive claims?**

1. C′1*, A1* and B-TM, with certified or derived operating regions.
2. The on-manifold matched relation-blind decision reversal (X4), together with path-specific mediation (C2).
3. Learned T1 models compute a causally used, directionally predictable, gauge-robust (population-level) relation feature.

**19. What are the strongest new negative results?**

1. T-RF: candidate-directed agreement rules are relation-free at the decision level. C′ has a decision-equivalent twin, and learned reconciliation models use no relation.
2. T-CO: dead bands are necessary. The historical C′1 had an impossible form.
3. T-B-IMP: bilinear gating cannot express K ≥ 3 agreement.
4. Decodability of the relation without use, in T2–T4.
5. The preregistered unit criterion is not sufficient for use (T3).

**20. Which original fatal problems have gained a genuine mathematical or identification result?**

| failure | rating |
|---|---|
| F1 | partial bridge |
| F5 | gauge-invariant population statement |
| F6, F7, F30 | procedural progress |
| F2, F3, F4, F25, F29 | sharpened with proofs |

See `TEN_FAILURE_PROGRESS_MATRIX.md`.

**21. Which remain unresolved?**

- **F1** for non-relational objectives and for pretrained models.
- **F4** at the behavioural level, which T-RF makes impossible for candidate-directed rules.
- **F29.**
- **F3** for K ≥ 3 learned representations.

**22. What exact research problems should be investigated next within Phase 1.5 Step 1?**

See `OPEN_RESEARCH_OBLIGATIONS.md`. In order:

1. Preregistered confirmation of T-RF's consequences.
2. Strength-dependent rules that are not threshold-representable.
3. Continuous T-RF.
4. Residual-rotation gauges.
5. A use-implying unit criterion.
6. The K ≥ 3 mechanism in TF2-AO.
7. Redundancy-aware set necessity.
8. Learned-model reachable matched contrasts.
9. Transfer.
10. A fresh review of any new mechanism.
