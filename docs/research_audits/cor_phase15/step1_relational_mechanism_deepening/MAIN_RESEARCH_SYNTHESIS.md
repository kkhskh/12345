# Main research synthesis — Investigation III (relational mechanism deepening)

**Post-review version.** Corrections from review round R1 are marked **[R: id]** (see `reviews/ADJUDICATION_R1.md`); corrections from round R2
(`reviews/R2_counterexample_defence.md`, adjudicated in `reviews/ADJUDICATION_R2.md`) are marked **[R2]**.
Not a thesis; no closure proposed. Verdicts of all earlier packages unchanged. Evidence categories: **[P]** proof,
**[N]** exact numerics, **[C]** prospectively confirmed, **[F]** prospectively falsified, **[I]** interventionally supported
mechanism, **[E]** exploratory, **[O]** open.

## 1. Connected account

Four relation families were studied at the three levels the mandate asked for.

- **Information (Level I).** In TRANS the likelihood factorises into an AB part (σ) and a BC part (τ); a composite decision
  is at chance unless both maps are informed [P]. In ADAPT, agreement counts are insufficient — position matters [P, N].
  For O/D, unweighted discrepancy is Bayes-order-compatible exactly when no smaller source set can outweigh a larger one
  [P], a condition stated in the semantic weights that discrepancy does not identify.
- **Implementation (Level II).** Cross-question relational pretraining transfers through *where and how* the aggregate is
  stored. **[R: CA1, R2]** Both curricula carry a causally used, additive one-hot agreement-count variable, at different
  sites and in different codes: P-count at the readout tokens with a shared code (5 seeds, IIA 0.60–0.72 [I]); T1 at the
  source tokens with slot-specific directions (3 parents, IIA 0.66–0.82 against a mismatched-count floor of 0.46 and
  capacity-matched nulls of 0.07–0.08 [I]). T1's data fit "each source stores its own group's count" equally well, so
  "the same variable" is not established; no cross-model swap was run. The pre-review claim that T1 stores "a different,
  non-count variable" was an artefact of an encoder shared across source slots and is withdrawn. GATE computes its relation by attention
  at the evidence tokens, stores it there as an abstract variable (held-out-x₀ probes 0.98–1.00), and decides by readout
  attention to them (attention mass to E 0.93–0.99), in three seed-specific layouts [I]. **[R: CA4]** The pre-review
  "jointly necessary backup pairs" are only contributing: ≥ 86 % of the gate effect survives ablating both. TRANS models represent σ (always) and τ (only with auxiliary supervision or a restricted prior) but did not compose
  them into decisions on the full task [I, E]; the restricted-prior model behaved as the training-prior Bayes rule, and
  **[R: CN4, FM2]** under that prior much of its accuracy needs no composition (Bayes without τ evidence 0.429).
  An ADAPT model with a T1 curriculum tracks a changing reliable group and uses positional evidence [C].
- **Theory (Level III).** A restrictive class RC (independent relation + generator-defined mediator with interchange +
  non-aggregate signature + signed margin consequence) contains T4 and GATE implementations and excludes TRANS ones. Across
  answer-supervised members (T1, GATE) the shared structure is "relation variable at source tokens → readout attention"
  [I per family; structural, not causal, across families]; **[R: CA10]** this is curriculum-dependent (P-count stores the
  count variable at the readout), not a property of the relation. **[R: SD7, SD8]** No structural signature predicted
  extrapolation on genuinely new units: the held-out Transformers share parents with development models, and a
  parent-lookup predictor (ρ 1.00, MAE 0.042) beats SAT (ρ 0.71, MAE 0.123). The frozen negative prediction Q3-D2 failed
  because of SAT [F]; this is a parent-shared replication, not a confirmation.

## 2. Answers to the 42 questions

**Learned representations**
1. P-count transfers cross-question aggregation of within-pair agreement, as explicit additive leave-q counts at the
   readout tokens [I]. **[R: CA1, R2]** T1 transfers a count variable stored slot-specifically at the source tokens (possibly
   only each source's own-group count) [I, 3 parents].
2. **[R: CA3]** Necessity: supported in the mean-replacement operator class once controls are per-position (T1 source: count
   subspace 1.65 vs variance-matched 0.00); pre-review "non-specific" was a pooled-statistics artefact. Not natural necessity;
   removals above 1.0 include collateral damage, so the count variable's necessity is not isolated [R2].
3. Partial sufficiency: writing only the counterfactual count change reproduces 60–72 % (P-count readout) and 66–82 % (T1
   source, per-slot; floor 0.46, so ≈ 91 % of the full patch's gain above floor for ref_T1p1) of counterfactual decisions [I];
   capacity-matched null encoders fail for T1 (0.07–0.08) [R2]; floors for P-count not measured.
4. The site follows the objective's read-out point (P-count's count head at readouts; T1's answer loss routes per-source
   reliability) — hypothesis, consistent with CA1, untested [O].
5. Yes: transfer 5/5 seeds (0.704–0.725) [C]; location 5/5 (readout); T1 location 3/3 (source) [I].
6. Agreement information from ≈ 1000 T1 steps; usable aggregate between ≈ 1700 and 2000 (one run, 2000-step protocol) [E].
   **[R: SD13]** Only "agreement before transfer" is supported; transfer rises before the count probe moves.
7. At 1000–1600 steps agreement is decodable but unusable under the 2000-step fine-tune protocol (not shown to be
   unusable asymptotically [R: SD12]); the old 1000-step protocol also under-rated usable features (P-count 0.654 vs 0.723) [E].

**Mechanism reconstruction**
8. An attention head at the evidence tokens reads the gate tokens and writes a gate-state variable (L0-h1 seed 1; L1-h3
   seed 2; L0-h2/h1 seed 0) [I].
9. The gate state is stored at the evidence tokens (decodable 1.00 even when answer-irrelevant); the readout then reads
   the evidence tokens through attention [I].
10. Readout attention in layer 1 (seed 0, primary) or layer 2 (seeds 1, 2); never the readout MLPs (≤ 0.07) [I].
11. **[R: CA4–CA6]** Seed 0's last-layer head transfers the flip when the layer-1 route is removed (conditional transfer 1.00;
    informative for seed 0 only). Ablating both seed-0 routes (mean ablation) leaves 86 % of gate flips followed, so further
    routes exist; resample ablation instead drops agreement to 0.37–0.39 by injecting foreign evidence [R2].
    In seed 2 one route dominates: ablation collapses the model to one gate state (flip-following 0.51) [I].
12. The abstract structure replicates (3/3); the layout and redundancy do not (frozen Q3-B2–B4 failed) [I, F].
13. Structurally similar to T4 A′ routing (relation variable at source tokens, readout attention); not testable as the same
    causal variable across tasks; GATE's variable is within-question [I/structural].

**Composition**
14. Not on the full 36-pair task (12k steps; 8k with map supervision); yes on a restricted 24-pair distribution [F/E].
15. σ: decodable at the 36-pair ceiling in the full and aux runs; **[R: CN1]** below the prior-matched ceiling (0.694) in the
    holdout run [I-probe with controls].
16. τ: discarded in the unsupervised full run (below random-network baseline); decodable at the supervised episode token in
    the aux run (guaranteed by the auxiliary head [R: CN2]; ≤ 0.44 at decision tokens).
17. Composite: not decodable beyond what a probe can compose from σ and τ [R: CN2]; never composed in full-task decisions.
18. Only in the restricted-prior model are σ- and τ-evidence causally used — jointly, within training pairs: it obtains 82 %
    of the accuracy that requires both maps' evidence [R2] — (map interventions follow at 0.64, the
    training-prior Bayes rule's own level [R: CN3]; masking drops 0.13–0.15, to the no-τ Bayes level 0.429 [R: FM2]) [I-input level].
19. No: held-out pairs at 0.246 (below chance), matching the training-prior Bayes rule (0.205) [C criteria; E interpretation].
20. Map supervision makes maps decodable at the supervised token but not used; restricting the pair distribution makes
    decisions learnable within the training class, largely without composition; no regime (one seed each [R: CN6]) produced
    generalising composition.
21. Under the product prior composite decisions need both maps informed (T2) [P]; **[R: FM3]** one chained AB/BC value pair
    leaves 2 composite candidates, a non-chained one 4; certainty can arise from a single chained pair of each type [P, N].

**General theory**
22. Shared by T1/T4 and GATE (answer-supervised curricula): relation variable computed and stored at source/evidence tokens,
    decision by readout attention to them; P-count stores its count variable at readouts instead [R: CA10]; no TRANS
    mechanism was shown to qualify (condition 2 untested for TRANS [R: FM10]).
23. Different: cross- vs within-question scope; site and code (shared vs slot-specific) of the transferable count variable
    [R: CA1, R2]; layers and redundancy.
24. A common abstraction survives only as the RC definition per family; no cross-family interchange is possible, so no common
    *causal* abstraction is established.
25. Invariant: exact head splits, per-token linear reparameterisations of the site; not invariant: seed (layout),
    architecture (DeepSets lacks readout attention), ablation operator (mean vs resample reverses GATE single-head
    results [R: CA4]), and cross-slot weight sharing in the encoder (CA1).
26. Decodable-but-unused maps (TRANS aux); dispensable salient attention (GATE s0); count variables at different sites
    and codes (P-count vs T1) [R: CA1, R2]; prior-fitting (TRANS holdout); insufficient count statistics (ADAPT); encoder-specification
    artefacts (CA1) and pooled-control artefacts (CA3) as methodological counterexamples.
27. Margin consequences follow behaviourally from pointwise Bayes-closeness; internally, from interventions on an RC mediator.
28. Directionally predictable only given generator semantics and only for mediated relations; represented-but-unmediated
    relations change outputs without predictable direction.

**Extrapolation and generalisation**
29. The P-A1 signature was an off-support probe reading; on-support measurements of the same idea reverse the
    cross-architecture reading [E].
30. **[R: SD7–SD9]** No. SAT's ρ 0.71 on 6 held-out Transformers replicates their parents' development values (parent lookup
    ρ 1.00, MAE 0.042; exact null p 0.068) and was never frozen as a positive prediction. The frozen Q3-D2 failed (because of
    SAT; GCD complied). The behavioural off-support probe A2 and architecture predict extrapolation [C: Q3-D1, D3]; A2's
    held-out rank agreement (ρ 1.00) is matched by parent lookup, so it beats parent sharing only on MAE (0.019 vs 0.042) [R2]. EX1 shows
    on-support behaviour does not determine extrapolation; it does not exclude on-support features predicting it [R: FM9].
31. DeepSets/Transformer split is architectural (15/15 DeepSets global); the within-Transformer spread is learned.
32. Under a change point, optimal decisions need positional history; from scratch one seed plateaued at the own-question
    level (step-unmatched [R: SD15]); with a T1 curriculum it matches exact Bayes and follows history interventions (two
    parents, second prospective; thresholds set after the exploratory run) [C].
33. Learned unknown-group mechanisms: not examined here [O]; Bayes-level limits are in the continuation.
34. Copying: a copier that copies independently per question at a rate matching honest agreement is unidentifiable [P]
    (**[R: FM6]** not every constant-rate copier: Markov or agreement-triggered copying changes the joint law); a disagreement-triggered copier is identifiable only
    through cross-group co-occurrence, weakly [N]; correlated errors for learned models not examined.
35. General: Bayes-closeness on support, readout-attention decisions in Transformers, curriculum dependence of
    cross-question latents. Task-specific: layouts, codes, composition failure, ADAPT's from-scratch failure.

**Original COR and distinctiveness**
36. No new consistency-to-decision theorem beyond OD2, which is a weighted-voting characterisation (K ≥ 3; false at K = 2 [R: FM5]).
37. Yes for every relation studied (gate rule, agreement pattern, maps, positions) — all defined from source semantics.
38. Reduce to known: all mathematics (Bayes, weighted voting, change points, label switching, group convolution);
    GATE redundancy is a known self-repair phenomenon; holdout behaviour is training-prior Bayes; decodable-but-uncomposed
    maps are an instance of the two-hop / implicit-reasoning literature [R: CN8].
39. New here: OD2's closing proof (restricted, short); abstract-variable interchange as a generator-defined identification
    method (variant of known methods), with a documented encoder-specification pitfall (CA1); prospective confirmations
    Q3-A and Q3-E′; several clean negative findings.
40. F1 (restricted bridges in three families), F3 (iff), F5 (identification by generator variables), F25.
41. F4 (weakly), F29, F30 and the general causal bridge remain unresolved.
42. Highest leverage: learned composition (open map #1), a causal test of graded vs saturated features with independent
    seeds (#2), a pooled-architecture mediator (#3), and learned unknown-group mechanisms (#7).

## 3. Documents
PCOUNT_CAUSAL_TRANSFER · GATE_FULL_CAUSAL_CIRCUIT · TRANS_COMPOSITION_RESEARCH · STRUCTURE_AND_EXTRAPOLATION ·
RELATIONAL_COMPUTATIONAL_CLASSES · ADAPTIVE_SOURCE_THEORY · GENERALIZATION_LANDSCAPE · CAUSAL_IDENTIFICATION_AND_INVARIANCE ·
THEOREMS_AND_COUNTEREXAMPLES · PROSPECTIVE_PREDICTIONS · BASELINES_AND_NOVELTY · COR_FAILURE_PROGRESS ·
OPEN_FOUNDATIONAL_RESEARCH_MAP · reviews/ · README.
