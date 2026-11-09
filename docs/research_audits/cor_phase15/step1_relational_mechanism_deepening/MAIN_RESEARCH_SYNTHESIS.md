# Main research synthesis — Investigation III (relational mechanism deepening)

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
  stored: P-count stores leave-q counts at the readout tokens in an additive code that is causally used (5 seeds [I, C]);
  T1 stores a different, non-count variable at the source tokens (3 parents [I]). GATE computes its relation by attention
  at the evidence tokens and decides by readout attention to them, in three seed-specific layouts, two with backup routes
  [I]. TRANS models represent σ (always) and τ (only with auxiliary supervision or a restricted prior) but did not compose
  them into decisions on the full task [I, E]; the restricted-prior model behaved as the training-prior Bayes rule.
  An ADAPT model with a T1 curriculum tracks a changing reliable group and uses positional evidence [C].
- **Theory (Level III).** A restrictive class RC (independent relation + generator-defined mediator with interchange +
  non-aggregate signature + signed margin consequence) contains T4 and GATE implementations and excludes TRANS ones. Across
  members the shared structure is "relation variable at source tokens → readout attention" [I per family; structural, not
  causal, across families]; codes, layers and redundancy differ. Structure did predict something prospectively for the first
  time — the within-Transformer extrapolation ranking via on-support colluder-code decodability (ρ 0.71, n = 6) — while a
  negative prediction to the contrary failed [F→ informative].

## 2. Answers to the 42 questions

**Learned representations**
1. P-count transfers cross-question aggregation of within-pair agreement, as explicit additive leave-q counts at the
   readout tokens [I].
2. Necessity: not established (mean-replacement is non-specific: variance-matched control removes more) [E].
3. Sufficiency: yes — writing only the counterfactual count change reproduces 60–72 % of counterfactual decisions [I, 5 seeds].
4. The site follows the objective's read-out point (P-count targets at readouts; T1 routes on per-source reliability) —
   hypothesis, untested [O].
5. Yes: transfer 5/5 seeds (0.704–0.725), location 5/5; T1 location 3/3 [C, I].
6. Agreement information from ≈ 1000 T1 steps; usable aggregate between ≈ 1700 and 2000 (one run, 2000-step protocol) [E].
7. At 1000–1600 steps agreement is decodable but unusable even with 2000 fine-tuning steps; the old 1000-step protocol
   also under-rated usable features (P-count 0.654 vs 0.723) [E].

**Mechanism reconstruction**
8. An attention head at the evidence tokens reads the gate tokens and writes a gate-state variable (L0-h1 seed 1; L1-h3
   seed 2; L0-h2/h1 seed 0) [I].
9. The gate state is stored at the evidence tokens (decodable 1.00 even when answer-irrelevant); the readout then reads
   the evidence tokens through attention [I].
10. Readout attention in layer 1 (seed 0, primary) or layer 2 (seeds 1, 2); never the readout MLPs (≤ 0.07) [I].
11. Seed 0's last-layer head is a backup: decisive only when the primary layer-1 route is removed (conditional transfer
    1.00). In seed 2 the last-layer head is the only route (ablation → 0.663) [I].
12. The abstract structure replicates (3/3); the layout and redundancy do not (frozen Q3-B2–B4 failed) [I, F].
13. Structurally similar to T4 A′ routing (relation variable at source tokens, readout attention); not testable as the same
    causal variable across tasks; GATE's variable is within-question [I/structural].

**Composition**
14. Not on the full 36-pair task (12k steps; 8k with map supervision); yes on a restricted 24-pair distribution [F/E].
15. σ: yes, at the statistical ceiling, every run [I-probe with controls].
16. τ: discarded in the unsupervised full run (below random-network baseline); represented at ceiling with supervision or the
    restricted prior.
17. Composite: decodable only under supervision/restricted prior; never composed in the full-task decisions.
18. Only in the restricted-prior model are σ- and τ-evidence causally used (map interventions follow Bayes 0.64; masking
    drops 0.13–0.15) [I-input level].
19. No: held-out pairs at 0.246 (below chance), matching the training-prior Bayes rule (0.205) [C criteria; E interpretation].
20. Map supervision makes maps represented but not composed; restricting the pair distribution makes decisions learnable
    within the training class; no curriculum produced generalising composition.
21. Composite decisions need both maps informed (T2); one value pair leaves 2 permutations, two identify a map (T3) [P].

**General theory**
22. Shared by T4 and GATE: relation variable computed and stored at source/evidence tokens, decision by readout attention to
    them; no TRANS mechanism qualified.
23. Different: cross- vs within-question scope; code and site of transferable variables; layers and redundancy.
24. A common abstraction survives only as the RC definition per family; no cross-family interchange is possible, so no common
    *causal* abstraction is established.
25. Invariant: exact head splits, linear reparameterisations of the site; not invariant: seed (layout), architecture
    (DeepSets lacks readout attention).
26. Represented-but-unused maps (TRANS aux); dispensable salient attention (GATE s0); same function, different codes
    (P-count vs T1); prior-fitting (TRANS holdout); insufficient count statistics (ADAPT).
27. Margin consequences follow behaviourally from pointwise Bayes-closeness; internally, from interventions on an RC mediator.
28. Directionally predictable only given generator semantics and only for mediated relations; represented-but-unmediated
    relations change outputs without predictable direction.

**Extrapolation and generalisation**
29. The P-A1 signature was an off-support probe reading; on-support measurements of the same idea reverse the
    cross-architecture reading [E].
30. On-support colluder-code decodability (SAT) predicted the within-Transformer ranking on 6 new models (ρ 0.71, beats
    constant); behavioural off-support probe (A2) and architecture are stronger [C-weak].
31. DeepSets/Transformer split is architectural (15/15 DeepSets global); the within-Transformer spread is learned.
32. Under a change point, optimal decisions need positional history; from scratch the model failed; with a T1 curriculum it
    matches exact Bayes and follows history interventions (two parents, second prospective) [C].
33. Learned unknown-group mechanisms: not examined here [O]; Bayes-level limits are in the continuation.
34. Copying: a copier matching honest statistics is unidentifiable [P]; a disagreement-triggered copier is identifiable only
    through cross-group co-occurrence, weakly [N]; correlated errors for learned models not examined.
35. General: Bayes-closeness on support, readout-attention decisions in Transformers, curriculum dependence of
    cross-question latents. Task-specific: layouts, codes, composition failure, ADAPT's from-scratch failure.

**Original COR and distinctiveness**
36. No new consistency-to-decision theorem beyond OD2, which is a weighted-voting characterisation.
37. Yes for every relation studied (gate rule, agreement pattern, maps, positions) — all defined from source semantics.
38. Reduce to known: all mathematics (Bayes, weighted voting, change points, label switching, group convolution);
    GATE redundancy is a known self-repair phenomenon; holdout behaviour is training-prior Bayes.
39. New here: OD2's closing proof (restricted, short); abstract-variable interchange as a generator-defined identification
    method (variant of known methods); prospective confirmations Q3-A, Q3-E′, SAT's held-out association; several clean
    negative findings.
40. F1 (restricted bridges in three families), F3 (iff), F5 (identification by generator variables), F25.
41. F4 (weakly), F29, F30 and the general causal bridge remain unresolved.
42. Highest leverage: learned composition (open map #1), a causal test of graded vs saturated features with independent
    seeds (#2), a pooled-architecture mediator (#3), and learned unknown-group mechanisms (#7).

## 3. Documents
PCOUNT_CAUSAL_TRANSFER · GATE_FULL_CAUSAL_CIRCUIT · TRANS_COMPOSITION_RESEARCH · STRUCTURE_AND_EXTRAPOLATION ·
RELATIONAL_COMPUTATIONAL_CLASSES · ADAPTIVE_SOURCE_THEORY · GENERALIZATION_LANDSCAPE · CAUSAL_IDENTIFICATION_AND_INVARIANCE ·
THEOREMS_AND_COUNTEREXAMPLES · PROSPECTIVE_PREDICTIONS · BASELINES_AND_NOVELTY · COR_FAILURE_PROGRESS ·
OPEN_FOUNDATIONAL_RESEARCH_MAP · reviews/ · README.
