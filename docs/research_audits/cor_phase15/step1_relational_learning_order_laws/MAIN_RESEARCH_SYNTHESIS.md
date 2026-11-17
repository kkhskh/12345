# Investigation XV: what governs the order in which networks learn relational trust?

**Status.** COR Phase 1.5, Step 1, provisional ≈ 25% (unchanged). No thesis is selected, no phase is closed, and the O/D framework is not established.

**Reviews.** The three reviews in `reviews/` (mathematics, learning dynamics and mechanisms, integrity) were written by independent agents. They are not human peer review. Every correction is adjudicated in `reviews/ADJUDICATION.md` and applied below.

**Dependencies.** The package depends on five earlier COR packages (`logs/DEPENDENCIES.json`) and is not standalone.

**Evidence labels.**
- **[proof]** mathematical proof;
- **[num]** numerical verification;
- **[dev]** development;
- **[frozen]** frozen confirmation;
- **[expl]** exploratory;
- **[spec]** speculation.

## What was done

**A task family with exact mathematics** (RELATIONAL_COMPLEXITY_AND_VALUE_MATHEMATICS.md). The four GATE variants vary two things:
- the number of sources that determine trust (gate arity a = 2 or 3);
- the gate's decision value (about 0.14 or 0.28 nats), through the noise-pair law ν.

The architecture, inputs, budget, recipe and Bayes NLL (0.288) are the same in every condition.

**Verified exactly, and independently by Review A:**
- The arity-3 gate has no pairwise shortcut.
- The decision needs interaction order exactly a + 1.
- The gate's value is matched across arity to within 0.09–0.9%.

**Predictions frozen before any confirmation run** (freeze commit `7957fee`, 05:04:03 UTC; the first run started at 05:04:17). Three hypotheses with distinct patterns: H1 (complexity first), H2 (decision benefit), and RACE (from development).

**Training.**
- 16 fresh blind-trained Transformers: 4 conditions × 4 seeds, 4000 steps each, snapshots every 25 steps (every 250 after step 2000).
- 4 fresh DeepSets models on T4, as a cross-task positive control.
- All 40 jobs (20 training, 20 measurement) completed with no replacements.
- Blinding, freeze order, the absence of early inspection, and byte-exact scorer reproduction were all verified by Review C.

## Frozen verdicts

| prediction | frozen verdict | what it rests on (after review) |
|---|---|---|
| **P1** complexity at matched value | **COMPLEXITY_DELAYS**, 8/8 pairs | Arity-2 gates at steps 500–1500 (8/8). Arity-3 gates at 2250–2500 (2/8) or not within 4000 steps (6/8, sitting exactly at the gate-free optimum). Seed pairs share streams and initialisation, so this is about 4 independent replicates. It holds in cumulative-learning-rate units too. The budget is about 2700–3000 effective steps under the decayed schedule. "Later" versus "never" is undetermined. |
| **P2** value at fixed complexity | **INCONCLUSIVE**, 5/8 | Post hoc: at arity 2, the lower-value gate came later in 4/4 pairs (ratios 1.4–2.6). This is confounded with an agreement cue that is stronger and overlaps the gate more (88% against 74%). Uninformative at arity 3 because of censoring. |
| **P3** stage pattern | **H1_COMPLEXITY_FIRST**, (+, +, +, +) | RACE and H2 are falsified as frozen. But in C and D, "stage present" is close to entailed by censoring, and in B the stage lasts 3 snapshots in 3 of 4 models. At an A_p threshold of 0.6 the pattern becomes RACE. **H1 is not robustly separated from RACE.** |
| **P6** DeepSets T4 | **CONFIRMED** | The trainability gate passed (4/4). Each model held the own-question rule for 7–10 snapshots (350–500 steps), then acquired the cross-question relation within 150–250 steps. |

## What was withdrawn after review

- **The learning-signal result.** "The initial gate SNR tracks arity" and ρ = −0.80 are withdrawn. The SNR estimator is about 1 under no signal, and every value lay in the null band (Review B F1). P5 and H3 are untested.
- **"Gate information before use" in D_s203.** It was the agreement cue (Review B F2, re-checked).
- **"Newly represented" in A, B and C.** The probes cannot separate the gate from the cue before the transition.
- **"Less abrupt at arity 3".** The snapshots there are 250 steps apart (Reviews B F7, C F4).
- **"Calibration-free" for G and A_p.** Their sizes scale with confidence (Review A F3).
- **A separate source-only stage in GATE.** None exists (Review B F6). It exists in DeepSets T4.
- **The fixed-sum explanation of the ν confound** (Review A F4).
- **The §7 counterexample about the gate-source law** (Review A F7).
- **"Development A had A_p ≥ 0.5 from step 200."** It is contaminated and unsupported (Review C F1).

## Answers to the 24 questions

**Mathematics**

1. **What defines relational complexity?** [proof] Two quantities, in log-linear interaction order:
   - the minimal number of sources that determine trust (gate arity a);
   - the minimal decision order (a + 1).

   They change together across this family's arity contrast, and at a = 2 the gate is itself pairwise, like agreement.
2. **What is the minimum source count for the trust relation?** 2 in A and B; 3 in C and D [proof + num: every proper subset of gate sources has MI 0 with g].
3. **What is the minimal interaction order of the Bayes decision?**
   - 3 for a = 2 and 4 for a = 3 [proof + num].
   - With a noisy gate it becomes 5, the full order [num, verified by Review A].
4. **What are the exact gains?** [num]
   - Gate gain: 0.277 (ν = 1/3), 0.139–0.140 (ν = 0.05).
   - Intermediate gain: 0.025 and 0.076–0.077.
   - The "2×" value contrast holds for Δ_gate (1.98). From the order-1 state the ratio is 1.40, and in accuracy it is 3.9.
5. **Were the comparisons built without shortcuts?**
   - Complexity: yes. Value is matched within 0.09–0.9%, and there is no low-order shortcut [proof + num].
   - Value: only with the ν confound.
6. **What remained confounded?**
   - Gate value against agreement-cue strength and overlap. This comes from the single knob ν, not from a fixed sum. A generator with separate reliabilities can decouple them (Review A).
   - Order-1 accuracy (0.79 against 0.87).
   - The gate's accuracy gain (0.109 against 0.028).

**Learning**

7. **Is source information learned first?**
   - DeepSets T4: yes, a real source plateau in 4/4 [frozen].
   - GATE variants: order-1 accuracy is reached first (t_S 150–175), but agreement reliance is already rising, with no separate plateau [frozen + review].
8. **When does an intermediate stage appear?**
   - In every condition under the frozen rule.
   - It is long when the gate is slow or censored, and only about 3 snapshots in most B models, where the gate follows quickly [frozen].
9. **When is it skipped?**
   - Not in this family at the frozen threshold.
   - XIV's GATE skipped it, but with a different agreement index. That suggests, but does not show, that skipping depends on task construction.
10. **Does value at fixed complexity change the order?**
    - The order of stages: no.
    - Timing at arity 2: later with lower value in 4/4 pairs (post hoc, confounded) [frozen P2 inconclusive].
11. **Does complexity at matched value change the order?**
    - The order of stages: no.
    - The gate is strongly delayed or not acquired within the budget [frozen P1].
12. **Is timing better predicted by complexity, value or signal?**
    - Complexity: the largest effect.
    - Value: a confounded effect at arity 2.
    - Signal: untested (measure withdrawn).
13. **Do abrupt transitions persist across seeds?**
    - Arity 2: 175–325 steps in 8/8.
    - DeepSets T4: 150–250 steps in 4/4.
    - Arity 3: not resolved by the snapshot spacing [frozen].
14. **Are the jumps genuine?**
    - At arity 2 and in DeepSets, yes. They span 7–13 snapshots at arity 2 and 3–5 in DeepSets (50-step spacing), matched by accuracy changes, and accompanied by the gate rising above the cue baseline at the transition [frozen + expl].
    - The sizes of G and A_p are partly confidence-dependent.
15. **Which predictions were falsified?**
    - RACE, as frozen.
    - H2's "no arity effect" and its stage pattern [frozen].
    - H1 survives P1 and P3. At arity 2, P2 leans against H1's auxiliary "no value effect" (post hoc, confounded).

**Mechanisms**

16. **Newly represented or newly usable?** Undecided [expl].
    - Before the transition, no site carries gate information beyond the agreement-cue baseline (0.88 or 0.74).
    - Information beyond the cue and its causal use appear together at the transition, in all 4 acquiring models.
    - The censored model stays at the cue baseline.
17. **What causes the switch?**
    - Observed: gate information appears at the evidence tokens by the layer-2 input, and is read out [expl].
    - Not established: where it is computed, and why it forms when it does [spec].
18. **Does the mechanism explain the differences between variants?** No. Where the gate forms, it forms the same way. Only timing differs [expl].

**Generalisation**

19. **Can T4 be learned from scratch under the tested setup?**
    - DeepSets: yes, 4/4 [frozen].
    - The Transformer was not retried; its earlier record is 0/3 [historical].
20. **What does the Transformer's T4 failure establish?** That this architecture and recipe stalled at the source stage. Not that T4's relation is necessarily learned later.
21. **Does T4 learning support the GATE-derived picture?**
    - The pattern of low-order structure first, then an abrupt relational leap: yes, in another architecture [frozen].
    - The intermediate stage: not tested.
22. **Is a restricted principle justified?** Only a narrow, observational one:
    - In this Transformer and task family, gate-free structure is learned first, and the gate later, abruptly at arity 2.
    - Raising gate arity from 2 to 3 at matched value delays it greatly, or prevents it within about 3000 effective steps.
    - Not a law, and not value-driven [frozen; spec beyond].
23. **Which counterexamples limit it?**
    - XIV's skipped stage.
    - B's 3-snapshot stage, and its flip to RACE at threshold 0.6.
    - Censoring cannot distinguish late from never.
    - Effectively about 4 replicates.
    - One recipe.
    - DeepSets is a different architecture.
24. **What is the next direction?** [judgement]
    - Decouple value from the agreement cue (separate reliabilities, or gate relevance).
    - Arity 4 at matched value.
    - A longer constant-rate budget or a curriculum for arity 3.
    - Cue-controlled probes and floor-conditioned stage rules (COR_FOUNDATIONAL_NAVIGATION_XV.md).

## Post-review corrections (summary)

These come from the 13 findings of Review A, the 13 of Review B and the 10 of Review C. All were accepted, or partly accepted with reasons. None is rejected outright. The frozen verdicts are unchanged; their interpretation is narrowed as above.

**Applied in:**
- OPTIMIZATION_AND_MECHANISM_ANALYSIS.md (§1 withdrawn; probe table with all sites and cue baselines; hypothesis table);
- LEARNING_TRANSITIONS_AND_FROZEN_CONFIRMATION.md;
- RELATIONAL_COMPLEXITY_AND_VALUE_MATHEMATICS.md;
- T4_FROM_SCRATCH_AND_GENERALIZATION.md (reference values 0.742 and 0.580);
- COR_FOUNDATIONAL_NAVIGATION_XV.md;
- an appended, clearly separated note in the freeze record TASK_VARIANT_EXPERIMENTAL_DESIGN.md, whose frozen text is left unchanged.

## Disclosures

- **Development overwrite incident.** It reached step 850, and one development trajectory is partly contaminated. It did not affect confirmation.
- **Mechanism-model selection.** C_s204 and D_s203 were used instead of seed 201, because seed 201 never acquired the gate in C or D.
- **Material written after the freeze, before scoring.** Exploratory and sensitivity scripts, and the mathematics document.
- **Limits of the scorers.** They check structure, not content. Content integrity rests on MANIFEST.sha256 and on Review C's recomputation of trajectory points.
- **Missing records.** There is no peak-memory record for the confirmation runs; the benchmark gave 0.71–0.74 GB.
- **Not run.** No neural runs of the noisy gate; mathematics only.
