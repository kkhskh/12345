# Main foundational result (continuation of the relational-computation atlas)

**What this is.** This is a connected account of what was newly established, corrected or refuted. It is not a paper thesis and does not claim closure.

**Unchanged verdicts.**
- `OBSTRUCTION_READY = NO`
- `PHASE_2 = NO GO`
- every atlas and Step 1 verdict

**Review status.**
- Two adversarial review rounds by separate subagents (R2: four reviewers; R3: two), adjudicated in `reviews/ADJUDICATION_R2.md` and `reviews/ADJUDICATION_R3.md`.
- Every review-driven correction is marked [R] in the documents.

## 1. The one-paragraph account

Ten models solve the latent-reliability task T4. They span pooling, key-tag routing, split representations and rereading. All are close to Bayes on the training generator, and that closeness accounts for everything they share:

- **Intervention signs.** On the filtered PP1 edits, the true effect exceeds the models' summed pointwise margin errors for 98.6–99.7% of edits, and there the signs agree with Bayes (C1). On an edit family with density ratio ≥ 7.7, every sign with |Δm| ≥ 1 is correct. This is an empirical property of pointwise accuracy. The distribution-level theorem we proved (C4) is valid but vacuous at the observed KL.
- **No shared mechanism beyond that.** Below the behavioural level nothing is shared:
  - The seed that looked non-local (seed 12) runs the common routing operation at reduced strength, plus at least three further cross-question channels.
  - Its tag acts partly through values, whereas seed 10's acts purely through routing. Under frozen patterns, the tag ablation removes 0.40 of the relational gain in seed 12 against 0.02 in seed 10.
- **Off the support** (p_c = 0), the implementations diverge, and only there.
  - The divergence correlates with *which agreement statistic* a representation carries linearly: graded counts versus a saturated full-agreement detector.
  - The direction follows architecture (all five DeepSets go global).
  - A frozen internal predictor of the within-Transformer spread **failed** prospectively (P-A1).
- **What the curriculum transfers.** Early layers trained only to compute leave-q pair-agreement counts across questions (P-count) transfer 88% of the curriculum gain. Those counts are the reliability sufficient statistic, but no answers are supervised. Within-question parsing transfers nothing (P-local).
  - In the T1 run, "fast-usable" transferability (under a 1000-step fine-tune) appears in one 1000-step window together with agreement detection. That protocol is confounded with fine-tune speed [R3], so the timing of when the features *exist* is not established.
  - The frozen probe criterion for *which* feature (P-B3) failed, because it probed the source tokens. P-count carries its counts at the readout tokens (decodable at 1.000).
- **At the Bayes level,** the decision-relevant semantics of agreement were mapped across copying, collusion, biased errors, mixtures and unknown groups, including exact nonidentifiability lines.
- **O/D.** Its unweighted discrepancy never strictly reverses the Bayes order exactly under a weighted-voting coincidence condition (reviewer-proposed; the "only if" part is unverified). It fails for zero-weight and copying sources, which are the relations this family is about.
- **A second family.** The GATE task has a relation among *non-answer* sources (a modular constraint, i.e. equality after a recoding) that gates which answer sources count. There the model matches Bayes. The relation is localised to evidence tokens and then to the readout residual. But the last-layer attention, although gate-dependent, is causally unnecessary [R3]. So relation-gated *routing* does not recur as the causal operation.
- **A third family.** A two-hop compositional relation (TRANS) was not learned in a truncated 4000-step run that was still in a learning transition.

## 2. Answers to the mandate's questions (§21), with the evidence grade

**Labels.**
- [C] prospectively confirmed;
- [F] prospectively falsified;
- [T] theorem;
- [N] exact numerics;
- [E] post hoc empirical.

| # | question | answer | grade |
|---|---|---|---|
| 1 | Which PP1 edits were valid, under which support notion? | 93,570 of 135,000 are in S1 (Bayes defined). 90,000 are in S2 = S3 (possible for the *same* world). 3,570 are in S1 \ S2. 41,430 are off-support. The atlas "on-support" meant S1 | [N] |
| 2 | When does approximate Bayes guarantee intervention signs? | Pointwise: \|Δm\| > \|e(x)\| + \|e(x′)\| (C1–C3). Distributional: C4 needs a density ratio and gives vacuous bounds at ε ≈ 10⁻³. Accuracy (CX1) and ECE (TEMP) guarantee nothing. In practice the coverage of C1 is 98.6–99.7% | [T, N, E] |
| 3 | What does the PP2 shift reveal about internal computation? | Nothing on the support: every model follows training Bayes there, confirmed on held-out models. Off support, it reveals which agreement statistic is carried (graded vs saturated), which is descriptive. Its frozen internal predictor failed | [C (support), F (P-A1), E] |
| 4 | What early-layer feature makes the curriculum work? | Cross-question aggregation of within-pair agreement (the leave-q counts, which are the reliability sufficient statistic), with no answer supervision (P-count 0.723 vs T1 0.742). It is not within-question parsing (P-local 0.576) and not generic initialisation (random 0.554). It is carried at the source tokens (T1) or the readout tokens (P-count). When it emerges is not established, because the dynamics protocol is confounded (R3) | [C (B1), F (B3), E] |
| 5 | What does seed 12 compute, and why did localised interchange miss it? | A′ routing (head 3) plus value reads of other questions (head 0), other questions' readout states, and its own readout residual. A 2-D tag interchange leaves those channels at base values. Mean-replacement necessity is partly LayerNorm-mediated and not tag-specific in magnitude | [E] |
| 6 | What survives exact head splitting and valid reparametrisations? | The quotient-game (Owen) attribution over exact pattern classes is invariant under exact splits (R1). Within-head gauge is trivially invariant at head-output sites. Pattern-changing function-preserving moves remain open (OF-5) | [T, N] |
| 7 | Are there nontrivial causal properties shared by all implementations? | **None found below the behavioural level.** The nearest is a 2-D source-token subspace whose ablation moves outputs toward the own-question posterior in every Transformer seed tested. Its necessity is matched in magnitude by a variance-matched control in seed 10, and its sufficiency is class-dependent | [E] |
| 8 | What is the strongest meaningful equivalence relation? | Functional equivalence on the support plus pointwise-margin sign agreement. Both come from each model's own closeness to Bayes (K1 in its empirical form) | [T, E] |
| 9 | Which properties are invariant and independently identifiable? | The input–output law on the support; quotient-game attributions under exact splits; routing patterns up to L1 | [T] |
| 10 | Which natural attributions are fundamentally non-unique? | Head-level attributions (I5, R2-c); query/key side (I6, conjecture); "necessity" of a subspace by magnitude (LayerNorm- and control-dependent); probe-based "codes" off support | [T, E] |
| 11 | When are relational statistics sufficient for decisions? | Counts plus own reports are sufficient under (U) and (O). (O) is needed (spread 0.44 with two answer-dependent groups) | [T, N] |
| 12 | When do they discard indispensable candidate information? | Always without X_q (J1). Also when two answer-dependent groups exist (cross-agreement), and under OD1′-violating weights for unweighted discrepancy | [T, N] |
| 13 | What assumptions determine the sign of a relation-conditioned margin response? | The role semantics and the candidate-conditioned configuration posterior (D). Without semantics there is no sign (D2). The interior support rule is guaranteed under T4 (Proposition SR, sketch) and fails at the full-agreement boundary (0.11; the paradox of unanimity) | [T, N] |
| 14 | Which generators give monotone, non-monotone or irrelevant agreement evidence? | Monotone without a higher-agreement role (U2). Non-monotone with colluders or copiers above a_rel. Non-quasi-concave for unions of roles (mixtures over α, 45/49 contexts). Exchangeable at ρ*(α) (I2). Biased errors split the one-vote boundary | [T, N] |
| 15 | Impossibility boundaries from unknown groups and dependencies? | The full T1 pairing is ≤ 1/3 identifiable (I1′). The role swap is exact at ρ* (I2). Overlapping groups under a one-block class are actively mis-explained | [T, N] |
| 16 | Why do different successful models develop different implementations? | **Unknown.** The architecture fixes the off-support direction. No measured internal quantity predicted the within-Transformer spread (P-A1 failed) | [F, open] |
| 17 | What does curriculum pretraining transfer? | See question 4 | — |
| 18 | Which mechanism properties survive changes in architecture, task, source organisation and error model? | Few. Causally used last-layer routing exists in latent-reliability A′ seeds. It does not exist in GATE (the pattern is gate-dependent but unnecessary), in DeepSets (no attention), or for TRANS composition (not learned). What recurs is early computation of a relation-derived variable carried at token residuals. Unknown-group learned mechanisms remain absent | [E] |
| 19 | Does the characterisation explain other relationship families? | Partly at the Bayes level. GATE is solved at Bayes level, through a mechanism different from latent-reliability routing. TRANS composition was not learned (truncated run). Answer-discrepancy O/D cannot express gating by non-answer sources | [E] |

## 3. What is new, and what is only re-labelled

See BASELINES_AND_NOVELTY.md for the full audit. No new mathematical theory is claimed.

**Known results, used as tools:** C1–C5, R1 (the Owen quotient game), V1 and OD1′ (weighted voting), the paradox of unanimity, latent-class identifiability.

**The contributions are:**
1. corrected scopes for the atlas's own theorems;
2. evidence separating pointwise Bayes-closeness from mechanism;
3. an operation-level account of seed 12;
4. a prospective dissection of what the curriculum transfers (P-B);
5. a failed prospective internal predictor (P-A1), reported as such;
6. a second relation family (GATE) with a localised relation variable, and a falsified recurrence of the routing operation;
7. a negative but inconclusive result on composition (TRANS, truncated run).

## 4. What is retracted or downgraded relative to the atlas, and within this continuation

**Atlas claims downgraded or corrected.**
- PP1 as an "implementation-independent causal law" is downgraded to a consequence of pointwise Bayes-closeness.
- The atlas errata (mean-ablation was run; pattern target specificity is uninformative) are in REPRESENTATION §5.

**Claims of this continuation retracted after review.** These are in git history:
- the "ρ = 1 singularity" (NaN artefact);
- the K2 abstraction counterexample;
- OD1's "iff trivial semantics";
- "biased errors do not move the one-vote boundary";
- "C4 explains PP1";
- "bilinear content × tag";
- the PP2 "colluder code enforces the configuration constraint" reading;
- the S1 \ S2 transformer-routing hypothesis;
- the GATE "three-stage path ending in head-0 query-side routing" and "same operation as latent reliability" (round R3);
- TRANS as a "neural, not statistical" failure (round R3);
- the PP3 "transfers without reliability supervision", "probes are not the transferred variable" and "transfer emerges abruptly" readings (round R3).

## 5. Where to go next

OPEN_FOUNDATIONAL_MAP.md lists the open problems with their dependencies. The highest-leverage items are:
- **OF-4:** a probe-free causal test of graded vs saturated agreement features as the cause of the extrapolation direction;
- **OF-2:** finer-grained dynamics with many seeds, to predict implementation class;
- **OF-10:** a curriculum for TRANS, testing whether composition is learnable;
- **OF-8:** a learned mechanism under unknown groups. T4-perm is the identifiable first target.
