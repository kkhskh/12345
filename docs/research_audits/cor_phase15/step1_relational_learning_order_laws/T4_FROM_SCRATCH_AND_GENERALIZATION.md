# T4 from scratch and generalisation (Workstream E)

## 1. Earlier record (primary sources)

Source: `step1_latent_reliability/MAIN_ANSWER.md`, and the checkpoint records in `step1_latent_reliability/checkpoints/main/`.

- **The T4 "M" Transformer from scratch failed.** It did not learn the task in 0/3 seeds at 6000 steps. All seeds stayed at the own-question level (accuracy 0.578), and one post hoc seed was still there at 24,000 steps.
- **DeepSets learned T4 to Bayes in 3/3 seeds.** Accuracy was 0.743 at 6000 steps, taking 183 s per run. Its per-question encoder, mean pool and readout match the factorisation of the sufficient statistic (Theorem S).
- **Reading of the old failure.** The Transformer's stall at the own-question level is *the source stage of this investigation*: the rule that uses only the target question's own reports, P(Y_q | X_q).

## 2. Decision for XV (made at FREEZE XV, before any run)

**No new Transformer T4 attempt.** This was a bounded stopping rule applied in advance:
- the documented 0/3 failure;
- about 50 minutes per 6000-step run on this machine, which would have displaced the primary 16-run design.

**Positive control instead.** Four fresh DeepSets T4 models (seeds 301–304), trained with the identical original recipe and dense snapshots.

**What this means.** It is a cross-*task* observation in a *different architecture*. It is **not** a same-architecture T4–GATE comparison.

## 3. T4-appropriate references and diagnostics

These are T4's own exact references. No GATE number is reused.

- **Bayes:** P(Y_q | episode), from `bayes.episode_posterior`. On the confirmation stream it has accuracy 0.742 (corrected after Review C F9; 0.751 was the development-stream value).
- **Source-stage rule:** the own-question rule P(Y_q | X_q), with no cross-question information. Accuracy 0.580; KL from Bayes 0.273 on the confirmation stream (corrected; 0.588 was the development-stream value).
- **Relational edit.**
  - At another question q′ ≠ q, one source of a random pair is set equal to its partner if the two disagree, or to a different value if they agree.
  - Edits with an undefined posterior are dropped; all 3000 edits on the confirmation stream were valid.
  - **G4** is the slope of the model's change on the Bayes change. The own-question rule gives 0; Bayes gives 1.
- **Competence:** final KL ≤ 0.02 and G4 ≥ 0.6.
  - G4 is attenuated even at near-Bayes KL: development seed 300 reached 0.84 at KL 0.001.
  - So the threshold of 0.6 was fixed from development, before confirmation.
  - Post-review (Review B F11): the attenuation reflects the model's larger error on the edited inputs (0.015 against 0.0013 on unedited inputs, as measured by Review B). It is not a fault in the measure.

## 4. Results [frozen, `logs/score_T4.json`]

**Trainability gate: passed.** All four models are competent: final KL 0.0013–0.0014, G4 0.83–0.84.

| seed | consecutive source-stage snapshots (50-step spacing) | t_G4 | width (G4 ≤ 0.1 → ≥ 0.5) |
|---|---|---|---|
| 301 | 8 | 750 | 200 |
| 302 | 9 | 750 | 150 |
| 303 | 7 | 700 | 150 |
| 304 | 10 | 900 | 250 |

**P6: CONFIRMED.** Every model first reaches the own-question rule (KL to it ≤ 0.03, G4 ≈ 0) and stays there for 350–500 steps. It then acquires the cross-question relational rule abruptly, within 150–250 steps. Development seed 300 showed the same pattern.

## 5. What this does and does not establish (questions 19–21)

19. **Can a T4 model learn T4 from scratch under the tested setup?**
    - The tested Transformer: not attempted again. The record is 0/3 [historical].
    - DeepSets: yes, 4/4 [frozen].
20. **What does the Transformer failure establish?**
    - That this architecture and recipe did not get past the source stage in 6000 (or 24,000) steps.
    - It is consistent with the GATE arity-3 models' censoring, which also stalled at the best rule without the trust relation.
    - It does **not** show that T4's relation is learned "later" in principle. It is an optimisation or trainability failure [historical + inference].
21. **Does T4 learning order (DeepSets) support the GATE-derived picture?**
    - **For the first part, yes:** source stage first, then an abrupt relational acquisition [frozen].
    - **The intermediate stage was not tested.** T4 has no analogue that was measured here. Its natural candidate, "trust the agreeing pair", is harmful in T4 because perfect agreement marks the colluder.
    - **The architectures differ**, so the shared pattern (source stage, then an abrupt relational leap) is a cross-task observation, not a law [frozen + spec].
