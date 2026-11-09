# R3 review: learning dynamics, feature identification, prospective-prediction hygiene

Reviewer perspective: learning dynamics (also covering feature identification and P-A/P-B hygiene).
Scope: TRANSFERRED_FEATURE_IDENTIFICATION.md (TFI), PROSPECTIVE_PREDICTIONS.md P-A and P-B (frozen text + outcomes), EXTRAPOLATION_AND_INTERNAL_STRUCTURE.md §5; code `src/tasks_ext.py`, `src/train_tasks.py`, `scripts/run_job.py`, `scripts/pp3_probes*.py`, `scripts/pp3_dynamics.py`, `scripts/pa_evaluate.py`; logs and checkpoints; atlas PP3.

Small checks run by the reviewer (scripts in `/tmp/r3check/`, not added to the package; each < 4 min, one process at a time):
- `suff.py`: is the P-count target a sufficient statistic of the leave-q configuration posterior? What accuracy can the P-group target support?
- `readout_probe.py`: the pp3 probes repeated at the **readout tokens** (positions 36–41) of `resid_pre2`, the same site and data streams as `pp3_probes.py`.
- `ft1000.py`: the dynamics fine-tune protocol (1000 steps, eval 5000 episodes of stream 6,300,000) applied to `pre_pgroup` and `pre_pcount`.
- `norms.py`: parameter norms of embeddings and layers 0–1 for every arm.

Severity: **Major** = a headline conclusion changes; **Moderate** = a claim needs rewording or a caveat; **Minor** = presentation.

---

## Summary

- **Hygiene: clean.** P-A and P-B verdicts follow the frozen text exactly. Frozen text was not edited. The evaluation and probe scripts were committed before the outcomes and have not changed since. I found no tuning after outcomes.
- **Two Major problems in interpretation.**
  1. The P-count target is the *exact sufficient statistic* of the reliability posterior. The claim that P-count transfers "without any answer or reliability supervision" therefore overstates what was shown.
  2. In P-count, the transferred aggregate *is* linearly present: at the readout tokens, it decodes perfectly. "Probes are not the transferred variable" is an artefact of probing at the wrong site.
- **One Major confound in the dynamics protocol.** Under the 1000-step snapshot fine-tune, P-count (0.654) and P-group (0.577) both *fail*, though both reach the threshold at 2000 steps. So the dynamics series measures how fast transfer happens, not whether the features are there. That undercuts "transferability emerges abruptly" and the "nothing transfers before 2000 steps" reading.

---

## Findings

### F1. P-count supervises the exact sufficient statistic of π_{−q} [Major]

**Evidence (`suff.py`, 20,000 episodes).** For both T4 (p_c = 1, ρ = 1) and T1 (p_c = 0), the exact leave-q configuration posterior π_{−q} is a deterministic function of the leave-q agreement-count vector (n_0^{−q}, n_1^{−q}, n_2^{−q}). The maximum deviation within a count cell is 2.7e-15 (T4) and 3.5e-15 (T1).

The reason is in the model itself. Per-question likelihoods depend on the reports only through the agreement indicators, because the rel, coll and noise pair likelihoods marginalised over y are functions of `xa == xb`. This was checked against the generator in `gen_family.pair_lik`.

**Consequence.** The P-count target `T = A.sum(1) − A` (`src/tasks_ext.py`) is exactly the sufficient statistic of the latent reliability (and colluder) posterior, in both the pretraining task and the fine-tuning task. What it does not supervise is the final step: combining π_{−q} with the own-question likelihood.

**Claims that need rewording.**
- TFI §2 finding 2, "without any answer or reliability supervision".
- The mandate answer, "It needs no reliability or answer supervision".

These should say "without answer supervision and without the reliability→answer combination, but with a target that is a sufficient statistic of the reliability posterior". The result is still interesting, since a label-free statistic suffices. But it is close to "supervising the posterior's sufficient statistic transfers", which is expected.

### F2. P-count's aggregate is linearly decodable, at the readout tokens [Major]

**Evidence (`readout_probe.py`).** Same probe recipe as `pp3_probes.py` (logistic or ridge, streams 9,100,000 / 9,100,001), applied at the 6 readout tokens of `resid_pre2`, with one probe per group:

| arm | count_loo acc, g0 / g1 / g2 (majority ≈ 0.37–0.39) | P_coll R², g0 / g1 / g2 |
|---|---|---|
| P-control | 0.377 / 0.365 / 0.385 | −0.01 |
| P-local | 0.388 / 0.378 / 0.388 | 0.05–0.08 |
| P-group | 0.401 / 0.464 / 0.489 | 0.23–0.32 |
| **P-count** | **1.000 / 1.000 / 1.000** | **0.93–0.95** |
| T1 s0 | 0.543 / 0.796 / 0.472 | 0.43–0.64 |

**Consequence.** Layers 0–1 of P-count write the exact leave-q counts into the readout tokens, which is where the pretraining loss read them. This is the input that the re-initialised layer 2 receives.

The following claims should be withdrawn and replaced with this measurement:
- TFI §2 finding 4 ("the aggregate is mostly not linearly present … may sit at readout tokens or in a nonlinear form");
- §4 "Where the P-count aggregate lives … not run";
- "Its exact representation is not identified".

**What the measurement shows.** The arms carry the transferred variable at *different sites*: T1 at the source tokens (count 0.895), P-count at the readout tokens (1.000). That is why the single-site criterion of P-B3 could not work.

**A post hoc suggestion, not to be scored.** Take, for each arm, the larger of its source-token and readout-token count decodability. This orders the arms as P-control ≈ P-local < P-group < T1, P-count, which matches the transfer ranking except that P-count sits slightly below T1.

### F3. The 1000-step dynamics fine-tune is strongly confounded with fine-tune speed [Major]

**Evidence (`ft1000.py`).** The dynamics protocol (1000 steps; eval on 5000 episodes of stream 6,300,000; own-question 0.573; same 0.6595 threshold), applied to the pretrained arms:

| arm | 1000-step fine-tune | 2000-step fine-tune (`ft_*.json`) |
|---|---|---|
| P-group | **0.577 (fail)** | 0.669 (ambiguous success) |
| P-count | **0.654 (fail)** | 0.723 (success) |

**Training-loss evidence.**
- In the 1000-step P-count run, the loss sits on a plateau (≈1.09 → 1.04) until about step 400 and escapes only around step 600–700.
- The 2000-step `ft_pcount` and `ft_pgroup` histories show the same delayed escape.
- T1 features fine-tune almost at once: in `ref_T1`, the loss is 0.58 by step 400.

**Consequence.** Under the dynamics protocol, "transfer" means "transfer within 1000 steps". Features that are useful but slower to exploit score as failures. So:
- The claims "Before the window, nothing transfers" and "Transferability emerges abruptly, in a single window (1000–2000 steps)" (TFI §3) are not established. The step-1000 snapshot (0.562) could plausibly pass under the 2000-step protocol, as P-count and P-group do. This was not run because a 2000-step fine-tune exceeds this review's compute limit.
- "Abrupt" is also not resolvable with snapshots at 1000 and 2000. That window spans a third of training, and T1 accuracy itself rises 0.650 → 0.735 within it.
- The 2000-step protocol has the same weakness relative to the atlas 3000-step protocol. Atlas (vii) used 3000 steps (`step1_relational_computation_atlas/scripts/run_training.py`). `ref_T1` shows that T1 is insensitive to the cut (0.742 at both lengths). That says nothing about the slow arms. P-group, still improving at step 2000 (loss 0.83 → 0.76 over steps 1000–2000), might pass clearly at 3000, which would weaken the "graded with aggregation" reading.

**Recommendation.**
- Re-run the step-1000 and step-2000 snapshots with the 2000-step fine-tune, or report transfer as an area under the fine-tune curve.
- Rerun P-group at 3000 steps.
- State in the dynamics section that the 1000-step protocol misclassifies P-count.

### F4. P-local is a valid discriminating control, with caveats [Minor/Moderate]

**What checks out.**
- `sample_plocal` draws n·Q independent Q = 1 T1 episodes and reshapes them, so each question has its own reliable group, and π_{−q} is the prior by construction. Cross-question sharing is indeed useless.
- Its pretraining accuracy of 0.649 matches the T1 own-question Bayes of 0.655 (`suff.py`).

**Caveats.**
1. P-local removes both of the atlas's alternatives (b), the cross-question attention scaffold, and (c), the aggregate, at once. P-local versus P-count therefore separates "within-question parsing" from "(b) or (c)", but not (b) from (c). P-group is the only arm that partly separates them (see F5).
2. P-local was pretrained for 6000 steps against 3000 for P-count and P-group. This is unlikely to matter, since it would favour P-local, but it should be stated as an asymmetry.
3. Initialisation and scale alternatives: `norms.py` shows the embedding norms (Eval 7.5–7.8, Esrc 9.9–10.2, Eq 10.1–10.4, Eread 4.5–4.6) and the layer 0/1 norms (26.4–27.5) matched across all five arms. All arms also share the same seed-0 initialisation key sequence and the same seed-7777 re-initialisation of layer 2. The authors' exclusion of optimisation and initialisation scale effects is therefore supported, but by P-local, not by P-control. TFI §2 finding 3 should cite P-local as the generic-pretraining control: a random frozen network controls for "any frozen features", not for "any pretraining".

### F5. P-group is not difficulty-matched, and its transfer exceeds its target's information [Moderate]

**Difficulty.**
- P-group reaches auxiliary loss ≈ 6e-4 by step 900. P-count sits on a plateau (loss 1.65) until about step 600–900 and solves the task by step 1200.
- Both saturate (aux accuracy 1.000) within 3000 steps. So "comparable steps" holds, but the computational demand differs by design.

**Information content (`suff.py`).** The best T4 plug-in that uses only question q+1 (exactly the information in the P-group target) achieves 0.622. That is *below* the 0.6595 threshold. P-B2 therefore predicted transfer beyond the target's information content, so it was a riskier prediction than P-B1.

**Outcome.** P-group's 0.669 > 0.622. The re-trained layer 2 must therefore aggregate over more than one question, using the agreement or attention structure that P-group's layers expose. This fits the (b) "cross-question scaffold" explanation at least as well as the "graded aggregation" explanation. TFI §2 finding 2 ("scales with how much cross-question agreement aggregation the pretraining demanded") rests on 3 points with one seed each, and should be flagged as post hoc and not separated from (b).

### F6. n = 1 seed per arm is more fragile than the eval SE suggests [Moderate]

- The evaluation SE is small: 20,000 episodes, so roughly 0.002.
- But the fine-tune outcome depends on a plateau-escape time (F3), and the pretraining of P-count has its own plateau escape. Both are seed-stochastic events.
- The 1000-step vs 2000-step comparison shows accuracy swings of 0.07–0.09 depending on whether the escape happens inside the budget.

**Claims that need a seed replicate.**
- The ordering P-count (0.723) < T1 (0.742).
- P-group's +0.009 margin.
- "88% of the T1 gain".

TFI §4 acknowledges the P-group case. The "strongest identification here" wording in §2 should carry the same caveat.

### F7. P-B verdicts were applied exactly per the frozen text [Pass; Minor notes]

**Freeze integrity.**
- `git diff 07f7752 a5b9a1a -- PROSPECTIVE_PREDICTIONS.md` contains only appended outcome lines.
- `pp3_probes.py` (first committed 9a9de43, 02:20), `pp3_probes_lib.py` and `pp3_dynamics.py` (14660e0, 02:45) were committed before the freeze (04:08:32) and have not changed since.
- The output files are dated after the freeze: probes 04:43, dynamics 05:22, ft_pcount 04:23, ft_pgroup 04:42.

**Verdicts.**
- **B1 PASS** (0.723) is correct.
- **B2 AMBIGUOUS** (0.669, within 0.02) is correct per the frozen rule.
- **B3 FAILED** is correct.
  - The failing margin is small (0.893 vs 0.90), and the frozen text gives no ambiguity band for the probe threshold, so the strict reading is right.
  - The failure is also substantively robust. P-local (0.876, fail) and P-count (0.893, success) differ by 0.017 in agree_q but by 0.147 in transfer, so agree_q does not discriminate whatever cut is used.
- **B4 PASS** is correct per the text. However, B4 had low prior risk: every probed feature (agree_q, count, perfect, P_coll) and T1 accuracy jump inside the same coarse window, and given F3 the transfer column is protocol-biased. The B4 pass should be described as non-diagnostic, not as weak support for agree_q.

**Minor notes.**
- The freeze note says `pre_pcount` was "at step 600". From the job's 912 s duration and its 04:18:47 end time, the step-900 log line (loss 0.566, the plateau escape) was written about 20 s before the commit. This is immaterial to the predictions, which concern transfer, but the note is slightly stale.
- I cannot rule out an unlogged probe run on the three arms that existed between 02:20 and 04:08. The P-B3 cut of 0.90 lies between P-local's later-observed 0.876 and T1's 0.997. No file evidence of such a run exists (`git grep` of every pre-freeze commit finds no agree_q values). I record this as unverifiable, not as a finding against the authors.

### F8. P-A verdicts were applied exactly; the EXTRAPOLATION §5 wording slightly overreaches [Minor/Moderate]

**Integrity.**
- The P-A frozen text (lines 1–58) is byte-identical to `dc1dd3b`.
- The sha256 values of `src/pp2_sig.py` (2b91afcd…) and `logs/pp2_frozen_fits.json` (498c0e60…) match.
- `pa_evaluate.py` was committed at 02:20, before the first held-out model finished (03:04), and has not changed since.
- The held-out protocols match the frozen text: TF T1 6000 steps then T4 3000 steps; DeepSets 6000 steps.

**Verdicts.** The code's verdict logic (A1 < min of B1–B7, including B6 as the mean over random draws) reproduces P-A1 FAIL, P-A2/3/4 PASS. The post-freeze R2 annotation was added before evaluation and changes no verdict.

**Overreach in §5.**
- "Within Transformers the spread is predicted by A2 … held-out MAE 0.03" and "H-none is rejected for the behavioural level" rest on n = 2 held-out Transformers. The held-out MAE of 0.026 is pooled with the two easy DeepSets, and the per-model Transformer errors are 0.029 and 0.059.
- A2 is a near-proxy (development Spearman 0.988) measured off support.
- Suggested wording: "consistent with H-none being false at the behavioural level (2 held-out TF, correct order)".

**On P-A1.** Its failure margin against B1 (0.126 vs 0.091, n = 4) is, as the annotation says, weak evidence. §5 states this correctly.

### F9. Smaller interpretive points [Minor]

- **TFI §3, the perfect_loo feature.** Calling it "colluder-relevant … even though T1 contains no colluders" over-reads it. In T1, n^{−q} = 5 is simply the top count bin, and its linear decodability follows from count decodability, which rises at the same time (0.39 → 0.60).
- **TFI §3, counts after saturation.** "The counts at source tokens keep improving after transfer has saturated … not the limiting quantity" is drawn from a protocol (F3) with a ceiling near Bayes (0.742–0.745) and a single run. A flat transfer curve near the Bayes ceiling cannot detect a dependence on further feature improvement.
- **Probe recipe.** `pp3_probes.py` fits a single linear probe shared across all 36 positions. A per-position probe could raise the source-token numbers for P-count. This is worth one sentence in the protocol description.

---

## Requested changes (priority order)

1. Replace TFI §2 finding 4 and §4 "where the P-count aggregate lives" with the readout-token measurement (F2). Revise the mandate answer to say that the transferred variable in P-count is the exact leave-q count vector at the readout tokens, which is a sufficient statistic of π_{−q} (F1).
2. Reword "no reliability supervision" (F1).
3. Add the F3 protocol-sensitivity result to the dynamics section. Withdraw "abrupt" and "nothing transfers before 2000", or re-run the snapshots under the 2000-step protocol. Mark the B4 pass as non-diagnostic.
4. Cite P-local, not the random network, as the control for generic pretraining effects, and add the norm match (F4). Flag that P-group's target supports only 0.622 and that (b) is not separated from (c) (F5).
5. Add a seed caveat to "strongest identification" (F6). Soften the "within Transformers" and "H-none rejected" wording in EXTRAPOLATION §5 (F8).
