# Review B: learning dynamics and mechanisms (COR Investigation XV)

**Reviewer.** An independent adversarial agent review. This is not human peer review.

**Scope.** I examined:
- the trajectories in `confirm_outputs/traj_*.json`;
- the scorers and the exploratory logs;
- the code in `scripts/` and `src/`;
- the four documents: LEARNING_TRANSITIONS_AND_FROZEN_CONFIRMATION.md (**LT**), OPTIMIZATION_AND_MECHANISM_ANALYSIS.md (**OM**), T4_FROM_SCRATCH_AND_GENERALIZATION.md (**T4**), and TASK_VARIANT_EXPERIMENTAL_DESIGN.md (**TD**).

**Checks run.** All were light CPU checks: no training, and no package file was modified. The scratch scripts lived in `/tmp/revB/` and are not part of the package. Each check is described below so it can be reproduced.
- **C1.** Stream identity of the paired generators, over all 4000 training batches of seed 201.
- **C2.** A_p and G of synthetic rules, computed exactly through `xvdiag.Diag` on the frozen evaluation stream.
- **C3.** Bootstrap of A_p, R², and KL to the gate-free and order-1 references, at B-stage checkpoints (forward passes).
- **C4.** A null calibration of the `xv_gradsig.py` SNR estimator: 64 batches, true-vs-null against null-vs-null, at initialisation.
- **C5.** The Bayes-optimal accuracy of predicting g from the evidence tokens alone (the "agreement ceiling").
- **C6.** Linear probes for g stratified by evidence agreement, on D_s203 checkpoints 1500, 2000 and 4000.
- **C7.** Diagnostics of DeepSets T4 G4: concentration of the edit responses, and KL on edited inputs.
- **C8.** Cumulative learning-rate reparametrisation of the P1 ratios.

Severity scale: **CRITICAL** (a stated conclusion is wrong) > **MAJOR** (a conclusion is materially overstated or rests on an invalid measurement) > **MODERATE** > **MINOR**.

---

## Findings

### 1. [CRITICAL] The gradient-signal SNR is at its own noise floor, so §1 of OM has no measured signal to interpret

**Evidence: the estimator.** In `xv_gradsig.py`:
- `signal = ||mean_b D_b||` and `noise = sqrt(sum var(D)/B)`, with B = 16.
- When the true expected gate gradient is zero, E‖mean D‖² = tr Σ / B. So **SNR ≈ 1 by construction**, not 0.

**Evidence: calibration (C4).** I compared true-vs-null with null-vs-null, where the second has a true μ = 0 by construction. The null-vs-null 16-batch SNR distribution is:

| configuration | mean SNR under the null | 5–95% range |
|---|---|---|
| a = 3, seed 201 | 0.99 | [0.68, 1.43] |
| a = 2, seed 201 | 1.02 | [0.70, 1.43] |
| a = 2, seed 202 | 0.99 | [0.76, 1.29] |

- The gradient differences have an effective dimension of only about 6–11, which is why the spread is wide.
- **Every reported SNR (0.75–1.54) lies inside or at the edge of this null band.**
- A debiased ‖μ‖² estimate is indistinguishable from the null-vs-null value. For example, at a = 3, s201: 4.2e-5 against noise/B of 2.3e-4, while the null-vs-null value is −2.0e-5 / 4.9e-5.

**Consequences.**
- The arity-3 values (0.75–0.89) sit *below* the null centre. A value below 1 cannot mean "less signal". It is sampling or structural variation of a statistic whose expectation is about 1 with no signal at all.
- `gate_frac` has the noise floor in its numerator, so it is equally uninterpretable.
- The Spearman −0.80 (p = 2·10⁻⁴) is inflated further. A/B and C/D share the initialisation, the training stream and the gradsig RNG (Finding 4), so the 16 values are about 8 near-duplicated measurements.

**Correction.**
- Withdraw "the initial gate SNR tracks the arity" and the Spearman table.
- State instead that the estimator did not detect a gate-attributable expected gradient at initialisation or at t_S in any model.
- If P5 is revisited:
  - report the debiased ‖μ‖² with a null-vs-null control, and use far more batches;
  - note that a first-order expected-gradient measure is not the quantity that governs escape from a higher-order saddle, so H3 is not tested by it even with a perfect estimator.

### 2. [CRITICAL] The D_s203 "information present before use" exception is the agreement cue, not g

**Evidence: the probe ceiling is not 0.5 (C5).** The probe chance level assumed in OM (0.5) is wrong for the evidence and readout sites. Pair agreement (x₃ = x₄) predicts g with Bayes accuracy:
- **0.883 at ν = 0.05** (A, D);
- **0.741 at ν = 1/3** (B, C).

The full evidence triple gives the same values.

**Evidence: the readout probe sits at that ceiling.** D_s203's readout probe (pre2@R) scores:
- 0.87 at step 2000;
- 0.88 at step 4000, after complete gate use.

It never exceeds the agreement ceiling.

**Evidence: stratified probe (C6).** Within each agreement stratum, pre2@R accuracy is at or below the majority rate at every checkpoint:

| step | agree stratum (majority rate 0.93) | disagree stratum (majority rate 0.85) |
|---|---|---|
| 1500 | 0.93 | 0.82 |
| 2000 | 0.92 | 0.83 |
| 4000 | 0.92 | 0.85 |

- An agreement probe at the same site scores 1.00.
- The readout therefore carries agreement and **no g-specific information at any time, including after the transition**.
- g-specific information appears only at E2 / post2@E at step 4000 (within-stratum 0.99–1.00).

**Evidence: the same confound elsewhere.** A_s201 post2@R = 0.85 at step 625 (follow 0.00; the ceiling is 0.883). OM did not report this site.

**Correction.**
- Delete the "1 of 4 models has g before use" exception.
- Report every probe against the agreement ceiling, or stratified by agreement, with a label-permutation null per site.

### 3. [MAJOR] Several mechanism "use" and "route" claims are tautological or mislabelled

**(a) "Use G1" is not evidence of a route.** It patches the gate tokens at the *layer-1 input*, which is the token embedding. That is just the input edit itself, so it equals following plus baseline by construction (0.87/0.90, 0.97/0.98, 1.00/1.00, 0.54/0.51).
- In the same way, `resid_pre1@E` and `resid_pre1@R` patches are no-ops: the donor's evidence and readout tokens are identical. Their post-transition value of 0 just restates that.
- So "the gate is computed at the gate tokens by the layer-1 input" is **not shown**.
- What the data do show is non-trivial and survives: by the layer-2 input, the flip-relevant information sits at the evidence tokens, not at the gate or readout tokens. pre2@E transfers about the full effect; pre2@G and pre2@R transfer 0.00–0.05 after the transition.

**(b) The baselines are mislabelled.**
- The "use" metric counts P(patched argmax = Bayes-flipped argmax). Its no-information value is P(original argmax = flipped argmax), and that value changes as the model improves (0.04 → 0.00 in A).
- **D_s203's baseline is about 0.03**, read from the pre2@G and pre2@R patches at step 2000. It is not 0.30. The OM table's "0.25 (baseline)" for D_s203 at step 2000 is actual gate use: follow there is 0.28.
- Report a transfer measure conditioned on original ≠ flipped.

**(c) The C_s204 sequence has no pre-transition checkpoint.**
- The selected snapshots are 2250, 2500 and 4000, because snapshots are 250 steps apart after step 2000.
- At 2250 following is already 0.25 and use E2 is 0.48, against a baseline of about 0.25.
- So C cannot support "absent before, formed during". The dense step-2000 snapshot existed but was not used.

**(d) Probe values below chance.**
- Values of 0.00–0.04 at pre1@G (a = 2) and 0.28–0.40 at E sites show that the held-out-x₀ probe anti-generalises systematically. "Chance 0.5" is not the null.
- A_s201 shows use E2 = 0.90 while probe E2 = 0.58. That is more consistent with mean-pooling over the three evidence tokens cancelling a token-specific code than with "use precedes representation".

**Correction.** Rewrite OM §2 to:
- keep "by the layer-2 input, gate-dependent information is carried at the evidence tokens (4/4 analysed models)";
- keep "no evidence of g-specific information before use at the probed sites (A, B, D after stratification; C unresolved)";
- drop the layer-1 "computation" claim, the D exception and the "probe after use" ordering claim, or re-test them with per-token, stratified, permutation-nulled probes.

### 4. [MAJOR] Verified: seed-paired B/C and A/D see identical evidence streams. The consequence is about 4 independent replicates, not 8 or 16

**Evidence (C1).** Over all 4000 training batches of seed 201, B against C and A against D have **100% identical** Y, g, x₀ and x₃–x₅. Only x₁ and x₂ differ: about 68% of entries.
- A against B is also nearly identical: x₃/x₄ match in 85.8% of entries, because only the noise pair under g = 0 uses ν.
- The initialisation is shared within a seed.

**What the documents get right.** This is a strong common-random-numbers control for the arity contrast.

**What follows from it.**
- **(i)** "t_S and t_I identical by seed" is a design consequence, not an empirical finding. The trajectories are close but not identical: at step 150, B_s201 has KL 0.336 and C_s201 has 0.354.
- **(ii)** P3's B and C calls (and A and D) are the *same* pre-gate trajectory scored twice. Given the shared trajectory, "B shows the stage" reduces to "t_G(B) came after the shared t_I".
- **(iii)** P1's "8/8" is 4 seeds × 2 positively coupled contrasts.
- **(iv)** P3's "16/16" is about 8 distinct pre-gate trajectories.
- **(v)** The bootstrap over 8 pairs treats coupled pairs as exchangeable. It is not a valid interval even descriptively.

**Correction.**
- Report counts as "4/4 seeds (each in both ν arms)".
- Resample seeds, not pairs.
- Describe P3 as 2 independent pre-gate trajectories per seed, truncated at different gate times.

### 5. [MAJOR] The shared trajectory favours a race or parallel reading over a strict "complexity-first" sequence

**Evidence.** In the shared B/C trajectory, A_p climbs slowly:
- from 0.5 at steps 350–475;
- to 0.9 at steps 1500–2000, in C.

B acquires the gate while A_p is only 0.51–0.62 (C3: KL to the gate-free rule 0.011–0.019, KL to order 1 0.031–0.036). That is, the gate overtakes an agreement cue that is roughly half learned.

**B's stage.** The doc reports B's stage as lasting "150–325 steps before the gate". That figure is t_G − t_I. The actual run of A_p ≥ 0.5 with G ≤ 0.1 is:
- 3 snapshots (75 steps) in B_s201, B_s203 and B_s204;
- 12 snapshots in B_s202.

In B_s201 and B_s204, the KL already falls below the gate-free floor while G ≤ 0.1 (steps 400 and 550–575). Gate use has started before the stage window closes.

**What this means for P3.** The H1 and RACE patterns differ only in B. B is decided by A_p crossing 0.5 for 2–3 snapshots: the verdict flips at threshold 0.6, and the development seed had a peak of 0.41. The verdict therefore rests on a threshold coincidence of two concurrent learning curves, not on a qualitative difference.

**Correction.**
- Report P3 as "H1 at the frozen threshold; H1 and RACE are not separated by these data".
- Replace "lower-order relations are learned first" with "the lower-order cue's *onset* precedes the gate. Its completion need not, and in B it does not."
- Correct the "150–325 steps" description.

### 6. [MAJOR] The GATE "source stage" is not a stage

**Evidence.**
- t_S is defined by accuracy alone. But order-1 accuracy equals gate-free accuracy (0.875 in A/D, 0.796 in B/C), so t_S cannot separate source-only from agreement use.
- At t_S, A_p is already 0.11–0.28.
- The number of snapshots with source-level accuracy, G ≤ 0.1 and A_p < 0.3 is:
  - 0–2 (at most 50 steps) in all A and D models;
  - 1–6 in B and C.
- KL passes through the order-1 level without pausing. In A_s201, KL is 0.261 at step 150, 0.203 at 175, and 0.201 at 200, against an order-1 reference of 0.217.
- t_S (150–175) also falls inside the 200-step learning-rate warm-up.

**Correction.**
- LT §3 Q7 should read: "Source-level accuracy is reached slightly before substantial agreement use; there is no plateau at the order-1 rule."
- The navigation document's "Source stage first: holds in all 20 models" holds as a plateau only for DeepSets T4 (Finding 11). For GATE it is an onset ordering within about 50–150 steps.

### 7. [MAJOR] Abruptness at arity 3 is a snapshot-density artefact, and the "7–30 snapshots" claim is false

**Evidence.**
- Snapshots are 250 steps apart after step 2000.
- C_s204's transition (G 0.11 → 0.78; KL 0.227 → 0.022) happens **within one 250-step interval** (2250 → 2500). Its "width 750" spans only 4 snapshots.
- D_s203's transition is partly dense (G 0.10 at 1925, 0.26 at 2000) and partly sparse.

**Wrong claims.**
- LT Q14, "25-step snapshots resolve every transition into 7–30 snapshots", is false for both successful arity-3 models.
- LT Q13, "arity 3 … less abrupt", is not supported. In addition, those transitions ran at 0.34–0.54 of peak learning rate (C8), against about 0.94–0.99 for most arity-2 transitions.

**Smaller point.** The width (G ≤ 0.1 → G ≥ 0.9) folds in a slow calibration tail. For example, B_s201's G goes 0.08 → 0.67 in 100 steps, then takes 200 more steps to reach 0.9. A 0.1 → 0.5 or KL-based width would be more faithful.

**Correction.** Report arity-3 widths as resolution-bounded (≤ 250 to ≤ 600 steps) and drop the comparative abruptness claim.

### 8. [MODERATE] Censoring: the cosine schedule makes "later or never" partly a schedule property

**Evidence (C8).**
- The learning rate anneals to 0 at step 4000. Steps 2500–4000 contribute 9% of the cumulative learning rate, and steps 3000–4000 only 2.7%.
- A censored model has effectively stopped optimising by about step 3000.
- G in the censored models is 0.00 at every snapshot. The successful arity-3 models showed no precursor either: G was 0.00 until about 250 steps before their jumps.
- So the trajectories carry no information that distinguishes "stuck" from "delayed".
- Successful arity-3 acquisitions (including development seed C) took place at steps 1725–2500, where the learning rate is 0.34–0.6 of peak.

**What survives.** P1 survives in cumulative-learning-rate units: every pair is still ≥ ln 1.5, with the smallest A/D pair at 0.44.

**Correction.**
- State that censoring is at about 2700–3000 effective steps under a decayed schedule.
- State that "later vs never" cannot be addressed by extending this run. It needs a constant-learning-rate continuation or a fresh, longer schedule, as the navigation document already proposes.
- Express time ratios in cumulative learning rate as well as steps.

### 9. [MODERATE] The P2 confound is mis-described

**The doc's wording.** The doc says the result cannot separate a smaller gate gain from a stronger agreement cue "that reduces the gate's residual value". But the gate gain is *defined* as gate-free NLL minus Bayes NLL, which is already the residual value after agreement. That reading is circular.

**The substantive confound.** Redundancy:
- agreement predicts g at 0.883 in A against 0.741 in B (C5);
- so after the agreement cue is learned, gate-specific gradient comes from the 12% of inputs where they conflict, against 26% in B.

Also:
- "usefulness" is 2× in NLL but 4× in accuracy (0.028 against 0.109);
- A and B also differ in order-1 loss.

**Correction.** Name redundancy and conflict-fraction as the confound. The design proposal (iii), varying gate relevance with the agreement cue fixed, is the right fix.

### 10. [MODERATE] The comparison with XIV's skipped stage is not like-for-like

**Evidence.**
- XIV's A_p used a different evidence layout and swap (gatelib `swap35`, with G = positions 0–1 and E = 2–5) and different references.
- Within XV, B's stage is itself marginal:
  - development seed 100 peaked at 0.41;
  - at threshold 0.6 the stage appears in 1/4 confirmation models.

**Correction.** Downgrade "XIV's skip was task-specific" (LT Q9, OM H5 row) to "not reproduced in this family. The A_p indices differ, and B sits near the threshold."

### 11. [MINOR–MODERATE] DeepSets T4: the source plateau is real; G4 attenuation reflects edited-input error

**Survives.** There is a true plateau:
- KL_own ≤ 0.03 for 350–500 steps, with KL_bayes ≈ the own-question KL (0.28 against 0.273);
- then KL_bayes drops from about 0.27 to 0.03 within 150–250 steps.

**G4 attenuation (C7).**
- At final KL 0.0013, the model's KL on the *edited* inputs is 0.0148 (s301) and 0.0138 (s302). That is about 11× higher: the edit produces lower-probability configurations where the model is less accurate.
- G4 is not driven by outliers: excluding the top 5% of |ΔBayes| gives 0.827 against 0.830.
- G4's "abrupt" part reaches only about 0.55. The rest (0.6 → 0.83) takes 5000 steps.

**Comparability.** The T4 width is defined to G4 ≥ 0.5, but the GATE width is defined to G ≥ 0.9. LT Q13 lists them side by side as if comparable.

**Correction.**
- Attribute the attenuation to off-distribution edits.
- Describe the acquisition as abrupt in KL and partial in G4.
- Harmonise the width definitions.

### 12. [MINOR] Is A_p valid as a stage index? Mostly yes, with one blind spot

**Survives (C2).** A_p is exactly 0 for:
- every order-1 rule with miscalibrated per-source weights;
- order-1 rules with temperature scaling;
- every log-mixture of order 1 with Bayes, and every probability mixture of the two (≤ 0.11).

It is exactly λ along the log path from order 1 to the gate-free rule. So a model that mixes order 1 and the gate does **not** produce a spuriously high A_p.

At the B-stage checkpoints:
- the bootstrap SD is 0.003–0.005;
- R² is 0.79–0.92;
- the model is about twice as close (in KL) to the gate-free rule as to the order-1 rule.

So the stage is genuine partial agreement use.

**The blind spot.** A "soft gate" (a confidence-c gate mixture) gives A_p 0.98 at G 0.10 (c = 0.6). The gate-free rule is the c = 0.5 member of that family. So A_p ≥ 0.5 with G ≤ 0.1 does not exclude weak gate use. In the plateau models, KL stays at or above the gate-free floor, which rules this out. At B's stage boundaries it does not (Finding 5).

**Correction.** Add "KL ≥ the gate-free KL" as a stage condition.

### 13. [MINOR] Other points

- **P2 is labelled correctly as inconclusive.** The arity-2 4/4 is 4 seeds with common random numbers.
- **The mechanism selection deviation** (C_s204 and D_s203 used in place of seed 201) is disclosed.
- **OM §1's "identical initial SNR for A/B"** is guaranteed by the shared initialisation, the shared gradsig RNG and the 86% identical data. It does not support "the difference arises during training". That conclusion may be true, but it is not evidenced here.

---

## Claims that survive

1. **P1, complexity delays [frozen].** At matched gate value, the arity-3 gate came much later (2250 and 2500 steps against 500–1500), or not before the schedule annealed (6/8).
   - This holds in 4/4 seeds and in cumulative-learning-rate units.
   - It is "later or not within an effectively ~3000-step annealed budget". It does not show "never".
2. **Censored models sit exactly at the gate-free optimum.** Their final KL equals the gate gain, A_p ≈ 1, and G = 0.00. They are not broken. Stuck and delayed are not distinguishable.
3. **Arity-2 transitions are genuinely abrupt.** G and KL fall together within about 100–200 steps after plateaus of 150–1300 steps. G is calibration-invariant, as stated.
4. **A_p detects genuine gate-free agreement use** in A, C and D (long plateaus at KL ≈ the gate-free floor) and weakly in B. The ordering "agreement *onset* before gate onset" holds in all 4 seeds × 2 ν arms.
5. **The paired-stream design is verified** and is a strong common-random-numbers control for the arity contrast.
6. **Mechanism.** After the transition, the gate-dependent flip information is carried at the evidence tokens by the layer-2 input, not at the gate or readout tokens (4/4 analysed models). The censored C_s201 shows no transfer. I found no evidence of g-specific information before use at the probed sites, once agreement is controlled.
7. **DeepSets T4.** It is trainable (4/4). It shows a true own-question plateau, then an abrupt KL drop. P6 is CONFIRMED as scored.
8. **P2 is correctly INCONCLUSIVE.** The H2 stage pattern is falsified at the frozen threshold.

## Claims to withdraw or restate

| claim | location | action |
|---|---|---|
| initial gate SNR tracks the arity; the Spearman table | OM §1 | withdraw (Finding 1) |
| D_s203 has g before use | OM §2 | withdraw (Finding 2) |
| gate computed at the layer-1 input; D baseline 0.30 | OM §2 | restate (Finding 3) |
| C shows formation during the transition | OM §2 | restate as unresolved (Finding 3c) |
| "8/8", "16/16" and the bootstrap over pairs | LT, TD | restate as 4 seeds (Finding 4) |
| B's stage lasts 150–325 steps; H1 rather than RACE | LT, OM | restate (Finding 5) |
| a source stage in all 20 models | LT Q7, navigation | restate for GATE (Finding 6) |
| 7–30 snapshots per transition; arity 3 less abrupt | LT Q13–14 | withdraw (Finding 7) |
| the residual-value description of the P2 confound | LT P2 | restate (Finding 9) |
| XIV's skip was task-specific | LT Q9, OM H5 | downgrade (Finding 10) |
