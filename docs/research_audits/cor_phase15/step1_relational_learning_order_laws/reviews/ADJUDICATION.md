# Adjudication of the three Investigation XV reviews

The three reviews were written by independent agents (A: mathematics; B: learning dynamics and mechanisms; C: integrity). They are agent reviews, not human peer review. Each finding is accepted, partly accepted, or rejected with a reason.

**What a correction can change.** The frozen verdicts are what the frozen rules returned on the confirmation outputs, and they are not re-scored. A correction can change how a verdict is interpreted, the strength attached to it, or any exploratory claim.

**Checks I re-ran myself before accepting:**
- B2 and C2: the agreement-cue baseline.
- B5: the intermediate-stage snapshot counts.
- C2: the omitted probe site.

## Frozen verdicts after review

| prediction | frozen verdict | status after review |
|---|---|---|
| P1 complexity | COMPLEXITY_DELAYS (8/8) | **Stands.** It also holds in cumulative-learning-rate units (B8). The effective sample is about 4 independent replicates, not 8 (B4). "Within budget" means about 2700–3000 effective steps (B8). |
| P2 usefulness | INCONCLUSIVE | **Stands.** The "4/4 at arity 2" figure is a post-hoc subgroup (C6). The confound is redescribed (A4, B9). |
| P3 stage pattern | H1_COMPLEXITY_FIRST | **Stands as the frozen output, interpretation downgraded.** In C and D the call is close to forced by censoring (A1). In B the stage lasts 3 snapshots in 3 of 4 models (B5, re-checked). H1 is not robustly distinguished from RACE. |
| P6 DeepSets T4 | CONFIRMED | **Stands.** Reference values corrected (C9). G4 < 1 is explained (B11). |

## Withdrawn claims

**W1 (B1, critical). The gradient-signal analysis is withdrawn.**
- With B = 16, E‖mean D‖² = ‖μ‖² + tr Σ / B, so the estimator's SNR is about 1 even when there is no signal.
- The reviewer's null-against-null control gives a 5–95% range of 0.68–1.43. Every reported value (0.75–1.54) lies in or at the edge of that band.
- The following are withdrawn:
  - "the initial gate SNR tracks the arity";
  - the Spearman −0.80 (also inflated by pairing, B4 and C3);
  - the t_S variant.
- Replacement statement: the estimator detected no gate-attributable expected gradient at initialisation or at t_S in any model. P5 and H3 are untested.
- The switch from the declared t_S statistic to the initialisation headline (C3) is moot, and is disclosed.

**W2 (B2 and C2, critical and moderate). The "information before use" reading is withdrawn and replaced.**
- I re-checked the agreement-cue baseline. The pair-agreement indicator [x₃ = x₄] alone predicts g with accuracy:
  - **0.883** at ν = 0.05 (A, D);
  - **0.740** at ν = 1/3 (B, C).
- So 0.5 is the wrong chance level for a linear probe.
- Probe accuracies before the transition, at every measured site, compared with these baselines:

| model | site | accuracy | baseline |
|---|---|---|---|
| A_s201 | post2@R | 0.85–0.88 | 0.88 |
| D_s203 | post2@R | 0.87–0.91 | 0.88 |
| C_s204 | post2@R | 0.70 | 0.74 |
| C_s201 (censored) | post2@R | 0.68 | 0.74 |
| B_s201 | post2@R | 0.48–0.51 | 0.74 |

- After the transition, decodability rises to 0.97–1.00 at the readout and 0.86–1.00 at the evidence tokens. Both are well above the cue baseline.
- **Corrected statement:** before the transition, no measured site carries gate information beyond what the agreement cue supplies (D_s203's 0.91 at step 2000 is within about 1.5 SE). Gate information beyond the cue, and its use, appear together at the transition, in all four acquiring models.
- So D_s203 is not an exception. The "newly represented" versus "newly usable" question cannot be settled with these probes, because a probe for g is contaminated by the cue. The omitted site (C2) is now reported.

**W3 (B7 and C4, major and moderate). The "less abrupt at arity 3" claim is withdrawn.**
- After step 2000 the snapshots are 250 steps apart, so the 600–750 step width at arity 3 is resolution-limited.
- "Every transition spans 7–30 snapshots" is false for arity 3. Abruptness is established only for the arity-2 and DeepSets transitions.

**W4 (B6). A GATE "source-only stage" is withdrawn.**
- In GATE the order-1 rule and the agreement rule are not separated by accuracy.
- A_p is already 0.11–0.28 when t_S is reached, and the error does not plateau at the order-1 level.
- t_S marks reaching order-1 accuracy, not a stage.
- A real source plateau is shown only in DeepSets T4.

## Accepted corrections

**From Review A**
- **A1 (major).** Accepted. Stated in the synthesis: P3 discriminates only in B, and the arity-3 ordering evidence rests on 2 acquiring models.
- **A2.** Accepted. A_p identifies "gate-free second-order reliance", not specifically [x₃ = x₄]. The cross-pair rule scores 0.70.
- **A3.** Accepted. "Calibration-free" is withdrawn. A_p and the size of G scale with confidence (temperature λ gives A_p = λ). The thresholds are therefore partly confidence thresholds.
- **A4 (with B9).** Accepted.
  - The confound is not a fixed-sum trade-off.
  - It comes from the single knob ν moving both gains. The reviewer shows a generator with separate pair and single-source reliabilities that decouples them.
  - The B9 framing is added: at ν = 0.05 the cue overlaps the gate, 88% against 74%.
  - The "2× larger gate gain" depends on the reference point, so the ratio is reported with its reference.
- **A5, A6, A8, A9, A12, A13.** Accepted as wording corrections.
  - A6: G = 0 holds exactly in the population and only approximately on the stream (|G| ≲ 0.01).
  - A9: the sampler error is 3.9e-4, as logged.
  - A12: the noisy-gate fit used orders 1–4; the reviewer verified that order 5 is minimal.
  - A13: paired models behaving alike before the gate is an observation, not a consequence of the shared stream.
- **A7.** Accepted. The §7 counterexample is wrong: a uniform last gate source keeps the values matched (0.2555 against 0.2541). It is removed.
- **A10.** Accepted. At a = 2 the gate and agreement are both pairwise. "Lower-order first" holds only in decision-order terms, and the arity contrast changes relation arity and decision order together.
- **A11.** Accepted.
  - H1's "no usefulness effect" and H2's stage pattern each needed an auxiliary assumption.
  - "H1 consistent with all frozen tests" is withdrawn. At arity 2, P2 leans against H1's prediction of no effect (4/4 post hoc, confounded).

**From Review B**
- **B3.** Accepted.
  - Layer-1 input patching reproduces the input edit, so it says nothing about route location.
  - D's baseline is about 0.03, not 0.30.
  - C_s204 has no checkpoint before its transition.
  - Retained: after the transition, gate information sits at the evidence tokens by the layer-2 input.
- **B4.** Accepted. About 4 independent replicates. Pair bootstraps and 16-model correlations are invalid.
- **B5.** Accepted, as in the verdict table above.
- **B8.** Accepted, as in the verdict table above.
- **B10.** Accepted. The comparison with XIV's skipped stage is not like-for-like, because XIV used a different agreement index. It is downgraded to suggestive.
- **B11.** Accepted. G4 attenuation reflects error on the edited inputs.
- **B12.** Accepted for future work. A soft gate can score high on A_p. A floor condition (error at or above the gate-free floor) should be added to future stage rules. The frozen rule is not changed.

**From Review C**
- **C1.** Accepted. The development-incident disclosure is corrected:
  - the overwrite reached step 850 over about 10 minutes;
  - `logs/dev_traj_contaminated_A.json` steps 25–475 belong to the overwriting run;
  - the claim "development A had A_p ≥ 0.5 from step 200" is withdrawn as unsupported.
  - Confirmation is unaffected.
- **C5.** Accepted. Wording on independence is corrected.
- **C7.** Accepted and disclosed. The scorers check structure, not content. The integrity of content rests on MANIFEST.sha256 and on Review C's recomputation of 8 trajectory points. Four imported sibling modules are not hashed, though git shows them unchanged.
- **C8.** Accepted and disclosed. Several scripts (exploratory, sensitivity, scoring wrappers) and RELATIONAL_COMPLEXITY_AND_VALUE_MATHEMATICS.md were written after the freeze but before scoring. None of them touched confirmation outputs.
- **C9.** Accepted. The T4 confirmation-stream references are Bayes accuracy 0.742 and source-rule accuracy 0.580, with KL 0.273.
- **C10.** Accepted.
  - No peak-memory record exists for the confirmation runs; only the benchmark's 0.71–0.74 GB.
  - The Spearman summary script is moot (W1).
  - `confirm_outputs/` (about 1.1 GB) is included in the archive.
  - `logs/VERIFY_CLEAN.txt` is produced by the final verification.

## Rejected

No finding is rejected outright. One clarification on B5: the reviewer argues for a race or parallel reading. I keep the frozen verdict (H1), because the frozen rule returned it. I do not claim the evidence prefers H1 over RACE.
