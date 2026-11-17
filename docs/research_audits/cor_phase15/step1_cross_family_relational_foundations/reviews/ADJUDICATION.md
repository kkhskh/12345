# Adjudication of the XIV reviews

**Reviewers.** Three independent agents wrote the reviews. None of them is external human peer review.
- **A**, mathematics: `REVIEW_A_MATHEMATICS.md`. Written mid-investigation, before X1 or Stage I was scored. Its findings led to the X1 amendment (FREEZE_X_AMEND, `b019ca9`) and to the corrections already in the documents.
- **B**, mechanisms: `REVIEW_B_MECHANISMS.md`, with its checks in `review_b_checks/`.
- **C**, integrity: `REVIEW_C_INTEGRITY.md`.

**No frozen verdict changes.** The reproduced verdicts are R-S, R-N and R-P CONFIRMED; X1a and X1a′ CONFIRMED; X1b and X1b′ FALSIFIED. The changes below concern interpretation and disclosure.

## Review A (mathematics)

| finding | decision | action |
|---|---|---|
| F1: X1a cannot separate its prediction from small-scale joint learning | ACCEPTED | Registered as amendment X1a′ (accuracy band, NLL clause, partial index A_p) before inspection. The original X1a is reported as frozen. |
| F2: the "order-12 trust relation" claim for T4 is wrong | ACCEPTED | Proposition 4 rewritten. In the log domain trust is a sum of per-question order-2 terms; the high order comes from marginalising over the configuration. |
| F3: the common principle is definitional | ACCEPTED | **Rejected as a principle.** It is retained only as a shared symmetry (A1) plus a GATE calculus. The claim that X1 tests ledger N12 is withdrawn. |
| F4: X1b does not follow without a magnitude assumption | ACCEPTED | Stated in CROSS_FAMILY_CONFIRMATION.md. The falsification fits that warning. |
| F5: the A reference table was wrong (Bayes has A = 0.71) | ACCEPTED | Table corrected; A_p added. |
| F6: uniqueness and relabelling scope missing from Proposition 3 | ACCEPTED | Proof completed. |
| F7: the reason given for Corollary 3 was insufficient | ACCEPTED | Replaced by a symmetry-plus-uniqueness argument. |
| F8: consequence (b) too broad; X3 cited a non-instance | ACCEPTED | Both corrected. |
| F9: ledger N2 omitted DeepSets 3/3 | ACCEPTED | Added. The learnability-by-order link is withdrawn. |
| F10: N3, N19, N20, N21, N24 and N12 stated too favourably | ACCEPTED | All six corrected. |
| F11: hash coverage | ACCEPTED | The amendment hashes `tasks_ext.py`, `train_tasks.py`, `fwd3.py` and `FREEZE_X.md`. |
| F12, F13: minor | NOTED | Proposition 6 is labelled near-trivial. |

## Review B (mechanisms)

| finding | decision | action |
|---|---|---|
| A1 (major): **both** controls raise the route's attention share during steps 0–100 (noise on E 0.134 → 0.22; sham 0.134 → 0.33) | ACCEPTED | R-S and R-N exclude "any masking or logit perturbation of these readout edges". Neither tests "any early reduction of route share or access". The missing control (diverting attention while keeping every route edge reachable) is listed as not run. |
| A2: sham validity was checked on trained (step-1211) models only, and is one-sided | ACCEPTED | Disclosed. |
| A3, A4: the E contrast and the minimal own-question-only block | NOTED | Listed as not run. |
| A5: on M, mix_mass *is* the attention on the intervened edges | ACCEPTED | Stated. The M flips are also shown by lik_RO1 (≈ 0.75 against ≈ 0.04 in the baseline). |
| A6 (major): late E blocks are not "no effect"; lik_RO1 drops to 0.09–0.15 at release and recovers within about 200 updates | ACCEPTED | Corrected: late blocks on E transiently disrupt the early route, which then recovers, in D = 21 and 801. The frozen δ (mix_mass) does not register this. |
| A7 (major): PI means different things in the two arms | ACCEPTED | Reframed: a block **during formation** selects the organisation; the same block **after formation** is mostly reversed. "Greater early persistence" is kept only in that sense. |
| A8: R-P rests on M (3 of 4 valid pairs) | ACCEPTED | Stated. Together with C finding 5. |
| B1 (major): "not one direction" is partly an artefact of a subspace shared across tokens; per-token 1-d DAS gives s1 1.00, s0 0.64, s2 0.60 | ACCEPTED | Corrected. A *shared* 1-d subspace never suffices. Per-token single directions suffice in s1 and not in s0 or s2. |
| B2 (major): "dimensionality different from T4" does not follow | ACCEPTED | Row changed to "not established". |
| B3: probe chance level not reported | ACCEPTED | Chance is 0.5; values below it indicate a lookup code that inverts on held-out x₀. |
| B4 (major): the s2 "decodable ≠ used" example is invalid, because resid_post2 is after the last layer | ACCEPTED | Restricted to s0 and s1, and worded as "a patch at a site where g is decodable does not transfer the decision". |
| B5: the decoy toggle changes the decoy in only 33% of episodes, and decoy and gate are dependent given x₀ | ACCEPTED | The decoy control is weak, so its "ideal 0" is asserted. Disclosed. |
| B6–B8: DAS controls not variance-matched; G/R-site control tests little; tie attribution not computed for DAS; ideal references asserted, not computed | ACCEPTED | Disclosed; the wording is weakened. |
| B9: no code bugs found | NOTED | — |
| C (major): no cross-family *mechanism* similarity is established; the "same" rows are designed-in mathematics or the general probe point | ACCEPTED | GATE §5 and the synthesis say so. Operator row: "not comparable". Learnability row: confounded by architecture and recipe. |
| D (major): s3 does not "stay at the order-1 ceiling"; with G ≤ 0.011 its NLL reaches 0.449 against 0.462, about 40% of the order-2 gain | ACCEPTED | Reworded: order-1 accuracy is reached first; agreement use then grows slowly (s3 partially, s4 hardly); the gate is acquired abruptly before agreement reaches the order-2 level. The X1a criteria cannot separate order 1 from partial order 2. |

## Review C (integrity)

| finding | decision | action |
|---|---|---|
| 1 (major, disclosure): training-loss logs of GATE s3 and s4 were on disk before the freezes, and were committed with FREEZE X (to step 2400) and with the amendment (to step 4000) | ACCEPTED | **Disclosed.** At 19:13Z, *before the amendment*, the investigator read the final line of `g_train_s3.out` ("step 4000 loss 0.1807"), which bears on competence. Plateau-stage lines were not read. The amendment's statement "before any X1 output was inspected" is qualified accordingly. The reviewer judges the effect on verdicts as small: competence was later met by both models. |
| 2 (moderate): the four step-2000 checkpoints needed by R-P were missing at FREEZE I, were silently skipped by `unit_metrics.py`, and were restored and measured by hand | ACCEPTED | Disclosed in COMPUTE_LOG and in T4_FINAL_CAUSAL_CONTROLS.md. The bytes were verified against the XIII-E backup, which kept its XIII-E-era timestamps. |
| 3 (major, tooling): the preflight P0 checks fail on the finished package; the verification script had a no-op check and did not test the amendment scorer | ACCEPTED | `verify_clean14.sh` fixed. Preflights are kept as freeze-time records and not re-run. The amendment scorer and the refusals with no output are now tested. |
| 4: `score_I.py` hash-checks only the package's own design copy | NOTED | Disclosed. |
| 5: R-P is mainly M; the intervals describe 4–6 related pairs; "route-specific" in the ledger overstates | ACCEPTED | Corrected. |
| 6: transcription slips (M late PI down to −0.011; NLL ranges; stale "s0–s2" wording) | ACCEPTED | Corrected. |
| 7: one compute-log entry was written after the event; some runs lack an rc; pre-restart run output was overwritten | NOTED | Disclosed. |
| 8: packaging; absolute paths in `logs/gate_mech/s0–s2.json` (the `ckpt` field) | NOTED | Retained as provenance. The verification scans scripts and src only. |
