# R1 — Adversarial review: statistical inference and learning dynamics

Scope: PROSPECTIVE_PREDICTIONS.md (frozen blocks and OUTCOMES), PCOUNT_CAUSAL_TRANSFER.md, STRUCTURE_AND_EXTRAPOLATION.md,
ADAPTIVE_SOURCE_THEORY.md, GENERALIZATION_LANDSCAPE.md. I also read the code and logs they cite. The checks I ran myself were all
small (no training). Their scripts are in `/tmp/r1/` and are not part of the package; each one ran in under 3 minutes.
Severity: **H** = the claim as written is not supported; **M** = the claim needs qualifying or a deviation is
under-reported; **L** = presentation or provenance; **OK** = checked and found sound.

## A. Freeze integrity and verdict fidelity

**1. OK. The frozen text is unchanged.** `git diff 6e8124b cb9bf2e` on PROSPECTIVE_PREDICTIONS.md adds lines only:
the Q3-E′ block in 43d9144, then the OUTCOMES. Every verdict matches its frozen criterion as computed. A1 5/5. A3: 0.717 < 0.740,
with every T1 parent ≥ 0.70. A4 3/3. D1: max DeepSets 0.215, TF mean 0.620. D2: SAT ρ 0.714 > 0.5 and MAE 0.123 < 0.166, so the
prediction fails. D3: ρ 1.00, MAE 0.019. E1: changed 0.656 < 0.660 + 0.015. E2: 0.488 < 0.6. E′1: 0.691 ≥ 0.675 and
|0.736 − 0.735| ≤ 0.02. E′2: 0.879. E′3: 0.881. The hashes check out: `src/graded_sig.py` is 928f3a7b…, `frozen_fits.json` is
f79131f0…, and neither file has been touched since 6e8124b. The Q3-E′ freeze commit is dated 12:48:29. From
train_seconds = 2525 s and the npz mtime of 13:30:36, the run started at about 12:48:31, so it began after the freeze. All 16
held-out D models finished between 13:53 and 16:34, after the 07:24 freeze. `heldout_outcome.json` was written at 16:40.

**2. L. Small freeze and provenance inconsistencies.**
(a) The frozen Q3-D text says the held-out models were "trained before this freeze". `progress.log` shows they finished after
it, and so does the freeze-state paragraph. This is harmless (it makes the test stronger) but it contradicts itself.
(b) `ref_T1p1/p2/p10/p11` finished between 06:46 and 07:19, before the 07:24 freeze. The T1-parent half of Q3-A3 therefore rests
on results that were already on disk. The claim that they had not been inspected is stated but cannot be verified.
(c) `scripts/eval_extrap.py` was first committed in cb9bf2e, after evaluation. `adapt_curriculum_p1.py` was first committed in
a886e90, after the Q3-E′ run, and its docstring is a copy-paste: it still reads "EXPLORATORY … from the atlas T1 s0".
(d) `progress.log` has "(parallel, reverse)" lane entries. The script that wrote them is not in `scripts/`. Some entries
are no-op lines logged before the corresponding json existed: "tf 42 … done" at 16:19:05, while the json is dated 16:34:18.
All `logs/train_heldout3_tf*.out` files are empty. From train_seconds and mtimes, each held-out TF appears to have been trained
exactly once, but the training provenance should be documented.

**3. M. Q3-A2 is reported as "PASSED on 2 seeds".** The frozen criterion names 3 seeds. The budget cut was logged at 07:42, before
the run, which is correct practice. Even so, the verdict should read "not evaluable as frozen (2/3 seeds run; both fail)" rather
than PASSED.

**4. M. Q3-E2 was operationalised after its freeze.** The frozen text describes "pairs of episodes … static-count reference
≤ 0.5 by construction". `eval_adapt.py` instead scores every member of a count class whose exact argmax varies. That code was
written after the freeze: `eval_s0.json` is dated 11:57, and the code was committed at 12:48. My check (`/tmp/r1/q3e2_ref.py`,
same stream and code path) found:
- The static rule scores **0.519**, not ≤ 0.5.
- The best possible count-sufficient rule (in-sample class majority) scores **0.566**.
- 150/3576 items (4.2 %) are exact ties in the exact posterior. 119/1508 classes are "mixed" only through float tie-breaking.

The verdicts do not change: the curriculum models score 0.88 against a 0.566 ceiling, and the from-scratch model 0.49. But the
"by construction" claim is false as implemented. The 0.566 reference should be reported and ties excluded.

**5. L. The history-intervention reference is missing.** The static rule "follows the new exact answer" in **0.507** of the
selected questions (`/tmp/r1/hist_ref.py`), against the model's 0.88 and the 0.8 threshold, so the test does discriminate.
Separately, the static answer "changes" on 4–6 % of the selected questions. This is float tie-breaking: the maximum change in the
static posterior is 3.5·10⁻¹⁵, and 6.3 % of static posteriors are exact ties. "Static unchanged" therefore holds only up to ties;
say so.

## B. Exact Bayes (ADAPT)

**6. OK. `bayes_adapt` is correct.** I checked it four ways:
- **Brute force:** a direct enumeration over the 33 configurations and 3 answers matches `post_y` to 10⁻⁶ on three episodes.
- **Calibration:** on 40,000 episodes, the mean max-posterior is 0.70347 and the accuracy is 0.70349.
- **Prior and leave-one-out:** the prior sums to 1 (1/9 per no-change configuration, 1/45 per change configuration). The
  configuration weight excludes question q and the own-question likelihood enters once, so nothing is double-counted.
- **Static rule:** `bayes_adapt_static` is invariant to permuting the other questions (maximum change 3.5·10⁻¹⁵), so it is count-
  sufficient as claimed. The sampler matches the likelihood: noise sources are uniform, and an erring reliable source is uniform
  over the K−1 wrong values.

## C. Q3-D held-out structure-to-extrapolation

**7. H. The held-out Transformer test is close to a test-retest of the development parents.** Each held-out TF (s40–45) shares
its T1 parent with a development TF (s0, s1, s2, s10, s11, s12).
- **A parent-lookup predictor nearly matches the behavioural probe.** Predicting each held-out model's local index by its
  parent's development value gives Spearman **1.00**, Pearson 0.98 and MAE **0.042**. That beats SAT (0.123) and the constant
  (0.166), and is close to A2 (0.019).
- **The features are equally parent-determined.** SAT and GCD correlate with the parent's development value at r = 0.95 and 0.97.

So the within-TF spread is almost entirely between-parent variance that was already in the development set. The effective number
of new independent units is about zero. The frozen block admits that the models are "not independent of their parents", but
several claims still go too far:
- STRUCTURE §3 says SAT "predicted unseen models' extrapolation prospectively" and was "the first structural predictor … to beat
  its baseline on held-out models".
- The synthesis lists "SAT's held-out association" among its prospective confirmations and tags answer 30 as [C-weak].

These should be stated as replication on new T4 trajectories from the same parents. Two further points:
- **Not significant even at face value.** The exact one-sided null p for ρ ≥ 0.714 with n = 6 is **0.068**.
- **Wrong category.** No positive SAT prediction was ever frozen, so this is an unpredicted observation, not a confirmation.

**8. H. The GCD "sign reversal" is an artefact of which models were in the development set.** In development, the within-TF GCD
Spearman is −0.12, and the positive frozen slope (+0.65) is driven by one leverage point, tf_s14 (GCD 0.61, local index 0.36).
Dropping it gives Pearson −0.55. On the **same six parents** in the development data, GCD Spearman is already **−0.66** (SAT
0.50). The held-out −0.83 therefore reproduces the development parents; it is not a change between development and held-out
models. STRUCTURE §3(iii) ("GCD's within-TF sign reversed between development and held-out") and answer 30 should be corrected.

**9. M. The Q3-D2 outcome is reinterpreted after the fact.** Under the frozen, signed wording (ρ > 0.5), GCD complied with the
prediction: ρ is −0.83 and its MAE of 0.178 is above the constant. The failure comes from SAT alone. "Both on-support signatures
carried strong rank information in the predicted-by-hypothesis directions" applies a directional hypothesis that was not frozen.
The failure is also weak evidence that the signal is real: under the null with n = 6, P(ρ > 0.5) = 0.149 per signature, and about
0.27 that at least one of two exceeds it. (The MAE condition offers no extra protection, because a single violation fails the
conjunction.) A further fragility: SAT's frozen within-TF slope is 14.9 over a SAT range of 0.949–0.961, so predictions move by
about 0.15 for every 0.01 of SAT.

## D. P-count transfer (Workstream A)

**10. OK, with a correction to one statement. The seed replication and the 86 % figure hold up.**
- **Arithmetic:** (0.717 − 0.576)/(0.740 − 0.576) = 0.861. A bootstrap over the 5 + 5 seeds gives a 95 % interval of 0.81–0.91.
- **Robust to the baseline:** using the own-question level (0.577) instead gives 0.860; using the random network (0.553) gives 0.878.
- **The gap is real:** Welch t = −5.05, p = 0.006, and every P-count seed is below every T1 parent (exact permutation p = 1/252).
  This supports Q3-A3 under the 2000-step protocol.
- **Correction:** "(5 vs 5 seeds)" is wrong. P-local, the baseline in the ratio, has 2 seeds.

**11. M. The pretraining-steps confound runs against the doc's framing.** A1 says the 3000-versus-6000-step difference "favours
T1". A7 says P-count is "nearly as good as full T1 at half the pretraining steps". But the unconfounded dynamics run shows T1 s0
at **2000** steps transferring **0.732**, which is 96 % of the gain and above all five P-count seeds (0.704–0.725 at 3000 steps).
On this one seed T1 is the more step-efficient curriculum, so the efficiency framing in A7 is not supported. One caveat: the T1
transition is sharp (0.643 at 1800 steps).

**12. M. The long fine-tunes rest on one seed and a ceiling.** `ftlong_T1_s0` reaches 0.7417 against Bayes 0.74175, so T1 is at
ceiling and the "gap 0.008" is bounded by that ceiling. The P-count gap shrinks from 0.019 to 0.008 between 2000 and 6000
fine-tune steps. Most of the 2000-step "residual gap" is therefore a difference in how fast the features can be used, which bears
on how Q3-A3's "the gap is real" should be read. "Asymptotic usability" is not shown: there is one length and one seed.
`ftlong_plocal` was cut, so "P-local does not transfer" and Answer 7 ("even a 2000-step fine-tune cannot use it") hold only for
the 2000-step protocol. They say nothing about whether the information could eventually be used.

**13. M. The A6 dynamics support the first step of the claimed ordering, not the second.** Bit-identity is verified: the maximum
parameter difference at step 2000 is 0.0. But it is one run with one fine-tune seed per snapshot, at 200-step resolution
across the transition.
- **Transfer moves before the source-token count probe does.** Between 1600 and 1800 steps, transfer rises 0.575 → 0.643 (41 %
  of its total rise), while source-token count_loo moves 0.402 → 0.407 (about 1 % of its rise).
- **T1 accuracy jumps at the same time**, so any probe that tracks competence would co-move with transfer.
- **The probed variable is the one T1 does not use.** The linear leave-q count code has IIA 0.04–0.08 at the T1 source tokens
  in A3/A5.

"Agreement detection → aggregation → transfer" is therefore supported for the first arrow only. The second is simultaneity
within the snapshot resolution.

**14. L. "Sufficient: yes" overstates the interchange result.** An IIA of 0.60–0.72 shows partial sufficiency. The abstract-variable
interchange also has no null control, such as an encoder fitted to permuted counts.

## E. ADAPT learned models

**15. M. The curriculum-versus-scratch conclusions rest on very small samples.**
- **From scratch:** one seed. It plateaus at the own-question level from 1500 to 6000 steps (0.654–0.657), which is evidence of a
  plateau, not of a general failure. Its 6000 steps are also not matched to the curriculum's 6000 T1 + 3000 ADAPT.
- **Q3-E′ thresholds:** set after the exploratory values (0.88 and 0.88) were seen. That is appropriate for a replication, but
  the E′2 threshold of 0.6 is loose. The confirmation covers n = 2 parents.
- **State probe:** after a change, the probe scores 0.58, against an exact-posterior ceiling of 0.61 and a static rule of 0.53.
  That is "close to the limit", but there is no input-level linear-probe baseline.

## F. Multiple comparisons and synthesis framing

**16. L. Chance only matters for one verdict.** There are about 19 frozen sub-criteria. Every pass clears its threshold by a wide
margin relative to evaluation noise. For example, E′1's margin is 0.031 against a required 0.015, with SE ≈ 0.002, and the
smallest A1 seed is 0.704 against 0.6595. Q3-D2 is the one verdict where chance is material (item 9).

**17. L. GENERALIZATION_LANDSCAPE contradicts itself and drops qualifiers.**
- It says "two-hop composition was not learned under any regime tried", while its own table row, and the frozen Q3-C verdict,
  say the 24-pair holdout run was "learned (on training pairs)".
- The GATE row "stage-1 at evidence tokens … yes (structure)" across 3 seeds is a post-hoc cross-seed reading, made after
  Q3-B2, B3 and B4 failed. The GATE doc says so; the landscape table does not.
- The SAT and "P-count code causally used" rows lack the qualifiers from items 7 and 14.

## Required corrections, in priority order
1. Restate the SAT and GCD claims as parent-shared replications (items 7–9). Remove SAT from "prospective confirmations"
   and remove the GCD sign-reversal narrative.
2. Report the Q3-E2 count-rule references (static 0.519, ceiling 0.566) and exclude exact ties. Report the static reference for
   the history intervention (0.507).
3. Relabel Q3-A2 as not evaluable as frozen. Fix the P-local seed count in the 86 % sentence. Withdraw the A7 "half the steps"
   efficiency framing.
4. Weaken A6's second ordering arrow and the "asymptotic" and "sufficient" wording.
