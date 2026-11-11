# Frozen prospective protocol — Investigation VI (perfect-agreement boundary)

Status: **FROZEN before any confirmation-cohort model existed.** The freeze is the Git commit that adds this file; the
SHA-256 hashes of every script that produces or analyses confirmation data are listed in §9 and in `logs/FREEZE_HASHES.txt`.
No criterion below may change after training starts. Changes found necessary later are reported as deviations, not edits.

## 1. Object of the claim

For a natural T4 episode (K=3, Q=6, three source groups of two, colluder agreement ρ=1) and a scored question q, a **4→5
edit** raises one non-colluder group g's leave-q agreement count from 4 to 5 by changing exactly one report: the second
source of g in g's only disagreeing question other than q is set to the first source's value. The world, X_q and every
other report stay fixed (verified: `logs/exact_boundary.json`). The edit is oriented toward **c_g**, g's common vote at q.
Pairs where g's two q-sources disagree are excluded.

For a model M, let m(c) = log p(c_g) − mean of log p over the other two candidates (a candidate log-probability ratio).
The observable response is Δ45 m_M = m(after) − m(before). The exact-Bayes response Δ45 m_B is the same quantity under
the exact posterior. The **excess** is E45 = Δ45 m_M − Δ45 m_B. A positive excess means M credits g's vote more than
Bayes does when g reaches perfect agreement. No fitted φ, s or gate enters any confirmatory quantity.

Regimes are defined by c_g's relation to the other groups' votes at q, before the edit:
- **R1**: c_g agrees with a colluder candidate (a group whose leave-q count is already 5). Bayes response is positive here (≈ +0.23).
- **R2**: c_g agrees only with the third group. Bayes response is ≈ −2.4.
- **R3**: c_g agrees with no group. Bayes response is ≈ −2.1.

The same construction gives F34 (3→4), F23 (2→3) and F0 (the F45 token changed to the third value, which preserves the
counts and Bayes exactly).

## 2. Direction and its provenance

- **Mandated direction (from the earlier s ordering).** T1-pretrained models show a stronger candidate-level update at
  perfect agreement than P-count models after Bayes correction: **B_T1 > B_P**.
- **Development evidence.** This comes from `logs/dev/`, using Investigation IV/V models and evaluation stream 15_000_000.
  It is development only and is never counted as confirmation.

| quantity (11 T1 vs 11 P development models) | T1 mean (SD) | P mean (SD) | standardised diff |
|---|---|---|---|
| B45 pooled over oriented natural pairs | +0.067 (0.241) | −0.010 (0.316) | 0.27 |
| B45 in R1 | −0.132 (0.201) | −0.727 (0.227) | 2.78 |
| B34 in R1 | −0.036 (0.077) | −0.192 (0.106) | 1.69 |
| B45 in R2 / R3 | +0.47 / +0.13 | +0.72 / +0.29 | −0.61 / −0.44 |
| F0 control | \|B0\| ≤ 0.012 for every model | | |

The pooled statistic mixes a regime where P models under-credit (R1) with regimes where low-quality models of both
curricula under-discount (R2/R3). B45 in R2/R3 tracks KL in both curricula; cT67, a poor T1 model, behaves like P there.
The R1 statistic does not track KL in development. Both are therefore frozen, with the R1 statistic as a co-primary
because it was selected on development data. The independent cohort is what confirms or refutes it.

## 3. Predictions (all one-sided, T1 > P unless stated)

**P1 (boundary response; co-primary, Holm-adjusted over the two).** Intention-to-train (ITT), at transfer step 4000.
- (a) **B45** = per-model mean of E45 over oriented natural F45 pairs, weighting the natural distribution (each natural
  episode equally; the predefined weighting).
- (b) **B45-R1** = the same mean, restricted to regime R1.
- Each is **confirmed** iff the exact permutation p (Holm-adjusted) < 0.05 **and** the lower end of the 95% model
  bootstrap CI of the difference > 0.
- Probability space: the same R1 contrast for Δp(c_g) model − Bayes is reported. A sign agreement is required for P5;
  it is not a separate test.

**P2 (specificity).** D-R1 = B45-R1 − B34-R1, T1 > P, confirmed by the same rule (unadjusted). D-pooled = B45 − B34 is
also reported.
- If P1(b) is confirmed but P2 is not, the claim is an attenuated response to count increases in general, not a
  boundary-specific operation.
- In development, B34-R1 already differs by curriculum (d = 1.7), so P2 is a genuine risk.

**P3 (training stage).**
- Contrasts at transfer steps 250, 500, 1000, 2000 and 4000 for B45-R1, B45 and D-R1. Step 250 is descriptive only.
- Confirmed iff the B45-R1 difference is positive at ≥ 3 of {500, 1000, 2000, 4000} **and** P1(b) is confirmed.
- The P-count arm is trained for the full 4000-step long-transfer arm. This is mandatory, and it is the arm whose late
  snapshots test whether the difference closes with more transfer.

**P4 (quality robustness).**
- (i) *KL-matched comparison.* Target = median KL-to-Bayes of the P models at step 4000. For each T1 parent, the
  snapshot (of the five) whose KL is closest to the target is selected, using KL only. B45-R1 is then compared between
  those T1 snapshots and the P step-4000 models with the same permutation and bootstrap rule. KL ranges and overlap are
  reported. Confirmed iff p < 0.05, CI lower > 0, **and** the ranges overlap. If they do not overlap, P4 is reported as
  *not testable* (it does not count as a failure).
- (ii) *ANCOVA* across all snapshots: B45-R1 ~ arm + KL. The arm coefficient is tested by permuting parent labels, with
  snapshots kept with their parent.
- Accuracy, ECE, KL-optimal temperature, margin scale (mean |centred logit|) and NLL are reported for every snapshot.

**P5 (contexts).**
- The T1 − P sign of B45 is reported within each of R1, R2 and R3.
- Frozen expectation: positive in R1, not positive in R2 and R3. These are the regimes where development suggests a
  quality-driven under-discount.
- P5 holds iff P1(b) is confirmed and the probability-space R1 difference is positive.

**P6 (causal computation).** P6 is *not* preregistered here. If P1(b) is confirmed, a separate mechanism protocol will be
frozen in a later commit before any causal analysis, using independent episodes (stream ≥ 16_000_000).

## 4. Cohort

- **Parents and seeds.** 8 T1 parents (seeds 70–77) and 8 P-count parents (seeds 80–87).
  - Pretraining is the original protocol: T1 for 6000 steps (`nets.train`); P-count for 3000 steps (`train_tasks.train`).
  - Common architecture ("M", transformer); identical T4 transfer (`train_ext.train` code path: layers 0–1 and
    embeddings frozen, layer 2 reinitialised, warmup-cosine over 4000 steps).
  - Snapshots at 250, 500, 1000, 2000 and 4000 (`scripts/cohort6_train.py`; equality with `train_ext.train` verified,
    max diff 0.0).
- **Independence.** Seeds are disjoint across curricula, so initialisation, pretraining data, transfer reinitialisation
  (seed + 7777) and transfer data streams all differ. Verified: 0 bit-identical non-constant initial arrays within or
  across arms (`logs/independence.json`).
- **Development use.** No Investigation IV/V checkpoint is a confirmation unit. All of them are development only.
- **Success rule** (`scripts/cohort_status.py`): step-4000 transfer accuracy ≥ 0.6595 and a finite loss.
  - Failed runs are kept and reported.
  - Primary analysis is **ITT** over all 16 parents. A parent with no step-4000 model is imputed worst case against the
    hypothesis (T1 → the minimum observed value, P → the maximum).
  - A successful-only analysis is secondary.
- **Order and compute.** Two CPU lanes claim parents in the order T70, P80, T71, P81, … (`scripts/lane.sh`). All starts
  and ends go to `logs/COMPUTE_LOG.txt`. A crashed job is resumed from its last completed stage, and the resume is logged.

## 5. Measurement

- `scripts/boundary_run.py conf xT{s}@{step} …` on 6000 natural episodes from evaluation stream **15_100_000** (never
  used before). Families are built with fixed RNGs.
- Quality is measured on the same natural set.
- Generator-level facts (oriented fractions, regimes, Bayes responses) are written once to `logs/conf/generator.npz`.

## 6. Statistics

- **Unit.** The parent model. Episodes are never treated as independent units for a curriculum claim.
- **Test.** Exact two-sample permutation over all C(16,8) = 12,870 label assignments, on the difference of means, one-sided.
- **Interval.** Model bootstrap: 20,000 resamples within arm, seed 6, percentile.
- **Multiplicity.** Holm over P1(a) and P1(b) only. The other predictions use their own stated rules and are reported
  with unadjusted p.
- **Power** (simulated from development SDs, 8 + 8, one-sided α = .025):
  - ≈ 1.0 for a standardised difference of 2.8 (the development R1 effect);
  - 0.97 at 2.0;
  - 0.45 at 1.0;
  - **0.08 at 0.27** (the development pooled effect).
  - P1(a) is therefore expected to fail even if the development picture is right. It is kept because the mandate
    specifies the pooled direction.
- **Script.** `scripts/confirm_analysis.py` (frozen) produces `logs/confirmatory.json`.

## 7. Outcome mapping (frozen)

| result | reading |
|---|---|
| P1(b) and P2 confirmed, P4 confirmed or not testable with ANCOVA arm p < .05 | boundary-specific curriculum difference in the observable 4→5 update, not explained by KL (outcome A/B-type) |
| P1(b) confirmed, P2 not | curriculum difference in count-update gain, not boundary-specific |
| P1(b) confirmed, P4(i) testable and not confirmed | difference explained by approximation quality / stage |
| P1(b) not confirmed | no reliable curriculum difference at the boundary; earlier s-ordering not supported observably |
| F0 control \|B0\| ≥ 0.02 for any model | measurement problem; confirmatory results reported but flagged |

## 8. What is explicitly not claimed

- No algorithm is attributed from a fitted parameter (s, φ, gate) in any confirmatory statement.
- Fitted-operator analyses (mixture, linear pool, log pool, profiles of s) are exploratory and are reported in
  FITTED_OPERATOR_IDENTIFIABILITY.md.

## 9. Frozen code hashes

See `logs/FREEZE_HASHES.txt` (SHA-256 of `src/boundary.py`, `scripts/boundary_run.py`, `scripts/cohort6_train.py`,
`scripts/lane.sh`, `scripts/cohort_status.py`, `scripts/confirm_analysis.py`, and the Investigation IV/V modules they
import), written in the same commit as this file.
