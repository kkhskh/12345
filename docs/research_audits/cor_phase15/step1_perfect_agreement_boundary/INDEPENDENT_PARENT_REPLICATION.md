# Independent-parent replication

> **Post-review [R2].** Findings R2-1 to R2-13 are applied: P1(a) is downgraded, the full P3 contrasts are reported, P5 is
> re-read, multiplicity is stated, the ANCOVA/KL caveats are added and the audit trail is disclosed. See §4. Pre-review text:
> commit 9389496.

## 1. Cohort

The protocol was frozen in commit 4d42d0a, with a hash addendum in 8d8e41a, before any cohort model existed. All 16 parents
were new. No Investigation IV/V checkpoint is a confirmation unit; those were used for development only (`logs/dev/`).

| | T1 | P-count |
|---|---|---|
| seeds | 70–77 | 80–87 |
| pretraining | 6000 steps, `nets.train("T1")` | 3000 steps, `train_tasks.train("pcount")` |
| transfer | T4, 4000 steps, layers 0–1 and embeddings frozen, layer 2 reinitialised (seed + 7777); snapshots at 250/500/1000/2000/4000 | identical |
| initialisation and data streams | seed-specific; 0 bit-identical non-constant initial arrays within or across arms (`logs/independence.json`) | |

**Training records**
- Each parent: `checkpoints/x{T1,P}{seed}/` holds `pre.npz`, `pre.json`, `step*.npz`, `step*.json` and `train.log`.
- `logs/COMPUTE_LOG.txt` records the two lanes.
- An environment restart at about 06:29 UTC killed the in-progress parents T72 and P82, both still in pretraining with no
  checkpoint. They were restarted from scratch with the same seeds. Their partial logs are kept as `train.log.interrupted1`.
- No other interruptions occurred.

**Final transfer accuracy (step 4000; success threshold 0.6595)**

| T1 seed | 70 | 71 | 72 | 73 | 74 | 75 | 76 | 77 |
|---|---|---|---|---|---|---|---|---|
| acc | .740 | .740 | .696 | .740 | .713 | .741 | .742 | .741 |

| P seed | 80 | 81 | 82 | 83 | 84 | 85 | 86 | 87 |
|---|---|---|---|---|---|---|---|---|
| acc | .722 | .738 | .735 | .725 | .722 | .728 | .716 | **.653 (failed)** |

**Failures.** One run failed: xP87 is below threshold at every snapshot. Its pretraining count-task accuracy (0.517) was the
lowest in the arm. It was kept, and it is included in the primary intention-to-train (ITT) analysis.

**Auxiliary (pretraining) records**
- T1 pretraining accuracies are 0.711–0.747. Bayes is 0.747.
- P-count auxiliary count accuracies are 0.517–1.0.
- Within the P arm, pretraining accuracy is unrelated to B45-R1 (r = −0.07).
- Within T1, the two weaker parents (72, 74) have the least negative B45-R1 (r = −0.57). That is the direction *away from*
  the P-count pattern.

## 2. Frozen confirmatory tests

Script: `scripts/confirm_analysis.py`, output `logs/confirmatory.json`. Unit: parent. Test: exact permutation over C(16,8).
Interval: model bootstrap.

| prediction | statistic | T1 − P | one-sided p | 95 % CI | frozen verdict |
|---|---|---|---|---|---|
| P1(a) | B45 pooled, ITT, step 4000 | +0.362 | .018 (Holm .018) | [0.089, 0.657] | confirmed under the protocol §4 / code ITT rule; **fragile** (§4.1) |
| P1(b) | B45-R1, ITT, step 4000 | +0.698 | 7.8e-5 (Holm 1.6e-4) | [0.586, 0.809] | **confirmed** |
| P1(b) | successful-only (8 vs 7) | +0.704 | 1.6e-4 | [0.580, 0.827] | consistent |
| P1(a) | successful-only | +0.440 | .003 | [0.204, 0.706] | consistent |
| P2 | D-R1 = B45-R1 − B34-R1 | +0.661 | 7.8e-5 | [0.520, 0.802] | **confirmed** |
| P2 | D-pooled (reported) | +0.284 | .090 | [−0.052, 0.668] | not significant |
| P3 | B45-R1 difference positive at 500/1000/2000/4000 | 4 of 4 (also +0.21 at 250) | each ≤ 1e-4 | | **confirmed** |
| P3 (reported, R2-3) | pooled B45 at 250 / 500 / 1000 / 2000 | −0.61 / −0.38 / +0.06 / +0.31 | .998 / .95 / .39 / .0499 | | opposite sign early |
| P3 (reported, R2-3) | D-R1 at 250 / 500 / 1000 / 2000 | −0.04 / +0.16 / +0.45 / +0.60 | .82 / .03 / 2e-4 / 1e-4 | | boundary specificity emerges from about step 500–1000 |
| P4(i) | KL-matched T1 snapshot vs P at 4000 | +0.678 | 7.8e-5 | | **confirmed**; KL ranges overlap (T1 matched 0.042–0.141, P 0.029–0.212) |
| P4(ii) | ANCOVA arm coefficient (B45-R1 ~ arm + KL, all snapshots) | +0.648 | 1.5e-4 | | consistent with an arm effect; this tests the sharp null only (§4.4) |
| P5 | probability-space R1 difference | +0.058 | 7.8e-5 | [0.043, 0.075] | formal rule met; the rule is nearly implied by P1(b) [R2-5] |
| P5 | frozen expectation "R2, R3 not positive" | R2 −0.095 (n.s., CI [−0.62, 0.45]); R3 +0.234 (p = .14) | | | partly mismatched (R3), and not testable as stated: "not positive" cannot be shown by a non-significant test |
| F0 control | max \|B0\| | 0.013 < 0.02 | | | passes |

**Separation.** Every T1 parent has B45-R1 in [−0.12, +0.06]. Every P-count parent has B45-R1 in [−0.96, −0.49]. There is no
overlap, and the pooled SD is 0.12 against a difference of 0.70.

**Power.** P1(a) was expected to be underpowered: development d was 0.27, giving power of about 0.08. It was confirmed
because the confirmation cohort showed a larger pooled effect (d ≈ 1.2) than development. That is reported as found. The
development-to-confirmation change in the pooled statistic comes mainly from R3, where the P-count arm's excess was smaller
in the cohort (+0.10 vs +0.29 in development).

## 3. Replay versus independent confirmation

- **Exact replay.** Rerunning `scripts/boundary_run.py` and `scripts/confirm_analysis.py` on the stored checkpoints
  reproduces `logs/confirmatory.json` bit-for-bit. Fixed streams are listed in `scripts/verify_clean_extraction.sh`.
  This is replay, not confirmation.
- **Independent confirmation.** The 16 parents above, trained after the freeze, with new seeds and new evaluation
  streams. The development models (Investigation IV/V) agree in direction and in R1 size (dev difference 0.60 vs
  confirmation 0.70). They are not counted.

## 4. Post-review qualifications [R2]

**4.1 P1(a) is fragile [R2-1, R2-2].** The frozen materials are inconsistent about ITT:
- The protocol §4 and the frozen code impute only a parent with *no step-4000 model*.
- The frozen `cohort_status.py` docstring says every failed run "enters ITT by worst-case imputation".

xP87 failed but has a model, so its observed value was used. Under the stricter docstring reading:
- P1(a) gives +0.307, p = .066, CI [−0.055, 0.643]: **not confirmed**;
- P1(b) gives p = 7.0e-4, CI [0.36, 0.79]: still confirmed.

Without the two weak T1 parents (72 and 74), P1(a) gives +0.19, p = .076. Its pooled confirmation therefore rests largely on
quality-driven under-discounting in R2/R3, which the protocol itself attributed to quality. The directional hypothesis is
supported **only through the R1 statistic**. This inconsistency is recorded as a deviation note; no frozen file was edited.

**4.2 Scope [R2-5].** The effect is confined to R1, which is 435 of 1346 oriented F45 pairs (32 %) and about 3 % of natural
decisions [R5-3].

**4.3 Multiplicity [R2-7].**
- More than 20 directional tests are reported.
- Every R1-family result (P1(b), P2 D-R1, P3 at each stage ≥ 500 except D-R1 at 500, P4, P6b, P6c; p ≤ 2e-4) survives
  Bonferroni over 25 tests.
- P1(a) (p = .018), pooled B45 at 2000 (p = .0499) and D-R1 at 500 (p = .03) do not.

**4.4 KL and the ANCOVA [R2-8, R2-9, R5-6].**
- KL is a post-treatment variable, and the linear ANCOVA is misspecified (arm × KL interaction ≈ −1.0).
- The reviewer's support-restricted fit (51 snapshots in the common KL range) gives an arm coefficient of 0.62, exact
  p = 7.8e-5.
- The matched KL means agree: 0.079 (T1) vs 0.088 (P).
- KL is nearly blind to this effect. An error of 0.74 on about 3 % of decisions adds roughly 1e-3 to KL. KL matching is
  therefore a check that *average* quality does not explain the difference, not a strong control.

**4.5 Statistics details [R2-10, R2-11].**
- The decision rule (p < .05 and 95 % CI lower > 0) has an effective one-sided α ≈ .025, matching the power analysis.
- The percentile bootstrap with n = 8 per arm is somewhat narrow; a Welch t for P1(a) (t = 2.35) agrees.
- All inference conditions on one shared episode set. This matters little given complete separation in R1.

**4.6 Replication scope [R2-6].** This is replication across independent seeds of **one** pipeline (architecture,
generator, trainers, evaluation design). It is not replication across implementations.

**4.7 Audit trail [R2-12, R2-13].**
- The `.npz` checkpoints and per-model arrays were untracked at review time and are committed in the post-review commit.
  `generator.npz` is deterministic, and it was regenerated identically by each measurement run.
- `bayes.py` and `ckpt.py` were not hashed at freeze. They are unchanged in git since 2026-11-08.
- `FREEZE_HASHES.txt` keeps two error lines, corrected by its addendum.
- **Interim looks.** Boundary measurements of finished parents ran from 10:09, while T77, P86 and P87 were still training.
  No decision depended on them; the outputs went to files that were not inspected before all training ended. This is
  disclosed here.
- Wording fix: ITT and successful-only agree in direction and are **similar**, not identical, in size (pooled 0.36 vs 0.44).
