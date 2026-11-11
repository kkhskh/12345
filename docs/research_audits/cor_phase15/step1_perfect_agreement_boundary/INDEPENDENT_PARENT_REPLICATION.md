# Independent-parent replication

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
| P1(a) | B45 pooled, ITT, step 4000 | +0.362 | .018 (Holm .018) | [0.089, 0.657] | **confirmed** |
| P1(b) | B45-R1, ITT, step 4000 | +0.698 | 7.8e-5 (Holm 1.6e-4) | [0.586, 0.809] | **confirmed** |
| P1(b) | successful-only (8 vs 7) | +0.704 | 1.6e-4 | [0.580, 0.827] | consistent |
| P1(a) | successful-only | +0.440 | .003 | [0.204, 0.706] | consistent |
| P2 | D-R1 = B45-R1 − B34-R1 | +0.661 | 7.8e-5 | [0.520, 0.802] | **confirmed** |
| P2 | D-pooled (reported) | +0.284 | .090 | [−0.052, 0.668] | not significant |
| P3 | B45-R1 difference positive at 500/1000/2000/4000 | 4 of 4 | each ≤ 1e-4 | | **confirmed** |
| P4(i) | KL-matched T1 snapshot vs P at 4000 | +0.678 | 7.8e-5 | | **confirmed**; KL ranges overlap (T1 matched 0.042–0.141, P 0.029–0.212) |
| P4(ii) | ANCOVA arm coefficient (B45-R1 ~ arm + KL, all snapshots) | +0.648 | 1.5e-4 | | supports |
| P5 | probability-space R1 difference | +0.058 | 7.8e-5 | [0.043, 0.075] | sign agrees: **holds** |
| P5 | frozen expectation "R2, R3 not positive" | R2 −0.095 (n.s.); R3 +0.234 (p = .14) | | | R2 as expected; R3 point estimate positive but n.s. (partial mismatch, reported) |
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
