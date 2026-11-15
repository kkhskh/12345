# Frozen confirmation and statistics (Workstream D)

Investigation XIII-B, Phase 1.5 Step 1.

## 1. What is being confirmed, and what kind of replication it is

**Units.** New fine-tuning seeds (seed 21, `native_train.py uP<p> full_c1 4000 21 label logs/eval_conf13.npz`, the same protocol as
XIII's seed 20) of the five complete XIII parents uP160, uP161, uP163, uP164 and uP165.

**What these replicates test.** They are **fine-tuning replicates** of already-pretrained parents, so they test fine-tuning variability.
They are **not** independent pretraining replicates and are not counted as such. No new pretraining cohort was run.

**Fresh inputs.** The replicates were launched first (23:29Z and 01:05Z). Their outputs were not opened before the freeze. Confirmation
analyses use new streams (79_000_000 and above) and a new evaluation set (`logs/eval_13b.npz`, stream 21_400_000).

**Development data.** Everything in the other four documents comes from XIII's seed-20 models and earlier pools. Thresholds were set there.

## 2. Freeze

FREEZE_SECTION

## 3. Scorer preflight

PREFLIGHT_SECTION

## 4. Confirmation results

(Filled after the confirmation queue completes.)

## 5. XIII's original verdicts (preserved, not re-scored)

| prediction | XIII verdict |
|---|---|
| P1–P4, P7 | met the frozen criteria on 2 strong-mixture parents |
| P5, P8 | falsified |
| P6 | exploratory, n = 1 |

Sensitivity: including uP164 as mixture-class would falsify P1 and P2.

XIII-B does not re-score XIII. It tests new hypotheses that XIII's results raised, on new fine-tunes.

## 6. Deviations and execution incidents

See `logs/COMPUTE_LOG.txt`. Development OOM kills and requeues are listed there. XIII-B's `lane.sh` records each job's own exit status,
which fixes XIII review 2's finding.
