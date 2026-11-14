# Investigation XIII, Stage 0: fresh-cohort training protocol

Frozen before any XIII cohort job started. Commit and hashes: `logs/STAGE0_HASHES.txt`.

## Interpretation of the mandate's "before training"

P-count pretraining contains **no T4 information**: it predicts agreement counts only. The mandate's
freeze-before-training requirement is therefore applied as follows.

- **This Stage-0 freeze, before pretraining**, fixes:
  - the parent pool;
  - replacement rules;
  - readiness criteria;
  - fine-tuning protocol;
  - the development/confirmation boundary.
- **The prediction freeze**, which comes before *any* T4 fine-tuning of the cohort, fixes:
  - predictions, tests, metrics and the scorer;
  - null-control preflight results;
  - all hashes.

This is the same two-stage structure as XII. It is declared here, in advance.

## Diagnosis that motivates the protocol change (`scripts/diag_pretrain.py`, `logs/diag_pretrain.json`)

All 18 previous P-count parents were examined: 11 complete and 7 incomplete.

- **Incompleteness is group-level.** Every incomplete parent has one or two groups at chance accuracy (0.27–0.39),
  while every other group is at ≥ 0.998.
- Loss curves fall in **stages**, consistent with groups being learned one at a time. Incomplete parents sit on a
  plateau when the cosine schedule ends.
- The unlearned group varies: group 2 in four parents, group 1 in two, group 0 in two. It is not a fixed positional
  defect.

Development test, after the first 900–1500 of 3000 steps of a fresh-schedule continuation from the stuck checkpoint:
- P153 left its plateau (loss 1.00 → 0.54);
- P151 did not (0.52).

**The remedy therefore partly helps. It is not guaranteed.**

## Protocol: condition "XIII-gated", not pooled with earlier cohorts

| item | value |
|---|---|
| pool | **6 parents, seeds P160–P165**. Never used in the program; existing parents are P100–107, P130–135, P140–143, P150–156 |
| block 0 | `cohort13_pretrain.py P <seed>`: XII's script unchanged; 3000 steps, original schedule and data stream |
| gating | if aux_acc < 0.99, continue with `continue_pretrain.py` for 3000 steps (fresh warmup-cosine, block-specific stream 31_000_000 + 1000·seed + 2 + 100000·block); at most **2** continuation blocks |
| readiness | **complete iff final aux_acc ≥ 0.99** (unchanged criterion) |
| replacement | **none**. No parents are added or removed for any reason. If fewer than 4 are complete, the achieved n is reported and claims are restricted |
| fine-tuning | **every** parent (complete and incomplete): `native_train.py uP<seed> full_c1 4000 20 label logs/eval_conf13.npz`. Seed 20 is new. Frozen-prefix (C1) fine-tunes are **not** run, to save compute; no XIII prediction needs them |
| evaluation set | `logs/eval_conf13.npz`, stream 21_300_000 |
| analysis streams | 73_000_000 and above (fixed with the predictions) |
| development models | xP81, yP130–134, wP140–143 and XII's vP150–156 (all analysed in development) |
| confirmation models | uP160–165 only |

## What will exist before the prediction freeze

- Pretraining outputs: pre.json, logs and checkpoints.
- **No T4 fine-tune is started before the prediction-freeze commit.** native_train.py writes T4 evaluation metrics
  during training, so none of those outputs can exist before that commit.
- Every launch time is in `logs/COMPUTE_LOG.txt`.
