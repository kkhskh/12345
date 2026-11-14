# Stage 0: fresh confirmation cohort protocol (frozen before any training of the cohort)

Investigation XII, frozen 2026-11-14 before any fresh-cohort job was started. Commit hash and file hashes are in
`logs/STAGE0_HASHES.txt`.

## Cohort

| item | value |
|---|---|
| parents | **P150, P151, P152, P153, P154**: P-count pretraining, `scripts/cohort12_pretrain.py P <seed>`. This is Investigation X's `cohort10_pretrain.py` with only the docstring changed. Seeds 150–154 have never been used in the program (checked: existing parents are zP100–107, zP130–135, zP140–143, zT1110–1113) |
| readiness criterion (prospective) | a parent is **complete** iff `pre.json` aux_acc ≥ 0.99. Incomplete parents are fine-tuned and analysed, but reported **separately** and excluded from aggregate pass rules. The only information read before the freeze of predictions is pre.json and the pretraining log |
| fine-tuning | `scripts/native_train.py vP15x full_c1 4000 10 label logs/eval_conf12.npz` and the same with `c1`. This is Investigation X's native_train.py unchanged. **Fine-tuning seed 10** (new; previous cohorts all used seeds 0/1, so the fine-tuning data stream is also new) |
| same-parent replicates | full_c1 **seed 11** on the **first two complete parents in seed order** (decided from pre.json only) |
| reserve | if fewer than 4 parents are complete, no extra parents are trained. The shortfall is reported, and aggregates use the complete parents only |
| evaluation set | `logs/eval_conf12.npz`, built by `scripts/build_eval_conf12.py` from stream **21_200_000** (unused in the program) with Investigation VII's unchanged `build_eval` |
| analysis streams | new natural streams 70_000_000+ for every XII confirmation analysis (specified with the predictions) |

## Information barrier

- No T4 fine-tuning of the cohort starts before `FRESH_CONFIRMATION_AND_STATISTICS.md` Part I (frozen predictions,
  procedures, statistics, criteria) is committed with hashes.
- native_train.py writes training curves and evaluation summaries. These are **not read** until the confirmation
  analyses run after the freeze.
- Development uses only existing models: xP81, yP130–134, wP140–143 and their seeds.
- Existing Investigation X/XI models are **never** relabelled as new confirmation.
