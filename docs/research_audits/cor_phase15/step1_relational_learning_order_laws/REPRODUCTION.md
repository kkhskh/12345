# Reproducing Investigation XV

**Not standalone.** XV imports code from five earlier COR packages (`logs/DEPENDENCIES.json`): the GATE trainer and forward pass, the XIV GATE loader, and the T4 DeepSets net and exact posterior. Extract the ZIP into a `cor_phase15/` directory that already contains them. Set `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"`.

**Verification.** `bash scripts/verify_clean15.sh ZIP /path/cor_phase15 /tmp/V.txt` runs the 16 mandated steps; the archived output is in `logs/VERIFY_CLEAN.txt`.

**Mathematics.**
- `python3 scripts/xv_math.py OUT a:beta[:nu] ...` computes the exact references.
- `python3 scripts/xv_math_verify.py` runs the independent checks.
- `python3 scripts/xv_noisy_math.py` covers Workstream F.

**Design.** `python3 scripts/xv_design.py` writes `logs/XV_design.json` and the training and measurement queues. The protocol is frozen in TASK_VARIANT_EXPERIMENTAL_DESIGN.md (commit `7957fee`).

**Confirmation.**
- Two lanes: `QUEUE=logs/queue_train.txt bash scripts/lane2.sh A`, then `QUEUE=logs/queue_measure.txt bash scripts/lane2.sh A`.
- Each GATE-variant run is `BLIND=1 python3 scripts/x_train.py COND SEED DIR`, about 33 minutes per 4000 steps at 2 steps/s.
- Each DeepSets T4 run is `t4_ds_train.py`, about 3 minutes.

**Scoring.** `python3 scripts/score_XV.py` and `python3 scripts/score_T4.py`. Both are fail-closed.

**Exploratory analyses.**
- `bash logs/expl/run_expl.sh` covers the learning signal and the mechanism checkpoints.
- `python3 scripts/xv_sensitivity.py` covers the threshold sensitivity.

**Streams.**
- Development: 97_400_000–97_900_000, 99_000_001, 99_200_000.
- Confirmation: 98_700_000 (GATE variants) and 98_800_000 (DeepSets T4).
- Training data: 41_000_000 + 1000·seed (GATE variants) and the original nets formula (DeepSets).
