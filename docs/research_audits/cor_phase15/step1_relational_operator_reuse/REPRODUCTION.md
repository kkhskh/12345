# Reproducing Investigation XVII

**Not standalone.**
- Place the package as `cor_phase15/step1_relational_operator_reuse/` next to the sibling packages in `logs/DEPENDENCIES.json`. That includes `step1_relational_acquisition_limits` (XVI) and `step1_relational_learning_order_laws` (XV), whose checkpoints the development replays regenerate.
- **Environment:** set `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"`. Bitwise determinism was verified on CPU.

## Stage 1 (O/D)

```
python3 scripts/xvii_od_check.py                 # Proposition S numbers -> logs/xvii_od_check.json
```

## Development (existing XVI models)

```
bash logs/expl/run_dev_replays.sh                # dense bitwise replays: C_s202, WB_low_s503, WA_const_s402 -> checkpoints/replay/
python3 scripts/xvii_slot_traj.py COND DIR OUT   # per-slot reconstruction
python3 scripts/xvii_dev_relabel.py COND CKPT A U OUT
python3 scripts/xvii_dev_attention.py COND CKPT OUT
bash logs/expl/run_devtrain.sh                   # slot-targeted and code-only training arms
```

## Freeze and preflight

```
python3 scripts/xvii_design.py
bash scripts/freeze_hashes.sh
python3 scripts/preflight_XVII.py
```

## Confirmation (about 14 lane-hours on 2 CPUs)

```
QUEUE=logs/queue_xvii.txt bash scripts/lane2.sh B1 &  QUEUE=logs/queue_xvii.txt bash scripts/lane2.sh B2 &
python3 scripts/score_XVII.py                    # fail-closed -> logs/score_XVII.json
```

The queue runs, in order: training (`xvii_train.py`, 20-step snapshots, BLIND = 1), the frozen analyses (`xvii_confirm_analysis.py` → `logs/analysis/`), and the learning arms (`xvii_learning_arm.py` → `learning/`).

## Verification

```
bash scripts/verify_clean17.sh ZIP COR_PHASE15_ROOT LOG     # archived as logs/VERIFY_CLEAN.txt
bash scripts/verify_parts17.sh PARTS_DIR                    # multipart reconstruction, archived as logs/VERIFY_PARTS.txt
```
