# Reproducing Investigation XVI

**The archive is not standalone.**
- Place it as `cor_phase15/step1_relational_acquisition_limits/` next to the sibling packages listed in `logs/DEPENDENCIES.json`. The XV package `step1_relational_learning_order_laws` must include its `confirm_outputs/` (the continuations start from six XV step-4000 checkpoints, hashed in `logs/FREEZE_XVI_HASHES.txt`).
- **Software:** Python 3.13, JAX, optax, NumPy, SciPy and scikit-learn, at the versions in `logs/DEPENDENCIES.json`.
- **Environment:** set `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"`. Bitwise reproducibility was verified with this setting on CPU.

## Mathematics (seconds to minutes)

```
python3 scripts/xvi_math_verify.py        # 7 checks -> logs/xvi_math_verified.json (all_pass)
python3 scripts/xvi_value_search.py       # 240-condition value-vs-cue grid -> logs/xvi_value_grid.json
python3 scripts/xvi_math.py OUT.json 4:0.9:0.9:0.3333333333333333:7 ...   # arity-4 orders 1-5
```

## Design, freeze and preflight

```
python3 scripts/xvi_design.py             # logs/XVI_design.json, XVI_conditions.json, queue_train.txt
bash scripts/freeze_hashes.sh             # logs/FREEZE_XVI_HASHES.txt
python3 scripts/preflight_XVI.py          # synthetic planted outcomes and refusals -> logs/preflight_XVI.json
```

## Training (about 22 lane-hours on 2 CPUs)

```
QUEUE=logs/queue_train.txt bash scripts/lane2.sh A1 &  QUEUE=logs/queue_train.txt bash scripts/lane2.sh A2 &
```

- Each queue line runs `scripts/xvi_train.py` with `BLIND=1`.
- Runs resume from `resume.pkl` after an interruption (bitwise deterministic: `logs/expl/resume_determinism_check.txt`). Clear `logs/claims/<key>` for any unfinished line before relaunching.
- To regenerate any intermediate checkpoint exactly:

  ```
  python3 scripts/xvi_replay.py COND SEED OUT SCHED STEPS STOP SAVE_LIST [INIT]
  ```

## Scoring and statistics

```
python3 scripts/score_XVI.py              # fail-closed -> logs/score_XVI.json (verdicts P1, P2, P3, P5)
python3 scripts/xvi_stats.py              # -> logs/xvi_stats.json
```

## Exploratory analyses (logs/expl/)

| script | what it does |
|---|---|
| `xvi_traj_offline.py` (run_xv_offline.sh) | XVI diagnostics on XV snapshots |
| `xvi_xv_original.py` | frozen definitions applied to XV |
| `xvi_gradsig2.py` (run_gradsig2_xv.sh, run_gradsig2_xvi.sh) | the rebuilt learning-signal estimator |
| `run_replays.sh`, `run_probes.sh` (xvi_probe.py) | cue-conditioned probes on replayed checkpoints |
| `xvi_expl_summary.py` | summary tables |

## Clean-extraction verification (the 16 mandated steps)

```
bash scripts/verify_clean16.sh ZIP COR_PHASE15_ROOT LOG     # output archived as logs/VERIFY_CLEAN.txt
```

Step 16 runs `scripts/check_reports16.py`, which checks the reports against the primary result files.
