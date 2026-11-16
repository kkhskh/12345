# Reproducing Investigation XIV

**Not standalone.** XIV reads models, modules and evaluation sets from sibling COR packages. The full list, with exact SHA-256 hashes, is in `logs/DEPENDENCIES.json`. Extract the ZIP into a `cor_phase15/` directory that already contains those packages.

**Environment.**
- Python 3.13, JAX and optax, numpy, scipy and scikit-learn; versions are in `logs/DEPENDENCIES.json`.
- Always set `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"`. Bitwise reproduction depends on the fixed summation order this gives.

## Clean-extraction check (the 13 mandated steps)

```
bash scripts/verify_clean14.sh /path/cor_inv14_cross_family_relational_foundations.zip /path/cor_phase15 /tmp/VERIFY.txt
```

The output of the archived run is `logs/VERIFY_CLEAN.txt`.

## Stage I (T4 controls)

- **Design.** `python3 scripts/i_design.py` writes `logs/I_design.json`, `logs/queue_I_train.txt` and `logs/queue_I_measure.txt`. These are frozen; see `logs/FREEZE_I.md`.
- **Training and measurement.** Run two lanes, each on the training queue and then on the measurement queue:
  ```
  QUEUE=logs/queue_I_train.txt bash scripts/lane2.sh A
  QUEUE=logs/queue_I_measure.txt bash scripts/lane2.sh A
  ```
  Each run is one `scripts/ft_factorial_c.py` command.
- **Scoring.** `python3 scripts/score_I.py`. The scorer is fail-closed. It writes `logs/score_I.json`.
- **Development and calibration.** `scripts/i_attn_dev.py`, `scripts/i_sham_dev.py`, `scripts/i_calib.py`.
- **Restored intermediate checkpoints.** Hashes are in `logs/I_restored_checkpoints.txt`. They are regenerable by replaying the XIII-E commands, because training is deterministic.

## Stage III (GATE)

- **Exact mathematics.**
  - `scripts/g_rules_exact.py`
  - `scripts/g_order_ceilings.py`
  - `scripts/t4_order1.py`
  - `scripts/math_verify.py`
  - `scripts/x_amend_refs.py`
- **New models.** `python3 scripts/g_train.py 3` and `python3 scripts/g_train.py 4`. Each takes about 40 minutes on one core without contention.
- **Frozen prediction X1.**
  - `python3 scripts/score_X.py` (FREEZE_X);
  - `python3 scripts/score_X_amend.py` (FREEZE_X_AMEND).

  Both are fail-closed. Preflights: `scripts/preflight_X.py` and `scripts/preflight_X_amend.py`.
- **Mechanisms.**
  ```
  python3 scripts/g_mech.py NAME CKPT 98200000 logs/gate_mech/NAME.json
  python3 scripts/g_das.py NAME CKPT SITE 98300000 logs/gate_mech/das_NAME.json
  ```

## Evaluation streams used

- **Development:** 97_000_000–97_500_000, 99_000_000, 99_100_000, 98_200_000, 98_250_000, 98_300_000.
- **Confirmation:**
  - 95_300_000 (Stage I measurements and sham validity);
  - 98_500_000 (X1).
