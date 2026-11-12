# Reproduction

**Environment**
- Python 3.13 with JAX (CPU) and scikit-learn.
- Run single-threaded: `export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"`.
- Earlier packages under `docs/research_audits/cor_phase15/` are read-only dependencies.

**Exact theory and controls (no models)**
- `python3 scripts/r1_information.py`, `scripts/formal_checks.py` → `logs/r1_information.json`, `logs/formal_checks.json`.

**Evaluation sets**
- `python3 scripts/build_eval.py dev 17100000` and `scripts/build_eval.py conf 17900000` → `logs/eval_{dev,conf}.npz`.
- Both evaluation sets are committed.

**Development (Investigation VI parents)**
- `scripts/eval_dev_parents.py`
- `scripts/layer2_reconstruction.py dev <models>`
- `scripts/dynamics_snapshots.py <models>`
- `scripts/accessibility_probes.py <models>`
- `scripts/separability_fit.py`
- `scripts/oracle_reader.py [steps]`
- `scripts/downstream.py PARENT SPEC REP [STEPS]`: queues in `logs/devqueue*.txt`; outputs to `checkpoints/dev/`.
- `scripts/downstream_postreview.py`: post-review option `C1_injna`.

**Confirmation**
1. **Stage 0:** `scripts/pretrain_lane.sh` → `checkpoints/yP9x/pre.npz`.
2. **Frozen runs:** `scripts/conf_lane.sh` over `logs/confqueue.txt` (`PHASE=conf scripts/downstream.py ...`) → `checkpoints/conf/`.
3. **Then:** `scripts/conf_mech.py`, `scripts/conf_probes.py`, `scripts/confirm7_analysis.py` → `logs/confirmatory7.json`.

**Verification**
- `scripts/verify_clean_extraction.sh`:
  - checks the manifest;
  - re-evaluates two committed confirmation checkpoints (a forward-pass reproduction of their step-4000 evaluation);
  - replays the frozen analysis from the committed per-run logs.

  This is replay and reproduction, not independent confirmation. The independent confirmation is the training of the 8 new
  lineages under the frozen protocol.
