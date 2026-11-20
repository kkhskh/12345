# Reproducing XIX

**Environment.**
- Two CPUs, JAX on CPU.
- Always set `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"`. Bitwise determinism depends on it.

**Dependencies.**
- Every external file XIX imports or reads is listed with its SHA-256 in `logs/DEPENDENCIES19.json`, with paths relative to `cor_phase15/`.
- That covers the measurement libraries, the harness libraries, the 12 XVIII reference checkpoints and the uP160 parent.
- The archive contains these files under their original paths. **It is not standalone with respect to the earlier investigations' full packages or the parent's pretraining history.**

## Pipeline

1. **Graph audit.** `python3 scripts/graph_audit19.py && python3 scripts/graph_table19.py` produces `logs/graph_audit19.json` and `logs/graph_table19.csv`.
2. **Scorer preflight.**
   - `python3 scripts/scorer_validation19.py 97960000 98960000 99960000`
   - `python3 scripts/scorer_validation_summary19.py`
   - `python3 scripts/scorer_planted19.py` (77 checks)
3. **Development.** The queues are `logs/queue_dev*.txt`, run with `QUEUE=<file> bash scripts/lane2.sh <name>`.
   - Tables: `scripts/dev_table19.py`, `scripts/dev_bandcheck19.py`.
   - Diagnostics: `scripts/s100diag19.py`, `scripts/compat19.py`, `scripts/distance19.py`, `scripts/state_diff19.py`.
4. **Design and freeze.**
   - `python3 scripts/conf_setup19.py logs/conf_spec19.json` writes the queues and `logs/conf/design19.json`.
   - `python3 scripts/freeze19.py` writes `FREEZE_XIX.json`. The committed freeze is `478a038`.
5. **Confirmation.** `bash scripts/conf_lanes19.sh C1 & bash scripts/conf_lanes19.sh C2`.
   - This runs 54 step-100 window states, then 99 continuations, each followed by `measure19.py`.
   - Claims are an md5 of the queue line. After a crash, delete the claims of the interrupted jobs and restart the lanes; runs resume from their last snapshot.
6. **Scoring.**
   - Primary: `python3 scripts/score19.py logs/conf/design19.json logs/conf/measure_c19.jsonl logs/conf/score_c19.json`
   - Sensitivity variants: add `--sens <variant>` for each of delta0, delta20, phicut70, phicut80, eps003, eps03, lin, evalB, thr_pm05, thr_strict, comp_strict, comp_loose.
   - Statistics: `python3 scripts/stats19.py logs/conf/score_c19.json logs/conf/stats_c19.json`
7. **Functional battery.**
   - `python3 scripts/func_battery19.py 93300000 logs/conf/func_battery_c19.json <run>@uP160@1211 ...`
   - `python3 scripts/score_func19.py logs/conf/score_c19.json logs/conf/func_battery_c19.json logs/conf/func_score_c19.json`
8. **Verification and archive.**
   - `python3 scripts/verify19.py` checks the 21 mandate items and writes `logs/VERIFY_XIX.json`.
   - `python3 scripts/build_archive19.py <outdir>` builds the archive parts and the audit archive.
