# Reproducing Investigation XVIII

**Not standalone.**
- Place this package as `cor_phase15/step1_pathway_selection_causal_identification/` next to the sibling packages listed in `logs/DEPENDENCIES.json`.
- Those packages supply the pretrained parents uP160–uP165, the XIII-B evaluation file, the XIV harness modules and the XIV development checkpoints.

**Environment.**
- Always set `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"`.
- Training is bitwise deterministic on CPU under this flag (`logs/replay_check.json`; Reviewer D's replays).

## Order of work

1. **Reconstruction and development.**
   - `scripts/replay_check.py`, `dev_behav.py`, `dev_necessity.py`, `dev_rhoL.py`.
   - Unit checks: `scripts/unit_modes.py`, `unit_modes2.py`, `unit_modes3.py`, `unit_sha_far.py`.
   - Development queues `logs/queue_dev*.txt`; outputs in `logs/dev/`.
2. **Freeze.**
   - `python3 scripts/xviii_design.py`, then `bash scripts/freeze_hashes18.sh`.
   - Preflight: `python3 scripts/preflight_XVIII.py` (ALL_PASS).
3. **Confirmation and later stages.**
   - `bash scripts/run_all18.sh`. Its stages: confirmation, H3, scoring, continuations, screening, parent rule, parents.
   - After the container restart, `scripts/resume18.sh` ran the remaining stages.
   - Each run is one `scripts/xviii_train.py` command followed by `scripts/measure18.py`.
4. **Scoring.**
   - Frozen: `scripts/score_XVIII.py`, `scripts/score_parents18.py`, `scripts/score_cont18.py`.
   - Post-hoc (labelled): `scripts/posthoc_nobehav18.py`, `scripts/posthoc_parents18.py`.
5. **Verification.** `bash scripts/verify_clean18.sh ZIP COR_PHASE15_ROOT LOG`; output archived as `logs/VERIFY_CLEAN.txt`.
6. **Report check.** `python3 scripts/check_reports18.py`.

## Compute

About 11 minutes per 1211-step run (including measurement) per CPU lane on a 2-vCPU Xeon. Run counts:
- 48 confirmation runs;
- 22 screening runs;
- 12 parent runs;
- 37 stability continuations;
- about 30 development runs.
