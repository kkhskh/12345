# Reproduction, dependency inventory and clean extraction (Investigation XII)

## This package is not standalone

**Bundled in the package:**
- all XII code (`src/`, `scripts/`);
- **all fresh-cohort checkpoints**: 7 parents (`checkpoints/zP15x/`) and 16 fine-tunes with every saved step
  (`checkpoints/native_train/vP15x_*`);
- the XII evaluation set (`logs/eval_conf12.npz`);
- the fresh E-subspaces (`logs/das/`);
- all raw outputs, queues, compute log, freeze hashes and reviews.

**Read from the surrounding `cor_phase15/` tree.** The tree must be complete. `scripts/deps_inventory.py` writes
`logs/DEPENDENCIES.json`, which holds:
- the SHA-256 of all **267 external data files**: development checkpoints, Investigation X DAS subspaces, eval_dev /
  eval_conf10, trajectory logs and XI decoy logs;
- the **11 external Python modules**, with path and SHA-256.

Sibling packages:
- `step1_causal_equivalence` (X);
- `step1_native_causal_construction` (XI);
- `step1_native_relational_mechanisms` (IX);
- `step1_r1_causal_origin_and_selective_rescue` (VII: nnlib, evalrun, r1lib, eval_dev);
- `step1_learned_relational_computation` (VIII: v8lib, used by the eval-set builder);
- `step1_latent_reliability` (bayes, ckpt);
- `step1_relational_computation_atlas` and `step1_foundational_atlas_continuation` (pretraining: train_tasks, nets);
- `step1_perfect_agreement_boundary`, `step1_curriculum_response_identification`,
  `step1_relational_mechanism_deepening`, `step1_cross_implementation_causal_equivalence` (transitive imports).

**Software:**
- Python 3.13.18;
- jax and jaxlib 0.12.0.1 (CPU);
- numpy 2.5.3;
- optax 0.2.8;
- scikit-learn 1.9.1;
- scipy 1.18.1.

Set `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"`.

## Order of execution

Every command, with start and end times and return code, is in `logs/COMPUTE_LOG.txt`.

1. **Stage 0.**
   - `STAGE0_PROTOCOL_FREEZE.md` (commit `a7f49aa`).
   - `scripts/build_eval_conf12.py`.
   - `scripts/cohort12_pretrain.py P <seed>` for 150–156 (`logs/queue_pre_*.txt`; P155/P156 via Amendment 1,
     commit `5425317`).
2. **Freeze.**
   - `FRESH_CONFIRMATION_AND_STATISTICS.md` Part I, `scripts/score_xii.py`, `scripts/stats_xii.py`,
     `scripts/make_conf_queues.py` and the queues (commit `87e2a9f`).
   - Hashes in `logs/FREEZE_XII_HASHES.txt` and `logs/AMENDMENT1_HASHES.txt`.
3. **Fine-tuning.** `logs/queue_train_T1.txt` and `T2`, running `scripts/native_train.py` and
   `scripts/rep_train.sh`.
4. **Confirmation analyses.** `logs/queue_conf_C1.txt` and `C2`, plus `scripts/conf_replicates.sh`,
   `conf_transfer.sh` and `conf_merge.sh`. The merge runs `score_xii.py` and `score_p6_text.py`.
5. **Development and exploratory analyses.** `logs/queue_dev_*.txt` and `logs/queue_explore.txt`. The analysis
   scripts include `circuit_l1.py`, `alt_ablation.py`, `dyn_mech.py`, `adapt_c2.py`, `vs_linear.py`,
   `vs_linear_randctrl.py`, `disc_ident.py` and `restore_nat.py`.
6. **Mathematics.** `scripts/math_xii.py` → `logs/math_xii.json` (deterministic, about 3 minutes).

## Clean-extraction verification

Script: `scripts/verify_clean.sh ZIP COR_PHASE15_ROOT LOG`. Result: `logs/clean_extraction_check.txt`. Its eight
steps:
1. extract into a fresh copy of the tree (with this package removed);
2. check the manifest;
3. re-run `math_xii.py`;
4. re-derive the verdicts;
5. recompute the CIs (fixed bootstrap seeds);
6. replay `alt_ablation` on a complete fresh parent;
7. check the dependency inventory, and scan for absolute paths;
8. record failures.

The run's outcome is recorded at the end of this file.

## Clean-extraction result (run 2026-11-14T15:12Z; `logs/clean_extraction_check.txt`)

All eight steps passed:

1. The ZIP extracted into a fresh copy of the tree, with this package removed.
2. Manifest OK: 582 files.
3. `math_xii.json` was regenerated identically (M1 max error 4e-15).
4. All verdicts were re-derived.
5. Verdicts, effects + CIs, and the check list were identical.
6. The alt_ablation replay on P150 matched the archive (max |diff| 0.0).
7. 267 external files: 0 hash mismatches, 0 missing, modules identical, 0 absolute local paths.
8. No failures.

The check was run on the ZIP before this section and its log were added. The final ZIP differs only by those two
files and the regenerated manifest, and its manifest was re-checked after extraction.
