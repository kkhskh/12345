# Reproduction, dependency inventory and clean extraction (Investigation XI)

## This package is not standalone

It must sit inside the **complete** `docs/research_audits/cor_phase15/` tree, because it reads models, subspaces and
evaluation sets from earlier packages.

**Direct dependencies**, found by `scripts/deps_inventory.py` → `logs/DEPENDENCIES.json`:

| sibling package | what XI reads |
|---|---|
| `step1_causal_equivalence` (Investigation X) | wP140–143 parents (`checkpoints/zP14x/pre.npz`) and fine-tunes (`checkpoints/native_train/wP14x_*`, seed-1 runs); DAS E-subspaces (`logs/das/*_E_L2_q7_k6*_R.npy`); `logs/eval_conf10.npz`; `logs/traj_alg.json`, `logs/traj_alg_conf.json` |
| `step1_native_relational_mechanisms` (Investigation IX) | yP13x parents and dev fine-tunes (`checkpoints/`), with all intermediate step checkpoints |
| `step1_r1_causal_origin_and_selective_rescue` (VII) | `nnlib`, `evalrun`, `r1lib` (imported), `logs/eval_dev.npz` |
| `step1_learned_relational_computation` (VIII) | imported by native.py (z-parent paths) |
| `step1_perfect_agreement_boundary`, `step1_curriculum_response_identification`, `step1_latent_reliability` (bayes), `step1_relational_computation_atlas`, `step1_relational_mechanism_deepening`, `step1_cross_implementation_causal_equivalence` | transitively imported modules |

`logs/DEPENDENCIES.json` lists:
- the SHA-256 of **every one of the 247 external data files** read (parents, all used step checkpoints, R files, eval
  sets, trajectory files);
- the **11 external Python modules** (path + SHA-256).

Missing files at inventory time: none.

**Software:**
- Python 3.13.18;
- jax and jaxlib 0.12.0.1 (CPU);
- numpy 2.5.3;
- optax 0.2.8;
- scikit-learn 1.9.1;
- scipy 1.18.1.

Environment: `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"`.

## Package-local code

- `src/ftlib.py`, `src/native.py` and `src/genlib.py` are copied from Investigation X with **one change**: model and
  parent paths now resolve the Investigation X package (`X10`). This was needed for the wP and seed-1 models.
- `scripts/lane.sh` is the queue runner.

## Scripts

| workstream | script | output |
|---|---|---|
| A math | `audit_x1.py`, `audit_x1_extra.py` | `logs/audit_x1.json`, `logs/audit_x1_extra.json` |
| B mechanism | `mech_construct.py`, `stage_interaction.py`, `factorial.py`, `factorial_ablate.py`, `restore.py`, `perm_compare.py`, `dyn_factorial.py` | `logs/mech_construct*.json`, `stage_interaction_dev.json`, `factorial_{dev,conf}.json`, `factorial_ablate_{dev,conf}.json`, `restore_{dev,conf_explore}.json`, `perm_compare_dev.json`, `dyn_factorial_{dev,conf}.json` |
| C transfer / identification | `cross_interchange.py`, `das_decoy.py` | `logs/cross/`, `logs/cross_conf/`, `logs/decoy/`, `logs/decoy_conf/` |
| D generalization | `generalize.py` | `logs/generalize_{dev,conf}.json` |
| verdicts | `score_conf.py` (frozen) | `logs/score_conf.json` |
| inventory | `deps_inventory.py` | `logs/DEPENDENCIES.json` |
| verification | `verify_clean.sh` | `logs/clean_extraction_check.txt` |

Every executed command is in `logs/queue_*.txt` and `logs/COMPUTE_LOG.txt`, with start and end times and return codes.
The freeze hashes are in `logs/FREEZE_HASHES.txt`.

## Clean-extraction verification

The procedure is performed by `scripts/verify_clean.sh`; results are in `logs/clean_extraction_check.txt`.

1. Copy the cor_phase15 tree, without this package, to a temporary root. Extract the ZIP into it.
2. Check the manifest: `sha256sum -c MANIFEST.sha256`.
3. Re-run the mathematics: `audit_x1.py` and `audit_x1_extra.py`, comparing the JSON with the archived copies.
4. Re-derive the verdicts: `score_conf.py`, which must reproduce `logs/score_conf.json` exactly.
5. Forward-pass / causal check: re-run `factorial_ablate.py` on wP140 full_c1 (confirmation stream) and compare with
   the archived `factorial_ablate_conf.json` entry.
6. Record missing dependencies: re-run `deps_inventory.py` in the extracted location and compare its file hashes with
   the archived inventory.

## Clean-extraction verification result (performed 2026-11-14T06:14Z)

The full log is `logs/clean_extraction_check.txt`. All six steps passed:

1. Extracted into a fresh copy of the cor_phase15 tree, with this package removed beforehand.
2. Manifest OK (209 files).
3. `audit_x1.json` and `audit_x1_extra.json` re-generated identically.
4. `score_conf.py` re-derived all six verdicts; `score_conf.json` identical.
5. Forward-pass causal check (factorial_ablate on wP140 full_c1): max |diff| vs archived = 0.0.
6. Dependency inventory: 247 external data files, 0 hash mismatches, 0 missing; modules identical.

The check was run on the ZIP *before* this log and this section were added. The final ZIP differs from the verified one
only by these additions and by the regenerated manifest. That final ZIP's manifest was re-checked after extraction
(see the final response).
