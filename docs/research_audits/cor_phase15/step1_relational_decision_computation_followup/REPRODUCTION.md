# Reproduction (Investigation XIII-B)

## Not standalone

The archive contains this package, `step1_relational_decision_computation_followup/`. It must be extracted next to the sibling
`cor_phase15` packages listed in `logs/DEPENDENCIES.json` → `sibling_packages_required`. The most important is XIII,
`step1_relational_decision_computation`, which holds:
- the uP160–165 parents;
- the seed-20 fine-tunes used for development;
- the XIII code base copied into `src/`.

Every external data file is pinned by sha256 in `logs/DEPENDENCIES.json`. The **seed-21 fine-tunes** (`checkpoints/native_train/*seed21*`)
and the **adapted two-colluder models** (`checkpoints/adapt2/`) are inside this package and are covered by `MANIFEST.sha256`.

## Environment

The exact versions are in `logs/DEPENDENCIES.json` (python, jax/jaxlib, numpy, optax, scikit-learn, scipy). Set
`XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"`. Everything runs on CPU; 2 cores and 8 GB were used.
Running too many jobs at once caused OOM kills, all of them logged in COMPUTE_LOG.

## Pipeline

1. **Seed-21 fine-tunes.** `python3 scripts/native_train.py uP<p> full_c1 4000 21 label logs/eval_conf13.npz` for p ∈ {160, 161, 163, 164, 165}.
2. **Development analyses** on seed-20 models: commands are in `logs/COMPUTE_LOG.txt` and `logs/queue_*.txt`. The scripts are `locate.py`,
   `path_chain.py`, `attn_profile.py`, `class_features.py`, `fit_family.py`, `algorithm_discrimination.py`, `l2_components.py`,
   `l1_graded.py`, `das_bal.py`, `adapt2.py`, `decode_J.py`, `jpath.py`, `censored_bayes.py` and `traj_resp.py`.
3. **Freeze.** The freeze consists of:
   - `logs/predictions_13b.json` and `scripts/score_13b.py`;
   - the preflight, `scripts/preflight_13b.py` (which uses `build_dev13b.py`);
   - the queues, `scripts/make_queues_13b.py` → `logs/queue_conf13b_K*.txt`;
   - the hashes, `logs/FREEZE_13B_HASHES.txt` (commit d7e74a4).
4. **Confirmation.** Run every line of `logs/queue_conf13b_K1.txt` and `logs/queue_conf13b_K2.txt` from the package root, directly or
   through `scripts/lane.sh`.
5. **Scoring and reporting.** `python3 scripts/score_13b.py`, then `python3 scripts/report_13b.py`. The report covers rules RB1–RB4 and
   addendum AQ1.
6. **Post-freeze and post-hoc analyses**, labelled as such:
   - `fit_powermean.py`, for development and for addendum AQ1;
   - `das_bal_site.py`;
   - `l2_decomp_all.py`;
   - `adapt2.py` with `AD_STEPS=4000` (the long control) and with `AD_BETA=1.0` (the β = 1 arms).
7. **Archive check.** `bash scripts/verify_clean13b.sh ZIP COR_PHASE15_ROOT LOG` runs the 10 steps; the result is in `logs/VERIFY_CLEAN.txt`.
