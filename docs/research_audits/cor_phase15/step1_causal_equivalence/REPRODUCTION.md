# Reproduction and clean extraction (Investigation X)

**Dependencies.** The package must sit inside the **complete earlier `docs/research_audits/cor_phase15/` tree**. It imports read-only from these sibling packages, and some of them import further earlier packages (for example `step1_perfect_agreement_boundary` → `step1_curriculum_response_identification` → its own dependencies). A first clean check with only six siblings failed for this reason; see `logs/clean_extraction_check.txt`. The direct dependencies are:
- `step1_native_relational_mechanisms` (Investigation IX: fine-tuned models, parents yP13x, native.py, genlib.py);
- `step1_learned_relational_computation` (Investigation VIII);
- `step1_r1_causal_origin_and_selective_rescue` (Investigation VII: nnlib, evalrun, eval_dev);
- `step1_latent_reliability` (bayes);
- `step1_perfect_agreement_boundary` (xP81).

Python 3 with jax (CPU), optax, numpy, scikit-learn. Environment:
`XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"`.

**Scripts.**

| purpose | scripts |
|---|---|
| math | `math_abstraction.py` (Theorem X1); `verify_ftlib.py` (intervention-library identity checks) |
| pretraining | `cohort10_pretrain.py` |
| evaluation set | `build_eval_conf10.py` |
| fine-tuning | `native_train.py` (copied from Investigation IX) |
| analyses | `algorithm_discrimination.py`, `das.py`, `das_eval.py`, `subspace_shift.py`, `traj_alg.py`, `head_profile.py`, `profile_perm.py` |
| verdicts | `conf_evaluate.py` |

Every executed command is in `logs/queue_*.txt` and `logs/COMPUTE_LOG.txt`.

**Evidence types.**
- **Archived evidence:** the outputs in `logs/` and `checkpoints/`.
- **Independent forward replay:** `verify_ftlib.py` re-derives the stored models' logits through independent nnlib code
  (max |Δ| ≤ 1e-5). Representative scripts are rerun during the clean-extraction check below.
- **Independent retraining:** the confirmation parents and fine-tunes, from new seeds.

**Clean-extraction verification (performed, 2026-11-14).** Result: manifest OK (all files), verdicts re-derived identically, forward replay ≤ 1e-5, and wP140 algorithm discrimination recomputed with max |diff| 0.
1. Unzip `cor_inv10_causal_equivalence.zip` into an empty directory next to the sibling packages.
2. Check the manifest with `sha256sum -c MANIFEST.sha256`.
3. Rerun `scripts/conf_evaluate.py` (verdicts must match `logs/confirmation_results.json`) and `scripts/verify_ftlib.py`.

The results of this check are in `logs/clean_extraction_check.txt`.
