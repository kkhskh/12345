# Reproduction and clean extraction (Investigation IX)

**Dependencies.**
- Read-only sibling packages: `step1_r1_causal_origin_and_selective_rescue` (Investigation VII: nnlib, evalrun, r1lib,
  eval_dev), `step1_learned_relational_computation` (Investigation VIII: v8lib, Stage-0 lineages zP100–107),
  `step1_latent_reliability` (bayes) and `step1_perfect_agreement_boundary` (xP80/xP81).
- Python 3 with jax (CPU), optax, numpy and scikit-learn.
- Environment: `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"`.

**Commands.** Every executed command is listed in `logs/queue_*.txt` and `logs/COMPUTE_LOG.txt`.

| task | command |
|---|---|
| math | `python3 scripts/verify_math.py`; `python3 scripts/verify_math2.py`; `python3 scripts/adaptive_sources.py` |
| Stage-0 parents | `python3 scripts/cohort9_pretrain.py P 130` … `135` |
| evaluation set | `python3 scripts/build_eval_conf9.py` |
| native experiments | `scripts/decode_compose.py`, `native_reader.py`, `native_train.py`, `mech_native.py`, `mech_routing2.py`, `probe_location.py`, `probe_votes.py` |
| generalization | `scripts/gen_readers.py` (`GEN_EVAL_OFFSET=5000000` for confirmation) |
| verdicts | `python3 scripts/conf9_evaluate.py`; `python3 scripts/conf9_evaluate_A1.py` |

**Verified reruns.** The xP81 full_c1 run was executed twice: the first run was lost at step 1803, and the second
reproduced identical curve values through step 1211. Lost confirmation runs were rerun with identical commands.
No other procedure is claimed reproducible beyond its single execution.

**Integrity.**
- `logs/STAGE0_HASHES` (pretraining) and `logs/PROTOCOL_HASHES.txt` (freezes).
- `MANIFEST.sha256` covers every package file. Check it with `sha256sum -c MANIFEST.sha256` after
  `git archive <commit> docs/research_audits/cor_phase15 | tar -x`.
