# Reproduction (Investigation XIII-C)

## Environment

- Python 3 with jax / jaxlib (CPU), optax, numpy and scipy, at the versions in `logs/DEPENDENCIES.json`.
- Training lanes set `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"`. Bitwise reproduction of training
  depends on this setting.
- The machine used had 2 CPU cores and 8 GB of RAM.

## External dependencies

These are NOT in this archive. They are pinned by SHA-256 in `logs/DEPENDENCIES.json`.

| path relative to `docs/research_audits/cor_phase15/` | contents |
|---|---|
| `step1_r1_causal_origin_and_selective_rescue/` | nnlib, evalrun, r1lib; dev eval set |
| `step1_latent_reliability/`, `step1_relational_computation_atlas/`, `step1_foundational_atlas_continuation/` | `nets.py`, `bayes.py`, `ckpt.py` (imported by nnlib) |
| `step1_relational_decision_computation/` (XIII) | XIII cohort parents `checkpoints/zP160..zP165/pre.npz`; seed-20 fine-tunes |
| `step1_relational_decision_computation_followup/` (XIII-B) | seed-21/22/23 fine-tunes; `logs/eval_conf13.npz` |
| further `step1_*` packages | listed in `DEPENDENCIES.json`; only files actually read are pinned |

## Steps

1. **Bitwise checks:**
   - `python3 scripts/repro_check.py uP161 22 50` reproduces XIII-B seed-22 step 50 exactly.
   - `python3 scripts/repro_resume.py` checks that a resumed run is bitwise identical to an uninterrupted one.
2. **Mathematics:**
   - `python3 scripts/math_identifiability.py`, giving `logs/math_identifiability.json`;
   - `python3 scripts/c_math.py`, giving `logs/c_math.json`.
3. **Factorial and confirmation training:**
   - command lines are in `logs/queue_devA.txt` and `logs/queue_conf.txt`;
   - each command is `python3 scripts/ft_factorial.py PARENT L W D [STEPS]`, with `FX_INTERV` for the interventions;
   - checkpoints are written to `checkpoints/ft/`.
4. **Measurements:**
   - command lines are in `logs/queue_*.txt`; `logs/COMPUTE_LOG.txt` has the exact chronology and exit codes;
   - streams: development 86_xxx_xxx, confirmation 88_xxx_xxx.
5. **Confirmation scoring:**
   - `python3 scripts/collect_13c.py` builds `logs/bundle_13c.json`;
   - `python3 scripts/score_13c.py logs/bundle_13c.json` gives `logs/score_13c.json`;
   - `python3 scripts/preflight_13c.py` runs the null and pass checks of the scorer.
6. **Clean-extraction verification:** `bash scripts/verify_clean13c.sh`, which runs the 12 mandated checks and writes `logs/VERIFY_CLEAN.txt`.

## Frozen protocol

| item | where |
|---|---|
| population declaration | commit 9ec86d6 (`logs/CONFIRMATION_POPULATION_13C.md`) |
| prediction freeze | commit d6281c6 (`scripts/predictions_13c.json`, `scripts/score_13c.py`, `logs/FREEZE_13C_HASHES.txt`) |
