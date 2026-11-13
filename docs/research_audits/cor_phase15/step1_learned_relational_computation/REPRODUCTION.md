# Reproduction and clean extraction (Investigation VIII)

## Layout

| path | content |
|---|---|
| `sources/` | mandate (`MANDATE_VIII.txt`) and starting point |
| `src/v8lib.py`, `src/readers.py` | relational ingredient features, eligibility m (Investigation VII definition), readers, pickle-free tree I/O |
| `scripts/` | every experiment (see the table below), lane and queue utilities, `conf_evaluate.py` |
| `checkpoints/zP100…107`, `zT1110…1113` | Stage-0 lineages (`pre.npz` and pretraining log), frozen before any hypothesis |
| `checkpoints/compose`, `compose_conf` | composition readers (dev / confirmation), `.npz` params + `.json` evals |
| `checkpoints/dev`, `conf` | Condition Z transfers and repairs; `*_ablate.json`, `*_zablate.json` are exploratory ablations |
| `checkpoints/gen`, `xtok` | holdout readers (json only); Condition X |
| `logs/` | evaluation set `eval_conf8.npz`, hashes, queues, compute log, run outputs (`run/<md5>.out`), summaries |

## Dependencies

- Investigation VII's package `../step1_r1_causal_origin_and_selective_rescue/`, read-only. It provides `r1lib`,
  `nnlib`, `evalrun`, `ckpt`, `eval_dev.npz` and the Investigation VI dev checkpoints xP80/xP81/xT70.
- Python 3 with jax (CPU), optax, numpy and scikit-learn.
- Environment: `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"`.

## Commands (run from this directory)

| result | command |
|---|---|
| math checks | `python3 scripts/relational_math.py` |
| Stage-0 pretraining | `python3 scripts/cohort8_pretrain.py P 100` … / `T1 110` … |
| composition (dev) | `python3 scripts/compose_train.py {linear,mlp,attn,attn2,relnet} {R,O} {label,bayes} STEPS SEED` |
| composition (conf) | `PHASE=conf python3 scripts/compose_train.py attn R label 16000 10` (etc., `logs/queue_conf.txt`) |
| reader mechanism | `[PHASE=conf] python3 scripts/mech_reader.py TAG…` |
| decodability predictor | `[PHASE=conf] python3 scripts/predictor_probe.py TAG…` |
| Condition Z / repairs | `python3 scripts/repair_train.py {dev,conf} PARENT {orig,injm,lca,radapt,gadapt,auxm,unfreeze1,lora1,deep2,bayes}` |
| repair ablations | `python3 scripts/repair_ablate.py PHASE PARENT radapt`; `python3 scripts/repair_zablate.py PHASE PARENT radapt` |
| holdout | `python3 scripts/gen_holdout.py {attn,relnet,mlp,lookup} STEPS 0` |
| Condition X | `python3 scripts/x_train.py {1,3} label STEPS 0` |
| confirmation verdicts | `python3 scripts/conf_evaluate.py` → `logs/confirmation_results.json` |

Queues (`logs/queue_*.txt`) list the exact commands that were run. `scripts/lane.sh` executes a queue, with at most two
lanes in parallel.

## Integrity

- `logs/STAGE0_HASHES.txt`: Stage-0 lineage hashes.
- `logs/PROTOCOL_HASHES.txt`: protocol, evaluation set, scripts and parents, hashed at the freeze (commits `4dd9d34` and
  `29f7569`).
- `MANIFEST.sha256`: every file in the package at the final commit.

## Clean extraction

```
git archive <final-commit> docs/research_audits/cor_phase15/step1_learned_relational_computation \
                           docs/research_audits/cor_phase15/step1_r1_causal_origin_and_selective_rescue | tar -x
cd docs/research_audits/cor_phase15/step1_learned_relational_computation && sha256sum -c MANIFEST.sha256
```

Training runs are deterministic given seeds, up to CPU floating-point nondeterminism. Two confirmation runs were
re-executed after environment restarts (`logs/COMPUTE_LOG.txt`).
