# Reproduction instructions (Investigation XIII-D)

## This package is not standalone

XIII-D reads pretrained parents, natural fine-tunes and the evaluation set from earlier COR Phase 1.5 Step 1 packages. It also reads XIII-C's factorial runs and adapt3 models, and Python modules from several of those packages.

The required sibling directories are listed in `logs/DEPENDENCIES.json` → `sibling_packages_required`. Each external file is listed there with its exact sha256.

Place this directory next to its siblings under `docs/research_audits/cor_phase15/`.

## Environment

- **Software:** Python 3.13, jax/jaxlib 0.12.0.1, numpy 2.5.3, optax 0.2.8, scipy 1.18.1. The exact versions are in `logs/DEPENDENCIES.json`.
- **Required setting:** `export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"`. All lanes set this. Bitwise determinism (`logs/determinism_check.json`) was verified under it, on CPU.
- **Hardware used:** a 2-CPU, 8 GB container. Expect about 60–100 min per 4000-step fine-tune when run alone.

## Seeds and their roles

**Fine-tuning** (`scripts/ft_factorial.py PARENT L W D`):

| seed | what it controls | how it is derived |
|---|---|---|
| L | fresh layer 2 | `init(L+7777)["layers"][2]` |
| W | fresh readout map | `init(W+7777)["WA"]` |
| D | training data stream | `default_rng(1000003*(D+1)+31782)` |

L = W = D = s reproduces the natural seed-s model bit for bit.

**Run options:**

| variable | effect |
|---|---|
| `FX_INTERV=name:T` | mask for steps < T |
| `FX_IWIN=a:b` | mask window |
| `FX_BRANCH=tag:k` | start from run `tag` at step k, including optimiser and rng state; needs `FX_SAVEOPT=1` on the source run |
| `FX_WA=` | readout-map transform |
| `FX_STOP=k` | stop at step k with the 4000-step schedule unchanged; such runs do not resume and write no final.json |

**Analysis streams.** Each analysis has a stream argument (`UM_STREAM`, `CA_STREAM`, `DD_STREAM`, `DR_STREAM`, `E4_STREAM`, `DT2_STREAM`, `RS_STREAM`/`RS_SETS`/`RS_TRAIN`, `AD_STREAM`). The values used are in the queue files and the predictions files.

**Stream ranges by role:**

| role | stream values |
|---|---|
| Confirmation | 91.x M (freeze 1), 92.x M (freeze 2), 94.1 M (freeze 3b); adaptation uses adapt4's default stream |
| Development | 90.x M, 93.x M |

## Re-running the confirmation

The queues are in `logs/`. Each line is one command, run from the package root.

**Run a queue:** `QUEUE=logs/<queue> bash scripts/lane2.sh NAME`. A lane claims each line through an md5 key in `logs/claims/`, so delete the claims before re-running.

| freeze | training | analysis |
|---|---|---|
| 1 | `queue_base.txt`, `queue_f1_train.txt` | `queue_f1_analysis.txt` |
| 2 | `queue_f2A_train.txt` (+ `queue_f2A_train2.txt`), `queue_carrier_adapt.txt` | `queue_f2_analysis.txt` (+ `queue_f2_retry.txt`) |
| 3a | `queue_f3a.txt` | — |
| 3b | `queue_f3b.txt` | — |

The `*_RETRY` / `FX_CONF` prefixes on some lines are inert environment variables, added only to give a re-queued line a new claim key.

## Scoring and statistics

```
python3 scripts/score_f1.py logs;  python3 scripts/score_f2.py logs
python3 scripts/score_f3a.py logs; python3 scripts/score_f3b.py logs
python3 scripts/preflight_f1.py; python3 scripts/preflight_f2.py; python3 scripts/preflight_f3a.py; python3 scripts/preflight_f3b.py
python3 scripts/uncertainty_13d.py          # threshold sensitivity, unit bootstrap, factorial summary
python3 scripts/determinism_check.py        # full-length replays vs originals
python3 scripts/deps_inventory13d.py        # external dependency hashes
B5_FINAL=logs/unit_metrics_f2.jsonl B5_OUT=logs/b5_transition.json python3 scripts/b5_transition.py   # exploratory
```

## Clean verification

`bash scripts/verify_clean13d.sh ZIP COR_PHASE15_ROOT LOG` performs the 13 mandated checks on a clean extraction. The output of the run done for delivery is in `logs/VERIFY_CLEAN.txt`.
