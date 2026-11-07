# COR Phase 1.5 Step 1: learned causal mediator identification

This package is standalone: it has no dependency on sibling directories.

**Principal document:** `MAIN_ANSWER.md`.

## Contents

| item | file or folder |
|---|---|
| frozen protocol (commit `ff180da`, before evaluation) | `PROTOCOL.md` |
| every per-model result, generated from the logs | `INTERVENTION_RESULTS.md` |
| mechanisms and causal graphs | `MECHANISMS_AND_CAUSAL_GRAPHS.md` |
| T-ATT (exact) and T-MED (conditional) | `DECISION_THEOREM.md` |
| comparisons | `DISTINCTIVENESS.md` |
| corrections (§H) | `ERRATA_PREVIOUS_PACKAGE.md` |
| F-matrix update | `FAILURE_MATRIX_UPDATE.md` |
| process record | `RESEARCH_LOG.md` |
| reviews | `reviews/` (independent verification, adversarial review) |
| checkpoints | `checkpoints/historical/` (90 recovered models, `MANIFEST.json`); `checkpoints/new/` (AGG-NL seeds 0–4; held-out seeds 7–9) |

## Code (`src/`)

| file | content |
|---|---|
| `model.py` | vendored training code; `semantics_v1.py` holds the historical generators (hashes in `sources/VENDORED_FROM.sha256`) |
| `generators.py` | task generators |
| `ckpt.py` | checkpoint loading and saving |
| `mediator.py` | candidate mediators and causal tests |
| `run_protocol.py` | runs the frozen protocol |
| `locus.py` | pattern-freeze locus analysis |
| `explore_followups.py` | exploratory follow-ups |

## Reproduce

```
python3 scripts/retrain_check.py                       # A: retraining is bitwise identical to the historical checkpoints
python3 scripts/train_new.py                           # new models (~20 min, 2 CPU cores)
python3 src/run_protocol.py E1 REL-XNOR,REL-GATE,REL-MIS,K3-GATE,AGG-LIN,K3-BAYES TF1,TF2,TF2AO 2,3,4 checkpoints/historical
python3 src/run_protocol.py E2 REL-XNOR,REL-GATE,REL-MIS,K3-GATE TF1,TF2,TF2AO 7,8,9 checkpoints/new
python3 src/run_protocol.py E3 AGG-NL TF1,TF2,TF2AO 0,1,2,3,4 checkpoints/new
python3 src/run_protocol.py E4a REL-XNOR,REL-GATE,REL-MIS TF1,TF2 2,3,4 checkpoints/historical gauss
python3 src/run_protocol.py E4b REL-XNOR,REL-GATE,REL-MIS TF1,TF2 7,8,9 checkpoints/new gauss
python3 src/locus.py E1 E2; python3 src/explore_followups.py; python3 scripts/make_tables.py
python3 -m pytest -q tests/
```

Requirements: Python 3.11+, numpy, scikit-learn, jax[cpu] 0.12, optax 0.2.8, pytest.

## Configuration

- Training: d = 16, 2 heads, d_h = 8, MLP width 32; 3,000 Adam steps with cosine decay from 3e−3; batch 512; initialization 0.02 for attention, 0.1 for others.
- Seeds and pair seeds are listed in `PROTOCOL.md` and `src/run_protocol.py`.

## Scope

This package covers Phase 1.5 Step 1 only. It uses no pretrained models and no sealed data.
