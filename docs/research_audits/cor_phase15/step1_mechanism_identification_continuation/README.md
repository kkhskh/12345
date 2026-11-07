# COR Phase 1.5, Step 1: deep causal-mechanism identification and generalization (continuation)

This package continues the A/B/C/C′ investigation in `../step1_causal_bridge_deep_investigation/` (HEAD `bec0a64`). That historical package is **not modified**. Its superseded claims are corrected here:

- C′1 is replaced by C′1*;
- (A0) is replaced by A1*.

It does **not** propose closing Step 1, and it does not start Step 2 or Phase 2. No pretrained model and no sealed benchmark was used.

## Start here

1. `MAIN_RESEARCH_RESULT.md`: the integrated result and its arrow-by-arrow record.
2. `FINAL_RESEARCH_SYNTHESIS.md`: answers to the 22 questions.
3. `TEN_FAILURE_PROGRESS_MATRIX.md` and `OPEN_RESEARCH_OBLIGATIONS.md`.
4. `ADJUDICATION.md`: the seven-role review and what changed as a result.

## Documents

| file | content |
|---|---|
| `CPRIME_CORRECTED_OPERATING_REGIONS.md` | C′ closed form, exact boundary, Theorem C′1*, certified regions, continuity obstruction |
| `MLP_NORMALIZATION_THEOREMS.md` | A1* under shared LN2; P1–P5; normalization-created interaction; cones |
| `SOURCE_SEMANTICS_AND_IDENTIFICATION.md` | generator, five consistency definitions, identification procedure |
| `MATCHED_RELATION_INTERVENTIONS.md` | relation vs aggregate, reachability, sufficient statistic, mediation levels |
| `DYNAMIC_ATTENTION_AND_DEEP_MECHANISMS.md` | Program B (K = 3 gates, bilinear impossibility, total-margin theorem, attacks) and Program C (mediation, composition) |
| `CONDITIONAL_CAUSAL_THEOREMS.md` | every decision theorem with its conditions; T-RF |
| `LEARNED_SYNTHETIC_MECHANISMS.md` | preregistered small trained models T1–T5 |
| `REPARAMETERIZATION_AND_BASELINES.md` | gauges, baselines, transfer |
| `PREREGISTRATION_LEARNED_MODELS.md` | preregistration and logged deviations |
| `RESEARCH_LOG.md` | commits and checkpoints |

## Reproduce

```
sh scripts/run_all.sh            # engineered experiments + tests (~2 min)
sh scripts/run_all.sh --train    # plus retraining all learned models (~35 min, 2 CPU cores)
```

Requirements: Python 3.11+, numpy, scipy, scikit-learn, jax[cpu] 0.12, optax 0.2.8, pytest.

Learned checkpoints are not committed (`.gitignore`). They are regenerated deterministically from the seeds. `MANIFEST.json` lists the SHA-256 of every committed file.
