# COR Phase 1.5, Step 1: Foundational Investigation II (atlas continuation)

**What this is.** This directory continues `../step1_relational_computation_atlas/` (base commit `e26a109`, branch `cor-phase15-step1-relational-atlas`). The original atlas is untouched.

**Historical verdicts are unchanged.** All of these still stand:
- `OBSTRUCTION_READY = NO`
- `PHASE_2 = NO GO`
- the Step 1 verdicts
- the atlas PP1–PP5 verdicts

**Out of scope.** Phase 2 was not opened. No pretrained models were used. No closure is proposed.

## Reading order

1. `MAIN_FOUNDATIONAL_RESULT.md`: the connected account.
2. `PP1_CALIBRATION_AND_INTERVENTION_THEOREMS.md`: support notions and calibration-to-sign (corrections 1–2).
3. `EXTRAPOLATION_AND_INTERNAL_STRUCTURE.md`: PP2, support versus distribution change, and the prospective P-A outcome.
4. `TRANSFERRED_FEATURE_IDENTIFICATION.md`: PP3 dissection and P-B.
5. `COMPUTATIONAL_CLASSES_AND_EQUIVALENCE.md`: seed 12 at operation level, and the equivalence ladder.
6. `REPRESENTATION_AND_CAUSAL_IDENTIFICATION.md`: split-invariant attribution, intervention targets, the causal graph, and errata.
7. `RELATIONAL_GENERATIVE_LANDSCAPE.md`: exact-Bayes phase diagrams and unknown groups.
8. `BEYOND_LATENT_RELIABILITY.md`: the GATE and TRANS families.
9. `THEOREMS_COUNTEREXAMPLES_AND_LIMITS.md`: the theorem registry.
10. `BASELINES_AND_NOVELTY.md`, `PROSPECTIVE_PREDICTIONS.md`, `COR_TEN_FAILURE_MATRIX.md`, `OPEN_FOUNDATIONAL_MAP.md`.
11. `reviews/`: subagent reviews and adjudications.
12. `sources/`: the mandate and the base commit.

## Reproduction

**Dependencies.** Python 3.13, numpy 2.5, jax 0.12 (CPU), optax 0.2.8, scikit-learn 1.9, scipy 1.18.

**Code layout.** Everything runs from this directory and imports code from the sibling packages `../step1_latent_reliability/src` and `../step1_relational_computation_atlas/src`, which hold the models, generators and analysis utilities.

| what | script | output |
|---|---|---|
| training jobs (PP3 pretraining, fine-tunes, GATE, TRANS, references) | `scripts/run_job.py JOB` | `checkpoints/JOB.{npz,json,log}` |
| held-out models | `scripts/queue2.sh` (calls the atlas `train_run.py` and `train_curriculum.py`) | `checkpoints/heldout/` |
| PP1 | `pp1_calibration.py`, `pp1_cx_matched.py`, `pp1_cx_targeted.py`, `pp1_learned_ece.py`, `pp1_highratio.py` | `logs/pp1_*.json` |
| PP2 / P-A | `pp2_dev.py`, `pp2_mechanism.py`, `pp2_probe_control.py`, `pa_evaluate.py` | `logs/pp2_*.json`, `logs/pa_outcome.json` |
| PP3 / P-B | `pp3_probes.py`, `pp3_dynamics.py` (+ `pp3_probes_lib.py`) | `logs/pp3_*.json` |
| seed 12 / classes | `s12_operations.py SEED OUT`, `s12_review_checks.py SEED OUT`, `class_shapley.py SEED HEAD OUT`, `abstraction_shift.py` | `logs/s12/`, `logs/class_shapley_s10.json`, `logs/abstraction_shift.json` |
| landscape | `landscape_phase.py`, `landscape_checks.py`, `review_checks_landscape.py`, `unknown_groups.py`, `od_bridge.py` | `logs/landscape/`, `logs/od_bridge.json` |
| decisions | `decision_ledger.py` | `logs/decision_ledger.json` |
| beyond reliability | `beyond_analysis.py`, `gate_mechanism.py` | `logs/beyond_analysis*.json`, `logs/gate_mechanism.json` |

**Clean-extraction check.** `bash scripts/verify_clean_extraction.sh`:
- exports the committed tree to `/tmp`;
- checks `MANIFEST.sha256`;
- re-runs the exact-Bayes and constructed-model scripts and one learned-model script;
- compares their outputs with the committed logs. The result of the last run is in `logs/clean_extraction_check.txt`.

**Compute budget.** Changes made before training are logged in `logs/COMPUTE_LOG.txt`. Checkpoints are committed (small, about 0.5 MB each).

## Provenance rules followed

- Predictions P-A and P-B were frozen in git before the evaluations they concern. The outcomes were appended, and the frozen text was not edited.
- Post hoc analyses are labelled as such.
- Review-driven corrections are marked [R]. Pre-review text is in git history.
- Bugs found are listed in the documents where they occurred.
