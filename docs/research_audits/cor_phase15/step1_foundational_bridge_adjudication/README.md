# COR Phase 1.5, Step 1: foundational causal-bridge adjudication

**Verdict:** `STEP1_NECESSARY_CONDITIONS_ONLY`. In the investigated classes, consistency has no causal role in Transformer decisions beyond ordinary normalization (softmax reallocation and the final norm). The original `O`/`D` mechanism should be retired. `OBSTRUCTION_READY = NO`; `PHASE_2 = NO GO`; Step 2 is not opened.

## Read in this order

1. `CAUSAL_BRIDGE_MAIN_ARGUMENT.md`: the connected argument and the arrow-by-arrow verdict
2. `TEN_FAILURE_GATE_MATRIX.md`: the acceptance criterion
3. `FINAL_STEP1_VERDICT.md`: the answers to the 48 mandate questions
4. `ADVERSARIAL_CROSS_EXAMINATION.md`: what the independent reviews found and what changed
5. `proofs/PROOFS.md`: theorems and proofs

## Supporting files

- **Analyses.** `EXACT_COMPUTATION_PARADOX.md`, `CAUSAL_IDENTIFICATION_ANALYSIS.md`, `DIRECTIONALITY_AND_CANDIDATE_RANKING.md`, `OUTCOME_LEAKAGE_AND_FREEZING.md`, `EXISTING_THEORY_COMPARISON.md`.
- **Dossiers.** `CANDIDATE_H1_RECONCILIATION.md`, `CANDIDATE_H2_CAUSAL_SUFFICIENCY.md`, `CANDIDATE_H3_EVIDENCE_COORDINATION.md`.
- **Registers.** `THEOREM_REGISTER.md` and `.csv`, `CANDIDATE_REGISTRY.csv`, `COUNTEREXAMPLE_REGISTER.md`, `SOURCE_REGISTER.md`, `ORIGINAL_FAILURES.md`.
- **State.** `WORK_STATE.md`, `RESEARCH_JOURNAL.md`, `OPEN_OBLIGATIONS.md`, `UNRESOLVED_SCIENTIFIC_QUESTIONS.md`, `NEXT_RESEARCH_DECISION.md`.
- **Code and logs.** `proofs/bridge_math.py`, `tests/test_bridge.py`, `reviews/` (four separate-context reviews), `counterexamples/README.md`, `logs/`, `scripts/`, `MANIFEST.json`.

## Reproduce

```
cd docs/research_audits/cor_phase15/step1_foundational_bridge_adjudication
python -m pytest tests -q          # numpy, scipy, pytest
python scripts/verify_manifest.py
```

Requirements: Python 3.10 or later, numpy, scipy, pytest. No GPU, no model weights, and no network access are needed.
