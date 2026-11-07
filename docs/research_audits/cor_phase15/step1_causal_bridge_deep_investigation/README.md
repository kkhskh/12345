# COR Phase 1.5, Step 1: deep causal-bridge investigation

**Verdict: `STEP1_ONLY_KNOWN_NONLINEAR_MECHANISMS_SURVIVE`.**

Transformer blocks can compute a relation between evidence sources and use it in a candidate decision. Four circuits demonstrate this: an MLP XNOR (A), a query–key gate (B), MLP-gated amplification (C), and conditional rereading through attention (C′). Each comes with a class theorem and proved boundaries.

No noncircular, independently identified, distinctive consistency-to-decision theory survives:
- the direction of each effect comes from sign hypotheses on the weights, and mirrored circuits are equally implementable;
- source polarity is fixed only through the outcome pathway;
- every mechanism is a known one.

O/D is not involved. `OBSTRUCTION_READY = NO`; `PHASE_2 = NO GO`.

## Read in this order

1. `MAIN_CAUSAL_BRIDGE_ARGUMENT.md`
2. `FINAL_SCIENTIFIC_DECISION.md`: claims, the 25 answers, coverage, the falsification specification, the next decision
3. `TEN_FAILURE_ADJUDICATION.md`
4. `THEOREMS_AND_COUNTEREXAMPLES.md`
5. `MLP_AGREEMENT_MECHANISM.md`, `QK_COMPOSITION_MECHANISM.md`, `MULTILAYER_RECONCILIATION.md`
6. `IDENTIFIABILITY_AND_REPARAMETERIZATION.md`, `BASELINES_AND_NOVELTY.md`
7. `ADJUDICATION.md`: what the four separate-context reviews found and what changed
8. `PREREGISTRATION.md`, `PREREGISTRATION_ADDENDUM.md`

## Files

- `src/`: `cbmath.py` (exact blocks, constructions, gauge maps), `experiments.py` (every preregistered experiment, plus labeled post-hoc, supplementary and addendum sections), `positive_control_ref.py` (independent pure-Python control)
- `tests/test_deep.py`: 31 tests
- `logs/`: `experiments.txt`, `results.json`, `positive_control_matrices.txt`, `pytest.txt`, `clean_room.txt`, `historical_files_unchanged.txt`
- `scripts/`: `dump_positive_control.py`, `write_manifest.py`, `verify_manifest.py`
- `reviews/`: R1–R4
- `sources/`: the two mandates, and the independent audit handoff with SHA-256 sums
- `SOURCE_REGISTER.md`, `RESEARCH_JOURNAL.md`, `MANIFEST.json`

## Reproduce

```
cd docs/research_audits/cor_phase15/step1_causal_bridge_deep_investigation
python src/experiments.py              # about 2 minutes; rewrites logs/experiments.txt and results.json
python -m pytest tests -q
python scripts/verify_manifest.py
```

Requirements: Python ≥ 3.10, numpy, scipy, pytest. No GPU, no weights, no network.
