# COR Phase 1.5, Step 1B — mechanism discovery

**Verdict:** `STEP1B_NECESSARY_CONDITIONS_IDENTIFIED`

Phase 2 remains **NO GO**. Official COR maps were not replaced. No model was run.

A statistic that does not move along a direction the logit uses cannot determine the sign of the response along that direction. That necessary condition applies to the old obstruction score and to any similar invariant summary. Positive constructions that do constrain responses — sufficiency of a representation, path contributions, mixed differences of a known map — are existing methods. They are not a repaired coboundary.

Start with `30_step1b_scientific_verdict.md`. Proofs N1–N3 are in `proofs/`. Tests: from this directory,

```text
python -m pytest tests/test_mechanisms.py -q
```

Do not load a checkpoint. Do not fit obstruction. Do not open confirmation.
