# Step 1C — Computational mechanisms

Phase 1.5, Step 1, continued. Not Step 2. Not Phase 2.

**Verdict:** `STEP1C_NECESSARY_CONDITIONS_ADVANCED`

The original signed bridge stays false. Four Step 1B sentences were wrong and are replaced in `01_step1b_error_corrections.md`: nonlinearity is not interaction, `M` does not enter `O`, a fixed readout need not keep its sign under LayerNorm, and `ker P_u` is not the kernel of a residual. The block equations, the intervention rows, and the four investigations are `04` through `09`. The necessary and sufficient list is `11`. The verdict and the answers to the forty-five questions are `23_step1c_final_verdict.md`.

`PHASE_2 = NO GO`. `OBSTRUCTION_READY = NO`. `COR_PHASE1_CONDITIONAL_THEORY_REPAIR_REQUIRED` is unchanged.

No Transformer was run. The checks are finite maps:

```text
python -m pytest docs/research_audits/cor_phase15/step1c_computational_mechanisms/tests/test_step1c_math.py -q
```

A passing check is arithmetic on those maps. It is not evidence about a trained model. The proofs are in `proofs/`. Review provenance is in `reviews/`: four subagents in this session, not external referees. Their objections and the repairs are in `20_review_cross_examination.md`.
