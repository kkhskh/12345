# 08R2-STATE-SEMANTICS mean-level correction

Read-only. Does not change STATE-SEMANTICS artifacts, verdicts, or official letters.

Residual \(R^2\) and causal \(\Delta Y\) measure different properties. The identities below are an arithmetic decomposition of **mean response levels**, not a decomposition into independent causal mechanisms.

`STAGE_ONLY` is target-stage-conditioned: it uses the intended target's player and occupancy. It is not a target-independent natural-state prior.

## Supplied pooled identity

From the archived equal-JS intervention means:

\[
\text{FULL}-\text{STAGE\_ONLY}=0.023613
\]

\[
\text{FULL}-\text{Editor A}=0.053713
\]

Therefore

\[
\text{STAGE\_ONLY}-\text{Editor A}=0.030099
\]

and

\[
\frac{\text{STAGE\_ONLY}-\text{Editor A}}{\text{FULL}-\text{Editor A}}\approx 0.5604.
\]

About 56% of FULL's mean improvement over Editor A is already obtained by STAGE_ONLY under equal-JS calibration.

The earlier statement that generic stage correction does not explain most of FULL because its residual \(R^2\) is only 0.07 is not justified. That \(R^2\) describes reconstruction of probe-null residuals, not share of mean \(Y\).

## Subgroups

A, B, exact, and nonexact shares must be recomputed from `state_semantics_interventions.csv` pair-contrast rows by

`python run_08r2_symmetric_target.py --stage 0`

The runner writes `symmetric_target_stage0_correction.md` under `results/08r2/symmetric_target/` without modifying SEM files.

## Unchanged letters

- `LEGAL_MASK_SUFFICES_IN_TESTED_FAMILY`
- `DONOR_FREE_NULL_STATE_TRANSFER_SUPPORTED`
- `FIXED_JS_TARGET_SPECIFICITY_SUPPORTED`
- `TARGET_SPECIFIC_NULL_EFFECT_SUPPORTED`
- `TARGET_SPECIFIC_AT_EQUAL_JS_INCONCLUSIVE`
- `08R2_VALIDATION_COMPLETE — NEGATIVE`
- `0A C / PARTIAL`
- `SECTION_5_3 = PARTIAL`
- `OBSTRUCTION_READY = NO`
