# 08R2-VISIBLE-NULL-FACTORIAL preregistration

Prospective factorial test of whether frozen target-conditioned nullspace effects depend on an independently set rowspace state.

Frozen before the new causal cohort is generated.

Preserved:

- `SYMMETRIC_TARGET_SWITCHING_NOT_SUPPORTED`
- `SYMMETRIC_TARGET_CENSUS_INFEASIBLE`
- `LEGAL_MASK_SUFFICES_IN_TESTED_FAMILY`
- `DONOR_FREE_NULL_STATE_TRANSFER_SUPPORTED`
- `FIXED_JS_TARGET_SPECIFICITY_SUPPORTED`
- `TARGET_SPECIFIC_NULL_EFFECT_SUPPORTED`
- `TARGET_SPECIFIC_AT_EQUAL_JS_INCONCLUSIVE`
- `08R2_VALIDATION_COMPLETE — NEGATIVE`
- `0A C / PARTIAL`
- `SECTION_5_3 = PARTIAL`
- `OBSTRUCTION_READY = NO`

This is not obstruction, not a compatibility map, and not confirmation.

## Seeds (do not reuse 8182 or 9182)

- Cohort: `10182`
- Prefixes: `10282`
- Joint packing: `10382`
- Bootstrap: `10482`
- Random-null controls: `10582`
- Label shuffle: `10682`
- Train holdout diagnostic: `10782`

400 games. 16 A + 16 B. 96 distinct roles. One predetermined draw. Stop if the joint witness fails.

## Intervention

`rho = nu = 3.5`. No JS dose selection. Total L2 `sqrt(24.5)` for the four factorial cells.

`I_T = d_T(T) - d_T(W)`, `I_W = d_W(W) - d_W(T)`.

## Gate (pre-outcome)

Coverage ≥ 29/32. Integrity. Mean `I_T` and `I_W` ≥ 0.005, both CI lower bounds > 0. Random and shuffled controls do not reproduce that pattern. No outcome-based exclusions.
