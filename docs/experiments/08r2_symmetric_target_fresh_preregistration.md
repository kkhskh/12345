# 08R2-SYMMETRIC-TARGET-FRESH preregistration

New, separately preregistered evaluation on an independently seeded simulator cohort.

The leftover-validation study remains `SYMMETRIC_TARGET_CENSUS_INFEASIBLE`. That protocol is not rewritten.

Seeds 8182 / 8282 were a design pilot only. They are not this experiment.

## Frozen seeds (committed before cohort generation)

- Cohort generation: `9182`
- Prefix subsample: `9282`
- Joint A/B packing: `9382`
- Bootstrap: `9482`
- Games: 400
- Prefixes per game: 6
- Simulator: Li-compatible random legal self-play, `max_moves=60`

Do not regenerate with other seeds after seeing packing or causal Y.

## Preserved letters

- `LEGAL_MASK_SUFFICES_IN_TESTED_FAMILY`
- `DONOR_FREE_NULL_STATE_TRANSFER_SUPPORTED`
- `FIXED_JS_TARGET_SPECIFICITY_SUPPORTED`
- `TARGET_SPECIFIC_NULL_EFFECT_SUPPORTED`
- `TARGET_SPECIFIC_AT_EQUAL_JS_INCONCLUSIVE`
- `08R2_VALIDATION_COMPLETE — NEGATIVE`
- `0A C / PARTIAL`
- `SECTION_5_3 = PARTIAL`
- `OBSTRUCTION_READY = NO`

Mean-level correction (arithmetic of average Y, not mechanisms):

- FULL − Editor A = +0.053713
- STAGE_ONLY − Editor A = +0.030099
- FULL − STAGE_ONLY = +0.023614
- Stage share = 0.5604

## Joint design

16 A + 16 B. 32 recipient games. 64 target games. 96 distinct games.

A saved 32-triplet witness is required. Separate A and B abundance is not enough.

Failed greedy is not an infeasibility proof. Stop with `FRESH_TRIPLET_CENSUS_INFEASIBLE` if the predetermined cohort cannot supply the witness.

## Gate (FULL, natural anchor)

Coverage ≥ 29/32. Mean \(A_T\ge 0.005\) and \(A_W\ge 0.005\), both CI lower bounds \(>0\), mean \(C>0\), both arms positive on ≥ 60% of complete triplets.

Positive \(C\) alone is not bidirectional switching.

No confirmation. No 08R3. No obstruction. No retraining.
