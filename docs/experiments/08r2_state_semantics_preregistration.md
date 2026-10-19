# 08R2-STATE-SEMANTICS preregistration

Exploratory mechanistic ablation of the frozen 08R2-NULL-STATE-TRANSFER map.

This is not independent confirmation. The 48 pairs have already been inspected.

Preserved letters:

- `DONOR_FREE_NULL_STATE_TRANSFER_SUPPORTED`
- `FIXED_JS_TARGET_SPECIFICITY_SUPPORTED`
- `TARGET_SPECIFIC_NULL_EFFECT_SUPPORTED`
- `TARGET_SPECIFIC_AT_EQUAL_JS_INCONCLUSIVE`
- `08R2_VALIDATION_COMPLETE — NEGATIVE`
- `0A C / PARTIAL`
- `SECTION_5_3 = PARTIAL`
- `OBSTRUCTION_READY = NO`

Do not overwrite NST, SPEC, COST, forensic, or fixed-JS archives. No confirmation. No 08R3. No obstruction energy.

## Reconstruction

Reuse frozen \(U_{64}\), \(\alpha=10\), pair hash

`39abdc79090211ab4a69e3e5f79f14319231705640f231dcc146233a9f5092e6`

Stop if reconstructed FULL directions disagree with the archived 48 vectors beyond \(10^{-4}\) L2.

## Ablations

Same training games. Frozen PCA. No hyperparameter search.

- FULL: archived board + legal + player + occupancy
- BOARD_ONLY: drop legal mask
- LEGAL_ONLY: drop board one-hot
- STAGE_ONLY: player + occupancy

Report residual \(R^2\) and causal \(Y\) separately.

## Decomposition

\(\delta_{\mathrm{full}}=\delta_{\mathrm{generic}}+\delta_{\mathrm{target}}\) with STAGE_ONLY as generic.

Equal-JS at 0.01 for each method. Separate \(t=1\) identity check. Effects need not add at equal JS.

## Crossover

Same recipient Editor A. Intended and original stage-matched wrong targets. Report \(A_T\), \(A_W\), \(C\), and both-arms-positive fraction.

## Success floor

Incremental advantage \(+0.01\) with paired-bootstrap CI lower bound \(>0\), coverage \(\ge 44/48\).

Exploratory labels only:

- `BOARD_GEOMETRY_INCREMENT_SUPPORTED`
- `LEGAL_MASK_SUFFICES_IN_TESTED_FAMILY`
- `FEATURE_SEMANTICS_MIXED`
- `STATE_SEMANTICS_INCONCLUSIVE`
- `STATE_SEMANTICS_ENGINEERING_FAILED`
