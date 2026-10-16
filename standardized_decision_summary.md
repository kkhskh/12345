# Experiment 2D Standardized Response Score: Out-of-Sample Validation

Standardized score: `S_z = z(pred_delta_m) - alpha * z(pred_delta_O) - gamma * z(cost)`,
with z-scores taken across candidate sites within each example.

Bootstrap and train/test splits are over **examples**, not over site-rows.
Spearman-with-Pareto is not the same quantity as Spearman-with-observed-delta-m.

- tuning examples: 32
- held-out examples: 32

## Selected on Tuning Split

- with obstruction: `raw_response_beta2.0_gamma0.001` (tune Spearman Pareto 0.9177)
- without obstruction: `attribution_score` (tune Spearman Pareto 0.9063)

## Held-Out Performance

- with obstruction: `raw_response_beta2.0_gamma0.001` Spearman Pareto 0.9143, AUC repair 0.9547, Spearman delta_m 0.9141
- without obstruction: `attribution_score` Spearman Pareto 0.9054, AUC repair 0.9791, Spearman delta_m 0.9055

## Paired Obstruction Lift on Held-Out Examples

- spearman_pareto: mean lift +0.0086 [-0.0013, +0.0180], positive in 96.3% of resamples
- auc_top_repair: mean lift -0.0245 [-0.0353, -0.0145], positive in 0.0% of resamples

## Decision

Decision: **PASS_WEAK**

PASS_STRONG requires a Pareto lift whose interval excludes zero **and** a positive
held-out repair-AUC lift. PASS_WEAK is a narrow label: a tiny positive mean Pareto
lift is not evidence that obstruction beats attribution at repair-site selection.
Repair-AUC lift is negative, so report obstruction as a monitor (`pred_ΔO` tracks
`actual_ΔO`) and attribution as the intervention predictor.