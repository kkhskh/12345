# 08R2-FIBER-COST preregistration

Frozen before new COST intervention outputs.

08R2-FIBER-SPEC remains a valid positive result at **matched residual L2**. This study asks a stricter question and does not revise SPEC.

Official status remains:

- `08R2_VALIDATION_COMPLETE — NEGATIVE`
- `0A C / PARTIAL`
- `SECTION_5_3 = PARTIAL`
- `OBSTRUCTION_READY = NO`

Confirmation sealed. No 08R3. No new editor. No obstruction energy.

## Question

Holding recipient context, frozen probe representation \(R(h)\), and Jensen–Shannon divergence from the same Editor A output:

does the correct-donor null direction still produce a larger intended-target \(Y\) than L2-unconstrained, JS-matched wrong-donor null directions?

Output JS is a scalar disruption budget. Matching it is not equality of output distributions. Residual L2 will generally differ and is recorded, not controlled.

## Reused objects

- Frozen 48-pair manifest hash `2789accfc4b0908168509f7a998f9e95ddb8a4f36f7a661aa1b9d5e4a11143d3`. If the SPEC JSON is absent, rebuild from the same seeds and accept only if this hash matches.
- The same correct donor, three wrong donors, and three random seeds
- Frozen 08c contrast matrix, rank tolerance `1e-8`, Editor A budget-16 settings

Do not resample pairs.

## Matching protocol

1. Reconstruct \(h_E\) and \(\delta_{\mathrm{correct}}=P_N(h_D-h_E)\).
2. Measure \(t=\mathrm{JS}(F_x(h_E),F_x(h_E+\delta_{\mathrm{correct}}))\).
3. For each wrong and random unit direction \(n\), bisect \(\alpha\in[0,\alpha_{\mathrm{hi}}]\) so \(\mathrm{JS}(F_x(h_E),F_x(h_E+\alpha n))\approx t\).
4. \(\alpha_{\mathrm{hi}}=4\|\delta_{\mathrm{correct}}\|_2\) for wrong donors and \(8\|\delta_{\mathrm{correct}}\|_2\) for random directions.
5. 10 bisection evaluations after the high-end probe. JS is treated as nondecreasing in \(\alpha\).
6. Among evaluated scales inside \(|\mathrm{JS}-t|\le\max(0.01,0.05\,t)\), keep the one closest to \(t\).
7. If the high endpoint cannot reach \(t\), reject that control. Do not substitute another direction.

Search evaluations are saved as a dose–response trace. They are not independent scientific units.

## Primary estimand

\[
\Delta_{\mathrm{JS}}
=
Y(h_{\mathrm{correct}})
-
\frac13\sum_{j=1}^{3}Y(h_{\mathrm{wrong},j}^{\mathrm{JS}})
\]

scored on the same intended legal sets. Also report the JS-matched correct-minus-random contrast, \(\Delta G\), \(\Delta Q\), achieved L2, and KL/JS.

## Gate

On pairs whose correct, three wrong, and three random interventions all pass invariance and JS matching:

1. Mean \(\Delta_{\mathrm{JS}}>0\)
2. Recipient-game clustered 95% CI lower bound \(>0\)
3. Mean \(\Delta_{\mathrm{JS}}\ge 0.05\)
4. Mean \(\Delta G>0\) and mean \(\Delta Q<0\)
5. JS-matched correct-minus-random mean \(>0\) with CI lower bound \(>0\)
6. No silent exclusions

Bootstrap: 2000, seed 82. Aggregate the three wrong and three random responses inside each pair first.

## Verdicts

- `TARGET_SPECIFIC_AT_EQUAL_JS_SUPPORTED` only if the gate holds.
- `TARGET_SPECIFIC_AT_EQUAL_JS_NOT_SUPPORTED` if the study completes and the gate fails.
- `TARGET_SPECIFIC_AT_EQUAL_JS_INCONCLUSIVE` if matching support is insufficient. Incomplete JS matching is a scientific outcome, not an engineering-file failure.
- `FIBER_COST_ENGINEERING_FAILED` for identity or implementation failure.

A negative COST result does not retract SPEC. A positive COST result still does not define a unique germ.
