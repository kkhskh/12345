# 08R2-FIBER-COST fixed-JS restoration

This is the original frozen equal-JS protocol. It does not replace or rewrite the donor-reference COST run.

Preserved:

- `TARGET_SPECIFIC_NULL_EFFECT_SUPPORTED`
- `TARGET_SPECIFIC_AT_EQUAL_JS_INCONCLUSIVE`
- `08R2_VALIDATION_COMPLETE — NEGATIVE`
- `0A C / PARTIAL`
- `SECTION_5_3 = PARTIAL`
- `OBSTRUCTION_READY = NO`

Do not overwrite `fiber_spec/`, `fiber_cost/`, or `fiber_cost_forensic/`.

This execution is a protocol-compliant exploratory follow-up on the same 48 pairs, not sealed confirmation.

## Intervention

For each of the 48 frozen pairs and each of the seven original nullspace directions \(\delta_j\):

\[
h_j(t)=h_E+t\delta_j
\]

\(\delta_j\) is the SPEC direction (correct null, three L2-matched wrong nulls, three L2-matched random nulls). \(t=1\) is the original SPEC scale. It is not the COST target.

Calibrate \(t\) independently so

\[
\mathrm{JS}(F_x(h_E),F_x(h_j(t)))=0.01
\]

including the correct donor. Accept only if \(|\mathrm{JS}-0.01|\le 0.0002\).

Do not set the target from the correct donor's \(t=1\) JS. Do not mark any row `reference`.

## Search

Frozen relative-scale grid:

`0.003125, 0.00625, 0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 1.5, 2.0, 4.0, 8.0`

Evaluate in increasing order. Take the first grid point with \(\mathrm{JS}\ge 0.01\), then bisect 16 times against the previous point. Search does not use \(Y\).

If the last grid point cannot reach 0.01, the direction fails and remains in the attempt record.

Reported L2 is \(\|t\delta_j\|_2\).

Primary target: `0.01`. Secondary targets `0.005` and `0.02` are labeled secondary and are not used to pick a winner.

## Gate

A pair is complete only if all seven directions calibrate and pass probe invariance.

Require 44/48 complete pairs (90%).

On complete pairs:

- Mean \(\Delta_{\mathrm{wrong}}>0\) and clustered 95% CI lower bound \(>0\)
- Mean \(\Delta_{\mathrm{random}}>0\) and CI lower bound \(>0\)
- Mean \(\Delta_{\mathrm{wrong}}\ge 0.02\)
- Mean \(\Delta G>0\) and mean \(\Delta Q<0\)

Bootstrap 2000, seed 82, clustered by recipient game.

Insufficient coverage is `FIXED_JS_TARGET_SPECIFICITY_INCONCLUSIVE` even if the complete-case effect is large.

Do not expand the scale grid after seeing results.
