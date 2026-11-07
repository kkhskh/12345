# 08R2-FIBER Scientific Report

Probe-visible versus probe-invisible decomposition of the Editor A vs full-donor gap.
This is an exploratory mechanistic follow-up on the already-examined 08R2 validation pairs.
It is not sealed confirmation and not a new editor competition.

Official labels unchanged:

- `08R2_VALIDATION_COMPLETE — NEGATIVE`
- Original 0A: **C / PARTIAL**
- `SECTION_5_3 = PARTIAL`
- `OBSTRUCTION_READY = NO`

Status: `FIBER_DIAGNOSTIC_COMPLETE`
Verdict: `PROBE_LOGIT_INSUFFICIENCY_SUPPORTED`

## Predefined tolerances

{'rank_tolerances': [1e-06, 1e-08, 1e-10], 'primary_rank_tol': 1e-08, 'probe_proba_atol_f64': 1e-08, 'probe_proba_atol_f32': 1e-05, 'parity_proba_atol': 1e-06, 'behavior_floor': 0.05, 'js_same_atol': 1e-08, 'js_meaningful': 0.001, 'repro_dp_atol': 0.01, 'repro_l2_atol': 0.25, 'null_component_min_norm': 1e-08}

## A. Probe contrast rank

- Contrast matrix shape: [124, 512]
- Numerical rank / rowspace dim: 124
- Nullspace dim: 388
- Primary relative tolerance: 1e-08
- Singular-value sensitivity: [{'tol': 1e-06, 'rank': 124, 'nullspace_dim': 388}, {'tol': 1e-08, 'rank': 124, 'nullspace_dim': 388}, {'tol': 1e-10, 'rank': 124, 'nullspace_dim': 388}]
- Orthogonality error: 2.213769528096421e-14
- Reconstruction error: 1.8051124144066715e-13
- Idempotence errors: R=2.2289127781203563e-14 N=2.390701775556995e-14
- W P_N residual: 1.4747542333852913e-13

## B. Probe-equivalent activations with different downstream decisions

- Matched primary pairs: 32
- Valid (invariance held): 32
- Invalid invariance: 0
- Pairs demonstrating insufficiency: 32
- Category A demo: 24 / 24 valid
- Category B demo: 8 / 8 valid

A pair demonstrates insufficiency only if the complete probe probability vectors agree within the predefined atol and the downstream distribution or Y-functional still moves by a predefined floor.

## C–E. Factorial contributions

- Mean E_R (probe-visible): 0.9401232916520245 median=0.8558366991109907 CI={'n': 32, 'n_clusters': 32, 'point': 0.9401232916520245, 'boot_mean': 0.9394556496690694, 'ci95_low': 0.777768766156113, 'ci95_high': 1.0951597841037095}
- Mean E_N (probe-invisible): 1.0936478252310151 median=0.988954259128544 CI={'n': 32, 'n_clusters': 32, 'point': 1.0936478252310151, 'boot_mean': 1.0952901621917304, 'ci95_low': 0.9641848885407162, 'ci95_high': 1.2338828620824651}
- Mean interaction I: -0.5859902052425803 median=-0.44101429740263653 CI={'n': 32, 'n_clusters': 32, 'point': -0.5859902052425803, 'boot_mean': -0.5850330039897664, 'ci95_low': -0.7480766948716903, 'ci95_high': -0.43728809054463597}
- Conditional E_R | N : 0.3541330864094442
- Conditional E_N | R : 0.5076576199884346
- A: E_R=1.1217658911038138 E_N=1.1857139771174368 I=-0.7071086373219085 targets=2
- B: E_R=0.39519549329665665 E_N=0.8174493695717493 I=-0.2226349090045957 targets=6

The Transformer is not assumed linear in activation space. Interaction is reported, not absorbed into main effects.

## F–G. Natural-anchor null perturbations

- Natural-anchor rows: 768
- Mean |ΔY| at ε=1 around donor: 0.00031627718645454647
- Same-norm random-null |ΔY| at ε=1: 0.00017336593519578594
- Probe invariance failures on anchors: 0

## H. Alignment versus generic disruption

- Mean JS(H00, H01): 0.3795247744144162
- Mean JS(H10, H11): 0.18536944756005633
- Mean |E_N|: 1.0936478252310151
- Wrong-donor null mean |ΔY| on the intended legal set: 0.6240248035195239

Legal-set Y movement is not the full output distribution. JS is the full-distribution comparison. A large JS with small Y is generic disruption.

## I. A versus B

- A pairs use 2 target games. Recipient-game CIs do not establish generalization across arbitrary targets.
- B pairs use 6 target games.

## J. History / distribution-shift confound

- Prefix-length and occupancy contrasts are in `08r2_fiber_matched_pairs.csv`.
- Training-stat diagnostic distances are recorded per cell. They are not proof of manifold membership.
- Off-manifold hybrids (H01, H10) can inflate effects relative to small natural-anchor moves.

## K. What a stronger germ would need

If probe-equivalent activations change F_x, a stronger germ must preserve information outside R(h). This diagnostic does not identify that information as board state, history, or policy residue. Small natural-anchor effects versus large hybrid H01 effects would favor distribution-shift over a stable nullspace coordinate.

## Claims this does not support

- The nullspace is not shown to be a unique mathematical game state.
- The board probe is not shown to be wrong about the board.
- COR / Section 5.3 is not confirmed or refuted. Confirmation remains sealed.
- Obstruction energy is not computed.
- No new state editor is selected.

## Verdict

`PROBE_LOGIT_INSUFFICIENCY_SUPPORTED`
