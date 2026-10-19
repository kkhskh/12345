# 08R2-NULL-STATE-TRANSFER preregistration

Frozen before held-out causal evaluation.

Preserved letters:

- `FIXED_JS_TARGET_SPECIFICITY_SUPPORTED`
- `TARGET_SPECIFIC_NULL_EFFECT_SUPPORTED`
- `TARGET_SPECIFIC_AT_EQUAL_JS_INCONCLUSIVE`
- `08R2_VALIDATION_COMPLETE — NEGATIVE`
- `0A C / PARTIAL`
- `SECTION_5_3 = PARTIAL`
- `OBSTRUCTION_READY = NO`

Do not overwrite SPEC, COST, forensic, or fixed-JS archives. No confirmation. No 08R3. No obstruction energy.

This is a donor-free transfer test, not a new decision germ and not sealed confirmation.

## Stage 0

Recompute G, Q, Y from archived fixed-JS 61-way logits using the same simulator legal sets. Stop if they disagree beyond \(10^{-6}\). Record that JS=0.005 and JS=0.02 were not in the primary archive.

## Representation

Symbolic \(b\): 64×3 player-relative board one-hot, 64-square legal mask, player-is-black, occupancy/64.

On training activations at `blocks.6.hook_resid_mid` last real token:

\[
z=P_Nh,\qquad z\approx\mu+U_ka,\qquad \widehat a(b)=Wb+c.
\]

Candidate family: \(k\in\{8,16,32,64\}\), ridge \(\alpha\in\{0.1,1,10,100\}\).

Select by 5-fold training-game-grouped CV residual \(R^2\). Never select on held-out causal \(Y\).

## Split

Training prefixes for fit. Training-game CV for selection. New validation pairs exclude FIBER, SPEC/COST, and those wrong-donor games.

Seed 4182. Target 32 A + 16 B, disjoint recip/target, unused games only.

Frozen pair hash after the local census:

`39abdc79090211ab4a69e3e5f79f14319231705640f231dcc146233a9f5092e6`

If Colab reconstruction disagrees, stop. Do not reuse old evaluation targets.

Secondary JS=0.005 and JS=0.02 were not in the uploaded primary fixed-JS archive and are not claimed here.

## Intervention

\[
\delta_{\mathrm{learned}}=\widehat z(b_t)-P_Nh_E.
\]

No \(h_D\) at evaluation time. Oracle \(P_N(h_D-h_E)\) is loaded only after learned directions are frozen.

Controls: wrong-target \(\widehat z(b_w)-z_E\), stage mean \(\mu_{\mathrm{stage}}-z_E\), random null, oracle, optional \(P_N\nabla Y\).

Calibrate every direction to JS=0.01 with the fixed-JS grid and \(|\mathrm{JS}-0.01|\le 0.0002\). Search does not use \(Y\).

Primary completeness: learned, wrong-target, mean-state, random. Require 44/48.

## Success

Mean \(Y_{\mathrm{learned}}-Y_{\mathrm{wrong}}\ge 0.01\) with CI lower \(>0\); same positivity vs random, mean-state, and Editor A; \(\Delta G>0\) and \(\Delta Q<0\) versus Editor A.

High residual \(R^2\) alone is not sufficient.
