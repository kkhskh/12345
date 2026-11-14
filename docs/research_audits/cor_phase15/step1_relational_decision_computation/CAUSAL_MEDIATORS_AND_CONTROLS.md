# Causal mediators and controls (Workstream B)

Investigation XIII, Phase 1.5 Step 1. Development (dev) numbers are labelled as such. Confirmation numbers are in §7 and
`FROZEN_PROTOCOL_AND_CONFIRMATION.md`.

> **Corrections after review 3** apply here as in NATIVE_DECISION_COMPUTATION (top box).
> - The P2 "decoy" is the *lowest-mass* non-mixture head, not a size-matched head.
> - The mixture-head sets differ between scripts. For example, uP165 uses {0,1,3} in `l2_mix_causal` vs {0,1} elsewhere, and uP161
>   includes head 2. The R2 head-definition check compared only empty vs non-empty lists.
> - "Each subspace follows its own rule" (§4, §6) is withdrawn; see correction 3 there.
> - Exploratory E-vs-A1 interventional check in the TRUE factorial (`scripts/l2_mediation_A1.py`, `logs/l2_mediation_A1_conf_expl.json`):
>
>   | parent | observed TRUE \|NIE\| | E-mixture prediction | A1-mixture prediction | fidelity R², E | fidelity R², A1 |
>   |---|---|---|---|---|---|
>   | uP161 | 0.966 | 1.018 | 0.324 | 0.959 | 0.597 |
>   | uP165 | 0.724 | 1.018 | 0.324 | 0.764 | 0.500 |
>   | uP162 | 0.608 | 1.018 | 0.324 | 0.623 | 0.334 |
>   | uP164 | 0.352 | 1.018 | 0.324 | 0.285 | 0.101 |
>
>   In CTRL, E and A1 predict the same thing, as they should.

## 1. Intervention classes (kept separate everywhere)

| class | meaning | XIII instances |
|---|---|---|
| I | input edits (on-support episodes) | TRUE/CTRL factorial designs (`l2_mediation.py`); S = 3 slice selection (`adapt_sel.py`) |
| II | natural-activation transplants (activations of another real episode) | donor attention rows (`l2_mix_causal.py`); xP → xAP mixture-row transplant (`l2_mediation.py`) |
| III | synthetic activations (mean patches, learned subspaces, ablations, uniform rows) | subspace mean patches (`score_contrast.py`); DAS interchange (`decoy_disc.py`); layer-1 mean ablation (`l1abl_mix.py`); uniform rows |
| IV | high-level do() on model variables | the counterfactual predictions of rules E/A1/PERM used as *targets* in `decoy_disc.py` |
| V | artificial training targets | selective adaptation to C = 2 (`adapt_sel.py`); this is a learning experiment, not a mediation claim |

Claims of the form "the network *uses* X" rest only on class I and II evidence. Class III results are reported as what a synthetic
intervention does. Several class III tests in XIII are shown to be **non-specific** (§4).

## 2. Criteria that replace XII's biased KL-side criterion

XII's restoration criterion compared KL to one side and favoured interventions that pushed outputs toward high entropy. XIII uses only
signed, entropy-neutral quantities on the centred log-probability vector cen(P) = log P − mean log P:

- **Fidelity.** The observed Δcen(P) is regressed through the origin on the Δcen(P) that a hypothesis predicts. This gives a slope (1 is
  correct scale) and R². A shuffled-prediction null (rows permuted) is computed with every fidelity value.
- **NIE / TE.** NIE = cen(P_patched) − cen(P_base) and TE = cen(P_source) − cen(P_base), reported as mean |·| and as the projection
  NIE·TE/TE·TE.
- **Difference in differences.** |NIE|_TRUE − |NIE|_CTRL over matched designs. TRUE flips eligibility; CTRL makes the same edit type
  without changing eligibility. The 95% CI is an episode bootstrap.
- **Fraction of a score contrast removed**, using the signed contrast between eligible and ineligible groups (§4).

**Preflight of every criterion against nulls** (`preflight_nulls.py`, before the freeze). The nulls are no-op, uniform output, random
change, and an alternative mechanism (A1). The preflight found two criterion defects:
- R² = 1 for an identically zero change;
- a "decoy ≤ 0.4 × mixture" arm passing at 0 ≤ 0.

Both were fixed before the freeze. Arms that still pass on a null on their own are listed in FROZEN_PROTOCOL §5. No primary prediction is
confirmed by any null unit.

## 3. A natural mediator found without DAS: mixture-head attention rows

Factorial (`l2_mediation.py`, dev). xAP is the episode with group h made eligible. xP is the same episode with h perfect but not agreeing
at q (e_h = 0). The intervention runs xAP but transplants each mixture head's readout attention row from the natural xP run (class II).
The decoy control is the same transplant in the non-mixture layer-2 head with the least q-token mass.

| dev model | TRUE mean \|NIE\| | decoy-head \|NIE\| | CTRL mean \|NIE\| | DiD [95% CI] | NIE projection on TE | ATT-MIX fidelity R² (TRUE) |
|---|---|---|---|---|---|---|
| xP81 | 1.039 | 0.015 | 0.162 | 0.877 [0.808, 0.946] | 0.38 | 0.98 |
| vP155 | 1.031 | 0.010 | 0.171 | 0.860 [0.792, 0.925] | 0.38 | 0.97 |
| yP130 | 0.642 | 0.210 | 0.194 | 0.449 [0.389, 0.508] | 0.17 | 0.53 |
| vP156 | 0.468 | 0.111 | 0.151 | 0.317 [0.278, 0.357] | 0.12 | 0.52 |

Transplanting only the attention row moves the output strongly, and in the direction the attention-mixture account predicts. The effect
is eligibility-specific (the DiD) and head-specific (decoy ≪ mixture). It is a **partial** mediator: the projection on the total effect
is 0.12–0.38, because the eligibility edit also changes values, other heads and the MLP. Mediation is reported as partial.

## 4. Residual information and non-specific synthetic tests

- **E is used, but redundantly represented.** Eligibility is decodable at 1.0 from the 442-dim complement of E. After layer-1 mean
  ablation the mixture heads lose their alignment with r_g (0.95–0.98 → about 0), so layer 2 does not reconstruct eligibility.
  Mean-patching E removes the score contrast (137–147%). Mean-patching the complement does not; it strengthens the contrast.
- **The Bayes decision is not in E.** It is decodable from the complement (0.81–0.92) and not from E (0.41).
- **Matched decoys break the specificity of mean patching.** A1 and PERM subspaces, trained by the identical procedure and budget, also
  remove 88–143% of the score contrast (dev, xP81 and vP155). Mean-patch removal therefore cannot certify E. P3c freezes this
  non-specificity.
- **DAS did not identify E** (development wording "each trained subspace follows its own rule" withdrawn after review 3; P8 falsified).
- **The natural mediator discriminates.** On donors where E and A1 differ, the transplanted rows follow E (R² 0.97–0.99), not A1
  (0.17–0.19). This is P1c.

## 5. Restoration after a distinct, discriminating pathway intervention

**Not included in the frozen confirmation set.** XIII's natural mediator is the attention row. The donor-row test (P1) is itself a
replacement-and-prediction test on that pathway. A separate *restoration* experiment was not frozen: degrade one pathway, then restore it
with a natural activation through a distinct pathway, with a discriminating prediction. That experiment remains **unresolved** in XIII.
XII's KL-side restoration results are not re-used as evidence.

## 6. Matched-budget learned decoys (summary)

| decoy | where | result (dev) |
|---|---|---|
| A1, PERM subspaces (das_decoy, same budget as E) | resid2, q-tokens | overlap E in 4–5 of 6 directions; not rule-specific on E≠A1 pairs; not specific under mean patching |
| random 6-dim subspace | resid2, q-tokens | ≤ 0.04 R² to any rule; ≤ 1.3% contrast removal |
| non-mixture layer-2 head (lowest q-mass) | attention row | \|NIE\| 0.01–0.21 vs 0.47–1.04 |
| shuffled prediction | any fidelity | R² ≤ 0.002 |
| CTRL factorial | input edit | \|NIE\| 0.15–0.19 vs TRUE 0.47–1.04 |

## 7. Confirmation

Fresh cohort, frozen; FROZEN_PROTOCOL §6.

| parent | TRUE \|NIE\| | decoy \|NIE\| | CTRL \|NIE\| | DiD [frozen CI] | two-sample bootstrap CI | ATT-MIX R² | NIE projection on TE |
|---|---|---|---|---|---|---|---|
| uP161 (mixture) | 0.966 | 0.013 | 0.208 | 0.758 [0.687, 0.822] | [0.689, 0.827] | 0.96 | 0.33 |
| uP165 (mixture) | 0.724 | 0.018 | 0.118 | 0.606 [0.543, 0.668] | [0.543, 0.671] | 0.76 | 0.25 |
| uP162 (incomplete, mixture) | 0.608 | 0.261 | 0.136 | 0.471 [0.418, 0.523] | [0.418, 0.523] | 0.62 | 0.23 |
| uP164 (borderline, excluded) | 0.352 | 0.170 | 0.165 | 0.187 [0.150, 0.227] | [0.148, 0.226] | 0.29 | 0.13 |

- **P2 is confirmed in 2/2** frozen units.
- Mediation stays **partial**: the NIE projection on the total effect is 0.25–0.33.
- **Effect strength tracks the mixture head's logit share.** Strong-share parents (uP161, uP165) show large, head-specific effects. The
  borderline uP164 shows a small effect that is not head-specific; it would fail P2.
- **Matched decoys** (P3c, P8b) met their frozen thresholds: mean patching is non-specific, and A1-trained subspaces fit A1 better than E on E≠A1 pairs. Given the overlap of the subspaces (review 3), P8b is not evidence of separate representation.
- **E-trained DAS subspaces** were less E-specific than predicted, which falsifies P8a.
- **E-vs-A1 evidence** comes only from the attention-row mediator:
  - observationally (P1c): 2/2 fresh parents and 2/2 development models;
  - interventionally in the TRUE factorial, as an exploratory check (box at top).
