# Causal mediators and controls (Workstream B)

Investigation XIII, Phase 1.5 Step 1. Development (dev) numbers are labelled as such. Confirmation numbers are in §7 and
`FROZEN_PROTOCOL_AND_CONFIRMATION.md`.

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
- **DAS is non-identifying between E and A1.** On E≠A1 pairs, each trained subspace follows its own rule. DAS still rejects PERM; see
  NATIVE_DECISION_COMPUTATION §6.
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
| A1, PERM subspaces (das_decoy, same budget as E) | resid2, q-tokens | follow their own rule on discriminating strata (A1) or reject (PERM); not specific under mean patching |
| random 6-dim subspace | resid2, q-tokens | ≤ 0.04 R² to any rule; ≤ 1.3% contrast removal |
| non-mixture layer-2 head (lowest q-mass) | attention row | \|NIE\| 0.01–0.21 vs 0.47–1.04 |
| shuffled prediction | any fidelity | R² ≤ 0.002 |
| CTRL factorial | input edit | \|NIE\| 0.15–0.19 vs TRUE 0.47–1.04 |

## 7. Confirmation

(Filled from the fresh cohort.)
