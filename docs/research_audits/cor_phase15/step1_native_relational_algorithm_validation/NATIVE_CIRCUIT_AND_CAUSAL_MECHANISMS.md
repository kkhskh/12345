# Native circuit and causal mechanisms (Investigation XII, Workstream B)

## What this document covers

Development models:
- 11 repaired models: xP81, yP130–yP134 (seed 0), xP81 and yP130 seed 1, wP140–wP143;
- C1 contrasts.

Every development number is **[dev]**. Fresh-cohort numbers are in §7 and `FRESH_CONFIRMATION_AND_STATISTICS.md`
**[conf]**.

The design throughout is the within-episode 2 × 2 factorial from XI, applied to a group h that disagrees at q:
- A: make h agree at q;
- P: make h agree at its one remaining disagreeing question, so that n_h goes 4 → 5;
- e_h = 1 only in cell xAP;
- CTRL design: n_h goes 3 → 4, so e = 0 in all four cells.

Outcomes at xAP:
- KL(Bayes ‖ model);
- KL(cf_e0 ‖ model), where cf_e0 is the Bayes formula with h treated as ineligible;
- pref_e0, the fraction of episodes closer to cf_e0;
- the remaining E-subspace interaction.

## 1. The previous ablation was not the only route to the result (B3)

`scripts/alt_ablation.py`, `logs/alt_ablation_dev*.json`. The table shows KL(Bayes) / KL(cf_e0) at xAP.

| intervention | class (math §4) | xP81 | yP130 | yP131 | yP132 | yP134 | wP140 | wP141 | wP142 | wP143 |
|---|---|---|---|---|---|---|---|---|---|---|
| intact | — | 0.00/0.40 | 0.00/0.40 | 0.01/0.40 | 0.01/0.40 | 0.01/0.41 | 0.00/0.40 | 0.01/0.40 | 0.01/0.40 | 0.01/0.40 |
| XI additive completion of layer 1 | III | 0.53/0.02 | 0.63/0.01 | 0.59/0.01 | 0.42/0.10 | 0.66/0.02 | 0.53/0.02 | 0.55/0.03 | 0.48/0.05 | 0.54/0.04 |
| **path_inA**: layer 1 fed the natural xA residual | II | 0.77/0.01 | 0.76/0.01 | 0.75/0.01 | 0.58/0.05 | 0.72/0.02 | 0.74/0.01 | 0.71/0.02 | 0.62/0.03 | 0.78/0.02 |
| **path_inP**: natural xP residual | II | 0.48/0.08 | 0.43/0.18 | 0.47/0.14 | 0.40/0.11 | 0.54/0.06 | 0.46/0.11 | 0.49/0.06 | 0.39/0.11 | 0.52/0.07 |
| resample layer-1 outputs from the natural xA run | II | 0.77/0.01 | 0.76/0.01 | 0.75/0.01 | 0.58/0.05 | 0.72/0.02 | 0.74/0.01 | 0.71/0.02 | 0.62/0.03 | 0.78/0.02 |
| **scrub**: E coordinates from a natural donor with the *same* e | II | 0.00/0.40 | 0.01/0.40 | 0.01/0.40 | 0.01/0.42 | 0.03/0.41 | 0.01/0.40 | 0.02/0.40 | 0.03/0.44 | 0.01/0.41 |
| **scrub**: donor with e_h = 0 | II | 0.74/0.01 | 0.49/0.04 | 0.59/0.02 | 0.35/0.07 | 0.19/0.17 | 0.66/0.02 | 0.31/0.07 | 0.19/**0.23** | 0.63/0.02 |
| mean-ablation of layer 1 | III, non-specific | 0.49/0.85 | 0.15/0.53 | 0.23/0.64 | 0.32/0.69 | 0.39/0.69 | 0.34/0.68 | 0.34/0.68 | 0.27/0.67 | 0.39/0.71 |

Findings:
- **H3 (intervention-dependence) is rejected for the layer-level claim.** Interventions that use only *natural*
  activations reproduce XI's result in 11 of 11 models: layer 1 fed a real input in which h is not perfect, or one in
  which h does not agree, or resampling from the natural run. The decision moves to cf_e0, not to some arbitrary
  broken state.
- Non-specific mean ablation moves the decision away from both references. The cf_e0 match is therefore a property of
  the conjunction's removal, not of perturbation per se.
- **Causal scrubbing.** E coordinates replaced from a *different natural episode with the same eligibility vector*
  leave the decision at Bayes (KL ≤ 0.03, 11/11). From a donor with e_h = 0 they move it to cf_e0 in 10/11 models;
  wP142 is borderline, 0.23 vs 0.19.
- The 6-dim E-subspace therefore behaves as a mediator whose *value*, not its exact coordinates, determines the
  decision.

**Restoration after a natural-activation removal.** This was added after the freeze (`restore_nat.py`,
`logs/restore_nat_dev*.json`, 8 models). Layer 1's conjunction is removed by path_inA, a natural-activation design.
Then only the 6 E coordinates of the intact xAP state are restored:

| | KL(Bayes) |
|---|---|
| ablated | 0.62–0.78 |
| **restore E (6 dims)** | **0.07–0.16** |
| restore random 6 dims | recovers ≤ 0.012 of the gap |
| restore the 442-dim complement | recovers 0.06–0.64 |

XI's restoration result is therefore not an artifact of its hybrid ablation.

## 2. Weight-level operations in layer 1 (B1)

### 2.1 Queries are irrelevant; group h's keys and values carry the conjunction

These rows use path interventions that feed xA input only to some attention inputs:

| intervention | KL(Bayes) / KL(cf_e0), all 11 models |
|---|---|
| xA input at the q7 **queries** only | 0.00–0.01 / 0.39–0.41: no effect |
| xA input at **keys and values** only | identical to full path_inA to 3 decimals |

The conjunction is computed from what the q7 tokens *read*: group h's tokens at q and at the other questions. It does
not depend on what the queries ask.

### 2.2 Attention scores are nearly additive; the interaction arises after the scores

`circuit_l1.py`. Per head, pre-softmax scores were measured for keys in three groups:
- group h's tokens at q ("hq");
- group h's tokens at the other questions ("hoth");
- all other tokens.

Findings:
- A shifts the scores of the hq keys by 0.4–1.2 logits; the sign is head-specific.
- P shifts the hoth scores by 0.05–0.25 logits.
- The *interaction* in the scores is only 2–8% of these main effects (max over heads, every model).
- The attention-output interaction is reproduced in most heads by applying softmax to **additive** scores. Summed
  over heads, the softmax-of-additive-scores term is 0.09–0.59 of the total E-interaction, against 0.14–0.65 for the
  full pattern term.
- The conjunction is created by softmax normalisation coupling an A-driven score shift with a P-driven shift, plus the
  bilinear product of attention weights with value changes. **It is not created by a multiplicative QK feature.**

### 2.3 "Pattern vs value" attribution inside a head is not identifiable

The head output is o = Σ_s a_s u_s. Its double difference splits exactly into pattern, value and cross terms. That
split depends on the reference cell. For xP81's heads 0–2:
- with values fixed at x00, the pattern term carries the whole interaction (0.27 / 0.20 / 0.10 of the E-interaction);
- with values fixed at xAP, it carries **0.02 / −0.01 / 0.00**.

Weight-level "pattern vs value" attributions of an interaction are basis-dependent and should not be reported as
mechanism. This is a cautionary methodological finding of this investigation.

The **intervention-level** facts are invariant:

| intervention | effect in development models |
|---|---|
| remove P's effect on the attention **patterns** only (patterns from xA) | decision stays Bayes-side in 10/11 |
| remove P's effect on the layer-1 **values** only | decision stays Bayes-side in 11/11 |
| remove both (keys/values from xA) | full cf_e0 shift in 11/11 |

**The perfectness signal reaches the conjunction redundantly**, through both the routing (patterns) and the content
(values). Either route alone suffices. The exception is wP143, where the pattern route is dominant: removing it gives
KL 0.49 / 0.11. This is redundancy *inside* attention (H4 at the sub-head level). It is invisible to XI's whole-layer
ablation.

### 2.4 MLP1: smooth amplification, not a thresholded AND and not a backup route

| model | attention share of E-interaction (direct write) | MLP1 share | MLP1 own nonlinear interaction | from attention (transmitted) | threshold-AND neurons (fraction / share) | LN2-frozen factor |
|---|---|---|---|---|---|---|
| xP81 | 0.62 | 0.38 | 0.04 | 0.33 | 0.025 / 0.15 | 1.00 |
| yP130 | 0.17 | 0.83 | 0.36 | 0.47 | 0.024 / 0.34 | 1.00 |
| yP131 | 0.62 | 0.39 | 0.07 | 0.32 | 0.022 / 0.15 | 0.93 |
| yP132 | 0.21 | 0.79 | 0.28 | 0.51 | 0.022 / 0.28 | 0.98 |
| yP134 | 0.64 | 0.36 | 0.02 | 0.34 | 0.043 / 0.30 | 1.04 |
| wP140 | 0.56 | 0.44 | 0.02 | 0.41 | 0.030 / 0.18 | 0.95 |
| wP141 | 0.22 | 0.78 | 0.06 | 0.71 | 0.042 / 0.21 | 0.76 |
| wP142 | 0.46 | 0.54 | 0.06 | 0.48 | 0.029 / 0.22 | 1.10 |
| wP143 | 0.58 | 0.42 | 0.03 | 0.38 | 0.037 / 0.13 | 0.95 |

- MLP1 mostly **carries forward and amplifies** the attention-formed interaction. Its own GELU curvature on additive
  inputs is small (0.02–0.07) except in yP130 and yP132 (0.28–0.36).
- Threshold-AND units (pre-activation crossing zero only in xAP) are rare (2–4% of unit-token pairs) and carry only
  13–34% of the MLP1 interaction. The operation is a **smooth interaction**, not a learned Boolean gate.
- Freezing LN2 statistics changes little (factor 0.76–1.10). LayerNorm is not the source.

**Direct vs total effect** of removing the attention interaction (XI additive completion of the head outputs only),
as KL(Bayes):

| MLP1 condition | KL(Bayes) |
|---|---|
| MLP1 frozen (direct effect) | 0.01–0.35 |
| MLP1 recomputed (total effect) | 0.24–0.63 |

The total exceeds the direct effect in every model (≥ 1.5× in 11/11). MLP1 *amplifies* the attention interaction.
This is the opposite of compensation.

Single layer-1 heads removed by natural path patching likewise give total ≥ direct, e.g. xP81: head 0, 0.28 direct
vs 0.39 total; head 2, 0.11 vs 0.34.

**No compensating (hydra-like) backup was found in the fixed forward computation.**

Limits of this test:
- it does not exclude routes that would appear only under joint removal of layer 1 *and* layer-2 components;
- it does not exclude routes acquired by retraining;
- restoration through the 442-dim complement of E recovers a minority of the decision (XI: 0.06–0.58), so some
  decision-relevant residual information outside E exists.

## 3. The causal chain, with the status of each link

| link | availability / decodability | causal transmission | computation | mediation | downstream use |
|---|---|---|---|---|---|
| source reports → agreement-at-q at q7 (layer 0) | decodable after L0 attention (XI) | removing L0's interaction: no effect | none (no interaction at mid0) | — | — |
| perfectness of h → q7 (layer 1 keys/values) | decodable after L1 attention (XI) | keys/values path necessary (§2.1) | — | — | — |
| conjunction (layer-1 attention softmax coupling + MLP1 amplification) | e decodable even in C1 (not evidence) | natural path patching | **interaction first appears in layer 1** (P1) | — | — |
| E-subspace at resid2 | — | — | — | scrubbing (same-e donors preserve; e_h = 0 donors switch); restoration (XI) | — |
| layer 2 → candidate evidence | — | — | — | — | decision tracks cf_e0 under removal: the other groups' weights (S − e_g) are what change |

The candidate-evidence step is stated at the level of the Bayes formula. The decision change under removal matches
the reweighting of the other groups' evidence. Layer 2's internal circuit was **not** dissected in this investigation.

## 4. Same-parent versus cross-parent organization

The development basis is XI §6:
- same-parent seed pairs share head identities: raw profile distance 0.40 vs 0.62 after best permutation across
  parents;
- transfer did not differ.

The fresh test is frozen prediction P5 (§7).

## 5. Learning dynamics: why the representation "precedes" behaviour (B5)

See `GENERALIZATION_AND_LEARNING.md` §2. Summary [dev, 9 models]:
- The decision uses the conjunction from the checkpoint at which it first exists (use onset = emergence ± 1
  checkpoint, 9/9).
- XI's ordering came from the behavioural *classifier*. Investigation X's D2 rule-closeness passes through an A2
  (threshold-4) phase before Bayes, which is a calibration of the whole decision, not a readout lag.

## 6. What remains uncharacterized

- **Layer 2.** Its internal mechanism was not dissected: how (S − e_g) reweighting is implemented.
- **The complement of E.** The 442-dim complement carries some decision-relevant residual information.
- **The mechanism of the sub-head redundancy.** Why both routes suffice is not explained: it might be one shared
  upstream feature read twice.

## 7. Fresh-cohort confirmation

Filled from `logs/score_xii.json` after the confirmation; see the end of this file.
