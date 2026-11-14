# Native decision computation in layer 2 (Workstream A)

Investigation XIII, Phase 1.5 Step 1. Throughout, "development" (dev) means the models xP81, yP130–134, wP140–143 and vP150–156
(Investigations X–XII). "Confirmation" means the fresh cohort uP160–165, whose results are in section 8 and in
`FROZEN_PROTOCOL_AND_CONFIRMATION.md`. Every number below is labelled as one or the other.

## 1. The question

Earlier investigations showed three things about the trained T4 models:
- they output the Bayes posterior p(y) ∝ Σ_g W_g L_g(y)(S − e_g) closely;
- eligibility e_g is linearly and causally represented at the layer-2 input (resid2);
- a C = 1 → C = 2 transfer needs re-learning.

They did **not** show *how* the final layer turns per-group evidence into the decision. XIII asks:
- which factorization the network implements;
- where it is implemented;
- whether that can be confirmed causally with natural activations and matched controls.

## 2. Candidate factorizations, and an identifiability limit found first

The exact rule can be written as:

- **(a) Mixture** over "group g is the reliable group", with responsibility r_g ∝ W_g (S − e_g) Σ_y L_g(y) and component L_g / Σ L_g.
- **(b) Pooled deep-sets readout** S·A_y − B_y, with A = Σ W L and B = Σ W e L (Investigation VII, Theorem 1).
- **(c) Per-group count factor** m_g = S − e_g.
- **(d) Pairwise comparisons** between groups.
- **(e) A direct lookup** on raw tokens.

**Identifiability limit (dev, `l2_scores.py`).** On the support of T4 with C = 1 and β = 1, the maps (e, S) ↔ (e, m) are bijective. The
cell-count table shows e = 0, m = 0 and e = 1, m = 2 never occur. No realizable input therefore separates "uses pooled S" from "uses
per-group m_g". Claims that distinguish (b) from (c) by behaviour or by natural interventions are **not identifiable** in this task,
and XIII makes none. Only synthetic, off-support interventions could separate them, and those would be class III/V evidence about an
artificial regime.

## 3. Two classes of layer-2 solution (dev, `l2_scan.py`)

Direct-logit attribution at the readout token (the final LayerNorm is linearised at the actual input) splits the 21 dev models into two
classes:

- **Mixture class (15 models).** One or more layer-2 heads put ≥ 0.8 of the readout token's attention on the six source tokens of the
  target question and carry ≥ 0.2 of the centred-logit variance. Per-group attention mass correlates 0.70–0.99 with the Bayes
  responsibilities r_g (≥ 0.8 in 13/15). The correlations are clearly higher than for the alternatives:
  - the separable discount A3 (W_g(1 − e_g)ΣL): 0.59–0.86;
  - eligibility count only (M): 0.45–0.59;
  - evidence weight only (W): about −0.4.
- **Pre-assembled class (6 models: vP150 s10/s11, yP132, yP134, wP141, wP142).** The resid2 skip path alone carries 0.48–0.71 of the
  logit variance, and no head is a mixture head. The decision is largely present in the residual stream before layer 2.

The class looks parent-determined: both seeds of vP150 are pre-assembled. All five incomplete-pretraining parents were mixture-class;
complete parents split 6 mixture / 5 pre-assembled. **This is a description of development heterogeneity, not a claim.** P6 tests it
only as an exploratory prediction. Layer-1 readout attention is never a mixture.

## 4. The attention-mixture mechanism (ATT-MIX), causal tests (dev)

Hypothesis: in mixture-class models the readout query attends to group g's tokens at q with weight ≈ r_g, and the values carry g's
candidate likelihood. The output is then ≈ Σ_g r_g L_g, which is form (a).

**Donor-row test (`l2_mix_causal.py`; class II natural-activation intervention).** For each base episode, the mixture heads' readout
attention row is replaced by the row of a natural donor episode with the same q. The observed change in centred log-probabilities is
compared with the change ATT-MIX predicts from the donor's Bayes responsibilities:

| model (dev) | R², Bayes r | slope | R², A3 | R², A0 | R², A1 (perfect-only) | R², shuffled |
|---|---|---|---|---|---|---|
| xP81 | 0.983 | 0.95 | 0.835 | 0.084 | 0.939 | 0.001 |
| vP155 | 0.965 | 0.97 | 0.818 | 0.073 | 0.919 | 0.002 |
| yP130 | 0.796 | – | 0.69 | – | – | ≈ 0 |
| vP156 | 0.69 | – | – | – | – | ≈ 0 |

The no-op re-insertion changes nothing (max |Δ| = 0). A uniform row over the q-tokens produces the change that the uniform
responsibility predicts (R² 0.88–0.94).

**E versus A1 at the natural mediator.** A1 ("perfect at the other questions", ignoring agreement at q) is the closest alternative to E.
Matched-budget DAS cannot separate it from E (section 6). The mixture rows can. On donors whose E and A1 eligibilities differ (n = 91
per model):
- the transplanted change follows the E-based responsibilities with R² 0.986 (xP81) and 0.966 (vP155);
- it follows the A1-based responsibilities with R² 0.19 and 0.17 only.

The native mixture weights use E.

**Scores (`l2_scores.py`).** The mixture-head pre-softmax score of group g regresses on log W_g with coefficient 0.57–0.69 and on log ΣL_g
with 0.36–0.55. The Bayes log-responsibility would have both coefficients equal to 1.
- **S = 1:** the score gap between the sole eligible group and the others is about 6 logits. Bayes exclusion is −∞, so the network
  approximates a hard exclusion with a finite gap.
- **S = 2:** the gap is 0.5–1.7, against 0.69 for Bayes.
- **Query versus keys:** taking the *query* from a donor with a different S removes the log-m contrast (0.53 → 0.12, −0.02, −0.02).
  Taking the *keys* from the donor removes part of it. The S-dependence is therefore carried largely by the query.
- **Same-S donor query:** a query from a donor with the same S still changes the output (0.52–0.78), so the query carries more than S.
- **Fit of the query models:** QUERY_POOLED R² 0.29–0.39 and QUERY_FULL 0.29–0.52 (`l2_query_swap.py`). These fit only partially, and
  the exact query–key decomposition is **unresolved**.

## 5. What layer 2 reads, and what it does not reconstruct (dev)

**Score contrast (`score_contrast.py`; class III).** Mean-patching the 6-dim E subspace at the q-tokens removes 137–147% of the S = 1
score contrast. A random 6-dim subspace removes ≤ 1.3%. The 442-dim complement makes the contrast *more* negative. **But**
matched-budget decoy subspaces trained by the identical procedure also remove most of it: A1 125–143%, PERM 88–101%. Mean-patching
removal is **not E-specific**, and P3c freezes that non-specificity as a prediction.

**Layer-1 mean ablation (`l1abl_mix.py`; class III).** Ablating layer 1 drops the mixture heads' correlation with r_g from 0.95–0.98 to
about 0, so layer 2 does not recompute eligibility from layer-0 features. Eligibility e_g is nonetheless decodable at 1.0 both from E
and from the 442-dim complement. The Bayes decision is decodable from the complement (0.81–0.92) but not from E (0.41). Eligibility is
represented redundantly. The head's score contrast depends on the E coordinates (above), which shows **use**, not exclusivity.

## 6. Subspace interchange (DAS) is non-identifying between E and A1 (dev, `decoy_disc.py`)

E, A1 and PERM subspaces were trained by `das_decoy.py` with an identical procedure and budget (L2, the 7 q-tokens, k = 6, 1200 steps).
They were then evaluated on two disjoint strata of discriminating pairs:

| subspace → follows rule (R²) | E≠A1 stratum: E | E≠A1: A1 | all pairs: E | all pairs: PERM |
|---|---|---|---|---|
| E-trained (xP81 / vP155) | 0.35 / 0.45 | 0.14 / 0.12 | 0.93 / 0.94 | 0.01 / 0.00 |
| A1-trained | 0.06 / 0.15 | **0.44 / 0.29** | 0.83 / 0.87 | 0.05 / 0.03 |
| PERM-trained | 0.01 / 0.01 | 0.49 / 0.47 | 0.35 / 0.30 | 0.28 / 0.27 |
| random 6-dim | 0.00 | 0.01 / 0.04 | 0.01 / 0.03 | 0.01 / 0.03 |

Each of E and A1 can be found as a patchable subspace that implements its own counterfactuals. Interchange accuracy alone therefore
cannot say which variable the network *uses*. It does reject the group-permuted rule. The natural mixture-row test of §4 is what
separates E from A1. This is the clearest methodological result of XIII: **a natural mediator with a discriminating prediction
identifies the variable where matched-budget DAS cannot.**

## 7. Two-colluder repair is local to layer 2 (dev, `adapt_sel.py`)

Layers 0–1 of a C = 1-trained model are frozen, so resid2 is fixed, and the model is adapted on C = 2 data (β = 1, 1000 steps). All
values are 3,000 fresh episodes × 6 questions.

| condition | xP81 acc | xP81 S = 3 acc | vP155 acc | vP155 S = 3 acc | KL (xP81 / vP155) |
|---|---|---|---|---|---|
| exact Bayes | 0.794 | 0.972 | 0.794 | 0.972 | 0 |
| native (no adaptation) | 0.328 | 0.329 | 0.329 | 0.333 | 1.50 / 1.45 |
| readout only | 0.603 | **0.891** | 0.643 | **0.793** | 0.49 / 0.45 |
| layer-2 attention only | 0.801 | 0.972 | 0.801 | 0.971 | 0.012 / 0.008 |
| layer-2 MLP only | 0.797 | 0.971 | 0.794 | 0.923 | 0.060 / 0.045 |
| layer 2 + readout | 0.801 | 0.972 | 0.804 | 0.972 | 0.005 / 0.003 |
| fresh layer 2 + readout | 0.798 | 0.972 | 0.800 | 0.972 | 0.005 / 0.005 |

There are two findings.
1. Adapting either the layer-2 attention or the layer-2 MLP alone suffices to reach the exact C = 2 posterior from the frozen C = 1
   features. The repair needs no change to layers 0–1.
2. A linear readout of the frozen resid2 already gets most S = 3 decisions right (0.79–0.89), but fails overall (0.60–0.64). About a
   third of S = 3 episodes are trivial: all groups agree at q (MATHEMATICAL_SUFFICIENCY §C2). The independent-copying product rule scores
   0.336 on the whole slice: 0.97 on the trivial part, 0.013 on the rest. The readout's 0.79–0.89 therefore implies that it gets most of
   the non-trivial, J_H-dependent part right. The cross-group equality information that J_H needs on the support is present in the C = 1 representation. What
   the C = 1 network lacks is the C = 2 *combination rule*.

**Post-freeze development observation (disclosed; no amendment).** The pre-assembled dev model vP150 finished its adaptation run at
18:14Z, after the prediction freeze at 18:04Z. It shows the opposite readout pattern:
- readout-only overall acc 0.757, KL 0.167;
- readout-only S = 3 acc only 0.541;
- attention-only, MLP-only and full adaptation all reach Bayes (0.80 / S = 3 0.97).

In the pre-assembled class, the frozen C = 1 resid2 supports the S ≤ 2 (product-rule) part linearly but not the J_H part. In the mixture
class it is the reverse. This predicts that the frozen P5c/P5d arms will **fail on pre-assembled confirmation parents**. P5 was frozen
for all complete parents, and it is scored as frozen.

A first development run was invalid: `optax.masked` let the "frozen" parameters take raw-gradient steps. It was fixed before the freeze
and is disclosed in FROZEN_PROTOCOL §2.

## 8. Confirmation on the fresh cohort

(Filled from `logs/score_xiii.json` when the cohort finishes.)

## 9. Dynamics: an internal predictor of the transition (exploratory)

`scripts/transition_predictor.py`. This analysis was not frozen.

**Internal predictor (ALIGN).** ALIGN uses attention only, no logits. At each saved checkpoint it is the maximum over layer-2 heads with
q-token mass ≥ 0.5 of corr(per-group attention mass, Bayes r_g).

**Behavioural measure.** This is the decision metric: the gap = distance-to-A2 − distance-to-B on the on-support families D1/D2. t50 is
the first checkpoint at which a curve reaches half its final rise. Checkpoints are log-spaced (50 … 4000 steps).

| dev model | t50 ALIGN | t50 behavioural gap | lead (steps) | KL(Bayes‖model) at t50 ALIGN |
|---|---|---|---|---|
| xP81 | 246 | 813 | +567 | 0.21 |
| vP155 | 366 | 546 | +180 | 0.13 |
| yP130 | 546 | 1803 | +1257 | 0.07 |
| vP150 (pre-assembled) | never (ALIGN = 0) | 813 | n/a | n/a |

In the three mixture-class models, the attention mixture forms **before** the behavioural A2 → B separation, and at the same time as the
large KL drop. In the pre-assembled model the transition happens without any mixture head. So ALIGN is a class-specific predictor, not
a necessary condition for the transition.

There are three caveats.
1. The behavioural gap is small in absolute terms (|gap| ≤ 0.01), so its t50 is noisy.
2. The checkpoint grid is coarse.
3. The ALIGN mass gate (0.5) makes ALIGN step-like.

These results are dev and exploratory. The same analysis on the fresh cohort is queued post-freeze (lane TPX) and is labelled
exploratory.
