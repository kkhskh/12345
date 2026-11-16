# XIII-D Review 2: mechanism and causality (independent, adversarial AI-agent review; not human peer review)

**Scope.** I read Workstreams A–E as reported in READOUT_INITIALIZATION_AND_LEARNING_CAUSALITY.md,
NATURAL_RELATIONAL_CARRIER_AND_NECESSITY.md, INFORMATION_ACCESS_AND_REPAIR.md, CAUSAL_CONTROL_AND_IDENTIFIABILITY_AUDIT.md and
FROZEN_CONFIRMATION_AND_STATISTICS.md, and checked them against the evidence below. I also read the prior objections in XIII-C
`reviews/REVIEW_3_MECH.md`.

- **Scripts:** `scripts/ft_factorial.py`, `route_status.py`, `unit_metrics.py`, `score_f1.py`, `score_f3b.py`, `c_carrier.py`,
  `adapt4.py`, `d_das2.py`, `d_routes2.py`, `e4_noninv.py`, `predictions_13d_f2.json` and `predictions_13d_f3b.json`. I also read
  `src/ftlib.py` and the shared `nnlib.block` / `nets.init`.
- **Logs:** `logs/known_outcomes.json`, `unit_metrics_{f1,f2,f3b,tdev,b5dev,fx_dev13d}.jsonl`,
  `route_status_{f1,f1_react,dev,dev_react}.jsonl`, `score_f{1,2,3a,3b}.json`, `c_carrier_{dev,f2}.json`, `adapt4_*/*.json` and
  `d_das2_f2.json`.
- **Loss histories:** the `curve.json` loss histories of the eight freeze-1 runs in `checkpoints/ft/`.

**What I ran.** Read-only Python on JSON files, plus one small numpy calculation of learning-rate mass under the
`warmup_cosine_decay_schedule(0, 1e-3, 200, 4000)` used in `ft_factorial.py`. I loaded no model, trained nothing and modified no
existing file. Any number below that does not appear in a report I computed from the named file.

## Summary verdict

The frozen verdicts are correctly scored, and XIII-D does real work on the XIII-C objections:
- it ran the discriminating W-only cells;
- it ran the release-time masks;
- it tested Haufe patterns against decoders;
- it ran the edited-token-excluded DAS sites;
- it ran the supply-position arms.

Several *interpretations* layered on top of the verdicts do not survive, however:
- **"Mixture is the default/attractor" and "early is conjunctive and fragile"** are not distinguishable from a simpler null: the
  organisation is a sensitive, interaction-dominated function of the whole fresh (L, W) draw, and changing either factor amounts to
  drawing a new initialisation (F1, F2). Some of the package's own numbers contradict the attractor claim directly (F3, F4).
- **The F1 early/late asymmetry** is confounded by how severe each mask is (F4).
- **The carrier's "decodable ≠ used"** is, as measured, largely a filter-versus-pattern geometry effect. The decoders' own decodability is
  never measured (F7).
- **Several components labelled "the" mechanism** (the DAS subspace, the K/Q split, the query side for supRO, layer 0's "direct access")
  are among several sufficient explanations, and the alternatives were not tested (F8–F11).

## Findings

### F1 (critical). "Mixture is the default; early is a conjunctive outcome needing the diagonal's W *and* L" is not separated from "changing a factor is a fresh random draw"

**Evidence.**
- `nets.init` draws WA and each layer from independent `jax.random.split` keys. A same-seed (L, W) pair therefore has no structural
  relation that a mixed-seed pair lacks. An off-diagonal cell is, a priori, simply another random initialisation of the fresh head.
- **Base rates.** I classified every natural diagonal in `known_outcomes.json` with the 4000-step rule:
  - early: 7 (uP160 s20, s21; uP161 s22; uP163 s20, s22; uP164 s21, s22);
  - mixture: 6 (uP160 s23; uP161 s20, s21; uP165 s20, s21, s23);
  - hybrid: 4 (uP161 s23, 0.42/0.92; uP162 s20; uP163 s21; uP164 s20, 0.52/0.87).

  Early is as common as mixture among random inits, so "mixture is the default outcome" is false as a population statement.
- **Off-diagonal cells (L ≠ W)** at 4000, or at 1211 for freeze 3b: 3 of 11 are early. The sources are the READOUT table,
  `unit_metrics_f2/f3b/fx_dev13d`, and uP163 22/21/22 = 0.314/0.913. Diagonals are early in 7 of 17. Fisher exact two-sided p = 0.69.
  The data show no detectable excess of non-early outcomes when a factor is "broken".
- **P1's four cells.** The hypothesis "each changed cell re-rolls the outcome with the parent's early rate" gives all four
  non-early with probability around 0.1–0.2. That is not distinguishable from the conjunctive model at n = 4.

**Why it matters.** P1, P10 and the "Revised description" lead FROZEN §Interpretation to say that "the early organisation is a
conjunctive and fragile outcome". A conjunctive *mechanism* needs a reason why the specific diagonal pair matters. Under the null, the
outcome is f(parent, L, W), with large L × W × parent interaction and no low-order structure. Each "necessary" factor then simply
reflects the fact that re-drawing it changes f.

**Correction.** Restate as: "The organisation is a strongly interaction-dependent function of the fresh (L, W) draw and the parent.
Single-factor changes away from an early diagonal produced non-early outcomes in 4 of 4 cells in uP160/uP161 and 1 of 2 in uP164. No
low-order factor model is supported." Delete "mixture is the default outcome", or support it with a base-rate test. Cheap
discriminating designs:
- several *random* W (or L) re-draws on a mixture diagonal, to estimate P(early | change);
- the same on an early diagonal.

The conjunctive model predicts that P(early) collapses only on the early side. The re-roll null predicts the same rate on both sides.

### F2 (major). "uP163: W decides / W is sufficient in both directions" is contradicted by the package's own W20 cell; the per-parent "which factor decides" table rests on 1–2 cells per parent

**Evidence.**
- uP163 s20 is early (0.795/0.169, `known_outcomes.json`).
- Placing that early diagonal's W on L21/D21 gives lik 0.608, mix 0.553 at 1211, classed NOT-EARLY (`score_f3b.json`, descriptive
  `uP163:fx:21:20:21:stop1211`).
- So in uP163, W22 is sufficient but W20 is not. "W decides in uP163" holds for one specific W, which is again what an interaction
  model predicts (F1).
- The FROZEN "combined pattern" (uP163: W; uP164: L; uP160/161: both) is built from these cells:
  - uP164: 2 cells, at step 1211;
  - uP160/uP161: 2 cells each;
  - uP163: 4 cells, with the W20 counter-example omitted.
- The L22/W21/D22 mirror cell that XIII-C M1 requested exists (0.314/0.913, `unit_metrics_fx_dev13d.jsonl`), but it is absent from the
  READOUT factorial table.

**Correction.**
- Add both uP163 cells to the table.
- Drop "in uP163, W is sufficient in both directions"; say "W22 is sufficient on L21/L22; W20 is not (on L21, at 1211)".
- Present the per-parent pattern as a description of specific cells, not as "which component decides".

### F3 (major). "The mixture organisation is an attractor at every stage tested" contradicts F1c/F1d and misreads F1f

**Evidence** (`unit_metrics_f1.jsonl`, `route_status_f1_react.jsonl`):
- **F1c/F1d** (mixture base, l2q for 0–100 or 0–250) end **early** and stay there for 3900 or 3750 unmasked steps. Final lik
  0.850/0.856, mix 0.158/0.164. At the earliest stage tested, the mixture organisation is therefore *not* an attractor.
- **F1f** does not "return to mixture". At 4000 it has lik_RO1 0.460 and mix 0.914, so it is classed hybrid; the mixture base itself
  has lik 0.034.
  - With the mixture route blocked, F1f's *unretrained* readout still gets **0.788** accuracy (`react.before_acc`). Pure-mixture units
    get 0.32–0.34 (F1a 0.333, F1e 0.322), and the early units get 0.86.
  - So F1f ends as a genuine dual-route model.
- "Returning towards" is also an extrapolation. Under the cosine schedule the learning rate at step 3500 is 4.2e-5, and only 20% of
  the total learning-rate mass lies after step 2000. The trajectory has essentially stopped by 4000.

**Correction.**
- Replace the sentence with: "A 100-step early block flips either organisation permanently (n = 1 per direction). A 600-step block
  at 546 flipped the early run to mixture. The same late block on the mixture run left a stable hybrid in which both routes are used
  (lik 0.46; blocked-route accuracy 0.79)."
- Withdraw "attractor".

### F4 (major). The F1e/F1f "asymmetry in time" is confounded with intervention severity

**Evidence** (`checkpoints/ft/*/curve.json`, training-batch loss):
- **F1e** (ro0 on the early base from 546): the loss jumps from 0.688 at step 500 to **1.090** at 600 and 1.068 at 700. That is
  chance level, ln 3 = 1.099. It only recovers to 0.60 by step 900. The ro0 mask removes all 36 source keys from the readout rows of
  a pretrained layer. The early model is reduced to chance and has to *relearn* the task under the mask. The 546–1146 window holds
  27.8% of the total learning-rate mass (my calculation).
- **F1f** (l2q on the mixture base from 546): the loss goes 0.599 → 0.765 → 0.626, a mild perturbation.
- The relearned solution under ro0 can only be the mixture, because the early route is unavailable. Its persistence after release is
  then ordinary lock-in of a freshly relearned solution. It is not evidence that "the early organisation is not committed even at 546".
- In F1f, by contrast, much of the mixture solution survives the block. The masked logits receive exactly zero gradient, so the head's
  own-q QK affinity is altered only indirectly and by weight decay. That alone could explain the partial re-emergence of the head.

**Correction.**
- Report the onset loss of each late block.
- Frame the asymmetry as "a catastrophic block (to chance) of the early route forces relearning into mixture; a mild block of the
  mixture route does not".
- To test commitment fairly, match severity: a partial ro0 (block a subset of heads or sources) tuned to give the same onset loss as
  l2q, or the reverse.
- Measure the l2q-head own-q mass at the moment of release, to separate "preserved QK" from "re-learned".

### F5 (major). "The early decision is made within the first 100 steps" and "there is no release-time window" overreach

**Evidence.**
- At steps 111 and 246 the unmasked early and mixture bases are not yet distinguishable (`unit_metrics_b5dev.jsonl`):
  - uP160 s21 at step 111: 0.139/0.184;
  - uP160 23/23/23 at step 111: 0.194/0.206;
  - at step 246: 0.336/0.231 vs 0.344/0.334.
- The 0–100 mask lies wholly in warm-up and covers **1.24%** of the total learning-rate mass (my calculation).
- A trajectory that is *sensitive* to a perturbation in steps 0–100 is not one whose decision is *made* by step 100. F1e shows that the
  early outcome can still be reversed at step 546.
- Only mask lengths of 100 and 250 were tested here, plus 600 in XIII-C. Windows shorter than 100 steps are untested, so "no window"
  means "no window ≥ 100 steps".
- There is no sham control: a 100-step mask on an edge that is in neither route, such as readout → episode token. Without one, the
  results cannot separate *route-specific* blocking from generic sensitivity to any early architectural perturbation.
  - The direction of each flip does point to route specificity: each mask flips toward the unblocked route. That argument is weakened
    for F1a by F1/F2: early outcomes flip under every perturbation tried, including WA sign flip and column permutation
    (`unit_metrics_tdev.jsonl`).
  - D-only changes never flip the class (`uncertainty_13d.json`), but they are on-distribution perturbations, not architectural ones.

**Correction.**
- Say "the outcome is sensitive to route availability within the first 100 steps (about 1% of the learning-rate mass), in both
  directions".
- Drop "decision made within 100 steps".
- Add a sham-mask arm and a 25- or 50-step mask before claiming a critical period.

### F6 (major). P2c ("no suppression") rests on one arm and on a within-run baseline; group-resolved presence *is* lower

**Evidence.**
- In `score_f1.py`, P2c's peak is the maximum over steps 246 and 546 *of the masked run itself*. F1b and F1d are non-diagnostic:
  F1b's peak lik is ≤ 0, and F1d is not mixture. So P2c is n = 1 (F1a).
- The proper counterfactual is the same parent in the early organisation. The uP160 runs in which the early route is used (F1c/F1d,
  `route_status_f1.jsonl`) have, at step 4000:
  - present_pooled_RO1 0.994;
  - present_group_RO1 **0.991**.
- F1a, where the route is unused, has 0.934 and **0.776**. F1b has 0.743 and 0.551, and the development ro0T600 run 0.582 and 0.452.
- So the group-resolved information at RO1 is clearly lower when the route goes unused, and the longer the early block, the lower it
  is. The pooled 3-d target hides this.

**Correction.**
- Report P2c as "confirmed in one arm, on the pooled measure, relative to the run's own post-mask peak".
- Add the group-resolved comparison against an early-organised sibling.
- Replace "the route was not suppressed. It is present and unread" with "pooled evidence remains decodable; group-resolved evidence is
  reduced (0.78 vs 0.99)".

### F7 (major). The carrier's "decodable is not the same as used" is, as measured, a filter-vs-pattern geometry artefact, and decoder decodability is never measured

**Evidence** (`c_carrier.py`, `c_carrier_{dev,f2}.json`).
- **How DEC is built.** DEC_m / DEC_r are the QR span of ridge weights fitted on raw, unstandardised 64-d token states with
  λ = n·1e-3. That is near-OLS, so the weights are dominated by low-variance, noise-cancelling directions.
- **The network is less sensitive to DEC directions than to random ones.** Every DEC set gives:
  - ablation damage_KL = **0.000** (dev, C-C1) or 0.001 (C-C2), against 0.005–0.019 for RAND0;
  - a SUFF patch output mean_abs of 0.024–0.093, against 0.026–0.095 for RAND.

  An interchange along a filter direction moves the state only by the projection of the donor–base difference onto that direction. When
  the signal is written along a pattern a and the noise is correlated, that projection is small for *any* reader aligned with a. A null
  along DEC is therefore expected whatever the network reads. It is the textbook Haufe result about the decoder, not a finding about the
  model.
- **DEC decodability is never measured.** The report says "the ridge decoder directions of m and log r read the relation out well; the
  content section shows m R² ≈ 0.7–0.8". The CONTENT block, however, decodes from the **DAS** projection (`proj = ... @ DIRS["DAS"]`),
  not from DEC. No held-out R² of the token-level DEC decoder is computed anywhere.
- **The site decodability claim is trivial.** "Candidate evidence decodable at R² ≈ 1 on the full site" follows because the site
  includes the q-source tokens, whose embeddings carry the source values.
- **ENC ≈ CCD.** ENC_m and CCD_m give nearly identical results everywhere (e.g. C-C1 0.846 vs 0.843; reverse 0.072 vs 0.072). Any
  "signal-direction" estimate works; the contrast is decoder-filter vs signal-direction.

**Correction.**
- Report the held-out decoding R² of DEC.
- Report the per-direction state variance and the patch displacement norm for every set, and add a variance-matched random control
  (random directions drawn from the state covariance).
- Rephrase P3c as: "Interchange along Haufe/class-mean patterns transmits the relation, while interchange along ridge-filter
  directions does not, as expected under correlated noise."
- Remove the general claim that "the model writes the relation along directions that differ from the best linear readout directions"
  as a model property. It holds for any linear Gaussian code with correlated noise.

### F8 (major). "The DAS subspace is necessary / is the carrier" is not unique: every high-sufficiency 3-d set passes the same necessity test

**Evidence** (`c_carrier_f2.json`, C-C1). Reverse natural-donor remaining slope:

| set | remaining slope |
|---|---|
| DAS | 0.059 |
| ENC_m | 0.072 |
| CCD_m | 0.072 |
| ENC_r | 0.087 |
| DAS_DECOY | 0.090 |

C-C2 is similar (0.038 / 0.067 / 0.069 / 0.069 / 0.081). So several differently built 3-d subspaces are each "necessary". That means
they share a dominant direction, or the reverse test is mostly a re-expression of sufficiency. DAS_DECOY, trained on a non-diagnostic
relation, is nearly as sufficient (0.78 vs 0.90) and nearly as necessary.

The report acknowledges "not m-specific". But "the natural-donor test … shows the DAS subspace is necessary" still implies an
identified carrier.

**Correction.**
- Report principal angles between DAS, DAS_DECOY, ENC_m and ENC_r.
- Repeat necessity for the *intersection* and for the orthogonal complements (DAS ⊥ ENC).
- State the claim as: "A low-dimensional relational channel (shared by DAS, the Haufe patterns and DAS_DECOY) is sufficient and, by
  natural-donor reversal, necessary at this site."

### F9 (minor–major). "How the work splits between keys and queries is model-specific" may be a property of which sufficient subspace DAS found

**Evidence.** The TRACE is run only for the DAS subspace, which is trained from a random init (`PRNGKey(11)`):
- C-C1: keys 0.886, queries 0.021;
- C-C2: keys 0.219, queries 0.764.

F8 shows that other subspaces are equally sufficient. The K/Q split could therefore differ between sufficient subspaces within one
model. The invariant reported ("keys + queries 0.94–0.96") is robust, but the "model-specific split" is not shown to belong to the model.

**Correction.** Run TRACE for ENC_m, DAS_DECOY and a second DAS seed. Call the split model-specific only if it is stable across
these.

### F10 (major). Information access: "L1 cannot" and "why layer 0" are budget-limited and have an untested depth alternative; supRO is not shown to act on the query side

**Evidence.** From `adapt4_f3a/*.json` and `score_f3a.json`:
- **Budget.** L0only and L1only were run only at lr 1e-3. P5a itself shows that lr changes outcomes qualitatively (supL2 0.168 at
  1e-3 vs 0.906 at 3e-3). The lr-3e-3 L1only arm was in the development scope (COMPUTE_LOG 20:02Z), but `adapt4_dev_lr3/` contains
  only `native`. "Layer 1 alone cannot either" is therefore an lr-1e-3 result, which is the exact failure mode P5a exposed for supply.
- **Depth.** In L0only, a retrained layer 0 is followed by a frozen layer 1 that already pools across questions (route Q). L1only has
  to compute the within-question four-source agreement *and* the cross-question count in one block. "Layer 0 has direct access to the
  token sources" is not the discriminating property, because layer 1's residual also contains the source embeddings. The alternative
  is the number of downstream compositional steps.
  - L0attn alone reaches 0.818 (`adapt4_f3a/..._lr0.001.json`).
  - The report says the subcomponent results are "see `logs/score_f3a.json` (descriptive)", but that file has `null` for L0attn and
    L0mlp, and L0mlp exists in no output.
- **supRO.** m added at the readout token enters the readout residual directly, so the trainable layer-2 MLP and WA can combine it with
  the attended per-group information without m touching the attention score. "Supplying m only to the readout token (query side)" and
  "either side of the layer-2 key–query score suffices" are interpretations; no trace was run on the supply arms.

**Correction.**
- Run L1only at lr 3e-3 (and 3000 steps) before stating that layer 1 cannot.
- Report L0attn = 0.818. Correct the pointer to `score_f3a.json`, and state that L0mlp is missing.
- Test the depth alternative with an "L1only + one fresh block after layer 1" arm or an "L0 frozen, L1 + L1′" arm.
- For P5b, run the K/Q/V/MLP path decomposition on the supRO model, or relabel it "readout-token supply" vs "source-token supply".

### F11 (major). Workstream E controls: the XIII-C M6 objection persists, and the blocking results are confounded with off-distribution damage

**E1.** At ALLX2 the DAS rule-discrimination R² values are at or below the model's *behavioural* R² in every family (`d_das2_f2.json`,
"behaviour"):

| unit | family | DAS R²_E / R²_A2 | behaviour R²_E / R²_A2 |
|---|---|---|---|
| uP164 s21 | E5 | 0.87 / 0.69 | 0.914 / 0.685 |
| uP164 s21 | E3 | 0.30 | 0.606 |
| uP164 s21 | QF4 | 0.94 / 0.21 | 0.964 / 0.221 |
| uP165 s20 | E3 | 0.79 | 0.902 |

"The rule comparison replicates … E beats A2 … by ≥ 0.15 R²" therefore restates the model's own behaviour. It is not a property of the
subspace. XIII-C M6 asked for the patch-minus-behaviour margin; it is not reported.

Also, R²_E = R²_A1 exactly in E3 and QF4, so E vs A1 is discriminated only by QF5.

*Correction:* report patch − behaviour margins, and restate E1 as "an 8-d subspace at edited-token-free sites transmits the model's own
eligibility response (own-share 0.53–0.89)".

**E3.** Functional blocking uses −1e9 masks with softmax renormalisation, which are off-distribution interventions.
- BLOCK_R masks layer-1 readout → **all 36** sources (`d_routes2.py` line 57, `[:36]`), including q's own sources. It removes
  non-relational candidate evidence too, which explains KL 0.135–0.404 under BLOCK_R.
- "Single open route ≤ 0.51" requires blocking two routes at once, which compounds off-distribution states.
- The path-patching vs blocking discrepancy is therefore just as consistent with "blocking damages the substrate" as with
  "interdependent components".

*Correction:* replace the masks with resample-ablation of each route's attention *output* (that route's head-output contribution taken
from a random other episode), and restrict BLOCK_R to other-question sources. Until then, P8's "interdependent" is a
blocking-specific description.

**E4.** SYM/PERM change rows and raise KL in every unit. A rejection of H_pos by an intervention that also damages the decision
(SYM1 KL 0.023–0.106) is not clean. MOVE could increase because of the damage. *Correction:* say "symmetrising the rows did not
reduce MOVE (and damaged accuracy)". Do not say "the asymmetric rows are part of the correct computation".

### F12 (major). Freeze-3b labelling, classification at 1211, and missing report sections

**Mislabelled cells.** The prediction text (`predictions_13d_f3b.json`) calls arm (a) L20/W21/D21 the "early diagonal's W on the
non-early L, D", but its D is 21, the early D. Relative to the early diagonal 21/21/21:
- arm (a) is an **L-only** change and became NOT-EARLY;
- arm (b), L21/W20/D21, is a **W-only** change and stayed EARLY.

FROZEN §Factorial says the opposite: "a change of L alone left the outcome early; a change of W alone (W21 placed on L20/D21) did not
produce early". The conclusion "uP164: L decides" is right, but the sentence describing it is wrong, and relative to 20/20/20 each arm
changes two factors.

**1211 vs 4000.** P10 classifies at step 1211 on the claim that "the organisation is settled by 546–1211". This package contains a
counter-example: F1f is EARLY by the P10 rule at 1211 (0.732/0.289) and hybrid with mix 0.914 at 4000.

Arm (b) (mix 0.162) is far from the boundary, so the verdict is probably robust. Still:
- arm (c) and the uP164 s21 reference sit at mix 0.258/0.279, the same as F1f's 0.289 at 1211;
- the arm (c) and s21 KL values (0.044/0.042) are close to the 0.05 competence bound;
- the uP161 s22 early reference used in the transform screen is non-competent at 1211 (KL 0.121, `unit_metrics_tdev.jsonl`).

**uP164 has no mixture diagonal.** uP164 s20 is hybrid at 4000 (0.524/0.871), so uP164 cannot inform "mixture is the default".

**Missing sections.** READOUT refers to the sections "Freeze 3b result" and "Transform screen" "below". Neither exists; the document
ends at line 194. The transform results that do exist are unreported:
- WA sign flip and column permutation each turn the early uP161 22/22/22 diagonal into mixture at 1211;
- scale 0.6 on a mixture diagonal is uninformative;
- rot1 and scale1.5 checkpoints exist without metrics.

The sign-flip result is informative. It preserves every spectral and column-space property of W_A and still flips the outcome, which
supports F1 (outcome depends on W's alignment with the parent/L, not on a W property).

**Correction.**
- Fix the labels.
- Continue the three uP164 cells to 4000. `resume.pkl` exists at 1211; a FX_BRANCH-style continuation is cheap relative to new runs.
- Write the two missing sections, and report the transform results as development.

### F13 (minor). P2a reactivation is partial and the cap miss is within noise

**Partial reactivation.** Reactivation lifts F1a and F1e from 0.333/0.322 to 0.592/0.596. The early-organisation ceiling is about
0.86 (F1c/F1d `before_acc`). After-KL is still 0.40–0.44. That recovers about half of the gap, so "show that signature strongly" should
be "partially".

**Noise.** The gain-with-RO1-removed values (0.054, 0.050) differ from the 0.05 cap by less than the binomial SE of a single accuracy on
2000 episodes (≈ 0.011). Neither pass nor fail is informative at that margin. The frozen verdict is fine; the narrative "fail only by
0.004" should add that passing would have been equally uninformative.

**What the probe shows.** WA-only retraining is a linear probe on resid3 with layer 2 fixed. It shows that the information is readable
at the output stage. It does not show that "the downstream readout's choice decides which route is used". Nothing isolates WA from
layer-2 attention as the locus of lock-in.

### F14 (minor). Smaller reporting issues

- **D-only effects.** "D-only mean 0.04" and "data order never changes the organisation" hold for the class. Continuous lik still
  moves by up to 0.18 (uP161 23/23/23 0.422 vs 23/23/20 0.242, `known_outcomes.json`), which is the XIII-C M2 point.
- **Measurement streams.** P1 references were measured on stream 86.1M and the cells on 92.1M. The effect is negligible given
  d ≈ 1.0 vs 0.01, but it should be stated.
- **Factorial pairs.** The pooled "W-only 0.44 / L-only 0.21" means use overlapping cells and selected contrasts. Report them per
  parent, without pooling.

## Alternative explanations: status

| alternative | ruled out? |
|---|---|
| F1 mask side-effects: loss spike at release | **Yes**: no release spike in any run |
| F1 mask side-effects: catastrophic damage at onset | **No** for F1e: loss at chance (F4) |
| F1 mask side-effects: zero-gradient preservation of the masked QK | **Untested** (F4) |
| Generic early-perturbation sensitivity (sham) | **Untested** (F5) |
| Classification at 1211 vs 4000 | **Not ruled out**: one in-package counter-example (F1f); P10 arm (b) is likely robust (F12) |
| "Mixture is an attractor" from 4–6 runs | **Does not follow**: contradicted by F1c/F1d and F1f, and confounded by severity (F3, F4) |
| "Mixture is the default" (population) | **Contradicted** by the base rates: 7 early / 6 mixture / 4 hybrid diagonals (F1) |
| On-distribution interventions | Natural-donor interchange and reversal: **acceptable**. Ablation and mean-replacement: acknowledged as off-distribution. Route blocking (E3), SYM (E4) and late ro0 (F1e): **off-distribution, and not acknowledged** |
| "Decodable ≠ used" as a basis artefact | Carrier (C): **yes, largely** (F7). Route status (B): **no**; presence is a full-state ridge and use is full-state patching, so that dissociation is basis-free |

## Claims that survive

1. **Training is deterministic.** The diagonal fine-tune is the natural seed model, and data order does not change the organisation
   *class* (10 D-only pairs).
2. **XIII-C's "the readout-map initialisation sets the organisation" is falsified** (P1; P10 adds uP164, where a W-only change from
   the early diagonal stays early). The outcome depends on the fresh (L, W) pair jointly with the parent, with strong interaction. No
   factor-level or W-feature-level rule is supported, and W_A sign flip and column permutation alone can flip an early outcome
   (development).
3. **Route availability during the first 100 steps (about 1% of the learning-rate mass) redirects the organisation permanently, in
   both directions** (F1a, F1c; n = 1 each). A late, catastrophic early-route block forces relearning into mixture. A late mild
   mixture-route block leaves a dual-route hybrid.
4. **In a mixture-organised run, pooled early-route evidence stays decodable at RO1 while its causal use (lik_RO1) is ≈ 0.** A WA-only
   probe recovers about half of the accuracy gap, and most of that depends on RO1. Group-resolved information is reduced, and longer
   early blocks degrade the route itself (F1b; development ro0T600).
5. **In early01-adapted two-colluder models, a low-dimensional relational channel at the matched site is sufficient and, by
   natural-donor reversal, necessary.** It acts on behaviour through the frozen layer-2 key–query score, not through values or the
   skip path (0.94–0.96), and the induced score change tracks Δlog r (0.82–0.93). Neither the channel nor the K/Q split is unique to the
   DAS solution (F8, F9). The encoder-vs-decoder contrast is a property of filter geometry (F7).
6. **Access, not capacity.** Supplying m (at either token set) enables a layer-2 + readout repair that fails without it, at lr 3e-3
   (0.906 vs 0.000). At lr 1e-3, a retrained layer 0, and its attention alone (0.818), repairs while layer 1 does not. The "why" and the
   lr-3e-3 L1 comparison are open (F10).
7. **True attention-row interchange of the mixture heads transmits a Bayes-shaped change** (R² 0.90–0.96). Shuffled and decoy rows
   transmit nothing, whether scored against the base pair or the borrowed pair, and NOOP is within tolerance (E2). This is the cleanest
   control result in the package.
8. **Path patching gives near-additive route shares with a class-dependent split.** Route O is about 0 in the mixture unit and
   0.3–0.5 in the early units. The functional-blocking "interdependence" reading needs non-masking ablations (F11).
9. **Symmetrising or permuting readout rows does not reduce the MOVE error** (P7). The source of that error is still unidentified.
10. **Frozen bookkeeping is sound.** The verdicts as scored (7 confirmed, 6 falsified) follow from the frozen scorers and inputs, and
    I found no scoring error. The problems above are interpretive and descriptive (F12 labelling), not violations of the protocol.
