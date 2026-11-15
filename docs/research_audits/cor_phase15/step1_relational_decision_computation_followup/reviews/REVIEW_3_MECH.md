# Mechanistic and causal review: XIII-B, step1_relational_decision_computation_followup

**Reviewer:** independent mechanistic-interpretability and causal-inference review.
**Access:** read-only for the package. Every check lives in `/tmp/review13b_mech/` (scripts and JSON outputs). Each check took under 5 minutes, used under 1.5 GB, and ran forward passes on 3000 or fewer episodes, apart from the fits, which used 6000 episodes on the addendum stream.
**Scope:** I read every file in the brief. Reviews 1 (math) and 2 (integrity) are taken as given, and their points are not repeated here except where the confirmation data change them.

## Checks run

| script | question | result |
|---|---|---|
| `pm_alt.py` | Profile of R² over ρ; competing one-parameter extensions of MIX4 (lapse floor on L_g, log-blend with unweighted pooling, output saturation); PM plus floor | Reproduces the addendum exactly (uP160 seed 21: ρ 0.569, R² 0.98561) |
| `pm_boot.py` | Paired bootstrap of R² differences at ρ̂ vs ρ fixed at 0.6 or 0.9 | |
| `pm_rescorr.py` | Residual correlation across units under MIX4 vs PM | |
| `das_target.py` | Re-scores the **saved** seed-21 E-trained DAS subspaces against alternative targets | Reproduces the frozen fid_E: 0.376 / 0.134 / 0.771 |
| `site_cover.py` | Share of the purely relational effect carried by resid2 at the DAS site (RO + 6 q tokens) vs other token sets, all five seed-21 units | |
| `route_stab.py` | Stream-to-stream variability of route_Q / route_R (class_features definition, n = 800) | |
| `fp16.py` | Whether adapt2's float16 storage of resid2 for the layer-2-side arms matters | Negligible: mean \|Δ cen log p\| 1.4e-4, no decision flips. **Ruled out** |
| `jpath_copy.py` | Re-runs jpath on dev uP161 early01 / full | The doc's numbers reproduce (0.99 / 0.49 / 0.42). The package has no log for them |

---

## Verdict summary

**The frozen confirmation is a mixed result, and it was scored honestly.** Three predictions were confirmed and three falsified, and AQ1 holds. Mechanistically:

**What holds up**
- **"Pre-assembled" is not a correct description of the early-readout class.** Layer 2 is needed for the relational decision in every class.
- **Candidate evidence reaches the readout token early in early-readout units.** This replicates on new seeds.
- **Readout signature and ρ are tied at the unit level.** The power-mean exponent ρ separates the classes reliably, and it follows the unit's *own* readout signature even when a fine-tuning seed moves a parent to a different class.
- **Mixture-head attention rows are sufficient carriers of the relational effect in late-readout units.** In seed 21 they carry Shapley 0.85–0.88 of the effect, with transplant projection 0.94–0.99 and fid_E 0.95–0.97.
- **Rows follow E, not A1, on the P0 family.** fid_E 0.87–0.89 vs fid_A1 0.04.
- **At β = 0.6, adapting layers 0–1 under a frozen layer 2 reaches Bayes and uses the cross-group relation (J) in all five units.** A layer-2 repair uses J in proportion to the unit's early-readout signature.

**What needs weakening**
- **The routing description.** It is a continuum that the fine-tuning seed can shift, not a categorical law.
- **"Combination by the layer-2 MLP".** This is not identified.
- **"Distinct combination operator".** ρ is a robust class marker, but it is not uniquely an operator. It is algebraically identical to tempered per-group evidence followed by output re-sharpening, and part of it is absorbed by a lapse floor on the evidence.
- **"Class inherited from the parent".** Two of five parents moved by more than 0.3 in lik_RO1 under a new seed.
- **"E is identified".** This holds only at a site that is adequate for late-readout units.

**What the three falsifications mean**
- **Q3 and Q4** fail on the same quantity: route-Q necessity in a single unit, uP161 seed 21. That unit moved half of its relational routing onto route R.
- **Q5** fails in early-readout units for reasons that are largely the design's:
  - the DAS site misses 43–51% of the relational effect in those units;
  - the target is contaminated by the model's own deviation from Bayes.

---

## 1. What is supported

### S1. "Pre-assembled" is inaccurate

- In dev, freezing layer-2 attention and MLP at the readout removes 0.91–0.97 of the relational effect in early units.
- Seed-21 class_features show lik_RO1 ≥ 0.82 in uP160 and uP164 but rel_RO2 of only 0.25–0.27.
- My site check (seed 21) finds that resid2 at RO carries only 0.37–0.49 of the model's own relational total effect in early units.

The relational decision is therefore not complete at the layer-2 input. **This correction is the most solid mechanistic result in the package.**

### S2. Candidate-evidence routing replicates

- lik_RO1 is 0.85 and 0.82 for early units vs 0.006 and 0.047 for late units.
- mix_mass is 0.17–0.28 vs 0.97.
- Relational transfer from the edited group's layer-0 outputs is 0.84–0.93 in all five units (Q3a).

### S3. The power-mean ρ is a robust class marker (AQ1)

| unit | lik_RO1 | ρ |
|---|---|---|
| uP160 | 0.85 | 0.57 |
| uP164 | 0.82 | 0.54 |
| uP163 (hybrid) | 0.37 | 0.69 |
| uP161 | 0.006 | 0.865 |
| uP165 | 0.047 | 0.87 |

Across the five units, corr(lik_RO1, ρ) = −0.99.

**Within-parent switches.**

| parent | seed 20 | seed 21 |
|---|---|---|
| uP163 | lik 0.85, ρ 0.60 | lik 0.37, ρ 0.69 |
| uP164 | lik 0.52, ρ 0.75 | lik 0.82, ρ 0.54 |

ρ follows the unit, not the parent. This is the strongest evidence that ρ reflects the solution, not parent identity or noise.

**Bootstrap** (`pm_boot.py`, paired over test episodes):
- uP160: R²(ρ̂) − R²(ρ = 0.9) = 0.0018, 95% CI [0.0015, 0.0021];
- uP161: R²(ρ̂) − R²(ρ = 0.6) = 0.0010, CI [0.0008, 0.0011].

So the small gains are not noise. Caveat: these CIs are conditional on the fitted parameters, with no refit per resample.

### S4. Mixture-head rows in late-readout units (Q4 a, b, d, e pass)

- **EA family:** Shapley R is 0.88 / 0.85. MIX-row transplant gives projection 0.99 / 0.94 and fid_E 0.95 / 0.97.
- **P0 family:** MIX rows give fid_E 0.87 / 0.89 vs fid_A1 0.04 / 0.04. Restoration gives 0.86 / 0.90 vs 0.03 / 0.05.
- **PERM rows** (same head, same mass and positions, wrong group assignment) give fidelity ≤ 0.35.

Content-specificity *within the mixture head* is therefore shown.

### S5. The two-colluder adaptation (Q6)

- **early01 (layer 2 frozen):** accuracy 0.791–0.798 against Bayes 0.796. J slope 0.87–0.95, R² 0.71–0.88, in all five units.
- **The layer-2 "full" repair's J slope rises monotonically with lik_RO1:**

| unit | lik_RO1 | J slope |
|---|---|---|
| uP160 | 0.85 | 0.53 |
| uP164 | 0.82 | 0.63 |
| uP163 | 0.37 | 0.11 |
| uP165 | 0.05 | 0.00 |
| uP161 | 0.006 | 0.00 |

This graded pattern is better evidence than the n = 2 per-class arms.

---

## 2. Design confounds, by claim

### (a) "Early-readout routing with layer-2 MLP combination"

**path_chain freezes components to their base values.** That is valid path patching, because base values are the natural no-edit activations. But it identifies only "is on the path", not "performs the combination":
- MLP2 is the last nonlinearity at RO, so it necessarily relays any attention-imported change. Freezing it removes the MLP's response to the attention change as well as any computation it does itself.
- Freezing L2 attention removes 0.68 and freezing MLP2 removes 0.76 in dev uP160. Both are large, which is the signature of a serial path.
- "Mostly by its MLP" is not identified. Identifying it would need a test of the multiplicative evidence × weight interaction inside MLP2, for example controlled-responsibility probes or linearised attribution.

**The described early-class route is incomplete** (`site_cover.py`, seed 21). Share of the model's own relational total effect carried by resid2 at each token set:

| token set | uP160 | uP164 | uP163 | uP161 | uP165 |
|---|---|---|---|---|---|
| RO | 0.49 | 0.37 | 0.57 | 0.60 | 0.32 |
| q tokens | 0.07 | 0.12 | 0.28 | 0.38 | 0.65 |
| RO + q (DAS site) | **0.57** | **0.49** | 0.84 | **1.00** | **0.94** |
| other-question source tokens | **0.39** | **0.54** | 0.17 | 0.00 | 0.05 |

In early units, 40–54% of the relational effect at the layer-2 input sits on *other-question source tokens*, and layer-2 attention at RO reads it there. Dev path_chain agrees (L2ATT@RO freeze 0.68 in uP160). The doc's account, "pooled into RO by broad layer-1 readout heads, then combined at RO", leaves out roughly half of the early-class relational route.

### (b) "Shared E-based weighting, different operator (ρ ≈ 0.6 vs 0.9)"

**ρ is not uniquely an operator.**
- PM: log p = (τ/ρ) log Σ w L^ρ. This is *identical* to a posterior mixture of tempered per-group likelihoods L^ρ followed by an output temperature τ/ρ.
- For uP160, τ/ρ ≈ 1.8, so the fitted model is p ∝ (Σ w L^0.57)^1.8.
- "Between sum and product" and "under-confident per-group evidence that is re-sharpened at the output" are the same function. Telling them apart needs internal evidence.

**ρ trades off with other parameters.**
- **Weight exponent α:** along the profile, α rises from 0.61 to 0.96 as ρ goes from 0.4 to 1.0.
- **Lapse floor on L_g:** with ε = 0.044, ρ moves from 0.57 to 0.67 in uP160.
  - A lapse floor alone (FLOOR, one parameter) recovers 59% of PM's gain in uP160: 0.98454 vs 0.98302 / 0.98561.
  - In uP164 it recovers none.
- **The R² surface is flat:** in uP160 it varies by only 0.0001 over ρ = 0.5–0.7.
- The class *separation* is robust (S3). The *value* "≈ 0.6" and its reading as an operator depend on the family chosen.

**Optimiser reliability.** Powell gives a nested superset a lower held-out R² than its subset:
- PMFLOOR < PM in uP164 (0.97216 vs 0.9738);
- FLOOR < MIX4 in uP164.

This suggests local optima. Use multiple starts or gradient-based fits before reporting ρ to two decimals.

**"E-based" is only partly resolved.**
- On D1/D2 the E rule (B) and the threshold-4 rule (A2) are 0.004–0.016 apart. Model distances to B and to A2 differ by ≤ 0.011, for example uP160 D1 B 0.248 vs A2 0.257.
- Q1b therefore establishes "agreement-at-q-gated eligibility (not A1, A0 or A3)", not the n_h = 5 threshold.
- Q1a passes uP164 at MIX R² 0.97001 against a 0.97 threshold.

### (c) "Attention rows are a specific natural mediator"

**The decoys are weaker than described.**
- TOP is the "contribution-matched" decoy, but in seed 21 its direct-logit share is −2e-5 (uP161), 0.009 (uP165) and 0.004 (uP163), against 0.22–0.62 for the mixture heads.
- In uP161 seed 21, **TOP and LOW are the same head (head 2)**. MAG rescales that same head.
- Rows of a head that writes nothing to the logits cannot pass, whatever their content. Q4b's "contribution-matched" wording is therefore empty. The informative decoy is PERM, judged by fidelity.

**The MEAN decoy is not a clean source-free control.**
- It averages donor rows over episodes whose q-token positions differ, so the "row" spreads mass over every question's tokens (L1 row change 1.5–1.7).
- That is an off-distribution pattern, not "the same row with no source information". The mean row should be computed per q position, or in q-relative coordinates.

**Restoration adds little beyond sufficiency.**
- RESTORE_MIX projection (1.01 / 0.97) ≈ single-component MIX transplant projection (0.99 / 0.94).
- Once the rows alone reproduce the effect, restoring them after an upstream removal must work. B3 already concedes "same path".
- The restoration arm does not test mediation independently.

**Shapley over the chosen partition.**
- R bundles query-side information (RO, route R) with key-side information (q tokens, route Q). The rows are where the two routes meet.
- A large R share is therefore expected under *either* routing. That explains why Q4a/b/d pass in uP161 seed 21 even though its routing moved.
- R vs V separates "which group" from "what evidence". It does not separate the upstream routes.

**fid is slope-adjusted R².**
- `RP.fid` fits the slope first, so its R² ignores magnitude.
- In the EA family fid_E ≈ fid_A1 (0.95 vs 0.94), so Q4c's fid_E ≥ 0.85 condition says nothing about E vs A1. Only the P0 family discriminates.

### (d) β = 0.6 adaptation

**Capacity and depth confound.**
- early01 trains two full blocks, including layer-0 attention, end-to-end from raw input.
- The "full" arm trains one block plus the readout on a frozen resid2.
- "Layer-2 adaptation is neither necessary nor sufficient" conflates *which layer* with *how many trainable blocks lie upstream of the readout*.
- Matched controls are missing:
  - layer 1 only, with layers 0 and 2 frozen;
  - layers 1 and 2;
  - a two-block head on frozen resid2.

**jpath "J enters at resid1 from the other-question tokens (0.99)" is true by construction.**
- J-pairs edit only other-question source tokens.
- resid1 at RO and at the q tokens is 0.00, because layer 0 does not carry the change there.
- The informative quantity is resid2: RO 0.49 + q tokens 0.42.
- "The frozen layer 2 reuses its C = 1 mixture operation" was not tested. That would need a row-transplant mediation test of J inside the adapted model.

**Missing provenance.** The early01 jpath output (`logs/jpath_dev_P161.json`) is not in the package; only `jpath_try.json` (native and full) is. I reproduced the numbers (`/tmp/review13b_mech/jpath_uP161__s20.json`).

**decode_J "frozen C = 1 models compute the cross-group identity".**
- The probe is ridge regression on the *concatenation* of all 36 source tokens, so the summation over questions that defines n4 is done by the probe.
- The result shows that within-question four-way-equality features exist at each question's tokens. It does not show that the network computes n4, nor that one attention read could recover it.
- So "present at source tokens" does not by itself explain why a layer-2 repair can or cannot use J.

**J-pair slope as evidence of "use".**
- It is good evidence that the output responds to cross-group equality at q' ≠ q in the exact posterior's direction: native slope 0, product rule exactly 0, early01 R² 0.71–0.88.
- It does not identify the statistic used, which the doc acknowledges.

### Other design points

- **DAS site** (q7 tokens) is inadequate for early-readout units; see §3, Q5.
- **das_bal "NOOP" is not a no-op.** `R = zeros + 1e-9`, but `jnp.linalg.qr` returns an *orthonormal* Q, so the patch replaces a degenerate 6-dimensional subspace whose first direction is uniform. Its magnitude is 0.11–0.22, against 0.15–0.23 for trained E. Relabel it, or use R = 0 with the QR step skipped.
- **class_features on fixed-seed episodes.** mix_mass and share are drawn from 75_700_001/2 (integrity D1). The models are new, so this is a disclosure and CI issue, not a confound. The bigger issue is that **no frozen metric carries a CI**; see the route_Q noise in §3, Q3.

---

## 3. The three falsifications

### Q3 (arm b: uP161 seed 21, route_Q 0.38 < 0.40)

**What it shows.** Routing is not fixed by the parent. In uP161 seed 21, route_R is 0.55–0.61 vs 0.26–0.33 in seed 20, and route_Q is 0.38–0.45 vs 0.63–0.71 (`route_stab.py`, three streams per model). The fine-tuning seed moved roughly half of the relational routing from "into the q tokens" to "into the readout token". A categorical routing law is not supported, even within the late class.

**What it does not show.**
- **That the threshold was clearly missed.** Stream-to-stream SD is about 0.03, and the other two streams give 0.42 and 0.45, which pass. The arm-level verdict is a near-threshold coin flip. The *substantive* finding is the shift from seed 20, which is well outside noise.
- **That late units stop using route Q.** 0.38–0.45 is still 4–8× the early units' 0.05–0.11.
- **That the readout mechanism changed.** The mixture rows still carry 0.88 (Shapley).

### Q4 (arm c: uP161 seed 21, removal 23% < 40%)

**What it shows.** Q4c and Q3b measure the same thing: removing route Q removes little because this unit routes the EA effect through route R. The removal fraction also depends on the family. In the same unit it removes 0.66 (EA_ctrl), 0.57 (P0) and 0.73 (P0_ctrl). So the falsification is specific to the EA family.

**What it does not show.** It is not evidence against the rows:
- in the same unit, RESTORE_MIX gives projection 1.01 and fid_E 0.98;
- decoy restorations stay at the removed level;
- the P0 rows are E-specific;
- preservation slope is 1.04.

The fair summary is: *rows are a sufficient and content-specific endpoint; the necessity of a particular upstream route is unit-dependent*. Q3 and Q4 are not independent falsifications. Count them as one event.

### Q5 (arm a: uP160 fid_E 0.38, uP164 fid_E 0.13; both late units and hybrid uP163 pass)

**What it shows.** At the frozen site and target, a 6-dimensional E subspace does not reproduce E counterfactuals in early-readout units. The thresholds were set only on late units (integrity M6).

**What it does not show: that E is absent from, or unmediated in, early units.** Two design artefacts predict the failure.

**(i) Site.** The q7 site carries only 0.57 (uP160) and 0.49 (uP164) of the relational effect, against 0.94–1.00 in late units (table in §2a). The rank order of site coverage matches the rank order of fid_E exactly:

| unit | site coverage | fid_E |
|---|---|---|
| uP161 | 1.00 | 0.77 |
| uP165 | 0.94 | 0.57 |
| uP163 | 0.84 | 0.54 |
| uP160 | 0.57 | 0.38 |
| uP164 | 0.49 | 0.13 |

**(ii) Target contamination.**
- The target is dE = cen(cfE_Bayes) − cen(P_model,base). It contains the model's own base-episode deviation from Bayes, which no interchange can produce. That deviation is larger in early units (KL to Bayes is 0.0043 / 0.0081 vs 0.0018).
- On these discriminating pairs, the *true* E change is small (mean |Δ| 0.078) and the base error is larger (0.154 in uP160, 0.209 in uP164). Base error makes up 24% and 38% of the target's variance, against 8.5% in uP161.
- Re-scoring the saved subspaces against the Bayes-to-Bayes change gives:

| unit | frozen target | Bayes-to-Bayes | PM-surrogate |
|---|---|---|---|
| uP160 | 0.38 | **0.49** | 0.50 |
| uP164 | 0.13 | 0.20 | 0.20 |
| uP161 | 0.77 | 0.81 | 0.81 |

- Post hoc, uP160 would clear the 0.4 bar. uP164 would not. The frozen verdict stands, but the falsification is mainly evidence about measurement validity.
- A related point: base agreement with cfE is already 0.84–0.87 (the NOOP agrees as often), so decision-agreement metrics are uninformative here.

---

## 4. Is "same function family, distinct approximation (ρ)" justified by R² gains ≤ 0.003?

**Partly.**

**For the claim**
- The gains are small but systematic: bootstrap CIs exclude 0, and ρ is clustered across 23 models. ρ also follows unit-level class switches (S3).
- **Simple calibration or temperature is excluded.** A global output temperature is already the free τ, and τ ≈ 1.0–1.03.
- **Output saturation is excluded.** A tanh soft-clip of MIX4 output (SAT) gains nothing (0.98306 vs 0.98302).
- **A two-pathway log-blend is excluded.** Mixture × unweighted pooling of all groups (BLEND) gains only 0.0003 in early units.

**Against it**
- **ρ explains only about 15% of MIX4's residual** in early units: 1 − R² goes from 0.0170 to 0.0144.
- **The residual correlation between early units falls only from 0.41 to 0.32 under PM** (natural episodes; cross-class 0.15–0.29 → 0.15–0.30). Most of the class-shared error is not captured by ρ.
- **ρ ≡ tempered component evidence plus output re-sharpening**, and a lapse floor on the evidence absorbs part of it in one unit (§2b).

**Wording that is supported.** "Both classes are well fitted (held-out R² ≥ 0.97) by reliability-weighted combinations with agreement-gated eligibility. Early-readout units show a reproducible, class-specific distortion, which the power-mean exponent captures as ρ ≈ 0.55–0.67 (depending on family) vs ≈ 0.87–0.94. Whether this is a different combination operator or mis-scaled per-group evidence is not resolved by input-output fits, and it accounts for a minority of the early class's deviation from Bayes."

---

## 5. Over-claims needing corrected wording

**O1. Class inheritance** (ALT §2 "Early-readout class (uP160, uP163 …)"; §4 "discrete early-readout cluster"; Q2 title "inherited from the pretrained parent").
- Seed 21 moved uP163 from early to hybrid (lik_RO1 0.85 → 0.37). It gained a strict mixture head (mass 0.91, share 0.56).
- It moved uP164 from hybrid to early (0.52 → 0.82, mixture head lost).
- **Correct to:** "the parent strongly biases, but does not fix, the class. 3/4 predicted parents kept their class; 2/5 parents moved by > 0.3 in lik_RO1 under one new fine-tuning seed. The signature distribution is clustered, but membership is a property of the fine-tuned unit." List uP163 as early in dev and hybrid in seed 21.

**O2. "Combination: … by layer 2, mostly by its MLP"** (ALT §2).
- **Correct to:** "layer 2 (attention and MLP, serially) is on the path of ≥ 0.9 of the relational effect. Which sublayer performs the evidence × weight combination is not identified. In early units, 40–54% of the relational effect at the layer-2 input sits on other-question source tokens and is read by layer-2 attention."

**O3. "This is a distinct approximation strategy … not only a different location"; "matches the class-shared error pattern"; the attention-vs-MLP rationale** (ALT §3b).
- **Correct to** the §4 wording above.
- Mark the attention/MLP rationale as a hypothesis.
- State that ρ ≡ inner evidence temperature with outer τ/ρ.
- Report the dependence of ρ on family (+lapse: 0.57 → 0.67).

**O4. "They agree on the relational variables used" / "E-based"** (ALT §3, §3b; Q1).
- **Correct to:** "agreement-at-q-gated eligibility. D1/D2 do not separate E from the threshold-4 variant A2 (distance differences ≤ 0.011)."

**O5. "Routing law … differs categorically"** (Q3 title; NOTES_DEV A).
- **Correct to:** "graded and seed-dependent. uP161 seed 21: route_Q 0.38–0.45, route_R 0.55–0.61 vs seed 20 0.63–0.71 / 0.26–0.33."

**O6. Mixture rows "survive magnitude- and contribution-matched decoys"; "restoration through an independent pathway"** (CAUSAL B1/B3; Q4 title).
- **Correct to:**
  - "specific relative to rows of heads with ≈ 0 direct-logit share (TOP, MAG, LOW), and content-specific relative to group-permuted rows of the same head (by fidelity)";
  - "the TOP decoy was not contribution-matched in seed 21 (share ≤ 0.01; TOP = LOW in uP161)";
  - "MEAN is an off-position average";
  - "restoration is implied by row sufficiency";
  - Q4's falsification concerns route-Q necessity, not the rows.

**O7. "E is identified"** (CAUSAL B2; Q5 title).
- **Correct to:** "identified at resid2 (RO + q tokens, k = 6) in late-readout and hybrid units only. In early-readout units this site carries ≈ 0.5 of the relational effect and the target includes model-vs-Bayes base error, so Q5's failure there is uninformative about E."
- Fix the stale promise at CAUSAL L113.
- Fix the NOOP label.

**O8. "Frozen C = 1 models compute the cross-group identity"** (JOINT C2).
- **Correct to:** "within-question four-way equality features are linearly decodable at each question's source tokens (probe on concatenated tokens; the probe performs the cross-question sum)."

**O9. "Layer-2 adaptation is therefore neither necessary nor, here, sufficient"; "The frozen layer 2 then reuses its C = 1 mixture operation"; "J-dependence enters at resid1 from the other-question source tokens"** (JOINT C3/C4).
- **Correct to:**
  - "a layers-0–1 repair (two trainable blocks) suffices under a frozen layer 2; a one-block layer-2 repair on frozen features does not use J in late units at this budget";
  - capacity is not matched;
  - reuse of the mixture operation is untested;
  - the resid1 statement follows from the J-pair construction.
- Add the jpath log.

**O10. Pre-confirmation text still in present tense.**
- "uP164 … is a hybrid, not a thresholding accident" (ALT §4) is contradicted by uP164 seed 21 (lik 0.82, clean early class).
- The FROZEN doc still lacks its §4 results.

---

## 6. Alternative explanations to state

1. **ρ as evidence mis-scaling.** Early-readout units read candidate evidence at layer 0 through heads that also pool other inputs. Tempered or lapse-floored L_g, followed by output re-sharpening, reproduces the power-mean fit.
2. **Q5 as site and target failure.** See §3. Both artefacts predict the observed rank order of fid_E.
3. **Q3/Q4 as one seed-specific routing shift.** A single event, not two independent falsifications.
4. **Q6 as depth or capacity.** early01 has two trainable blocks upstream of the readout. The class effect on layer-2 repair could be "is J's within-question feature already pooled at positions a single block can read". The graded slope pattern is consistent with this, and it is also what the authors propose. It is not a distinct operator or route claim.
5. **Shapley R large under any routing.** Rows are where the query and key routes meet.

## 7. Missing controls

1. **Uncertainty for every frozen metric.** Stream or bootstrap CIs (route_Q stream SD ≈ 0.03). Several verdicts sit within one SD of their threshold: Q3b, and Q1a for uP164.
2. **DAS for early units at an adequate site** (RO + other-question source tokens, or all tokens), with:
   - the target defined as a Bayes-to-Bayes change (or the model's own counterfactual);
   - a true no-op;
   - a per-position MEAN.
3. **A contribution-matched decoy with real direct-logit share**, for example the mixture head of the opposite family's rows, or a second mixture head swapped with permuted groups.
4. **Matched-capacity adaptation arms** (layer 1 only; layers 1–2; a two-block layer-2 side), plus at least one alternative learning rate or initialisation for the late-unit layer-2 repair.
5. **Mediation of J in early01-adapted models.** Do the frozen mixture rows carry the J effect (row transplant on J-pairs)?
6. **An internal test of the operator.** With two eligible groups whose L_g conflict, vary the responsibilities through controlled patches of the eligibility features. Sum vs product predicts different curvature of log p in the weight. Separately, probe per-group evidence magnitude at layer 2 to test the mis-scaling account.
7. **Power-mean fits with multiple starts or a gradient-based optimiser**, plus KL-loss fits (R² on centred log-probabilities overweights confident episodes), and refit-bootstrap CIs on ρ.
8. **More fine-tuning seeds per parent** before calling class membership "inherited".
9. **Alternative Shapley partitions:** split R into query-side and key-side contributions, and include layer-1 components.

## 8. Minor

- `jpath` early01 results have no log in the package. They reproduce: `/tmp/review13b_mech/jpath_uP161__s20.json`.
- adapt2's float16 resid2 is harmless (checked).
- l2_components `decoy_meta` shows TOP = LOW for uP161 seed 21. Q4's "decoy" margins are meaningless there.
- The EA family cannot discriminate E from A1 (fid_E ≈ fid_A1). Say so wherever an EA fid_E is cited as E evidence.
- PM κ in seed 21 differs by class (early 1.11–1.12, late 0.35–0.37), but dev κ shows no class pattern. Do not interpret it either way.
- ALT §2 lists uP164 dev among neither class but uses it in tables beside early units. Keep the class assignment per unit and per seed.

---

## Must-fix list

1. **Q5 is not evidence against E in early-readout units.**
   - The q7 site carries only 0.49–0.57 of their relational effect, against 0.94–1.00 in late units.
   - The target includes model-vs-Bayes base error: 24–38% of target variance.
   - Re-scored against the Bayes-to-Bayes change, uP160's fid_E is 0.49 (frozen 0.38) and uP164's is 0.20.
   - Restrict "E is identified" to late-readout and hybrid units at that site, and fix the das_bal NOOP, which is not a no-op.
2. **Count Q3b and Q4c as one event.** Both are route-Q necessity in uP161 seed 21, and that unit moved routing to route R (route_R 0.55–0.61). Q3b's miss is within stream noise (0.38 / 0.42 / 0.45). Withdraw "categorical routing law". Report that the rows still pass sufficiency, specificity and E-vs-A1 in that unit.
3. **Withdraw "class inherited from the parent" as stated.** 2/5 parents switched signature under one new seed: uP163 early → hybrid, and it gained a mixture head; uP164 hybrid → early. Fix the early-class membership lists and the "uP164 is a hybrid" sentence.
4. **ρ wording.**
   - Report it as a robust class-specific distortion that follows the unit's own signature.
   - Do not call it a "distinct combination operator". It is algebraically equal to tempered per-group evidence plus output re-sharpening, and its value depends on family (+lapse: 0.57 → 0.67).
   - It explains about 15% of the early class's MIX residual (residual correlation 0.41 → 0.32).
   - Mark the attention-vs-MLP rationale as a hypothesis.
5. **Layer-2 mechanism wording.**
   - "Combined mostly by its MLP" is not identified: path patching shows serial membership only.
   - Add the missing early-class route: 40–54% of the relational effect at the layer-2 input sits on other-question source tokens and is read by layer-2 attention.
6. **Decoys and restoration.**
   - TOP was not contribution-matched in seed 21 (share ≤ 0.01; TOP = LOW in uP161).
   - MEAN is an off-position average.
   - Restoration follows from row sufficiency.
   - Rephrase the Q4 specificity claims accordingly.
7. **Adaptation claims.**
   - Disclose the capacity mismatch: early01 trains two blocks, full trains one.
   - "Reuses the mixture operation" is untested.
   - The resid1 J-entry statement is true by construction.
   - decode_J's probe does the cross-question sum.
   - Add the missing jpath log.
8. **"E-based" means "agreement-at-q-gated".** D1/D2 do not separate E from A2.
9. **Add CIs for the frozen metrics.** Several verdicts sit within about one SD of threshold (Q3b; Q1a for uP164 at 0.97001).
