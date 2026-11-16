# Review 1 (mathematics and statistics): Investigation XIII-D

**Reviewer type.** This is an agent review, not human peer review. Its brief was to try to falsify the package's claims.

**Scope.**
- Posterior and sufficiency statements: two-colluder generator, Proposition S, LOCALQ.
- Metric definitions: J_DM slope, own_share, SRCF/SRCFREE, rel_to_JDM_patch, reverse/restore, MOVE, SHUFD_vs_borrowed.
- The encoding-versus-decoding comparison.
- `scripts/uncertainty_13d.py`.
- Every report, as it stood at 23:45–23:55Z, including MAIN_RESEARCH_SYNTHESIS.md, which was being edited during the review.

**Method.**
- I read the code and the raw logs.
- I ran four small numpy checks (`reviews/review1_scripts/`; each runs in under a minute and uses about 100 MB). They import only `src/genlib.py` and `src/xii_bayes.py`. No model was loaded and nothing was trained.
- The checks build an **ideal carrier**: the model is the exact posterior, and the patched subspace carries the relational statistic exactly and nothing else. This gives the value each metric takes when the hypothesis is perfectly true. Several metrics turn out to have reference values far from the 1 or 0 the reports assume.

Run each script as `python3 reviews/review1_scripts/<script>.py src`.

---

## A. What I verified and found correct

**V1. Proposition S holds.** Take a J_DM base episode, replace its m with the donor's m in the exact log-weights, and recompute. This reproduces B(donor) to within 4.9e-15 over 2400 pairs (`ideal_carrier.py`, "PropS check").
- The proof in XIII-C `logs/NOTES_MATH_C.md` (lines 41–49) is correct, including A = β/K + (1 − β)/K⁴ and B = (1 − β)/K⁴.
- The Corollary-P coefficient is log(A/B) = log(1 + βK³/(1 − β)) = log 41.5 = 3.726. Correct.
- One wording point: "exactly 0" (INFORMATION_ACCESS_AND_REPAIR.md line 13) means zero to floating-point precision, not symbolically exact. This is harmless.

**V2. The LOCALQ gap is reproduced by an independent implementation.** `localq.py` computes the exact posterior from only the self-agreement bits and the values at q*, factorised over questions with the two colluders' joint agreement probabilities.
- Over 5 × 20,000 episodes, the full posterior's accuracy is 0.7940 and the local posterior's is 0.7589.
- The gap is **3.51 ± 0.09 points**. This is consistent with the 3.6 ± 0.07 carried over from XIII-C.

**V3. The shared-copying likelihood is correct.**
- `xii_bayes.posterior_shared` matches the sampler `genlib.sample`: one copy event and one copied value per question, shared by all four colluder sources.
- colH = β·1[all four equal]/K + (1 − β)/K⁴ is the correct per-question likelihood.

**V4. The MOVE invariance is correct.** Under the C = 1, β = 1 generator, a MOVE edit preserves every group's agreement count n_g and the values at q. The closed-form posterior depends only on those, so the invariance is exact (1.1e-16, `e4_noninv.py` line 108).

**V5. J_D0 and DECOY pairs have an exact Bayes change of 0.**
- The J-pair relabelling permutes the values of one group at every question other than q. It therefore keeps all self-agreement bits and the values at q*. By Proposition S, J_D0 pairs (m unchanged) have an exact change of 0.
- The DECOY edit moves one source of a non-agreeing group to the third value. This keeps the group non-agreeing and cannot create an all-four-equal set. Exact change 0 as well.

**V6. The Haufe pattern is computed correctly.**
- `c_carrier.py` line 107 computes A = Σ_x W Σ_ŷ⁻¹, which is the Haufe et al. (2014) forward model.
- After the QR step, span(A) = span(Σ_x W), so the Σ_ŷ⁻¹ factor has no effect.
- The ENC_* and CCD_* results are almost identical (C-C1 ENC_m 0.8458 vs CCD_m 0.8427; C-C2 0.7636 vs 0.7646), as expected, since both are close to span Cov(x, T).

**V7. Verdict bookkeeping is correct.**
- 13 frozen predictions: 7 confirmed, 6 falsified. Each scorer's arithmetic agrees with the raw logs.
- The bootstrap numbers in `uncertainty_13d.json` agree with the inputs.
- The 48 freeze-2 perturbations are 24 thresholds × 2 scalings.

---

## B. Findings

### 1. CRITICAL: the "source-free" control (SRCF) is described wrongly, and its value is what a perfect carrier predicts

**Where.**
- NATURAL_RELATIONAL_CARRIER_AND_NECESSITY.md:
  - line 246: "SRCF: source-free interchange; the q-source tokens are not patched";
  - lines 279–280: "The source-free interchange already carries 0.59–0.65 of the effect. The readout token alone holds much of it. DAS beats SRCF by 0.24–0.32, so the q-source tokens add to it";
  - Limits: "Much of the effect lives at the readout token".
- MAIN_RESEARCH_SYNTHESIS.md line 60: "a source-free interchange already carries about 0.6 of the effect".
- FROZEN table: "DAS − source-free slope".

**What the code does.**
- `c_carrier.py` line 143: `patched(Xa[TE], q[TE], Za[un][TE], P)`.
- The site mask is the same one used everywhere (`site_mask`: the 6 q-source tokens **and** the readout token).
- Only the donor changes. It is the layer-2 state of an **unrelated** episode `un` with the same q (lines 6–8 and 99).
- So the q-source tokens are patched. SRCF is an unrelated-donor control, not a site ablation.

**Why its value is not "a share of the effect".**
- SRCF is scored against dB = B(X_b) − B(X_a).
- Overwriting the base's relational content with an unrelated episode's content produces a change ≈ f(u) − f(X_a).
- This change shares the −B(X_a) term with dB, so its slope on dB is positive even when the donor carries no information about the pair.

**Ideal-carrier check** (`ideal_carrier.py`):

| quantity | ideal carrier | observed (dev; C-C1; C-C2) |
|---|---|---|
| SUFF slope | 1.000 | 0.85; 0.90; 0.82 |
| SRCF slope | **0.720** | 0.652; 0.587; 0.586 |
| SRCF R² | 0.288 | 0.21 (both confirmation models) |
| SRCF mean \|Δ\| / mean \|dB\| | 1.080 / 0.768 | 1.09 / ≈ 0.8 |

So the observed SRCF is about what a perfectly specific m-carrier gives with an unrelated donor. It is not evidence that the readout token "holds much of the effect", and it says nothing about the q-source tokens. The ideal DAS − SRCF excess is 0.28; the observed 0.20–0.32 is on that scale.

**Correction.**
- Describe SRCF as "the same subspace, same site, with an unrelated same-q donor".
- Withdraw the readout-token and q-source-token interpretations, and the "~0.6 of the effect" limit in MAIN_RESEARCH_SYNTHESIS.md.
- Report SRCF against the ideal reference of 0.72. Better, score it by R² or by its projection onto the *unrelated donor's* expected change.
- To answer the token question, run an actual site contrast (RO-only versus QS-only DAS).

### 2. MAJOR: the P4b logic ("synthetic ablation is not specific because it damages unrelated behaviour") is invalid

**Where.**
- NATURAL_RELATIONAL_CARRIER_AND_NECESSITY.md lines 301–303.
- predictions_13d_f2.json P4(b).
- MAIN_RESEARCH_SYNTHESIS.md line 56.

**The flaw.** The "unrelated" damage set is 800 natural β = 0.6 two-colluder episodes (`c_carrier.py` lines 131–133). Their decisions *depend on m*, so a carrier that is perfectly specific and on-manifold must change them when m is removed.

**Ideal check** (`ideal_carrier.py`): replacing m by its mean, or by 0, in the exact posterior gives KL(B ‖ B | m = m̄) = **0.347** on natural episodes.
- Observed DAS damage: ablate 0.35 / 0.57 / 0.51, meanrep 0.54 / 0.50 / 0.45.
- The frozen bound "≥ 0.2" is met by the ideal carrier.
- So P4b's "confirmation" cannot tell non-specific damage apart from correct removal of a used variable.

**Correction.**
- Compare the damage to the ideal value of about 0.35. Only the excess (≈ 0.1–0.2) is a candidate off-manifold artefact.
- Alternatively, measure damage on episodes whose posterior does not depend on m (for example all m_H equal, or a C = 1 control set).
- Restate P4b as "consistent with ablation removing a used variable, plus an excess of ≈ 0.1–0.2 KL".

### 3. MAJOR: the "restore" test has a reference value of ≈ 0.46, not 1

**Where.** NATURAL_RELATIONAL_CARRIER_AND_NECESSITY.md line 303: "Restoring after mean-replacement recovers only 0.24–0.35 of the slope. The subspace's contribution is not separable from the rest of the residual stream once that stream has been moved off-distribution".

**What the code does.**
- `c_carrier.py` line 91, restore mode: x + (μ − x)PPᵀ + (z_s − μ)PPᵀ = x + (z_s − x)PPᵀ, with z_s = Z_b.
- In the donor run, x = Z_b, so the donor run is *unmodified*.
- The test therefore compares an intact donor with a mean-replaced base. Its delta contains the base's relational contribution (the meanrep damage) as well as the pair effect.

**Ideal check:** an ideal carrier gives a restore slope of **0.462** (R² 0.33).
- Observed 0.32 / 0.35 / 0.24, which is about 0.36 / 0.28 of the unpatched slope.
- The shortfall against the ideal is real but modest. It is not the "non-separability" the report claims.

**Correction.**
- Either apply restore to both runs (meanrep on both, then add each run's own component back), so that the null reference is 1.
- Or report the ideal reference and interpret only the deficit relative to 0.46.

### 4. MAJOR: the E1 specificity criterion (DAS own_share − SRCFREE own_share ≥ 0.15) is ill-posed, and P6's interpretation is wrong

**Where.**
- CAUSAL_CONTROL_AND_IDENTIFIABILITY_AUDIT.md lines 76–81: "the source-free interchange alone recovers 0.92 of the effect, more than DAS … the effect is carried by tokens that the source-free control also patches".
- MAIN_RESEARCH_SYNTHESIS.md line 79.

**What own_share is.**
- own_share = Σ d·own / Σ own² (`d_das2.py` line 134). This is a regression of the patched change on the model's own change.
- It is not normalised by the size of the patched change, so a large perturbation with weak alignment can score high.
- In uP165 s20, family E3: SRCFREE mean |d| = 1.70 against DAS mean |d| = 0.186, and the family's own change is small.
- 0.92 is therefore roughly |d|/|own| ≈ 9 times a cosine of about 0.1. It is not "recovering 0.92 of the effect".

**Ideal check** (`srcfree_e1.py`; C = 1 exact posterior, subspace carrying all agreement counts):
- An unrelated-donor SRCFREE already has own_share **0.50 (E5)** and **0.40 (E3)**, while ideal DAS = 1.
- The "≥ 0.15 excess" threshold is therefore not anchored to anything.
- SRCFREE patches **the same tokens** as DAS (`site_mask` with the same `ed`), so "carried by tokens that the source-free control also patches" is not a meaningful contrast.

**Consequences.**
- The verdict as scored stands (P6 FALSIFIED).
- What the falsification means changes. The frozen metric measures magnitude-weighted alignment, not eligibility specificity.

**Correction.**
- Rescore SRCFREE by R² against the own change, or by cosine.
- Report the ideal reference of about 0.4–0.5.
- Withdraw the token-location reading.

### 5. MAJOR: "decodable ≠ used" is overstated; the decoder result is the generic Haufe effect, and key controls are missing

**Where.**
- NATURAL_RELATIONAL_CARRIER_AND_NECESSITY.md lines 272–274: "the central finding … the model *writes* the relation along directions that differ from the best linear readout directions"; and "the content section shows m R² ≈ 0.7–0.8 from the site".
- OPEN_COR_FOUNDATIONAL_QUESTIONS.md line 185.
- MAIN_RESEARCH_SYNTHESIS.md lines 46 and 113.

**Evidence.**
- The ridge decoder (λ = 1e-3·N, `c_carrier.py` line 106) loads on low-variance, noise-cancelling directions. The data show the model is *less* sensitive to the DEC subspace than to random subspaces:

  | test | DEC_m | RAND0 |
  |---|---|---|
  | ablation damage KL (dev / C-C1 / C-C2) | 0.000 / 0.0004 / 0.0014 | 0.011 / 0.018 (confirmation models) |
  | SRCF mean \|Δ\| | 0.04–0.12 | 0.10–0.35 |

  This is the textbook case where decoder weights differ from activation patterns (Haufe 2014). It is a property of the activation covariance, not evidence of where the model "writes".
- ENC_m ≈ CCD_m ≈ ENC_r ≈ CCD_r ≈ DAS_DECOY ≈ DAS, with a total spread in SUFF slope of only 0.12 (C-C1: 0.78–0.90) and 0.07 (C-C2: 0.75–0.82). *Any* subspace that covaries with m transfers the effect. No variance-matched control (the top-3 principal components at the site) was run.
- The cited R² ≈ 0.7–0.8 is from the **DAS 3-d projection pooled over 7 tokens** (21 features, `c_carrier.py` line 157). It is not from the decoder directions. The decodability of m from the DEC subspace was never reported.
- J_DM relabellings change every cross-group equality, not only m. The J_DM response therefore confounds m with other relational changes. This also explains why DAS_DECOY works.

**Correction.**
- Restate the finding as: "interchange along the ridge-decoder directions has no effect; those directions are low-variance and the model is insensitive to them, as expected from Haufe's result".
- Add a top-PC control and report ‖(z_d − x)P‖ for each set.
- Remove "the model writes the relation along …" as a finding about the model.
- Keep P3c as a scored pass, with the interpretation narrowed.

### 6. MAJOR: "a 3-d subspace" is a design choice, not a finding

**Where.** MAIN_RESEARCH_SYNTHESIS.md line 46; INFORMATION_ACCESS_AND_REPAIR.md line 90: "carries m in a 3-d subspace".

**Evidence.**
- K = 3 is hard-coded (`c_carrier.py` line 35). No other k was tested.
- At the full site, m is decoded at 0.96 / 0.87, against 0.76 / 0.70 from the 3-d projection.
- INFORMATION_ACCESS line 90 extends the claim to supply and L0only repairs, which were never carrier-analysed. Workstream C covered only early01 adaptations.

**Correction.** Say "a k = 3 subspace (k fixed by design) is sufficient at 0.82–0.90". Restrict the claim to the early01 adaptations.

### 7. MAJOR: SHUFD_vs_borrowed has no power by construction

**Where.**
- CAUSAL_CONTROL_AND_IDENTIFIABILITY_AUDIT.md line 99: "Shuffled rows explain nothing, whether scored against the base pair or against the borrowed pair. So no generic signal transfers with a row".
- P6 (E2).
- MAIN_RESEARCH_SYNTHESIS.md line 81.

**Why.**
- A change in the attention rows acts through the **base** episode's values.
- The borrowed pair's dB lives in the candidate-value space of a *different* episode.
- So even a perfect row carrier cannot reproduce dB[perm_d].

**Ideal check** (`shufd_ideal.py`): with rows carrying exactly the role responsibilities, and output equal to the exact posterior (checked to 2e-16):
- TRUE R² = 1.000;
- SHUFD vs base R² = 0.007;
- SHUFD vs borrowed **R² = 0.003**.

The observed values (0.000–0.005) are the ideal-carrier values. The arm passes under every hypothesis that can be built from attention rows, so it adds nothing to TRUE and SHUFD-vs-base.

**Correction.** Drop the "no generic signal" sentence, or replace the reference with the change the borrowed role-weight shift would cause *in the base context* (in the ideal model, out(r_b + Δr_borrowed, ℓ_base)).

### 8. MAJOR: P7 builds the intervention's confound into its success criterion, and some reported numbers are wrong

**Where.**
- predictions_13d_f2.json P7.
- `score_f2.py` line 93 (requires `KL_nat(SYM1) > KL_nat(BASE)`).
- CAUSAL_CONTROL_AND_IDENTIFIABILITY_AUDIT.md lines 243–249.
- MAIN_RESEARCH_SYNTHESIS.md line 84.

**Logic.**
- "H_pos is false" is confirmed only when symmetrisation *damages* the model. In the confirmation models, SYM1 raises KL 5.8–22× (for example 0.0049 → 0.106 in uP163 s20).
- A damaging intervention can raise MOVE through generic disruption, which hides any reduction caused by removing the asymmetry.
- The package uses exactly this argument against synthetic ablation in Workstream C. Here it treats damage as part of the confirmation.
- In addition, SYM equalises only the attention *weights*. Value vectors keep their positional content, so a symmetrised row still does not make the readout invariant to MOVE. H_pos is therefore tested only in its weights-only form.

**Number errors** (`logs/e4_noninv_f2.json`):

| report statement | data |
|---|---|
| "Symmetrising never reduces MOVE" | SYM2 reduces MOVE by 7.2 % (uP164 s22) and 4.8 % (uP160 s20); SYM12 by 1.7 % (uP164 s22); dev uP160 s21 SYM2 by 1.8 % |
| "always increases KL" | dev uP161 s21 SYM2 KL 0.0018 = base |
| "Permuting the rows increases MOVE 2.4–5×" | PERM1 ratios are 3.38–5.01× in confirmation and 3.79–7.78× in dev; "2.4" is the reduction value −2.38 (×3.38) misread as a ratio |

**Correction.**
- Restate P7 as "SYM1 did not reduce MOVE at a KL cost of 0.02–0.10. The test cannot exclude H_pos, because the intervention is off-distribution and touches only the weights".
- Fix the numbers.

### 9. MAJOR: the P8 criterion cannot tell "independent" from "interdependent", and the blocking argument is confounded

**Where.** CAUSAL_CONTROL_AND_IDENTIFIABILITY_AUDIT.md lines 219–226; P8; MAIN_RESEARCH_SYNTHESIS.md line 83.

**Evidence.**
- The frozen criterion is "single open route < 0.7 and |interactions| ≤ 0.2". Independent additive routes with the measured path-patching shares (0.66 / 0.27 / 0.02; 0.25 / 0.30 / 0.52; 0.11 / 0.43 / 0.50) also satisfy it. The criterion only excludes redundancy.
- "Independent is ruled out by the blocking" relies on a post-hoc comparison: block-R cost against R's path share.
- Blocking is off-distribution. KL rises to 0.135–0.404 when R is blocked and to 0.31–0.64 when all three routes are blocked. This is the same confound the package uses to dismiss ablation.
- "Blocking R costs 0.6–0.7 of the slope" is, normalised to the unblocked slope, 1 − 0.364/0.957 = 0.62, 1 − 0.387/0.880 = 0.56 and 1 − 0.312/0.879 = 0.65. That is 0.56–0.65.

**Correction.**
- Say that P8 confirms only "not redundant".
- Present the non-additivity under blocking as confounded by blocking damage.
- Fix the range.

### 10. MAJOR: "the decision is made within the first ~100 steps" overstates what the masks show

**Where.**
- READOUT_INITIALIZATION_AND_LEARNING_CAUSALITY.md, B section: "The early decision is made within the first 100 of 4000 steps"; and "That choice is made within the first ~100 steps".
- OPEN_COR_FOUNDATIONAL_QUESTIONS.md line 182: "decided in the first ~1–3% of training".
- MAIN_RESEARCH_SYNTHESIS.md lines 33 and 112: "decided within the first ~2.5% of training".

**Logic.**
- That a 100-step block flips the outcome shows the trajectory is *sensitive* in [0, 100]. It does not show the outcome is *decided* there.
- F1e shows that the early organisation can still be converted permanently by a block over 546–1146 (14–29 % of training).
- For the mixture base, insensitivity is shown only at 546 (F1f), not at 100.

**Correction.** Say "perturbations within the first 100 steps suffice to flip the organisation in both directions; the mixture organisation is insensitive to a block at 546; the early organisation is not committed even at 546".

### 11. MAJOR: capacity and impossibility claims go beyond the evidence, and one is a non sequitur

**Where.**
- INFORMATION_ACCESS_AND_REPAIR.md line 84: "Layer 2 cannot compute m from what it receives, even with full retraining".
- Same file, line 90: "It does not depend on the parent's original organisation: all repair units were mixture models".
- OPEN_COR_FOUNDATIONAL_QUESTIONS.md line 191: "It cannot be changed by retraining layer 2 or layer 1 alone at matched capacity".
- OPEN_COR_FOUNDATIONAL_QUESTIONS.md line 178: "depends on joint initialisation properties, not on capacity".

**Evidence.**
- "Cannot" is supported by two optimiser settings (lr 1e-3 and 3e-3, 1000 steps; XIII-C also 3000 steps). That shows failure to learn under those budgets, not impossibility.
- The package's own table (INFORMATION_ACCESS lines 94–98) shows layer-2 retraining reaching 0.378 in an early model, which contradicts the unqualified OPEN line 191.
- You cannot conclude "does not depend on the organisation" from a sample that contains only one organisation.
- Capacity was never varied in Workstream A.

**Correction.**
- Write "did not learn under the tested budgets".
- Restrict to mixture parents.
- Delete the organisation-independence sentence and the "not on capacity" clause.

### 12. MAJOR (interpretation): P5b's "query side / key side" labels are not established

**Where.** INFORMATION_ACCESS_AND_REPAIR.md lines 41–42 and 68–71; P5b text.

**Logic.**
- m supplied at the readout token can act through MLP2 and the residual path into W_A, not only through layer-2 queries.
- m supplied at the q-source tokens can act through values as well as keys.
- No trace was run on the supRO or supQS models.
- "Either side of the layer-2 key–query score suffices" is therefore an interpretation, not what the arm measures.

**Correction.** Say "either supply position suffices". If the score-side claim is wanted, run the TRACE decomposition on the supRO and supQS checkpoints.

### 13. MAJOR (statistics): no sampling uncertainty on any measured quantity; "95 % intervals" with n = 3 are the min–max range; sensitivity is incomplete

**Where.** FROZEN_CONFIRMATION_AND_STATISTICS.md, "Threshold sensitivity" and "Effect sizes"; `scripts/uncertainty_13d.py`.

**Evidence.**
- With n = 3, the percentile bootstrap places at least 1/27 = 3.7 % > 2.5 % of the mass on each all-min and all-max resample. Every n = 3 "95 % interval" in the table is therefore exactly [min, max]. Examples: DAS sufficiency [0.821, 0.903] equals the values (0.850, 0.903, 0.821); reverse [0.038, 0.059].
- Effect sizes pool development units, which were used to set the thresholds, with confirmation units. The E4 pool also mixes classes: uP161 s21 is a mixture unit in an early-class error test.
- Threshold scaling is not sampling uncertainty. Several verdicts sit inside the plausible sampling error of their statistic, and no standard error was computed for any of them:
  - P2a: F1a and F1e miss by 0.004 and 0.0005 in an accuracy difference;
  - P6: 0.13 against 0.15, from 250 test pairs per family;
  - P4: RAND0 0.79 against 0.7.
- `rescore` does not check the subprocess return code. A crashing perturbed scorer would leave the previous score file in place, which would then be read as the result. The scorers are wrapped in `safe()`, so this is latent rather than realised.
- The scorer writes the live `logs/score_*.json` repeatedly, so "the frozen files were never touched" is true only of the prediction files.
- Freeze 3b (P10) is not included in the sensitivity analysis, and the document does not say so.

**Correction.**
- Label n ≤ 4 intervals as "range".
- Separate development units from confirmation units.
- Add episode- or pair-level bootstrap standard errors for the statistics behind P2a, P4 and P6.
- Check return codes.
- State that P10 was not covered.

### 14. MINOR: the uP164 result is stated backwards in FROZEN, and cell naming is inconsistent

**Where.** FROZEN_CONFIRMATION_AND_STATISTICS.md lines 118–121: "a change of L alone left the outcome early; a change of W alone (W21 placed on L20/D21) did not produce early".

**What the data say.** Taking the early diagonal 21/21/21 as reference:
- 21/20/21 (W changed) stayed EARLY;
- 20/21/21 (L and D changed) was NOT-EARLY.

READOUT_INITIALIZATION_AND_LEARNING_CAUSALITY.md has this right ("W only → EARLY"). The FROZEN verdict row and predictions_13d_f3b.json name 21/20/21 the "L-only cell", which is the opposite convention to READOUT. Note also that the inference "L decides" in uP164 rests on a single L + D cell.

**Correction.** Rewrite the FROZEN sentence and use one naming convention throughout.

### 15. MINOR: wrong inference from the step-0 W_A features, and pseudo-replication

**Where.** READOUT_INITIALIZATION_AND_LEARNING_CAUSALITY.md: "At step 0 no parent has candidate evidence at the readout token (R² ≈ 0)"; and the n = 19 correlations.

**Evidence.**
- R2_RO is the fraction of variance of the *random-W_A-projected* logits explained by candidate evidence (`w_features.py` lines 37–43). It measures W_A's alignment, not whether evidence is present. Presence needs a fitted probe, as used in B3.
- S_RO depends only on (parent, W): uP163:22:21 equals uP163:21:21.
- ‖W_A‖ takes 4 distinct values over the 19 points.
- S_RO1 was the pre-declared principal feature, yet the report leads with S_RO0. Several features were screened with no multiplicity control.

**Correction.**
- Drop the "no candidate evidence" sentence, or measure presence with a fitted probe.
- Report the effective sample size: 4 values of ‖W_A‖, 17 (parent, W) cells.

### 16. MINOR: the transform-screen attribution is too narrow

**Where.** READOUT_INITIALIZATION_AND_LEARNING_CAUSALITY.md: "what matters is the orientation of W_A relative to the fixed layer 2 (L22)"; MAIN_RESEARCH_SYNTHESIS.md line 31.

**Logic.**
- Negation and column permutation also change how W_A reads the layers-0–1 readout-token features. The early route bypasses layer 2.
- The data equally support "orientation relative to the pretrained early features".
- The unmodified early reference at step 1211 has KL 0.121, which fails the package's own competence rule at 1211 (≤ 0.05).

**Correction.** List both readings and flag the reference's competence.

### 17. MINOR: code defects in the carrier CONTENT section and head selection

- `c_carrier.py` line 159: `RP.cen(np.log(PBn + 1e-12))` takes the log twice (`cen` applies log itself), which produces NaN. This is why the "posterior" content row is NaN in every unit.
- NATURAL_RELATIONAL_CARRIER_AND_NECESSITY.md line 289 says "the full 192-d site". The full-site decoder uses 7 tokens × 64 = **448** features (line 164), fitted unregularised on 2000 samples.
- TRACE selects mixture heads by mass ≥ 0.8 only (line 184). It does not use the package's single rule `RP.mixture_heads` (mass and share).

### 18. MINOR: the null-preservation metric is lax and is not compared with the model's own null error

**Evidence.**
- On null pairs, the DAS patch reproduces 65–81 % of the *model's own* null response (`rel_to_own_null` 0.65 / 0.77 for C-C1 and 0.67 / 0.81 for C-C2). This number is not reported.
- The bound "≤ 0.30 of the J_DM patch response" is about 3× the model's own null/J_DM ratio (≈ 0.08 / 0.13).
- Random and decoder sets also pass: rel 0.17–0.45.
- DAS_DECOY's NULL_DEC is evaluated on its own training pairs (`c_carrier.py` lines 128 and 140), so it is in-sample.

**Correction.** Report rel_to_own_null. Describe "null preservation" as faithfulness to the model's own null response.

### 19. MINOR: the "reverse" interchange is misdescribed, and the normalisation is inconsistent

**Where.** NATURAL_RELATIONAL_CARRIER_AND_NECESSITY.md lines 250 and 299 ("the matched null-partner's projection"; "a matched pair in which m does not change"); P4 text.

**What the code does.** `c_carrier.py` line 150: the donor X_b is run with the **same pair's base** projection Z_a, where m differs. Under Proposition S the ideal value is 0 (`ideal_carrier.py`: −0.000).

**Normalisation.**
- "Removes 94–96 %" divides by 1. Relative to the unpatched slope (≈ 0.99 in C-C1, ≈ 0.85 in C-C2), the figure is 94 % / 95.5 %.
- "Random subspaces 3–21 %" (MAIN line 56) is also divided by 1. Relative to the unpatched slope it is 2 % / 7 %.

**Correction.** Fix the description and state the denominator.

### 20. MINOR: metric wording in E1 and E2

- E1: own_share is described as "the fraction of the full interchange effect recovered" (CAUSAL_CONTROL line 138). It is the projection on the model's own donor-minus-base change. FULLSITE own_share is itself only 0.52 (uP165 QF4) to 0.99.
- E2: "The true row interchange explains ≥ 0.90 of the change" (line 199). That is R² against Bayes, and uP161 s20 has R² 0.898. TRUE own_share is 0.73–0.96.

### 21. MINOR: stale and missing pointers

- INFORMATION_ACCESS_AND_REPAIR.md lines 59 and 88 cite "descriptive results in `logs/score_f3a.json`" for L0attn / L0mlp. That file has `[null, null]`, because it was scored at 23:40 and the arms finished at 23:50.
- The arms exist in `logs/adapt4_f3a/uP165_s_20_b0.6_lr0.001.json`: L0attn slope 0.818 with 16,512 parameters; L0mlp 0.008. These support the attention hypothesis, and MAIN_RESEARCH_SYNTHESIS.md already uses them.
- Rerun `score_f3a.py` and update the pointer.
- INFORMATION_ACCESS line 20 ("The math reviewer re-checked them (`reviews/`)") was written before any review existed. This review now covers Proposition S and LOCALQ (V1, V2). The ρ non-identifiability proof and the E identification were not re-derived here.

### 22. MINOR: overstatements about the factorial

- **"All four ended in the mixture organisation" (P1).** A-C4 (uP160 23/21/23) has lik 0.157 > 0.15, so it is *hybrid* under the frozen class rule.
- **"In uP163, W is sufficient in both directions."** The freeze-3b descriptive cell uP163 L21/W20/D21 is NOT-EARLY (0.608 / 0.553), although uP163's diagonal 20/20/20 is early. W sufficiency in uP163 holds for W22 only.
- **"D never matters."** It holds at the class level, but D-only changes move lik_RO1 by up to 0.18 within hybrid cells.
- **L changes confounded with D.** Every "L change" from an early diagonal in uP160, uP161 and uP164 also changes D. Each cell is a single deterministic trajectory, so "necessary" and "sufficient" are single-draw statements.
- **P10 read at step 1211.** The protocol asserts the organisation is "settled by 546–1211", but F1f is still moving after 1146 (0.732 / 0.289 at 1211 → 0.460 / 0.914 at 4000). The uP164 "non-early" reference has lik 0.615 at 1211.

---

## C. Verdict: which claims survive

**Survive as stated.**
- Proposition S and its proof.
- The exact shared-copying posterior.
- The LOCALQ gap (independently reproduced: 3.5 ± 0.09).
- The MOVE invariance.
- Exactness of the J_D0 and DECOY nulls.
- Determinism of training.
- The W-only falsification (P1) and the parent-specific L/W dependence (P10), with the naming fix in finding 14.
- DAS sufficiency (0.82–0.90, far above random).
- Necessity under reverse interchange, with the description fixed.
- The key + query routing of the carried effect (0.94–0.96).
- Supply succeeding versus the no-supply L2 arm failing at lr 3e-3.
- L0only (and L0attn) succeeding versus L1only failing.
- The TRUE row-interchange R² and NOOP.

**Survive only in narrowed form.**
- "Decodable ≠ used" (finding 5): decoder directions are low-variance and are not read; any m-covarying subspace transfers the effect.
- "3-d subspace" (finding 6): k = 3 fixed by design.
- The critical-period statement (finding 10).
- P5b (finding 12): position, not score side.
- P7 (finding 8): no reduction at a KL cost; not decisive against H_pos.
- P8 (finding 9): not redundant; the rest is confounded.
- "Layer 2 / layer 1 cannot" (finding 11): did not, under the tested budgets, in mixture parents.

**Do not survive.**
- The SRCF interpretations: readout token holds much of the effect; q-source tokens add 0.24–0.32; "~0.6 of the effect" (finding 1).
- P4b's inference that the ablation damage shows non-specificity (finding 2).
- The non-separability reading of "restore" (finding 3).
- The E1 reading that the effect is "carried by tokens the source-free control also patches" (finding 4).
- "No generic signal transfers with a row" as evidence from SHUFD_vs_borrowed (finding 7).
- "Symmetrising never reduces MOVE" and "2.4–5×" (finding 8).
- "Does not depend on the parent's original organisation" and "not on capacity" (finding 11).
- The reversed uP164 sentence in FROZEN (finding 14).
- "No parent has candidate evidence at the readout token at step 0" (finding 15).

The frozen verdict *labels* (7 confirmed, 6 falsified) are computed correctly from the frozen rules. For P4 (arm b), P6 (E1 specificity and E2 decoy arms), P7 and P8, however, the frozen rules do not test what the verdict sentences claim. Their meaning should be restated as above.
