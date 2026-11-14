# Review 3: mechanism and causal inference (Investigation XIII, Phase 1.5 Step 1)

Reviewer: independent mech-interp / causal-inference review. I did not modify any package file. All new numbers below come from the
package JSONs or from light re-analysis scripts in `/tmp/review_mech/`:
- `sens.py`: frozen scorer re-run with a modified unit set;
- `enea1.py`: donor-test episodes regenerated from stream 74_100_000. The regenerated Bayes predictions match the stored
  `per_episode_pred_donor` to 2e-7. These are re-analyses of stored observed changes, with no forward passes;
- `enea1_attn.py`: one forward pass over the 93 E≠A1 donor episodes, for uP161, uP165 and uP164;
- `s2.py`: Bayes score contrasts on the score_contrast stream.

## Verdict summary

- **The bookkeeping is sound.**
  - The frozen verdicts are reproduced exactly.
  - The uP164 sensitivity claimed in FROZEN §6.3 is reproduced: P1 and P2 are falsified; P3, P4 and P7 are confirmed 3/3 at share ≥ 0.19.
  - The falsifications of P5 and P8 are reported honestly.
- **The confirmed mechanism claims are true for two parents (uP161, uP165).** The wording overstates in three ways.
  1. **What the interventions add.** They show that the natural mixture-head attention row is causally used as mixture weights over
     per-group likelihood values. The "≈ Bayes responsibilities" part, and in particular the E-vs-A1 discrimination, is an
     observational property of the rows the model produces. The causal test adds only "and these rows are used". It does not identify
     an upstream variable.
  2. **The P3 "E coordinates" claim.** The E, A1 and PERM subspaces share 4–5 of their 6 directions, so the claim is not about E.
  3. **The DAS conclusion.** The P8 data do not show that "A1 is represented", nor that "each subspace implements its own rule". What
     A1-trained subspaces do on E≠A1 pairs cannot be told apart from what PERM predicts, and a source-free mean patch matches it.
- **Generality.** The mechanism is a property of 2/5 complete fresh parents, plus 1 incomplete one. It sits on a knife-edge class
  threshold. "Confirmed" must always be qualified as "confirmed in the 2 strong-share mixture-class parents (2/2 under the frozen rule)".
- **The exploratory transition predictor has no demonstrated predictive value.** The KL drop reaches t50 at or before ALIGN in all 7
  mixture-forming models. The "lead" is a property of the late, noisy gap metric.

## Supported

1. **P1: the attention row is a causally used mixture weight.** Within uP161 and uP165:
   - the no-op is exact (≤ 1e-6);
   - shuffled R² is 0.000;
   - a uniform row gives R² 0.86–0.88 against the uniform-weight prediction;
   - donor rows give R² 0.915 and 0.933 with slopes 0.92–0.95.

   The head output therefore behaves as Σ_g w_g·(per-group likelihood) for the inserted weights w. This is the genuinely causal content,
   and it is solid in these two parents.
2. **The natural rows exclude the sole eligible group (E, not A1).**
   - All 93 E≠A1 donors have the same configuration: S = 1 and S_pf = 2.
   - The dominant heads put a mean of 0.03 mass on the sole eligible group. E predicts 0; A1 predicts 0.77.
   - Observational corr(mass, r_E) is 0.88–0.995; corr(mass, r_A1) is about −0.33.
   - In the causal fidelity, the bootstrap CI of R²(E) − R²(A1) over donor episodes is [0.64, 0.80] (uP161) and [0.67, 0.82] (uP165).
   - Rescaling each episode separately leaves the gap (Bayes 0.975–0.979 vs A1 0.72–0.76), so it is not a scale artefact.
   - n = 93 is adequate for the sign, but it covers exactly one configuration.
3. **P2: the mediation design is reasonably matched.**
   - TE magnitudes are almost identical in TRUE and CTRL: 1.25 vs 1.16–1.18.
   - The TE-normalised mediated fraction is 0.25–0.33 in TRUE vs 0.025–0.08 in CTRL.
   - So "eligibility-specific, partial" is supported for uP161 and uP165. The two-sample bootstrap equals the frozen CI.
4. **P4a/c and P5b.**
   - Layer-1 mean ablation abolishes the alignment.
   - The decision is linearly decodable from the complement of E (0.82–0.89 in all six parents), not from E.
   - Attention-only and MLP-only layer-2 adaptation on frozen resid2 reach Bayes in 5/5 complete parents, and in uP162.
5. **Reporting honesty.** The documents now disclose:
   - the n = 2 limitation;
   - the uP164 sensitivity;
   - the single-head decoy;
   - the post-freeze status of the stats, report and transition scripts;
   - the vP150 observation that anticipated the P5 failure.

## Not supported / over-claims (must-fix wording)

**M1. "Attention ≈ Bayes responsibilities" is presented as causally established.** (Synthesis §1.3 and §3.1; NATIVE §4 and §8.)
- **The fidelity criterion certifies the value pathway.** A uniform row passes nearly as well (R² 0.86–0.88 vs 0.92–0.93). P1a
  therefore certifies "the output is a mixture over the inserted weights", not that natural weights equal r_g.
- **The identification of r_g comes from P7-type correlation.** The donor row is the model's own attention, which correlates
  0.86–0.98 with r^d.
- **The direct test was computed but not reported.** `mass_resp`, the prediction from the donor's actual attention mass, scores R²
  0.868 (uP161) and 0.927 (uP165). That is *below* the Bayes prediction (0.915, 0.933), which means the downstream path reshapes the
  weights; equal-weight averaging over heads of different share also contributes.
- **The attention is a tempered approximation.**
  - Scores regress on log W with a coefficient of about 0.6, not 1.
  - At S = 2 the head's eligible-vs-ineligible score contrast is +1.4 to +2.9, while the Bayes log-r contrast on the same stream is +4.85.
  - At S = 1 the confirmation contrast is −2.1 to −3.2, not the "about 6 logits" quoted from dev.
- **Only the dominant head meets P7.** In uP165, two of the three heads the scorer classes as mixture heads (h0 and h3) have corr(B)
  0.52–0.54. They would fail P7b individually (B − A3 is 0.02–0.06; B − M is 0.02–0.05).
- **Suggested wording:** "Mixture-head attention rows are causally used as mixture weights over per-group likelihood values (P1a,
  uniform-row control). The natural rows of the dominant head approximate a tempered version of the Bayes responsibilities (corr
  0.86–0.98; P7, correlational)."

**M2. "A natural mediator with a discriminating prediction identifies the variable where matched-budget DAS cannot."** (NATIVE §6,
"clearest methodological result"; CAUSAL §4 and §7; Synthesis §1.5 and §3.11.)
- **P1c's discriminating power is observational.** On the 93 E≠A1 donors, the attention itself already separates E from A1 (above).
  The causal step only adds downstream gain.
- **uP164 shows this directly.** Its head separates E from A1 just as well observationally (corr 0.989 vs −0.31). It "fails" P1c
  (0.217 vs 0.089) only because the head's downstream effect is weak (slope 0.29). After per-episode rescaling, its Bayes and A1 R²
  are equal (0.624 vs 0.622).
- **What P1c establishes:** the network's computed mixture weights implement E-type exclusion in one configuration. It does not
  establish which resid2 variable is upstream. E = A1 ∧ agree-at-q, so the query–key circuit could combine an A1-like feature with an
  agreement feature.
- **Suggested wording:** "The natural attention pattern, which is causally used, implements E-type exclusion (S = 1, S_pf = 2
  configuration; 2/2 strong parents). This is an observational discrimination plus a causal-use test, not identification of a
  represented variable."

**M3. "Each of E and A1 can be found as a patchable subspace that implements its own counterfactuals" / "DAS finds subspaces that
implement whichever variable they are trained for (A1 5/5)" / "correlated variables are both represented".** (NATIVE §6; Synthesis
§3.4 and §3.11.)
- **A1 and PERM cannot be told apart on the E≠A1 stratum.** In `decoy_disc_conf.json`, every subspace (E-, A1- and PERM-trained)
  scores R²(A1) within 0.05 of R²(PERM) there. The PERM-trained subspace "follows A1" (0.29–0.60) as well as A1-trained ones do
  (0.50–0.56).
- **The slopes are small.** A1-trained subspaces have slopes of 0.21–0.32 to the A1 counterfactual. They implement about a quarter of
  its magnitude.
- **A source-free mean patch matches.** A mean patch of E with no source information reaches R²(A1) 0.59–0.69 and R²(PERM) 0.57–0.66
  on all discriminating pairs. That equals or beats the A1-trained subspace's R²(A1) on all pairs (0.59–0.66).
- **What "following A1" means here:** the generic signature of erasing or blurring source-specific eligibility at this site, not an
  implemented A1 variable. Random-subspace nulls are not the right null for P8b; the mean patch is.
- **The subspaces largely coincide.** Principal-angle cosines between the E and A1 subspaces are about 0.98, 0.95, 0.9, 0.8, 0.6 and
  0.3. They are close to the same subspace, so "matched-budget decoys" is a misnomer.
- **What P8 does show:**
  - E-trained subspaces, trained on unstratified pairs where E≠A1 is rare, do not acquire E-specific behaviour on the stratum that
    matters;
  - they fail even the all-pairs E fidelity of 0.8 in uP163 and uP164;
  - "DAS at this site and budget did not recover E-specificity" is supported;
  - "both variables are represented" is not.

**M4. P3: "The score contrast … is carried by the E coordinates" / "depends on the E coordinates".** (P3 title; Synthesis §1.6 and §3.3.)
- **The decoys do the same.** A1 and PERM patches remove 88–221%, and the three trained subspaces share most of their span.
- **Removal above 100% is a sign of off-manifold states.** The contrast reverses sign: E removes 155–222%, while the 442-dim complement
  *strengthens* it by 141–177%. This is not a neutral "removal".
- **Suggested wording:** "Mean-patching a 6-dim region of resid2 at the q-tokens, common to the E-, A1- and PERM-trained subspaces,
  abolishes (indeed reverses) the contrast; random 6-dim subspaces do not. The test does not single out E."

**M5. P2 title: "than size-matched non-mixture heads".** (predictions_xiii.json and FROZEN §4.)
- The decoy is the single head with the *lowest* q-token mass. It is selected to be the least likely to change, so it is the opposite
  of size-matched. "Head-specific" is therefore a weak claim.
- Two of the four mixture-like parents show substantial decoy |NIE|: uP162 at 0.26 (43% of the mixture NIE) and uP164 at 0.17 (48%).
- **Wording:** "than the lowest-q-mass non-mixture head". The title is frozen, so add the correction as a note.

**M6. "Pairwise or lookup … not tested causally beyond what the ATT-MIX fidelity leaves unexplained: 7–9% of variance in strong
parents".** (Synthesis §3.2.)
- R² there is the fraction of variance in the *row-swap-induced changes*. It says nothing about the variance of the decision computed
  through other routes; mediation is itself partial (25–33%).
- Delete this sentence, or say "of the variance of the changes induced by row transplants".

**M7. "No reconstruction after layer-1 ablation".**
- **The intervention is very large.** Every layer-1 head and MLP1 output is replaced by its token-wise mean. That correlation collapses
  is the expected result, and post-ablation q-mass and entropy are not reported. If the row becomes near-uniform, corr ≈ 0 holds almost
  by construction.
- **A promised measure is missing.** The docstring promises a behavioural residual (TRUE − CTRL decision contrast after ablation), but
  `l1abl_mix.py` does not compute it.
- **Wording:** "with all layer-1 outputs mean-ablated, layer-2 attention has no layer-0-only route to Bayes-aligned weights". Avoid
  "does not reconstruct eligibility" as a general statement.
- **P4b is near-trivial.** e_g is decodable at 1.0 from a 442-dim complement of a 6-dim subspace of 448 dims, and the null is
  majority-class at 0.64. The ALTMECH preflight also passes it. Call it "consistent with redundancy", not evidence.

**M8. P5 interpretation.**
- **What "local to layer 2" means.** P5b supports "layer-2 attention-only or MLP-only adaptation on frozen layers 0–1 is
  *sufficient*". A *fresh* layer 2 also reaches Bayes, so the result says that the frozen C = 1 resid2 features are sufficient input
  for one transformer block. It does not show that the native layer-2 mechanism is "repaired" or reused. No arm tests necessity, for
  example by adapting layers 0–1 with layer 2 frozen.
- **Wording:** "C = 2 can be solved from frozen C = 1 resid2 by adapting layer-2 attention or MLP alone (5/5)". Avoid "repair is local
  to layer 2".
- **The readout pattern is not a clean class split.**
  - uP162 is mixture-class, yet fails P5d (S = 3 readout acc 0.66 < 0.7).
  - uP164 is mixture-like and behaves like the pre-assembled parents.
  - Across the six parents, Spearman ρ(resid2 logit share, readout S = 3 acc) ≈ −0.77 (n = 6; my calculation).
  - "Depends on the solution class" should read "covaries with how much of the logit bypasses layer 2 (n = 6, post hoc)".
- **The J_H claim is falsified as a general claim.** NATIVE §7 finding 2 says "the cross-group equality information that J_H needs on
  the support is present in the C = 1 representation". It holds in 2/5 complete parents, and only for a linear readout at the readout
  token. Restrict it to those parents. Also note that a linear-readout failure does not show the information is absent.

**M9. The "native mediator uses E rather than A1" headline is stated as 2/2 confirmation.** Note that the confirmed E-vs-A1 result
covers one configuration: S = 1 with one perfect-but-disagreeing group.

## Sensitivity (uP164 and uP162)

Frozen scorer rerun (`/tmp/review_mech/sens.py`). The pass rule requires ceil(0.75·n) units: 2/2, 3/3, or 3/4.

| unit set (mixture-only predictions) | P1 | P2 | P3 | P4 | P7 |
|---|---|---|---|---|---|
| frozen {161, 165} | conf | conf | conf | conf | conf |
| + uP164 (share ≥ 0.19) | **fals** (a, b, c fail on 164) | **fals** (a, b, c fail) | conf | conf | conf |
| + uP164 at frozen 0.2 | fals | fals | conf | conf | fals (top_mix undefined → None) |
| + uP162 (incomplete) | conf 3/3 | **fals** (b: decoy 0.261 > 0.4×0.608) | conf | conf | conf |
| all four mixture-like {161, 162, 164, 165} | conf (3/4) | **fals** (b 2/4) | conf | conf | conf |

uP164 per-arm values:
- P1a: R² 0.325, slope 0.29;
- P1b: Bayes − A3 = 0.038;
- P1c: 0.217 − 0.089 = 0.128;
- P2a: lower bound 0.150;
- P2b: decoy 0.170 > 0.141;
- P2c: 0.285.

uP164 passes P3 and P4, and P7 when defined (corr 0.939; B − A3 = 0.114; B − M = 0.405). Its mediated fraction is 0.125 (TRUE) vs 0.02
(CTRL). So it is still eligibility-specific in ratio terms; it fails on magnitude and head-specificity.

How to read this:
- **The 0.001 margin decides the verdicts.** P1 and P2 are confirmed only because uP164's head share is 0.199 rather than 0.2.
- **High alignment does not imply strong mediation.** uP164 has a head with corr(B) 0.94 whose causal role in the output is weak. This
  cuts against reading P7-type alignment as evidence of mechanism.
- **The verdict tracks logit share, not the mechanism's form.** The mechanism verdict effectively tracks how much of the logit the head
  carries.

Recommended wording:
> "P1–P4 and P7 met their frozen criteria in both mixture-class complete parents (2/2; the minimum allowed). The result is fragile:
> the next most mixture-like complete parent (uP164, logit share 0.199 vs 0.2) fails P1 and P2, and including it falsifies both. The
> incomplete mixture parent uP162 passes P1 but fails P2b. The mechanism is established for parents in which the mixture head carries
> a large share (≥ 0.3) of the logit, n = 2."

Parent-level t-intervals at n = 2 (for example, the DiD mean CI is [−0.29, 1.65]) carry no population information and should not be
quoted as CIs.

## Missing controls

1. **The direct attention test.** Report `mass_resp`, the prediction from actual donor attention, next to `bayes_resp`. Weight it by
   head logit share instead of an equal-weight head average.
2. **An observational baseline for P1c.** Report corr(mass, r_E) vs corr(mass, r_A1) on the E≠A1 donors. This makes clear that the
   discrimination is observational. Cover more than one E≠A1 configuration; currently every donor has S = 1 and S_pf = 2.
3. **An E-vs-A1 test inside the factorial.** In the TRUE design, A1 is constant between xP and xAP (h is perfect in both). An A1
   mechanism therefore predicts TRUE ≈ CTRL row effects. Computing the A1-based ATT-MIX prediction for the TRUE NIE would give a class
   I+II discriminating test with n = 400 per parent, far stronger than the n = 93 donor subset. It is not reported.
4. **Size-matched decoy heads.** Use a non-mixture head matched on q-mass or logit share, and transplant all non-mixture heads together.
   Use one shared head definition per unit across all scripts. Currently uP165 uses heads {0, 1, 3} in three scripts and {0, 1} in
   `l2_mediation`, and uP161 includes head 2 (share 0.158), which the scorer's rule excludes. The R2 check compares only empty vs
   non-empty lists, so it reports "no mismatch" for both.
5. **Subspace overlap and the right null for P8.**
   - Report principal angles between the E, A1 and PERM subspaces.
   - Add the source-free mean patch as the null on the E≠A1 stratum.
   - Train DAS on a *stratified* E≠A1-balanced pair set, to test whether an E-specific 6-dim subspace exists at this site at all.
6. **Layer-1 ablation.**
   - Report post-ablation q-mass and entropy.
   - Use resample ablation instead of mean ablation.
   - Ablate heads1 vs MLP1 separately and only at the q-tokens.
   - Add the promised behavioural residual.
7. **P5 necessity.** Adapt layers 0–1 with layer 2 frozen. Analyse whether the adapted model reuses the mixture heads; Synthesis
   §3.7 says this was not analysed.
8. **Seed replication.** There is one fine-tune seed per fresh parent. Is the class a property of the parent or of the fine-tune seed?
   Dev shows both outcomes across parents, but the fresh cohort has no within-parent replication.
9. **A restoration experiment through a distinct pathway.** It is acknowledged as not run.

## Exploratory transition predictor (section 9)

- **ALIGN is not built from the decision metric.** It uses no logits; it is the maximum over gated heads of corr(attention mass, r_B).
  It is, however, aligned with the training target by construction (r_B is a factor of the Bayes posterior), so co-movement with
  behaviour is expected rather than surprising.
- **The mass gate makes ALIGN an onset detector.** ALIGN is 0 until some head reaches q-mass 0.5, then jumps to 0.37–0.61. Its t50 is
  essentially "the first checkpoint where some head attends to the q-tokens".
- **The leads are a few grid steps or none.**
  - Confirmation leads are 0, 1, 3 and 4 grid intervals (uP164, uP165, uP162, uP161).
  - The gap t50 sits at the same checkpoint (813) in 6 of the 10 models, mixture or not, which suggests schedule-driven timing.
  - The gap is non-monotone, with an early dip, and |gap| ≤ 0.02. Its t50 shifts by one or two grid steps depending on whether lo is
    taken at step 50 or at the dip (uP161: 1211 → 813; uP162: 2686 → 1803).
- **The critical null fails.** The t50 of KL(Bayes‖model) is at or before ALIGN t50 in **all 7** mixture-forming models:
  - equal in xP81 and vP155;
  - 1–3 grid steps earlier in yP130, uP161, uP162, uP164 and uP165.

  A generic loss metric "leads" the gap at least as much as ALIGN does. The "internal predictor of the transition" framing (NATIVE §9
  title; Synthesis §1.8, "rises at or before the behavioural separation") has no demonstrated predictive value. Restate it as: "ALIGN
  onset coincides with or follows the main KL drop and precedes the noisy D1/D2 gap; pre-assembled models make the same transition
  with ALIGN ≡ 0."
- The confirmation runs contradict the dev claim of "at the same time as the large KL drop": ALIGN lags KL in 4/4 fresh mixture models.

## Minor

- **The S = 1 score gap.** "About 6 logits" (dev; Synthesis §3.1) is not reproduced: the confirmation score-contrast intact S = 1 is
  −2.1 to −3.2. Either report the confirmation value or explain the different measure.
- **The S = 2 comparison.** "S = 2 gap 0.5–1.7 against 0.69 for Bayes" compares the head's raw contrast with the m-factor alone. The
  full Bayes log-r contrast is +4.85 (it includes log W); the head's is +1.4 to +2.9, which is tempered.
- **The decoy is a single head.** CAUSAL §3 states "decoy ≪ mixture" without noting that the decoy is selected to be minimal.
- **The P1b margin over A3 comes from 9% of donors.** Over all donors, Bayes ≡ A3 on the S = 1 donors (91%). The entire P1b margin comes
  from the 177 S = 2 donors (A3 R² ≈ 0 there). Note this.
- **The mediation populations differ.** TRUE and CTRL draw different base populations (n_h = 4 vs 3), and the bootstrap is unpaired.
  This is fine given matched |TE|, but say so.
- **Grid units.** `transition_predictor` t50 uses the last checkpoint as "final", which is sensitive to final-checkpoint noise. Report
  leads in grid steps, not raw steps.
- **The decision is linearly decodable before layer 2 in every class.** Decodability from the complement is about equal in
  pre-assembled (0.885, 0.888) and mixture parents (0.82–0.85). The class difference is about *use* (logit routing), not *content*.
  "The decision is largely present in the residual stream before layer 2" should not be used to distinguish the classes by content.
- **Report the configurations covered.** Report the E≠A1 donor composition (all S = 1, S_pf = 2) wherever "n = 93" is quoted.

## Must-fix items (short list)

1. Reword "attention ≈ Bayes responsibilities" (M1). The causal tests show the rows are *used* as mixture weights. The ≈ r_g part is
   correlational and tempered, and holds for the dominant head only (uP165's h0 and h3 have corr about 0.53). Report `mass_resp`.
2. Reword the "natural mediator identifies the variable where DAS cannot" headline (M2). P1c's E-vs-A1 discrimination is observational:
   attention on the E≠A1 donors already separates E from A1, including in uP164. It covers one configuration and does not identify an
   upstream represented variable.
3. Remove "DAS subspaces implement whichever variable they are trained for" and "both variables represented" (M3). The A1-subspace
   behaviour is identical to the PERM prediction on that stratum, has slope about 0.25, and is matched by a source-free mean patch. E,
   A1 and PERM subspaces share 4–5 directions.
4. Reword P3 "carried by E coordinates" to "a 6-dim region shared by E/A1/PERM-trained subspaces; mean patch overshoots" (M4).
5. Correct "size-matched non-mixture heads" in P2: the decoy is the lowest-mass head (M5). Disclose the head-set inconsistencies across
   scripts (uP165 {0, 1, 3} vs {0, 1}; uP161 head 2) and that the R2 check is class-level only.
6. Delete or fix "7–9% unexplained" as a bound on pairwise or lookup mechanisms (M6).
7. Narrow "no reconstruction" to "no layer-0-only route under full layer-1 mean ablation". Add post-ablation mass (M7).
8. P5:
   - "local repair" becomes "frozen C = 1 resid2 suffices for an adapted (or fresh) layer 2";
   - the readout pattern is graded and post hoc, not class-determined (uP162 fails P5d);
   - restrict the J_H-in-C = 1-representation claim to 2/5 parents (M8).
9. Qualify every "confirmed" as "2/2 strong-share mixture parents; falsified for P1/P2 if uP164 (share 0.199) is included; uP162 fails
   P2b". Stop quoting n = 2 t-intervals as CIs.
10. Transition predictor: KL t50 precedes or equals ALIGN t50 in all 7 models, so drop "internal predictor of the transition". ALIGN
    is a mass-gated onset detector on a coarse grid.
