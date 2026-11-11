# Reviews and adjudication

Five independent reviewers examined the final results (pre-review commit 44f17a6): `reviews/R_formal.md` (13 findings),
`reviews/R_statistical.md` (11), `reviews/R_mechanistic.md` (12), `reviews/R_counterexamples.md` (9), `reviews/R_positive_theory.md` (11).
Their scripts are in `reviews/checks/`, outputs in `logs/review_checks/`. Codes: FO formal, ST statistical, ME mechanistic, CX
counterexample, PT positive theory. Corrections in the documents are marked **[R: id]**; the pre-review text is preserved in commit
44f17a6. **No frozen verdict is changed.**

**Verification.** Findings that change a headline were re-checked by me before acceptance: the route-blocking magnitudes (ME1–2, CX9)
recomputed from the frozen arrays (`|N|` with the readout route blocked: cT62 0.114 → 0.447, cT64 0.178 → 0.407, cT65 0.159 → 0.419;
R-error 0.095 → 0.220 etc.; cT67 0.194 → 0.163); the seed pairing of initialisations (ST2) by reading both trainers (both call
`init(…, seed)`) and the reviewer's bit-identity check; the remaining findings by reading the reviewers' scripts and outputs.

## Accepted — headline changes

| id | finding | consequence |
|---|---|---|
| ME1, ME2, ME3, CX9 | Route "shares" are projections; readout-only and readout-blocked null responses are 2.5–3.9× the natural one and nearly opposite; blocking the readout route makes models *further* from Bayes (null-pair KL 0.008 → 0.097 in cT62) and doubles relevant-response error; T1 readouts carry the total count S (R² 0.58–0.83) and not the Bayes-relevant difference d, and per unit ΔS the readout route responds alike in both families. | **The "selectively blockable curriculum-specific nuisance route" is withdrawn.** What remains: in T1 models two non-additive, partly cancelling routes (a source-token route and a readout total-count route) jointly produce a near-Bayes output; null responses are imperfect cancellation of a count-sum signal — an approximation bias, not a separate pathway. P5(a) passed as frozen but is pooled and is positive in 5 of 6 P-count models too (ST4, ME). |
| FO1, FO2, CX1 | The class test measures closeness to Bayes (𝔽_mix contains Bayes; Bayes + arbitrary errors fits it at 0.79–0.87); no ε was frozen; a log-opinion pool with the same parameter count fits as well for P-count models. | **"Same algorithmic factorization across curricula" is withdrawn.** Supported: T1 models are close to Bayes and prefer 𝔽_mix over a log-pool by 0.04–0.11; P-count models are not characterised by these classes (linear vs log pooling undetermined); no algorithm claim for P-count. |
| ST1, PT (quality/stage) | "Explained by quality" was assigned by a non-rejection with power ≈ 0.2 at the observed effect; the manipulation changes transfer budget (stage, schedule), not KL alone; matched only in median. Robust result: within-parent training-stage effect, coh_T(2000) − coh_T(500) = 0.248, CI [0.079, 0.423]. Quality and stage are not separable here. | P2(b) verdict stands as frozen, but its reading becomes: **the T1 coherence is a late-training (near-convergence) property of the same parents; whether its cause is KL or stage is not identifiable.** By the frozen reading P2(b) is only interpreted if P1 replicates, which it did not. |
| ST2 | Same-seed T1 and P-count runs start from bit-identical shared weights and share the transfer seed (not documented in COHORT_PROTOCOL.md). Pairing-respecting permutation: p = 0.0625. | Design flaw recorded; P1 still fails. The same coupling holds for Investigation IV's same-index P-count/T1 models (P2/T2). |
| PT (φ ordering), ME (φ), CX2, ST5 | The *sign* of s = φ_rel(5) − φ_rel(4) is parameterisation-dependent for P-count (forcing s = 0 costs ≤ 0.2 % loss in 4/6; 3/6 flip under a log-pool) and stage-confounded; the 13/13 count included failed runs. The *ordering* is robust: every T1-lineage model has higher s than every P-count model, at full budget and at 500 steps (exact p = 0.0022 each); for T1, s > 0 is identified. | Sign claim withdrawn; ordering stated as the strongest post hoc curriculum-associated finding and the leading candidate for the next frozen prediction. |
| CX7 | P1b's one-factor test gives p < 0.05 in 200/200 simulated worlds without curriculum structure. | P1b is invalid as a test; its "T1-specific shared component" reading is dropped. |

## Accepted — wording and smaller corrections

- FO4: answer 12 restated — exact posterior matching plus pointwise interchange is *sufficient*; with T4's posterior floor (p* ≥ 0.015)
  average KL does bound average null responses; only pointwise bounds fail.
- FO5: the error-increment identity is elementary (by definition of the error); its novelty claim is removed.
- FO (minor): Proposition 1 holds for κ, B within the finite domain's limits and gives (2/3)B; TV row and average-log-odds row corrected;
  CE2 needs β > 0; CE4 needs null pairs ≤ 10 % of the set; Proposition 2's "or a limit" removed (the classes meet at V-independent maps);
  ρ in π-formulas is the likelihood ratio 3.88, not the generator's ρ = 1; split-half ceiling needs a Spearman–Brown correction
  (ME: cT67 0.81 → 0.61, cP61 0.74 → 0.58).
- FO3: computational equivalence is under-specified (alignment class, mediator-level interventions); it is not used for any conclusion.
- CX3: τ ≈ 0.75 is reproduced by exact Bayes with a 10 % probability floor; the "under-confident count evidence" reading is withdrawn.
- CX5, CX6, CX8: coherence rising with quality is reproduced by unrelated approximators, but not as a function of KL alone (a polynomial
  approximator at equal KL is anti-correlated); saturation alone cannot produce the coherence; cross-arm coherence (0.174) ≈ within-P
  (0.186) argues against curriculum-specific orthogonal errors in this cohort.
- CX4: two exact 𝔽_mix models with null responses correlated at −0.985 — strengthens Counterexample 2's lesson.
- PT (positive result): inside 𝔽_mix, null-invariance holds iff φ_rel is affine on 0–4 and the collusion gate is hard (Proposition 3,
  added to FORMAL_DECISION_EQUIVALENCE.md); all 12 models share increasing φ_rel on 0–4, a strong but leaky gate, and slope ≈ 1.0.
  At matched KL, P-count errors align with the near-Bayes T1 error direction as well as 500-step T1 models do (lineage effect +0.02,
  p = 0.78).
- ST3: bootstrap SE (0.104) < jackknife SE (0.130); answer 11's "a large effect is unlikely" is weakened (jackknife upper ≈ 0.40).
- ST6, ST7: P2(a), P3(a) and P4 are non-rejections at 6 + 6, not confirmations of equality.
- ST8: the success rule filters on an outcome-linked variable (transfer accuracy tracks KL); applied to parents not budget variants.
- ST9–11: `cohort_quality.json` mixes median definitions (b* = 500 either way; its script was inline — now archived as
  `scripts/cohort_quality.py`); `logs/dev/stats_dev6.json` was regenerated after outcomes (development file only); the frozen power
  statement was not computed (PT simulation: power ≈ 0.7 at a true Δ of 0.15).
- ME (minor): Cm touches the same tokens as N in only 36 % of pairs; T1 abstract interventions omit the readout copy; 𝔽_vote is a weak
  comparator; 15 % of R pairs have two count-5 groups; "layers 0–1 frozen" is not the reason patch-all reproduces the counterfactual.

## Rejected or partly rejected

None rejected. PT's suggestion that "did not replicate" is too strong is accepted as wording: the cohort excludes the Investigation IV
effect size (0.45) and neither confirms nor excludes a smaller effect of ≈ 0.15.

## Unresolved disagreement

PT reads the P-count/T1 alignment at matched KL as a positive common result ("errors are inherited and become shared as training
converges"); CX5 shows that unrelated approximators can also produce quality-dependent coherence. Both are recorded; neither is
promoted to a conclusion.
