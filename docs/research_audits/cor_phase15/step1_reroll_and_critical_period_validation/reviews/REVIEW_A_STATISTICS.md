# Review A: statistics and mathematics (agent review, not human peer review)

Package: XIII-E (`step1_reroll_and_critical_period_validation/`), building on XIII-D.
Scope: the re-roll null and crossed design (Workstream A), the R1 statistic, its permutation and bootstrap, independence, intervals, model
comparisons; the C1 ideal references and the C2–C4 geometry and k-sweep; overstatement in REROLL_VS_STRUCTURE.md,
EARLY_TRAINING_CAUSALITY.md and CAUSAL_METRIC_AND_SUBSPACE_AUDIT.md.
Method: I read the scripts and logs, and ran numpy/scipy re-analyses on logged values only. Nothing was trained, no model was loaded, and no
existing file was modified. The checks are reproducible from `logs/score_A.json`, `logs/manifest_A.json`, `logs/c_audit.json`,
`logs/c1_ideal_refs.json` and `logs/score_B.json`.

## What I verified as correct

- **Frozen hashes.** `sha256sum -c` on FREEZE_A2_HASHES passes for every file. FREEZE_C_HASHES passes except `logs/c_audit.json`
  (finding 18: harmless).
- **R1 reproduces exactly.** I enumerated all 8! permutations per parent and convolved the two exact null distributions. One-sided exact
  p for mean ρ_W = 0.65 is **0.0054** (the scorer's Monte Carlo value is 0.0061). For ρ_L the exact p is 0.543.
- **No confound from the shared parent.** In `scripts/ft_factorial.py` the W seed enters only through
  `WA0 = init(WS+7777)["WA"]`, L only through `init(LS+7777)["layers"][2]`, and D only through the data generator. There is no other
  randomness (no dropout), and evaluation is fixed. Within a parent, the two contexts of a seed share two things:
  - the parent weights, which are constant across seeds and so cannot create a correlation across seeds;
  - the identical WA tensor, which is the treatment itself.

  Permuting seed labels within each parent is therefore the right null for "no transferable component". There are no exact ties in
  mix_mass. Near-ties inside the early cluster (0.172–0.203) and the mixture cluster (0.93–0.96) add rank noise under the null, but no bias.
- **Interval formulas are correct.** Clopper–Pearson intervals, Newcombe (method 10, Wilson-based) intervals and the Monte Carlo
  Freeman–Halton homogeneity test are all implemented correctly.
- **Proposition S holds to 4.9e-15.** The C1 ideal values follow correctly from the exact posterior.
- **R5 and R6 are scored as their frozen code specifies.** The C-C1 record in today's `c_audit.json` is identical to the one at the freeze
  commit `87177ad`.

## Findings

### Workstream A: re-roll versus structure

**1. [major] The design separates a purely interactive seed effect from one with a within-parent main effect. It does not separate
"re-roll" from "structured selection".**
- Training is deterministic, so a fresh W necessarily "matters" beyond being a fresh draw.
- Rejecting R1-W shows only one thing: the outcome as a function of (W, L, D) has a component in W that persists across two (L, D)
  contexts of the same parent. That is a W main effect.
- Such an effect is fully compatible with the ordinary meaning of re-roll: a random draw whose value matters, with no identifiable
  property. The report itself says no W property was found (post hoc features; R2 not frozen).
- "Identifiable structure in the H1/H3 sense" (REROLL_VS_STRUCTURE §A8) therefore overstates the result.
- *Correction:* restate H0 as "the effect of the re-drawn tensor is purely interactive with the fixed components (no cross-context main
  effect)". Report the R1-W result as "a within-parent W-seed main effect exists". Reserve "structure" for a test with a pre-specified
  property, i.e. a frozen R2.

**2. [major] The W-versus-L asymmetry behind "Outcome C: mixed" is never tested.**
- The contrast rests on one significant and one non-significant result.
- Stratified bootstrap of ρ_W − ρ_L (20 000 draws): 95% CI **[−0.18, 1.38]**, P(diff ≤ 0) = 0.06.
- *Correction:* say "W shows a transferable component; L shows none detectable; the data do not establish that W and L differ."
  Drop or qualify "L behaves like a re-roll" as a contrast with W.

**3. [major] The R1-L bound ("rules out a latent correlation ≥ 0.6") depends on an unchecked assumption.**
- The frozen power simulation (`logs/A_power.json`) assumes both parents are equally informative.
- In uP161 the L outcomes sit at the ceiling: mix 0.90–0.97, lik ≤ 0.03 except one run. The per-parent bootstrap CI for ρ_L(uP161) is
  [−0.92, 0.85]. The uP161 term mostly adds noise and dilutes the mean.
- I re-ran the frozen decision rule (300 reps, 1000 permutations and 1000 bootstraps each):

  | true transfer c in uP160 | true transfer c in uP161 | P(NO-TRANSFER) |
  |---|---|---|
  | 0 | 0 | 0.72 |
  | 0.6 | 0.6 | 0.04 (reproduces the frozen sim) |
  | 0.6 | 0 | **0.40** |
  | 0.8 | 0 | **0.20** |

- So a strong L transfer present only where it is observable (uP160) would return NO-TRANSFER 20–40% of the time.
- The script behind `A_power.json` is not in the package, so the frozen numbers cannot be regenerated.
- *Correction:*
  - state the bound as conditional on both parents being informative;
  - report the per-parent intervals;
  - weaken "R1-L rules out ρ ≳ 0.6" to "R1-L found no transfer; a moderate L effect confined to uP160 is not excluded";
  - add the power script.

**4. [major] R1-W is a pooled result. Neither parent is significant alone, and uP160's contribution does not survive coarser outcome
measures.**
- Exact per-parent one-sided p: uP160 ρ = 0.55, **p = 0.086**; uP161 ρ = 0.76, **p = 0.018**.
- uP160 shows no transfer on the other pre-specified outcome axis: Spearman of lik_RO1 across contexts = **0.00**.
- At the class level uP160 shows no concordance either:
  - mix > 0.5 binarised, Fisher one-sided p = 0.50;
  - same class 4/8 against 3.0 expected.
- REROLL_VS_STRUCTURE says the uP160 signal is "whether the run lands early or mixture". That is not supported beyond two seeds (601 and
  606), which are mixture in both contexts.
- uP161 depends on seeds 604 and 608, the only values below 0.9:
  - with both dropped, ρ = 0.49;
  - the E161:W608 run is the one incompetent run (KL 0.061 > 0.05). Workstream B's rule would have excluded it.
  - Excluding that seed's pair: uP161 ρ = 0.64, p = 0.069; combined exact p = **0.014**.
- *Correction:*
  - report per-parent p-values and the competence-excluded p next to the pooled p;
  - delete or qualify the "early vs mixture" reading for uP160;
  - say the result holds as a pooled test (p ≈ 0.005–0.014), carried mainly by degree-of-hybridness in uP161.

**5. [major] The uP161 part of the W transfer may measure convergence speed at step 1211, not final organisation.**
- The uP161 spread being correlated (0.80–0.96) is about the size of the 1211→4000 drift in mix (up to +0.03 in clear cases; +0.39 and
  +0.35 for the borderline runs, `score_A.json` stability).
- Only W601 was continued. Whether 604 and 608 remain the most hybrid seeds at step 4000 is unknown.
- REROLL_VS_STRUCTURE notes drift for borderline values, but not that this drift touches the seeds that drive the statistic.
- *Correction:* continue W604, W608 and two high-mix seeds in both uP161 contexts to step 4000 and recompute ρ. Until then, describe the
  uP161 component as "transfer of 1211-step hybridness".

**6. [minor] The post hoc "does not transfer across parents" claim is a null with almost no power.**
- Cross-parent ρ = 0.02 with n = 8: approximate 95% CI [−0.71, 0.73]. The uP161 per-seed means span only 0.85–0.96.
- The eight feature correlations (4 features × 2 parents) are uncorrected.
- *Correction:* replace with "no evidence of cross-parent transfer; this test cannot exclude it".

**7. [minor] R_route "AUC 0.59, no predictive value" overstates the null.**
- Hanley–McNeil 95% CI ≈ [0.41, 0.76].
- The AUC pools both parents, and all 15 EARLY runs are in uP160, so the pooled AUC mixes in a parent effect.
- *Correction:* report "not distinguishable from 0.5" and give the within-parent (uP160) AUC.

**8. [minor] Several interval procedures ignore that cells share seeds.**
- **Newcombe intervals:** the early- and mixture-anchor cells of a parent use the same 8 seeds, so the comparison is paired. Given the W
  main effect, a paired interval (e.g. Tango or Newcombe-paired) is appropriate. The independent-sample interval is somewhat
  conservative for W.
- **Pooled Clopper–Pearson [0, 0.11] for 0/32:** treats 32 runs as independent, but they are 16 distinct seeds crossed with contexts,
  and seed effects exist. The table is labelled "descriptive" but still prints an inferential interval. Widen it (cluster by seed) or
  drop it.
- **Homogeneity test:** assumes the 32 outcomes are exchangeable across anchors, which shared seeds violate. The claim "almost all comes
  from the parent difference" was stated but never tested. I tested it: within-parent anchor homogeneity gives uP160 W p = 0.15,
  L p = 0.12; uP161 W p = 0.29, L p = 1.0. This is consistent with the claim, at low power. Add it.

**9. [minor] Anchor-selection wording.**
- "Says nothing about instability; it says the anchor was an unusual draw" asserts that one of two readings is correct.
- FREEZE_A2 states that the design cannot separate these readings (H2 is confounded with outcome-selected anchors).
- Also, the anchor's "early" label uses the step-4000 rule (lik ≥ 0.6), while re-draws use the step-1211 rule (lik ≥ 0.4).
- *Correction:* "is consistent with regression to the parent base rate".

**10. [minor] The "95% bootstrap upper bound" is a decision statistic, not a confidence bound.**
- It is a percentile bootstrap of Spearman ρ at n = 8, with duplicated pairs.
- Its meaning comes entirely from the Gaussian-latent simulation: 300 reps, Monte Carlo SE ≈ 1.5–3 percentage points per cell.
- *Correction:* label it accordingly.

### Workstream C: causal metrics and subspaces

**11. [major] The C1 ideal references assume the unit's own response equals exact Bayes (own slope 1). Observed values are compared
without normalising by the unit's own fidelity.**
- Restore slope = slope(L_b − L_a on ΔB) + slope(L_a − L_meanrep(base) on ΔB), i.e. own slope + meanrep term.
  - Ideal: 1 + (−0.538) = 0.462.
  - With an own slope of about 0.85–0.9 (SUFF is 0.82–0.90), the observed 0.24–0.35 implies a meanrep term of about −0.55 to −0.66:
    at or beyond the ideal.
  - The "modest shortfall" may therefore be largely a normalisation artefact.
- SRCF has the same issue (observed 0.59 against ideal 0.72), as do other slope metrics.
- *Correction:* report the unit's own slope (L_b − L_a on ΔB) for the J_DM test pairs, and compare observed/own with ideal. Until then,
  withdraw "the modest shortfall remains" as a finding.

**12. [major] The mean-replacement baseline is pooled over token positions, and the zero-ablation reference is not identified.**
- In `c_audit.py` (and the XIII-D `c_carrier.py`), `mu_site = Xs.mean(0)` averages the 6 q-source tokens and the readout token
  together. Mean-replacing the readout token's projection with that pooled mean is off-distribution by construction. It is an obvious
  candidate for the "unresolved 0.10–0.15 excess damage".
- The ideal zero-ablation KL (0.3471) equals the mean-replacement KL (0.3473) only because the exact posterior is invariant to a shift of
  m that is the same for all three pairs (both set m equal across pairs). A real model's zero point need not decode to an equal-m state,
  so the "zero-ablation excess 0.16–0.22" has no valid ideal reference.
- `SigS` and `SigN` (C2) also pool positions, so between-position mean differences count as "noise".
- *Correction:*
  - recompute damage with per-position means;
  - mark the zero-ablation comparison non-diagnostic;
  - note that signal/noise variances include position effects.

**13. [major] The decoy-trained subspace argument ("m-specific carrier contradicted"; "the model's general cross-group relation
direction") lacks the quantity it depends on.**
- I regenerated the DEC pairs with the audit's own construction and seed (numpy only). On all 300 pairs:
  - m at q is unchanged (0/300);
  - the exact posterior is unchanged (mean |ΔB| = 0.0).
- DAS_DECOY is trained with target sm(L_b), i.e. to reproduce the model's own non-Bayesian response to decoy edits.
- Neither the size of that response (|L_b − L_a| on DEC pairs) nor DAS_DECOY's held-out fit is logged. If the response is small,
  DAS_DECOY is weakly constrained, and its convergence to the high-signal subspace could reflect optimisation bias: the same
  initialisation family and 200 Adam steps.
- *Correction:* before calling the channel "not m-specific" or "general relational", report:
  - the model's own DEC-pair response;
  - DAS_DECOY's fit on held-out DEC pairs;
  - a DAS trained on shuffled-target pairs as a control.

**14. [major/minor] The "Σ-whitened" overlap is mislabelled, and the claim about it is false.**
- `pang(..., M=chol(Σ)ᵀ)` maps A → LᵀA. That gives the Σ-weighted inner product, which up-weights high-variance directions. It is not
  Σ^{-1/2} whitening. The whitening matrix `Wh` is computed but never used.
- The report says these overlaps "are higher for the DAS pairs and leave the ordering unchanged". The data contradict this:

  | unit | pair | Euclidean | Σ-weighted |
  |---|---|---|---|
  | C-C2 | DAS\|DAS_s2 | 0.690 | 0.658 (falls) |
  | C-C2 | DAS\|DAS_DECOY | 0.605 | 0.663 |
  | C-C2 | DAS\|ENC_m | 0.581 | 0.700 |
  | C-C2 | DAS\|CCD_m | 0.582 | 0.717 |
  | C-C1 | DAS\|VMRAND0 | 0.163 | 0.432 |
  | C-C1 | DAS\|DAS_DECOY | 0.325 | 0.358 |

  - In C-C2, DAS|DAS_s2 drops and falls from first to last among these four.
  - In C-C1, DAS|VMRAND0 overtakes DAS|DAS_DECOY.
  - Random baselines rise to 0.22–0.26.
- *Correction:* relabel the metric as "Σ-weighted", correct the sentence, and give the random baseline in that metric.

**15. [minor] C4 sweep wording and uncertainty.**
- "Adding dimensions beyond that gains ≤ 0.03" fails for C-C1, where "that" is one direction: SUFF goes from 0.865 (k = 1) to 0.924
  (k = 6), a gain of +0.059. The k = 1→2 step alone is +0.031.
- No C4 quantity has an interval. Seed stability rests on one pair of seeds.
- SUFF is non-monotone (C-C2: 0.821 at k = 5 against 0.807 at k = 6), so optimisation noise is about 0.01–0.02.
- The mean principal cosine is not comparable across k, because its random baseline changes with k.
- R5's falsification itself is robust: 0.667 against a threshold of 0.85.
- *Correction:*
  - "≤ 0.03 beyond 2 directions in C-C2; ≈ 0.06 beyond 1 direction in C-C1";
  - add bootstrap intervals over test pairs and at least 3 DAS seeds;
  - give the random-subspace baseline for the mean cosine at each k.

**16. [minor] Parts of R6 are true by construction.**
- The Haufe pattern A = Σ W Cov(Ŷ)⁻¹ is aligned with the signal covariance, while the filter W avoids noise. So clause (c) (signal
  variance ratio) and clause (d) (DAS overlap with ENC rather than DEC) are near-algebraic consequences.
- The informative clause is SUFF(DEC) ≈ 0 against SUFF(ENC) ≈ 0.75.
- The ratios quoted in the text are loose. "Signal variance about 1/10 of ENC, noise about 1/40" is actually 1/17 and 1/76 for C-C1, and
  1/7.7 and 1/39 for C-C2.
- *Correction:* give the exact ratios, and say which R6 clauses carry evidential weight.

**17. [minor] Hash note.** `logs/c_audit.json` fails its frozen hash because C-C2 was appended after the freeze. I confirmed that the
frozen C-C1 record is identical once both files are parsed as JSON (`git show 87177ad:logs/c_audit.json`). Add a sentence to FREEZE_C.md or the package README
so that later auditors do not read the mismatch as tampering.

### Workstream B: early-training causality (statistics only)

**18. [minor/major] The dose-response claim "50 steps flips 6 of 6" overstates the mixture base.**
- M-base 50-step runs have lik 0.250, 0.156 and 0.175 against the mixture threshold lik ≤ 0.15. Two of the three are within 0.03 of the
  boundary.
- They are classified at step 1211, where A5 shows borderline runs drifting towards mixture.
- Under the package's own +0.05 lik sensitivity shift, those two would be MIXTURE, i.e. not flips.
- They were not continued.
- *Correction:* "50 steps produces partial (hybrid) displacement on E; on M it is borderline in 2 of 3".

**19. [minor] Sample sizes and what the M-base control rules out.**
- Fisher one-sided p values:
  - 3/3 against 0/3 per base: p = 0.05;
  - 6/6 against 0/6 pooled: p = 0.0011;
  - late 4/4 against 0/4: p = 0.014.
- All runs come from one parent with fixed L and W; only D varies. The replication is over data orders only. The report's Limits section
  says so, but R3 "reproducible" should say "across 3 data orders of one parent".
- The M-base matched noise (Δloss ≈ 0.002–0.014) is close to a no-op. On M, R3's specificity contrast is real mask against almost
  nothing, and stronger noise does flip M (2/2). Specificity is therefore better supported on E than on M.

## Which claims survive

| claim | status |
|---|---|
| R1-W: re-roll null rejected, i.e. a transferable within-parent W-seed component exists | **survives as a pooled test** (exact p 0.005; 0.014 excluding the incompetent run). Not shown in either parent alone; mostly 1211-step hybridness in uP161 (findings 4, 5). "Identifiable structure / structured selection" does not survive (1) |
| R1-L: no transferable L component | **survives as "none detected"**. The quantitative bound c < 0.6 does not survive as stated (3) |
| W differs from L ("Outcome C: mixed" as a W/L contrast) | **not established** (2) |
| Parent difference dominates; uP161 never early | **survives** (the within-parent tests I added are consistent) |
| No early/mixture-anchor asymmetry detected; H2 neither supported nor excluded | **survives**. Intervals should be paired (8) |
| W effect is parent-conditional / does not transfer across parents | **downgrade to "no evidence"** (6) |
| No W-only spectral statistic or R_route predicts the outcome | **survives as "not detected"** (6, 7) |
| A5: step 1211 is reliable for clear cases | **survives**. Does not cover the uP161 transfer seeds (5) |
| A2: matching seeds are not physically aligned | **survives** |
| R3: early 100-step route masks redirect organisation; sham neutral; matched noise does not | **survives** for 100-step windows, across 3 data orders of one parent. Weaker on the M base (19) |
| R4 falsified: late matched blocks give a lasting hybrid; strong noise also redirects | **survives** (n = 2 per late cell) |
| Dose dependence (25 → 50 → 100 steps) | **survives qualitatively**. "6 of 6 at 50 steps" overstated (18) |
| C1: SUFF, reverse, E2-TRUE and DAS own-share are discriminating and near ideal | **survives** |
| C1: restore shortfall; excess meanrep/zero-ablation damage | **not established**: normalisation and position-pooled mean (11, 12) |
| C1: P4b, P6 criterion, SHUFD and JD0 non-diagnostic | **survives** |
| C2 / R6: decoder filters decode m but do not transmit; "decodable ≠ used" retracted | **survives**. Part of R6 is true by construction (16) |
| C3: DAS/ENC/CCD share a signal-aligned core; decoder span near-orthogonal; K/Q split stable within a model | **survives** (Euclidean metric) |
| C3: Σ-whitened overlaps "higher, ordering unchanged" | **false as stated** (14) |
| "m-specific carrier contradicted"; "general relational direction" | **not established** until the decoy-response size and fit are reported (13) |
| R5 falsified; "3-dimensional" withdrawn; 1 (C-C1) and 2 (C-C2) stable directions | **survives qualitatively**. The "≤ 0.03" gain wording is wrong for C-C1, and there is no uncertainty quantification (15) |
