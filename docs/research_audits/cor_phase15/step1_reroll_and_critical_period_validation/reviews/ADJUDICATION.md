# Adjudication of the XIII-E reviews

Three independent agents wrote the reviews: A (statistics), B (mechanisms and controls) and C (integrity). None of them is external human peer review. Each finding below is marked one of three ways:

- **ACCEPTED**: the documents are corrected.
- **PARTLY ACCEPTED**: the wording is corrected and the substance is kept, with the reason given.
- **NOTED**: no change is needed.

**No frozen verdict changes.** R1-W FALSIFIED, R1-L CONFIRMED, R3 CONFIRMED, R4 FALSIFIED, R5 FALSIFIED and R6 CONFIRMED all stand as scored. What changes is how they are interpreted.

**Frozen freeze records are left as they are.** The files `logs/FREEZE_*.md` are listed in the freeze hash lists, so they are not edited. Corrections to them are recorded here and in `logs/COMPUTE_LOG.txt`.

## Post-review exploratory measurement (one, prompted by finding B-1)

Review B noted that the eight late-window runs were classified only at step 4000. The short-window runs were classified at step 1211. Checkpoints for the late runs at steps 1211, 1803 and 2686 existed but had not been measured.

I measured them with the frozen `unit_metrics.py` settings. The command is in COMPUTE_LOG, labelled POST-REVIEW EXPLORATORY, and the results are in `logs/unit_metrics_Blate_traj.jsonl`. No training was done.

| run (late window 546–1146) | step 1211 lik / mix | 1803 | 2686 | 4000 |
|---|---|---|---|---|
| M base, l2q, D 23 | 0.71 / 0.29 (**EARLY**) | 0.52 / 0.85 | 0.44 / 0.90 | 0.43 / 0.91 (hybrid) |
| M base, l2q, D 801 | 0.71 / 0.38 (**EARLY**) | 0.45 / 0.88 | 0.43 / 0.92 | 0.42 / 0.93 (hybrid) |
| E base, ro0p k = 2, D 21 | 0.40 / 0.90 (hybrid) | 0.33 / 0.93 | 0.36 / 0.95 | 0.36 / 0.95 (hybrid) |
| E base, ro0p k = 2, D 801 | 0.40 / 0.82 (hybrid) | 0.34 / 0.88 | 0.36 / 0.90 | 0.34 / 0.90 (hybrid) |
| all 4 late-noise runs | stay in their base class at every step | | | |

**What this shows.**

- **Mixture base.** Measured 65 steps after release, the late block gives a full flip to early, which is the same class change as the 100-step early block. It then partly reverses to a hybrid within about 600 steps. The early 100-step block's flip does not reverse: the B continuations held their class to step 4000.
- **Early base.** The late partial block gives a hybrid that is stable from release onwards and keeps drifting slightly towards mixture.

**Effect on the conclusions.**

- On the mixture base, early and late blocks differ in **persistence and reversibility**, not in the size of the immediate effect.
- The documents said the difference was "one of degree", and that claim is withdrawn.
- R4 FALSIFIED still stands. A late route block still leaves a lasting change, and no early window closes.
- This is exploratory. There is one parent, 2 runs per arm, and D 23 repeats XIII-D F1f.

## Review A (statistics)

| # | Finding | Decision | Action |
|---|---|---|---|
| A1 | Rejecting R1-W shows that a W effect transfers within a parent. It does not show an *identifiable* structure, which a re-roll whose value matters could also produce. | ACCEPTED | Reworded to "a W component that transfers within a parent". "Identifiable structure" is withdrawn. The W property is unidentified, as already stated. |
| A2 | The W-versus-L contrast is untested. The bootstrap CI for ρ_W − ρ_L is [−0.18, 1.38]. | ACCEPTED | Outcome C is described as one significant and one non-significant result, not a demonstrated W > L difference. |
| A3 | The R1-L bound assumes transfer of equal size in both parents. With transfer only in uP160 (c = 0.6 or 0.8), the frozen rule still returns NO-TRANSFER 35% and 27% of the time. The power script was missing. | ACCEPTED | The bound is now conditional on homogeneous transfer. The original pre-freeze simulation ran inline and was not saved. It is reconstructed in `scripts/a_power_sim.py`, which writes `logs/A_power_reproduced.json`; the reconstruction matches the frozen `A_power.json` within Monte Carlo error and adds the heterogeneous scenarios. |
| A4 | R1-W is a pooled result. Per parent, p is 0.086 (uP160) and 0.018 (uP161). uP160 shows no transfer in lik or class. The single incompetent run sits at a leverage point. | ACCEPTED | Reported. The pooled result is the frozen test; its p is 0.010 to 0.014 with the incompetent run excluded. "Pushes early versus mixture" in uP160 is withdrawn. |
| A5 | uP161's contribution may reflect convergence speed at step 1211. | ACCEPTED as an open alternative | Listed as unresolved. Those seeds were not continued. |
| A6, A7 | The cross-parent ρ = 0.02 and R_route AUC 0.59 have wide CIs (±0.7 and [0.41, 0.76]). | ACCEPTED | Reworded from "does not" and "no predictive value" to "not detected; low power". |
| A8 | The intervals ignore that cells share seeds. | ACCEPTED | Noted as a caveat. The intervals are anti-conservative for pooled and difference quantities. |
| A9, A10 | Wording on anchor selection; the "95% bootstrap upper bound" is a decision statistic, not a confidence bound. | ACCEPTED | Reworded. |
| A11 | The C1 ideal references assume an own slope of 1, so the restore shortfall may be a normalisation artefact. | ACCEPTED | The restore shortfall moves from "unresolved finding" to "possibly a normalisation artefact". |
| A12 | The mean-replacement baseline is pooled over positions, and the zero-ablation reference is not identified. | ACCEPTED | The "excess ablation damage" is downgraded to "not interpretable as measured". |
| A13 / B C-1 | Decoy pairs change neither m nor the exact posterior (verified on 300 and 491 pairs). DAS_DECOY's sufficiency may only reflect the site's output-sensitive subspace. | ACCEPTED | "m-specific carrier: contradicted" becomes "**not established**". "The model's general cross-group relation code" is withdrawn. |
| A14 / B C-2 | The "Σ-whitened" overlaps use a Σ-weighted inner product, and the ordering does change. | ACCEPTED | The claim is removed and corrected. The Euclidean shared-core result is unaffected. |
| A15 / B C-3 | C4 wording: C-C1 gains +0.059 from k = 1 to k = 6, and no interval is given. | ACCEPTED | Corrected. |
| A16 | R6 clauses (c) and (d) hold nearly by construction. | ACCEPTED | Noted. The informative clause is (a)/(b): the decoders decode m but transfer nothing. |
| A17 / C3 | The `c_audit.json` hash fails after C-C2 was appended. The C-C1 record is unchanged. | ACCEPTED | Disclosed. |
| A18 / B B-8 | "50 steps flips 6/6" overstates the result. Two runs on the mixture base are within 0.03 of the threshold, and none was continued. | ACCEPTED | Reported as "3/3 on E (hybrid); 3/3 on M, 2 at the boundary". |
| A19 | Sample sizes. | NOTED | Already stated as one parent and 3 data orders. |

## Review B (mechanisms and controls)

| # | Finding | Decision | Action |
|---|---|---|---|
| B-1 | The late and short-window runs were classified at different times after release. | ACCEPTED, and measured | See the measurement above. "Difference of degree" is withdrawn and replaced by a difference in persistence and reversibility on the mixture base. |
| B-2 | On the early base the two arms used different interventions (full ro0 for 100 steps early; ro0p k = 2 for 600 steps late), and the late Δmix is as large as the early one. | ACCEPTED | "Greater early sensitivity on both bases" is withdrawn for E. The E arms are not comparable in strength. |
| B-3 | Noise is injected at the layer-2 input, not where ro0 acts. It favours the early organisation. The mixture-base σ is the bottom of the grid. | ACCEPTED | R3's specificity is narrowed. What survives: on E, ro0 pushes towards mixture, which no noise condition does. On M, the l2q block flips the outcome with almost no loss disruption. The noise arm on M is uninformative. |
| B-4 | The sham is a check that the masked code path is identical, not the off-route-edge sham the mandate's B1 asked for. | ACCEPTED | Relabelled "code-path identity check". **The off-route sham was not run; this is a deviation from the mandate.** It adds nothing to R3 beyond excluding harness artefacts. |
| B-5 | Severity was matched only at onset. | ACCEPTED | Stated. |
| B-6 | Noise leaves lasting changes outside the two-feature class (e.g. the share in E's layer-2 MLP). | ACCEPTED | "Noise changes nothing" becomes "noise does not change the route class". |
| B-7 | The strong-noise "early" runs are degraded. | ACCEPTED | Stated. |
| B-8 | The dose–response rests on near-threshold hybrids, and all the windows fall inside warm-up. | ACCEPTED | See A18. The warm-up confound is listed. |
| B-9 | Calibration provenance: the late noise σ values were set by hand. | ACCEPTED | Disclosed. The values are recorded in the frozen B_design note. |
| B-10 | The reactivation results were not reported in the B document. | PARTLY ACCEPTED | The D = 801 reactivation result is in EARLY_TRAINING_CAUSALITY.md §Interpretation 4; the remaining logged rows are in `logs/route_status_B_react.jsonl`. |
| A-1 | R1-W is carried mainly by uP161. | ACCEPTED | See A4. |
| A-2 | `a2_init_audit.py` hard-codes the sub-key indices it claims to verify. | PARTLY ACCEPTED | The KS and tensor-reuse tests are empirical and stand. The independence of the key layout comes from reading `nets.init`, not from the audit; the document now says so. |
| A-3, A-4 | Wording on regression to the base rate; threshold sensitivity is partial. | ACCEPTED | Reworded and noted. |
| C-4 | The observed C values have no uncertainty. | ACCEPTED | Noted as a limitation. |

## Review C (integrity)

Overall verdict: **PASS**. All the confirmation-before-freeze timing checks pass. All three preflights reproduce. All three scorers reproduce bitwise and refuse all 9 incomplete inputs.

| # | Finding | Action |
|---|---|---|
| 2 | Checkpoints in progress were committed in three freeze commits (a4f66b7, b710b38, 87177ad), not just one. `resume.pkl` remains in the history of all three. | Disclosed in COMPUTE_LOG and in FROZEN_PROTOCOL. The files hold only weights and loss histories, and none was read before scoring. The history is not rewritten. |
| 3 | The `c_audit.json` frozen hash fails because C-C2 was appended. | Disclosed. The C-C1 block is identical when compared parsed. |
| 4 | `ft_factorial.py` was edited temporarily. | Already disclosed. |
| 5 | Count slips: FREEZE_A2 says 2 of 64 runs had finished at the commit; the log shows 4. FREEZE_B says 34 short-window runs; the design has 33. | The errata are recorded here and in COMPUTE_LOG. The frozen files are left as they are. |
| 6 | `score_A.py` checks the continuation rule only weakly. | Noted. Review C re-derived the rule independently and it matches. |
| 7–10 | Informational and positive findings. | Noted. |

## Effect on the main conclusions

- **Workstream A.** "Outcome C (mixed)" stays. Its basis is now stated precisely: one pooled within-parent W transfer, and L transfer not detected. The W − L contrast is untested, and the W effect depends strongly on the parent.
- **Workstream B.**
  - The route-specific redirection by early route blocks survives. On E it is directional; on M it occurs without loss disruption.
  - "No closing critical period" (R4 falsified) survives.
  - Early and late now differ in persistence and reversibility on M. They are not comparable on E.
- **Workstream C.**
  - These survive: the discriminating metrics, R6's informative clause, the Euclidean shared core, the stable K/Q split and R5's withdrawal of "3-d".
  - "m-specific" moves from "contradicted" to "not established".
  - The whitened-overlap claim is removed.
  - The restore shortfall and the ablation excess are downgraded.
