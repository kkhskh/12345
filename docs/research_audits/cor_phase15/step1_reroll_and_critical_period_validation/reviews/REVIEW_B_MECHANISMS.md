# Review B: mechanisms and experimental controls (Investigation XIII-E)

This is an independent agent review, not human peer review. Its aim was to falsify the conclusions of REROLL_VS_STRUCTURE.md (A), EARLY_TRAINING_CAUSALITY.md (B) and CAUSAL_METRIC_AND_SUBSPACE_AUDIT.md (C).

**Method.** Light reading of JSON logs and scripts, plus three small numpy checks:
- per-parent R1 permutation tests;
- a decoy-pair exact-posterior check, run with the package's own `src/genlib.py`, `src/xii_bayes.py` and `src/relpairs.py`;
- reading the training-loss traces in `checkpoints/ft/*/curve.json` and `stop_done.json`.

No model was trained or evaluated, and no existing file was modified.

**Severity scale.**
- **Major:** a stated conclusion is not supported as worded.
- **Moderate:** a conclusion needs qualification, or an analysis is missing.
- **Minor:** wording, provenance or reporting.

---

## Workstream B: early-training causality

### B-1 (Major). The late "hybrid" is measured at a different post-release time, and the "difference of degree" reading is not established

**Evidence.**
- Short windows are classified at step 1211, which is 1111 steps after release. Late windows (546–1146) are classified at step 4000, which is 2854 steps after release (`score_B.py`: `st = 1211 if kind == "short" else 4000`).
- XIII-D measured the same late l2q run (F1f) at both times (`../step1_causal_algorithm_selection_validation/READOUT_INITIALIZATION_AND_LEARNING_CAUSALITY.md`, F1 table):
  - **at step 1211 it was EARLY (lik 0.732, mix 0.289);**
  - at step 4000 it was hybrid (0.460 / 0.914).
- So, 65 steps after the late block ended, the mixture base had been **fully flipped**. It then partly reverted towards mixture.
- The XIII-E late runs have step1211/1803/2686 checkpoints (e.g. `checkpoints/ft/uP160_fx_L23_W23_D801_l2qT0_br546_w546-1146_B_ll2q/`), but only step 4000 was measured (`logs/unit_metrics_Bfinal.jsonl`).

**Consequence.** The data fit "late block → full flip → partial reversion after release" at least as well as "late block → weaker effect". The early/late difference then lies in **persistence after release (reversibility)**, not in the degree of the immediate effect.
- "A full flip early, a hybrid late", "greater sensitivity early" and "a stable hybrid" are not shown.
- "Stable" is untested. At step 4000 the cosine schedule has brought the learning rate to zero, so the state is frozen by the schedule, not shown to be an attractor.

**Replication count.** The M, D = 23 late-l2q run reproduces XIII-D F1f (same configuration, deterministic harness; the small metric differences, 0.431 vs 0.460, reflect the measurement stream). The claim "reproduces F1f, now with 2 data orders" therefore rests on **one new run** per late route condition (D = 801).

**Correction.**
1. Measure all 8 late runs at steps 1211, 1803, 2686 and 4000.
2. Report trajectories of lik and mix.
3. Compare short and late runs at an equal post-release interval.
4. Restate the result as "late blocks also redirect the organisation, and the redirection is partly reversed after release; early redirection is not reversed (4 of 4 continuations)".

### B-2 (Major). On the early base, the early and late arms use different interventions, and the late effect on mix is as large as the early one

**Evidence.**
- The early arm is full ro0 for 100 steps. The late arm is a partial block, ro0p with k = 2, for 600 steps.
- Cumulative learning-rate mass of the windows, computed from the schedule in `ft_factorial_b.py` (warm-up 200 steps, cosine to 4000):
  - 0–100: ≈ 0.025;
  - 546–1146: ≈ 0.55 (≈ 22×).
- Continuous outcomes (`score_B.json` rows, change from the D-matched baseline):

| arm | Δmix | Δlik |
|---|---|---|
| E early 100-step ro0 | +0.76 to +0.79 | −0.77 |
| E late ro0p k = 2 | **+0.74 to +0.78** | −0.47 to −0.48 |

**Consequence.**
- A *weaker* late block moved mix_mass as far as the full early block did. Only lik differs.
- The time at which the window falls is confounded with the kind of intervention and with the window's length and learning-rate mass.
- "Greater sensitivity early … holds on both bases" is therefore not supported for E. On M (same l2q, but 100 vs 600 steps), the comparison is still confounded with window length and with B-1.

**Correction.**
- Run equal-length late windows (100 steps) with the same intervention as the early arm.
- Where full late ro0 cannot be severity-matched, say that E's early/late comparison is not identified.

### B-3 (Major). The noise control is neither location-matched nor direction-neutral, so R3's "specificity" is weaker than stated

**Evidence.**
1. **Location.** Noise is added to all 43 tokens at the layer-2 input (`ft_factorial_b.py`, `logits_X`, `iname == "noise"`). ro0 acts on layer-0 readout→source attention. The two perturbations act at different sites.
2. **Direction.** The same σ = 2.32 for 100 steps:
   - turns the mixture base EARLY (2 of 2 competent runs);
   - leaves the early base EARLY (3 of 3).

   The disturbance is therefore biased towards the early organisation. On E, a null result from an early-favouring disturbance is expected whatever the mechanism, so "0/3 flips" on E cannot separate route-specific from general effects.
3. **Mixture base.** The "matched" σ = 0.25 is a floor clamp, not a match:
   - `DIST_M.target` = 0.0054 lies below the smallest grid value;
   - `match_sigma` returns `SIG[0]` (`b_calib.py`).

   That σ leaves every continuous measure unchanged at step 1211 (e.g. l2_mlp 0.143–0.150 against baselines 0.141–0.155). On M, R3 compares l2q with an essentially null perturbation. The only informative M disturbance, strong noise, flips M in the **same** direction as l2q.

**What survives.** Directional specificity on E: ro0 drives E towards mixture, which no noise condition does. On M, the defensible claim is "l2q redirects the organisation without measurable loss disruption". It is not "route-specific as opposed to general disturbance".

**Correction.**
- Rewrite R3's interpretation in those terms.
- Add a site-matched disturbance: noise on the layer-0 attention output at the readout tokens.

### B-4 (Major). The sham is a code-path identity check, not the neutral sham the mandate specified

**Evidence.**
- `mask_for`, `iname == "sham"`, returns a zero bias, and the run is bitwise identical to plain training (`score_B.json` `sham_bitwise: true`; metric rows identical).
- The mandate (`sources/MANDATE_XIIIE.txt`, B1) asks for a sham that "does not specifically block either hypothesized information route" while matching "gradient and activation disruption".
- XIII-D's stated gap was "a mask on an edge in neither route".

**Consequence.**
- A bitwise-identical run cannot be informative about anything except the implementation. It adds no evidence to R3.
- The doc's claim "It rules out any effect of the masking code itself" is true but narrow.
- The control that would actually test route specificity, an equal-size attention block on off-route edges, was not run. For example:
  - an l2q-shifted mask (readout 36+q blocked from the sources of q+1);
  - an equal-count block of source→source edges in layer 0.

**Correction.**
- Relabel the arm as "implementation-equivalence check".
- List the off-route edge sham as missing, and do not count the sham towards R3.

### B-5 (Moderate). Severity was matched at onset only; during the window it diverges

Severity was matched on a fixed batch, on frozen checkpoints, with one fixed noise sample (`b_calib.py`, `PRNGKey(7)`). The single-batch training losses in `curve.json` (`hist`, every 100 steps) show how the disturbances behaved during the windows.

| condition | step | loss | baseline |
|---|---|---|---|
| late M, l2q | 600 | 0.765 / 0.777 | ≈ 0.56–0.60 |
| late M, matched noise σ = 2.06 | 600 | 0.656 / 0.635 | ≈ 0.56–0.60 |
| late E, ro0p | 600 | 0.817 / 0.793 | 0.610 |
| late E, matched noise σ = 4.0 | 600 | 0.848 / 0.839 | 0.610 |
| early M, strong noise (D = 801) | 100 | 1.106 (≈ ln 3) | 0.972 |
| early M, strong noise (D = 801) | 500 | 0.866 | 0.616 |
| early M, strong noise (D = 801) | 800 | 0.693 | 0.577 |

- **Late M:** l2q's disruption during the window is about twice the matched noise's. The model adapts to the noise quickly but not to the mask.
- **Late E:** the two conditions are roughly matched.
- **Strong early noise on M:** loss sits at chance during the window, followed by a learning delay of several hundred steps. No real mask produces anything comparable.

**Consequence.**
- The late-M route-versus-noise contrast is not severity-matched in training.
- R4's "strong noise flips" clause compares a much more severe disturbance.
- "At about 86 times that noise variance … it acts like a mixture-route block" is an interpretation without a measurement. A severe delay of learning is an equally good explanation.

**Correction.**
- Report the disruption across the whole window (mean Δloss over the window, or parameter displacement relative to baseline at the end of the window).
- Rematch on that measure, or state the mismatch.

### B-6 (Moderate). "Noise does nothing" holds only for the two-feature class; noise leaves lasting mechanistic changes

**Evidence** (`unit_metrics_B.jsonl`, `unit_metrics_Bfinal.jsonl`):

| condition | measure | value | baseline |
|---|---|---|---|
| E matched noise, step 1211 | l2_mlp | 0.045–0.093 | 0.378 / 0.494 / 0.379 |
| E late noise, step 4000 | l2_mlp | −0.15 / −0.18 | 0.42 / 0.40 |
| M late noise, step 4000 | l2_attn | 1.03 / 1.04 | 0.80 |

**Why the measures favour the masks.** The class uses exactly the quantities the masks block:
- mix_mass is layer-2 readout attention to the own-q sources, which is the edge set l2q blocks;
- lik_RO1 is read at the readout token after layer 0, which is what ro0 blocks.

So the outcome measure is aligned with the route masks and not with the noise.

**Correction.**
- Qualify "flips 0/3" and "changed it in none" as class-level.
- Add a symmetric outcome distance (for example, distance over the full metric vector, or the functional KL between final models) for every arm.

### B-7 (Moderate). The strong-noise "early" outcome is not the natural early organisation

**Evidence** (at step 1211):

| measure | strong-noise early (M) | l2q-early (M) | natural early (E baselines) |
|---|---|---|---|
| slope_W | 0.60–0.67 | 0.73–0.79 | 0.79–0.84 |
| KL | 0.044–0.051 | 0.027 | 0.024–0.030 |
| l2_mlp | 0.10–0.15 | 0.49–0.53 | 0.38–0.49 |

One strong-noise run was excluded at KL 0.0514. The competence bound (0.05) sits only just above this cluster.

**Consequence.** These runs are degraded states at the edge of competence that pass the class thresholds. Counting them as flips to "early" in R4 over-reads them.

**Correction.** Report them as "degraded, early-like" and add the continuous comparison.

### B-8 (Moderate). The dose–response counts rest on near-threshold hybrids that were not continued, and dose is confounded with warm-up

**Evidence.**
- The M 50-step "flips" have lik 0.156 (D = 801) and 0.175 (D = 802), against the mixture threshold of 0.15: margins of 0.006 and 0.025.
- No threshold-sensitivity analysis was run for B.
- A5 shows that borderline runs drift towards mixture between 1211 and 4000, with lik falling. These two runs could therefore revert to the baseline class.
- None of the 25- or 50-step runs was continued.
- All early windows lie inside the 200-step warm-up:
  - cumulative learning-rate mass for 25 : 50 : 100 steps is ≈ 1 : 4 : 16;
  - ro0's disruption jumps from 0.07 at step 50 to 0.54 at step 100 (`b_calib.json`).

  The "dose" is therefore confounded with whether the window covers the time the early route forms.

**Correction.**
- Add a ±0.05 threshold-sensitivity table for B.
- Continue the 50-step runs to step 4000.
- Describe the dose effect as "window must extend past ≈ step 50–100", not as a graded dose.

### B-9 (Minor). Calibration provenance, a misquoted range, and shared noise

- **Late σ values.** For both late noise conditions, the frozen calibration output (`b_calib.json` `match`, `B_design.json` `calibration`) gives NOISE_LATE_E = NOISE_LATE_M = σ 8.0, which is saturated, with target 1.29. The runs used hand-set σ = 4.0 and 2.06, matched against a different target: l2q on M, 0.46.
  - This is disclosed in FREEZE_B.md and in the `B_design.json` note.
  - However, the E late target is borrowed from the other base, and the stale `match` block should be annotated.
- **Misquoted range.** "At steps 0–100, l2q barely moves the loss (Δ ≈ 0.002–0.014)": on the mixture base the range is 0.0025–0.0095. The value 0.014 is l2q on the *early* base.
- **Shared noise.** Every noise run uses the same realisation (`PRNGKey(424242)` folded with the step index). The D replicates of a noise arm therefore share the disturbance, and only the data order varies.

### B-10 (Minor). Reactivation results exist but are not reported

`logs/route_status_B_react.jsonl` holds the reactivation measurement. EARLY_TRAINING_CAUSALITY.md defers it to MAIN_RESEARCH_SYNTHESIS.md, which is not in this package.

| model | readout-only retraining, mixture route blocked | acc | KL |
|---|---|---|---|
| ro0-flipped (D = 801) | — | 0.35 → 0.55 | 0.79 → 0.47 |
| ro0-flipped (D = 801) | additionally, RO1 information removed | 0.34 → 0.36 | — |
| early baseline | — | ≈ 0.87, unchanged | — |

The retained early-route information is therefore only partly usable by the readout. This qualifies the "information persists in pooled form" paragraph. Report it in the B document.

---

## Workstream A: re-roll versus structure

### A-1 (Moderate). R1-W is carried mainly by uP161's within-mixture variation; uP160 alone is not significant

My recomputation uses the same statistic, a within-parent permutation (20 000 draws) and the `score_A.json` table.

| parent | ρ | one-sided p |
|---|---|---|
| uP160 | 0.55 | **≈ 0.085** |
| uP161 | 0.76 | ≈ 0.018 |

- The uP161 M-context mix_mass spans only 0.897–0.956, and almost every run is mixture.
- uP160 class concordance is 4/8, against 3.0 expected.
- "uP160: whether the run lands early or mixture" is therefore not supported on its own.
- In uP161 the W-norm correlates with the outcome (ρ = −0.69, `A_posthoc_W_features.json`). A trivial W-scale → attention-sharpness explanation of the uP161 component is not excluded.

**What is robust.** The pooled verdict holds. Excluding the incompetent E161 W608 run (KL 0.061), which R1 includes even though `competent` is computed, gives mean ρ = 0.62 and p ≈ 0.010.

**Correction.**
- Report the per-parent p-values.
- Describe the W effect as "mostly graded mix_mass within mixture in uP161; suggestive in uP160".
- Test W-norm as a covariate.

### A-2 (Minor). The randomisation audit hard-codes what it claims to verify

In `a2_init_audit.py`:
- `key_index` (`WA = 5`, `layer2 = 20–25`, `distinct = True`) is written as a literal taken from reading `nets.py`. It is not computed. The `keys` variable is created and never used.
- `tensor_reuse` checks one seed (21) and only the first 50 elements.

The conclusion is still correct by construction: W and L come from independent threefry sub-keys, and D from a separate numpy generator. The KS tests (p 0.27–0.99; means 0.216 = √(3/64)) confirm it.

The audit tests W–L(init) coupling only. Where R1-W found an effect, W–parent interaction, the audit says nothing, and should say so.

### A-3 (Minor). "Regression toward the parent's base rate" in uP161

uP161's natural early rate is 1/4 with n = 4 (Clopper–Pearson [0.006, 0.81]). The re-draws give 0/32, which is *below* even that rate. The data support "early is rare under uP161 re-draws; the E161 anchor was an unusual draw". The natural base rate is too imprecise to call this regression towards it.

### A-4 (Minor). Threshold sensitivity is partial, and the classification rules differ between 1211 and 4000

- `score_A.py` recomputes only the EARLY counts. It never shifts the mixture lik threshold (0.15).
- The early rule is lik ≥ 0.4 at step 1211 but lik ≥ 0.6 at step 4000 (`cls4000`). A5's "9 of 10 keep their class" therefore compares under two different rules.

None of the 10 continuations changes class because of the lik rule (L708 changes on mix). The asymmetry should still be stated, and the same rule applied as a check.

---

## Workstream C: causal-metric and subspace audit

### C-1 (Moderate). "Decoy" pairs carry no exact-posterior change, so DAS_DECOY's sufficiency is not evidence of a shared relational code

**My check.** I rebuilt the C-audit decoy-pair construction with the package's own `src` modules (491 pairs). The decoy edit changes:
- m in **0%** of pairs;
- the exact posterior by **exactly 0** (mean |ΔB| = 0).

**What DAS_DECOY is.** It is trained (`train_das("DEC", …)`, target = the model's own donor output) to reproduce the *model's* residual, non-Bayesian output change on pairs where the ideal output does not move. It nevertheless reaches SUFF 0.78–0.80 on m pairs.

**Competing explanations.**
- The model's relational code is shared across relation types (the doc's reading).
- DAS optimisation recovers the site's dominant output-sensitive subspace whatever the training variable.

The second is not excluded. If it is right, DAS sufficiency at this site is not selective for any variable.

**Correction.**
- Add controls: DAS trained on JD0 pairs (m unchanged), and DAS trained with shuffled donors.
- Change "P3 'm-specific carrier': contradicted" to "not established: DAS sufficiency here is not variable-selective".
- Mark "the model's general cross-group relation direction" as a hypothesis.

### C-2 (Minor). The whitened-overlap claim is false

The doc says the Σ-whitened overlaps "are higher for the DAS pairs and leave the ordering unchanged". The `c_audit.json` C3 values (`overlap_cov`) contradict this:

| unit | pair | Euclidean overlap | whitened overlap |
|---|---|---|---|
| C-C1 | DAS–VMRAND0 | 0.16 | **0.43** |
| C-C1 | DAS–DAS_DECOY | 0.33 | 0.36 |
| C-C2 | DAS–DAS_s2 | 0.69 | 0.66 (falls) |
| C-C2 | DAS–ENC_m | — | 0.70 (above DAS–DAS_s2) |

- In C-C1, the variance-matched random set overtakes the decoy set after whitening.
- The whitened random floor is 0.22–0.28, not ≈ 0.05.

**Correction.** Correct the sentence, and give the whitened random floor.

### C-3 (Minor). Numerical wording in C2 and C4

- **"Adding dimensions beyond that gains ≤ 0.03".** In C-C1, SUFF rises from 0.865 at k = 1 to 0.924 at k = 6, a gain of 0.059.
- **Seed stability.** It rests on 2 seeds with no intervals. Two different metrics carry the same name: overlap in C3 and mean principal cosine in C4.
- **Variance ratios.**
  - "Signal variance ≈ 1/10 of ENC" is actually 1/17 (C-C1) and 1/7.7 (C-C2).
  - "Noise variance ≈ 1/40" is 1/76 and 1/39.

### C-4 (Minor). Observed values have no uncertainty

The observed values compared against the ideal references have no sampling intervals. The doc acknowledges this. Every "near ideal" or "below ideal" verdict should be read as descriptive.

---

## Checks that passed

- **Independence.** W, L and D are independently generated (A-2 caveats aside). The KS results and the 0.216 means are reproduced from the log.
- **Sham.** It is bitwise identical to the baseline at step 1211 on both bases (scorer check; identical metric rows).
- **Reported values.** Every continuous value in the B tables matches the `score_B.json` rows. All 41 runs plus the 4 continuations are present, and the four continuations keep their class with KL ≤ 0.005.
- **Late E matching.** ro0p (Δ 0.39) and σ 4.0 (Δ 0.48) are matched at onset, and are also roughly matched in-training at step 600.
- **A5 claims.** The A5 numbers (9/10 agree; |Δmix| ≤ 0.03 in the 8 clear cases; borderline drift 0.45 → 0.84 and 0.57 → 0.92) match `score_A.json`.
- **C-audit numbers.** R5 (ratio 0.667) and R6 (all clauses) are scored as stated. The K+Q totals are 0.91–0.96. The C4 table matches `c_audit.json`.

## Claims that survive

1. **Re-drawing L behaves like a re-roll.** No cross-context transfer is detected (ρ ≈ −0.02). Latent correlations of about 0.6 or more are excluded.
2. **Re-drawing W is not a pure re-roll** when pooled over the two parents (p ≈ 0.006; ≈ 0.01 without the incompetent run). The effect is parent-conditional, mostly graded within mixture in uP161, and not significant in uP160 alone.
3. **"Matching" seed pairs are not physically aligned** at initialisation.
4. **A 100-step early route block reproducibly redirects the organisation on both bases** (3/3 data orders each), and the redirection persists after release (2/2 continued). The sham shows that the masking code path is not responsible.
5. **On the early base, the effect is directionally specific.** ro0 drives the organisation towards mixture; no noise condition does.
6. **On the mixture base, l2q redirects without measurable loss disruption.** Immediate loss disruption is therefore not the mechanism. A layer-2-input disturbance of much greater severity produces an early-like but degraded state.
7. **Route blocks late in training (546–1146) also change the step-4000 implementation** on both bases, whereas onset-matched noise leaves the lik/mix class unchanged. **Early sensitivity alone is therefore not the whole story; the "uniquely early critical period" interpretation is correctly revised.** Whether early and late differ in degree or in reversibility is open (B-1, B-2).
8. **Reverse-swap necessity (P4a), the true row interchange (E2), the K/Q route (P3e) and low-dimensional sufficiency (P3) survive** their ideal references. The "3-dimensional" claim is correctly withdrawn. The decoder-geometry account (R6) is confirmed on the held-out unit, and "decodable but not used" is correctly withdrawn.
9. **The non-diagnostic verdicts are sound:** P4b, P3a excess, P6 criterion, E2 decoy-row controls and the weak form of null preservation.

These claims need the qualifications above: R3 specificity (B-3, B-4), "difference of degree" (B-1, B-2), "stable hybrid" (B-1), "noise does nothing" (B-6), the dose count (B-8), "m-specific carrier contradicted" (C-1) and the whitened-overlap ordering (C-2).
