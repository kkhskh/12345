# R_statistical — independent statistical review of Investigation V (mandate §13, "Statistical reviewer")

Scope: package at HEAD 44f17a6 (cohort protocol 981af04, freeze db60086). Recomputations use the archived JSON only:
`reviews/checks/rv_stat_recompute.py` → `logs/review_checks/rv_stat_recompute.out`; `reviews/checks/rv_stat_init_coupling.py`
→ `logs/review_checks/rv_stat_init_coupling.out` (initialisation only, no training). Total CPU < 10 s.
Severity: **H** = changes a headline claim or its grade; **M** = needs correction or qualification in the documents; **L** = minor.

## Verdict table (every OUTCOMES row recomputed from logs/conf/*.json)

| prediction | logged | recomputed | frozen verdict correct? | reviewer note |
|---|---|---|---|---|
| P1 primary | Δ 0.149, p 0.039 (924), CI [−0.047, 0.341] | identical; 90 % CI [−0.021, 0.316]; bootstrap seeds 0–3 give lower bound −0.047…−0.055 | **yes, FAILED** | robust to α-mismatch and bootstrap seed (F3) |
| P1 all 16 | Δ 0.094, p 0.130 (12870) | identical | yes | — |
| P2(a) | adj Δ −0.004, p 0.53, CI [−0.44, 0.35] | identical; overlap region holds 1 T1 and 5 P-count models | mechanically yes | uninformative, not "no effect" (F6) |
| P2(b), b* = 500 | Δ_q −0.100, p 0.95, CI [−0.28, 0.09] | identical; b* = 500 under either median definition | mechanically yes ("explained by quality") | the label is a non-rejection rule; overstated (F1) |
| P3(a) | Δ_R −0.004, p 0.53 | identical | yes (no difference) | absence of evidence graded [C] (F7) |
| P3(b) | 2000 +0.300 p 0.004; b* +0.367 p 0.056 | identical; rank test 2000 p 0.004, b* p 0.13 | yes | not significant at b* also without the degenerate ratio |
| P3(c) | 0.872 < 0.9 | 0.8719 (TT 0.931, PP 0.927) | yes, FAILED | point criterion, no uncertainty |
| P4 | both CIs include 0 → agreement | identical (Nabs CI [−0.21, 0.68]) | yes | agreement of two low-power non-rejections (F7) |
| P5(a) | Wilcoxon p 0.0005 | 0.00049 | yes, PASSED | pooled test, not an arm test (F4) |
| P5(b) | diff 0.017, p 0.44 | identical | yes | — |
| P6 | 0.083 / 0.086 / 0.335 | paired parent bootstrap coh(2000) − coh(500) = 0.248, CI [0.079, 0.423] | descriptive | the one robust stage effect (F1) |
| P1b | EXCESS_TT +0.135 p 0.003; b* +0.038 p 0.74 | identical | yes | secondary, uncorrected |
| Pre-registered reading | Outcome C | Outcome C (P1 fails) | **yes** | P2(b) is only consulted if P1 replicates |

No arithmetic or bookkeeping error was found in any frozen verdict. The problems are in interpretation and grading.

## Findings

**F1 (H) — "Approximation quality explains what remains [C: P2(b)]" and "fully removed by quality matching" are overstated.**
Evidence: (i) The frozen P2(b) rule declares *explained by quality* whenever the CI includes 0 **or** Δ_q ≤ 0: it is a
non-rejection rule, not a test of the quality hypothesis. With model-level SE ≈ 0.10–0.13 (bootstrap / jackknife), power to detect
a true Δ_q equal to the observed P1 effect (0.149) at the frozen level is ≈ 0.2, so the rule would return "explained" most of the
time even if the curriculum effect were real at that size. (ii) The manipulation is the transfer budget, not KL: 500- vs 2000-step
transfers differ in steps, warm-up/cosine schedule (scaled with steps in `train_ext`), accuracy and training stage. Quality is
not actually matched: T1-b500 KL {0.025, 0.054, 0.056, 0.071, 0.098, 0.226} vs P-count {0.068, 0.074, 0.090, 0.109, 0.111, 0.188};
four of six T1-b500 models are better than the P-count median. (iii) At this "matched" point T1 is *less* coherent than P-count
(0.086 vs 0.186; two-sided perm p = 0.10). Coherence is not a monotone function of KL across arms (T1: KL 0.21→0.083,
0.064→0.086, 0.013→0.335; P-count: 0.099→0.186), so the data do not show that coherence is "explained by" quality. What they
do show is a within-parent training-stage effect: paired parent bootstrap coh_T(2000) − coh_T(500) = 0.248, CI [0.079, 0.423].
(iv) Under the pre-registered reading P2(b) decides between Outcomes A and B only if P1 replicates. P1 failed, so P2(b) is not a
confirmatory finding in this package.
Correction: change the grade of item 2 in MAIN_RESULT (and §17.3, SHARED_AND_NONSHARED §2, INDEPENDENT_PARENT_REPLICATION) to
[E]/"frozen 'persists' criterion not met". Say "the T1 coherence depends on transfer stage (P6). The quality-matched contrast
has no positive effect and its point estimate is reversed; quality and stage are not separated." Drop "removes the coherence
entirely" and "fully removed by quality matching".

**F2 (M) — The arms are seed-paired, not independent of each other.** For the same seed s, `train_tasks.init` calls
`nets.init(size, Env("T1",K=3,Q=6), s)`, so cT_s and cP_s start from **bit-identical weights (46/46 shared arrays, seeds 60 and 61
checked; the control seed s vs s+1 shares only the 20 constant arrays)**. They also share the T4 transfer seed: layer-2 re-init is
`N.init(.., seed+7777)` and the data stream is the same. COHORT_PROTOCOL.md and INDEPENDENT_PARENT_REPLICATION.md describe "fresh
random initialisation" but do not mention this coupling. Observed: same-seed cross-arm similarity is 0.210 (n = 5) against
0.169 for different-seed pairs. Within-arm independence holds: seeds 60–67 are distinct and fresh, and nothing is loaded from
older checkpoints. The unrestricted 924-relabelling test treats the 12 models as exchangeable. A test that respects the
pairing (swap labels within the 5 complete seed pairs, 32 relabellings) gives one-sided p = 0.0625 (minimum attainable 0.031).
P1's verdict does not change, but its "p < 0.05" half depends on ignoring the pairing.
Correction: document the pairing as a common-random-numbers design. Report the paired permutation alongside the unrestricted
one. Remove "independent" for cross-arm comparisons.

**F3 (L) — P1 inference details (all verified, no change of verdict).** The permutation test is exact over C(12,6) = 924, one-sided
as frozen, and includes the identity. The two-sided p would be 0.078. Both procedures resample whole models, so dependence among
the 15 + 15 + 36 pair correlations is respected. However: (a) the joint rule (one-sided p < 0.05 AND lower limit of a two-sided 95 % CI
> 0) is effectively a one-sided α = 0.025 test for its second part; the 90 % CI [−0.021, 0.316] also includes 0, so the
failure does not depend on this. (b) With 6 models per arm the percentile bootstrap is narrow. Its SE is 0.104, against a
delete-one-model jackknife SE of 0.130 (normal one-sided p 0.13). The CI upper bound behind "a large effect (≥ 0.45) is
unlikely" (§17.11) is therefore optimistic; jackknife ≈ 0.40, still below 0.45. (c) Leave-one-out: dropping cT67 (the only
T1 parent with P-count-like KL) raises Δ to 0.25; no other model moves Δ beyond 0.10–0.19. (d) The permutation null is "same
distribution"; arms differ grossly in KL, so p = 0.039 is not specific to curriculum-dependent coherence.
Correction: state (a) and (b) in INDEPENDENT_PARENT_REPLICATION "Precision". Use the jackknife bound for the "≥ 0.45 unlikely" sentence.

**F4 (H) — "A curriculum-specific route [C: P5(a)]" is mis-graded.** The frozen P5(a) is a pooled one-sided Wilcoxon test that N and R
profiles differ, run over all 12 models. It contains no arm comparison. Recomputed p = 0.00049 (11/12 positive), so the PASSED
verdict is correct. The statistic is conservative under equal profiles (split-half distances use half samples), so the pass is
credible as "N and R routes differ in the typical model". Two problems follow. (i) P-count alone also gives Wilcoxon p = 0.031
(5/6 positive, excess 0.04–0.20), which contradicts "P-count models use one route for both" (MAIN_RESULT 6, SHARED §3,
DOWNSTREAM §2). The defensible statement is "much smaller separation". (ii) The curriculum association, "4 of 6", and the
26–56 % / 34–65 % figures come from `route_selectivity.py`, which is post hoc. A post hoc exact permutation of the arm
difference in excess gives 0.426, one-sided p = 0.0097. That is suggestive, but it is not frozen and uses a direction chosen after looking.
Correction: grade "N and R routes differ" [C], and grade "curriculum-specific readout route" and every percentage [E].
Replace "one route" with "small but mostly positive separation".

**F5 (H) — Post hoc φ-sign result (MAIN_RESULT 7, APPROXIMATION §4, SHARED §4).** The package's own P2(b) logic would attribute the
coherence to quality/stage, yet the same confound is waved away here. Evidence: in the same T1 parents, s > 0 holds in only 2/8
at 250 steps and 4/8 at 500 steps. Across all 32 cohort-phase models, corr(s, log KL) = −0.62. At the only quality-"matched" point
(b500, successful parents vs successful P-count) the table is [[4,2],[0,6]], Fisher p = 0.061. "Not explained by quality among
full-budget models" rests on one model (cT67) and two P-count models. Among full-budget models curriculum and KL are almost
collinear (overlap: one T1 model), so quality cannot be ruled out there. The headline Fisher test (p = 0.0014) also departs from
the primary unit definition: it includes the four failed runs. And s is one of several fitted contrasts, chosen after the outcomes
were seen (selection is not adjusted). The labelling as post hoc is correct. "Stable curriculum-dependent approximation bias"
is not supported.
Correction: describe it as "a sign pattern of fully transferred T1 models that is quality/stage-dependent within T1". Report the
b250/b500 counts and the matched Fisher p next to the 0.0014. Keep OPEN_QUESTIONS 1, adding a quality-matched arm to the prospective test.

**F6 (M) — P2(a) "no curriculum effect after adjustment".** The frozen non-identifiability clause fires only if the ranges do not
overlap. They overlap through a single T1 model (cT67, KL 0.121), so the rule was applied correctly. However, the pair regression
on sum/|diff| of KL is then identified almost entirely by extrapolation, and the CI [−0.44, 0.35] is wider than the effect being adjusted.
Permuting labels while KL stays attached to models also makes the permutation p hard to read when KL nearly determines
arm. Correction: report P2(a) as "uninformative (one overlapping T1 model)", not as evidence of no effect.

**F7 (M) — Non-rejections graded as confirmations.** P3(a) "Bayesian-relevant response errors are not curriculum-specific [C]"
(CI [−0.21, 0.18]) and P4 "agreement … H4 not supported" (Nabs CI [−0.21, 0.68]) are both inferences from failing to reject at 6 + 6.
P4's frozen "both include 0" clause is met trivially at low power. Correction: write "no difference detected (CI ±0.2)".
Downgrade the grade or state the CI wherever these are cited as positive findings (MAIN_RESULT 5, §17.4–5).

**F8 (M) — Outcome-dependent selection.** (a) The success rule is pre-registered, so there are no forking paths. It conditions on
transfer accuracy, which is almost a function of KL (failed runs KL 0.23–0.25). That truncates exactly the covariate P2
studies; the all-runs analysis (Δ 0.094, p 0.13) should be cited next to every primary number. (b) The rule is applied to parents
but not to budget variants. bcT67_b500 (accuracy 0.651 < 0.6595, split-half ceiling 0.19) enters the b* arm. Dropping it gives
Δ_q −0.074, CI [−0.33, 0.14], so the verdict is unchanged. (c) Budget variants of the failed parents cT60/cT66 were trained and
analysed but excluded, consistent with the frozen text. No other inclusion choice was found that depends on outcomes.

**F9 (L) — b* rule and provenance.** b* = 500 holds under both definitions: |log-ratio| 0.45 vs 0.75/2.05 with successful
parents, and 0.16 vs 0.94/1.75 with all 8. `cohort_quality.json` is internally inconsistent: its T medians use all 8 parents
(0.253/0.085/0.017) while its P median uses successful runs only (0.099). The OUTCOMES text reports successful-only T medians
(0.21/0.064/0.013). The script that wrote `cohort_quality.json` (and applied the success rule) is not in `scripts/`.
Correction: archive it, and make the json match the frozen "successful parents" definition.

**F10 (L) — Freeze integrity: holds.** (a) Hashes in `logs/freeze_hashes.txt` match all eight frozen files at HEAD. (b) The text
above "OUTCOMES" is byte-identical to db60086; the diff only appends. (c) No cohort evaluation output existed at the freeze:
cohort training started at 17:49 and the freeze is at 18:33:43, while the first cohort run ended at 18:47 and the first
`logs/conf/model_*` file is from 19:15. (d) All analysis jsons are dated ≥ 01:00:08, the logged start of the confirmatory
analysis. (e) The cohort protocol (981af04) precedes all training. Limits: "not read" during 19:15–01:00 cannot be verified,
because `logs/conf_watcher.out` printed per-model accuracy/KL/fit numbers throughout. `logs/dev/stats_dev6.json` was regenerated
after the outcomes (01:02). The development primary is unchanged but N2/robustness entries differ and a key disappeared, which
shows the file committed at the freeze was produced by an earlier `cri_stats.py` than the hashed one. Correction: note both in
REPRODUCTION_README.

**F11 (L) — Multiplicity and power.** There is one primary. Roughly 12 secondary/robustness p-values plus two post hoc analyses are
uncorrected, as frozen. P3(b) at 2000 (0.004), P1b (0.003) and P5(a) (0.0005) would survive Bonferroni over 8 secondaries
(0.006); P3(b) at b* and the post hoc tests would not. The frozen power statement ("detected with high probability") was not
backed by a calculation and assumed 8 + 8. At 6 + 6, power with the jackknife SE is ≈ 0.93 for Δ = 0.45, 0.64 for 0.30 and 0.21
for 0.15. The Investigation IV estimate (0.45, at the minimum attainable p of 1/20) was itself selected for significance, so a
smaller replication effect is expected. The "≈ ⅓" comparison should not be read as decay.
Also, the probability-space contrast (−0.06) is one of six frozen robustness variants. Margin, z-score and cosine reproduce 0.149
(p 0.039), Spearman gives 0.12. "Not robust to the output space" is fair, but this one variant should not be singled out as a finding.

## Summary of required corrections to documents

1. P2(b) → not [C]. Re-word as "stage-dependent; quality and stage not separated; matched contrast reversed" (F1).
2. Document the seed pairing across arms and add the paired permutation (F2).
3. Route claim: pooled route difference [C]; curriculum specificity and percentages [E]. Remove "P-count uses one route" (F4).
4. φ-sign: report the within-T1 budget instability and the matched Fisher p. Withdraw "not explained by quality" (F5).
5. Change P2(a), P3(a) and P4 from "no effect/agreement" to "uninformative/no difference detected" (F6, F7).
6. Archive the `cohort_quality.json` producer and fix its median definition. Note the post-freeze dev-stats regeneration (F9, F10).

**Overall:** the pre-registered verdict (Outcome C: primary replication failed) is correct and robust: every frozen number
recomputes, the bootstrap seed does not matter, the 90 % CI includes 0, and the seed-paired test gives p = 0.0625. The secondary
narrative claims more certainty than the design supports. "Explained by quality", "curriculum-specific route [C]" and "stable
approximation bias" should be downgraded. What the data robustly support is that T1 null-response coherence appears between 500
and 2000 transfer steps within the same parents.
