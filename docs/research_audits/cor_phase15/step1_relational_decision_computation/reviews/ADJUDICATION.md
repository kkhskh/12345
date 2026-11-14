# Adjudication of independent reviews (Investigation XIII)

Each finding is listed with its disposition.
- **Accepted** means the finding was applied and the corrected text is in the documents.
- **Accepted (disclosure)** means it is recorded and cannot be undone without a new experiment.
- **Partly accepted** means only part of the finding was applied, for the reason given.

## Review 1: mathematics (`REVIEW_1_MATH.md`)

| finding | disposition |
|---|---|
| The counterexample x₂ also had B = C at one question, which inflated the shift | **Accepted.** B₂ changed to differ from both A and C. New exact posterior (0.059, 0.926, 0.015). Document and script updated |
| Hard-rule accuracy counted NaN rows as answers | **Accepted.** Accuracy now reported on defined rows, with the undefined fraction. 0.397 became 0.612 on 22.6% defined rows (C = 1, β = 0.6), and 0.361 became 0.428 on 30% (C = 2, β = 0.6) |
| "Any per-group rule" / "joint statistic required" over-claim | **Accepted.** Restated for *fixed low-dimensional* per-group statistics. Added: a rich per-group encoding can represent anything; the six-question kernel has rank 729 (computed) |
| "Needed exactly on S = 3" too broad | **Accepted.** Restated as S = 3 and not all agreeing at q (5.5%). On the all-agree third, the product rule is exact |
| C1 "shown by exact enumeration" overstated | **Accepted.** Closed forms were checked by Monte-Carlo and by the reviewer's re-implementation; enumeration verifies the exact posterior |
| Rank computed at β = 1 only; C ≥ 2 → C = 2, G = 3; "most of the input space" | **Accepted.** All three corrected |
| The exhaustive test shares the across-question independence assumption and uses q = 0 only | **Accepted (disclosure).** Closed by the reviewer's 10⁶-sample chi-square and q = 1 checks; noted in the document |

## Review 2: experimental integrity (`REVIEW_2_INTEGRITY.md`)

| finding | disposition |
|---|---|
| `lane.sh` logs rc=0 regardless of job status | **Accepted.** Rule R1: crash scan of every run output before scoring. Result: none. Script not edited (it was hashed and running) |
| Head-definition mismatch between scorer and scripts | **Accepted.** Rule R2: reported per parent. One mismatch (uP164, borderline) found and reported with a sensitivity analysis. Review 3 noted that R2 compares only empty vs non-empty lists; head sets differ (disclosed) |
| Merge can read partial files | **Accepted.** Rule R3: every merged file has 6 units and no None class (checked) |
| Verdict rules hide failed arms and call arms with missing nulls diagnostic | **Accepted.** Rule R4: failed arms listed; strict diagnostic flag. All primary nulls were computed in confirmation |
| vP150 post-freeze dev result bears on P5; committed `score_xiii_pre.json` was overwritten | **Accepted (disclosure).** Committed file restored; post-freeze rescoring kept separately; disclosed before scoring. P5 was then falsified as anticipated |
| das_decoy training streams reused; one decoy head; DiD CI holds CTRL fixed; Stage-0 empty eval file; fixes 4–5 missing from log | **Accepted (disclosure).** Two-sample bootstrap added (nearly identical); log entries added retroactively, marked as such |
| Preflight is a plumbing check; ALTMECH values hand-set | **Accepted (disclosure).** Stated in FROZEN_PROTOCOL §5b |

R1–R4 were logged at 18:56Z. At that time the only cohort output the author had opened was the exploratory uP160 transition log.

## Review 3: mechanism and causality (`REVIEW_3_MECH.md`)

| must-fix | disposition |
|---|---|
| 1. "Attention ≈ Bayes responsibilities" is partly correlational; report mass_resp | **Accepted.** Reworded everywhere to "rows causally used as mixture weights; match to Bayes observational, dominant head, flattened". mass_resp reported (0.868 / 0.927) |
| 2. P1c is observational, one configuration | **Accepted.** Reworded. Added the reviewer's suggested interventional check (E vs A1 prediction inside the 400-episode TRUE factorial) as **exploratory**: favours E in all 4 mixture-like parents |
| 3. Remove "DAS implements whichever variable"; report overlap and source-free null | **Accepted.** Claim withdrawn; overlap (4–5/6) and source-free patch reported |
| 4. P3 is a "shared 6-dim region", with overshoot | **Accepted** |
| 5. P2 decoy is the lowest-mass head; head sets differ between scripts | **Accepted (disclosure)** |
| 6. Delete the "7–9% unexplained" bound | **Accepted** |
| 7. Narrow "no reconstruction" | **Accepted.** Attention mass after ablation and the behavioural residual were not computed (stated) |
| 8. P5 = sufficiency of frozen features; readout pattern graded and post hoc; J_H claim 2/5 | **Accepted** |
| 9. Qualify "confirmed"; no n = 2 t-intervals as CIs | **Accepted.** The `stats_xiii` t-intervals at n = 2 are labelled ranges, not CIs |
| 10. Drop "internal predictor of the transition" | **Accepted.** KL halfway point at or before ALIGN in all 7 models stated |
| Missing controls (size-matched decoy, balanced-pair DAS, resample or partial ablations, a P5 necessity test, more seeds) | **Not run** (compute and time); listed in OPEN_COR_FOUNDATIONAL_MAP |
