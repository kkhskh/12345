# Frozen predictions — Investigation V

Frozen after development analyses on Investigation IV's models (P0–P4, T0, T1, T2, T10, T11) and on development budget
variants of T2, T10, T11 (`logs/dev/`), and **before any output of the independent cohort has been read** (cohort training
started at commit 981af04 under COHORT_PROTOCOL.md; only the queue log has been looked at). Text above "OUTCOMES" is not
edited after this commit. Code: `src/cri.py`, `scripts/cri_run.py`, `scripts/cri_stats.py`, `scripts/null_structure.py`,
`scripts/transfer_budget.py`, `scripts/cohort_train.py` (hashes in `logs/freeze_hashes.txt`).

## Units, data, cutoff

- **Statistical unit: the trained model** (independent training run). Cohort: cT60–cT67 (T1) and cP60–cP67 (P-count), successful
  runs only (COHORT_PROTOCOL.md success rule), all runs also reported.
- **Quality-manipulation variants (P2b, P6):** for every successful cohort T1 parent, the identical transfer with 250 and 500
  steps (`transfer_budget.py`, seed = parent seed): `bcT{s}_b250`, `bcT{s}_b500`. These are within-parent variants, never
  pooled with their own parent as independent units; each budget set is analysed as its own arm of independent parents.
  A P-count long-transfer arm was not affordable (≈ 2.7 h); its absence is recorded.
- **Data.** Evaluation stream 13_100_000 (2000 episodes, one scored decision per episode), encoder fit stream 13_110_000,
  surface: first 300 episodes × 36 same-world count vectors × 4 realisations. All families are generator-defined
  (`cri_run.py` docstring); valid sets depend only on the generator.
- **Cutoff.** The confirmatory analysis is run once all 16 cohort runs have finished; if compute ends first, on the largest seed
  prefix 60…60+n−1 completed in both arms, n ≥ 6. The stopping decision is made without reading any cohort output.
- **Inference.** Exact permutation of arm labels over models (C(n_T + n_P, n_T) relabellings; one-sided where a direction is stated
  below, else two-sided) and a model bootstrap (models resampled within arm, pairs of distinct draws, 4000 replicates, 95 %
  percentile CI). Episode-level intervals are never used for curriculum claims. With 8 + 8 models the permutation resolution is
  1/12870; the development effect (Δ ≈ 0.38–0.49, pair SD ≈ 0.15) would be detected with high probability; effects below ≈ 0.15
  may not be.
- **Multiplicity.** One primary test (P1). P2–P5 are pre-specified secondary tests, each reported with its own p; no correction is
  applied and none of them alone is treated as decisive.

## Hypotheses → predictions

| | P1 replication | P2 after quality control | P3 response operator | P4 abstract vs input | P5 pathways / nuisance |
|---|---|---|---|---|---|
| H1 curriculum-specific downstream computation | Δ > 0 | Δ persists in quality-matched comparison | arm difference in operator family fit; R-error coherence differs | agree | — |
| H2 shared computation, curriculum-dependent approximation error | Δ > 0 possible | Δ vanishes when quality is matched | operator difference explained by quality; common structured bias | agree | N and R share routes |
| H3 parent/seed dependence | Δ ≈ 0 | — | — | — | — |
| H4 representation / intervention artefact | — | — | — | **disagree** | — |
| H5 common operation + different nuisance pathways | Δ > 0 possible | — | common structured bias | agree | N routes differ from R routes, or matched-content control explains N differently by arm |

## Predictions and criteria

**P1 (primary) — replication.** Family N (input-level Bayesian-null minimal edits), centred-logit responses, Pearson pair similarity;
Δ = coh_T − coh_P. *Replicates* iff one-sided permutation p < 0.05 **and** bootstrap 95 % CI lower bound > 0.
Robustness (reported, not decisive): margins, probabilities, per-model z-scores, Spearman, cosine, argmax agreement; families N2,
C1–C3, Cm.

**P2 — approximation quality.**
(a) Report KL(Bayes ‖ model) per model. If the arms' KL ranges overlap, report the KL-adjusted Δ (pair regression with sum and
difference of KL) with permutation p; if they do not overlap, the adjusted Δ is declared **not identifiable** from observational
variation.
(b) *Quality-matched contrast.* b* ∈ {250, 500, 2000} is the T1 budget whose median KL is closest in log scale to the P-count
cohort median KL (chosen from KL alone). Δ_q = coh_T(b*) − coh_P. *Curriculum effect persists* iff one-sided permutation p < 0.05
and bootstrap CI lower bound > 0; *explained by quality* iff the CI includes 0 or Δ_q ≤ 0.

**P3 — Bayesian-relevant responses and the response operator.**
(a) Coherence of R-family response errors (response minus per-model-scaled exact Bayes response): Δ_R with the P1 tests.
(b) Operator-family fit on the input-level V-surface: mix_rel = R²_within(Bayes-mixture family)/split-half ceiling. Arm difference
(two-sided permutation) at budget 2000 and at b*. H1 predicts a difference that survives at b*.
(c) *Common structured bias.* Coefficient vectors of the N-response regression on generator features (`null_structure.py`): the
mean between-arm coefficient-vector correlation ≥ 0.9 is the frozen criterion for "a structured null-response component common to
both curricula" (development value 0.97).

**P4 — input-level vs abstract.** Δ on family Nabs (covariate-corrected encoders at the Investigation IV sites). *Agreement* iff the
P1 and Nabs bootstrap CIs both exclude 0 with the same sign, or both include 0. Disagreement supports H4.

**P5 — downstream pathways and nuisance.**
(a) Per model, excess = distance between the N and R sufficiency profiles over position sets {Rq, Qs, Oth} minus a split-half
null. One-sided Wilcoxon over all cohort models: p < 0.05 → N and R travel by different routes (H5 route evidence).
(b) Matched-content control: f_Cm = ⟨Δ_Cm, Δ_N⟩ / ‖Δ_N‖² per model (share of the N response reproduced by touching the same
report tokens without changing V). Arm difference, two-sided permutation, reported.

**P6 — training stage.** Coherence of T1 null responses at budgets 250, 500, 2000 (pre-specified schedule; descriptive).
No P-count stage schedule is available.

**Secondary — one common factor (P1b).** EXCESS_TT = mean over T–T pairs of (observed − one-factor-predicted) correlation, factor
estimated without the pair (`cri_stats.py`). Positive with permutation p < 0.05 → a T1-specific shared component beyond the factor
common to all models.

## Pre-registered reading

- **Outcome C (non-replication):** P1 fails.
- If P1 replicates: **Outcome B** (approximation explains) if P2(b) says *explained by quality*; otherwise the curriculum
  difference survives quality matching. **Outcome A** additionally requires a causally identified operation (P5 positive with a
  specific pathway), else it is reported as "replicated, quality-robust, mechanism not identified".
- **Outcome E** component: P3(c) passes (common structured bias) — reported alongside whatever P1/P2 give.
- **H4** if P4 disagrees. **Outcome D** for the curriculum-vs-quality question if P2(a) is not identifiable and P2(b) is
  inconclusive (CI includes 0 with point estimate > 0.1).

---
# OUTCOMES

Confirmatory analysis started after all 16 cohort runs finished (COMPUTE_LOG: "cohort complete; confirmatory analysis begins").
Logs: `logs/conf/cohort_quality.json`, `primary.json`, `allruns.json`, `budget_b500.json`, `budget_b250.json`, `null_primary.json`,
`p5b.json`. Successful runs (transfer accuracy ≥ 0.6595): T1 cT61, cT62, cT63, cT64, cT65, cT67; P-count cP61, cP62, cP63, cP64,
cP66, cP67 (6 + 6; failed: cT60 0.627, cT66 0.622, cP60 0.652, cP65 0.639). Six successes per arm → no replacement seeds.
b* = 500 (median KL of T1 budget variants of successful parents: 250 → 0.21, 500 → 0.064, 2000 → 0.013; P-count median 0.099).

| prediction | criterion | result | verdict |
|---|---|---|---|
| **P1 (primary)** replication | one-sided perm p < 0.05 **and** bootstrap CI lower > 0 | Δ = 0.149 (coh_T 0.335, coh_P 0.186, cross 0.174); perm p = 0.039 (924 relabellings); CI [−0.047, 0.341] | **FAILED** (CI includes 0) |
| P1 all 16 runs (reported) | — | Δ = 0.094, p = 0.130, CI [−0.066, 0.301] | — |
| P2(a) KL overlap / adjusted Δ | ranges overlap → report adjusted | T1 KL 0.008–0.121, P-count 0.068–0.188 (overlap: cT67); adjusted Δ = −0.004, p = 0.53, CI [−0.44, 0.35] | no curriculum effect after adjustment |
| **P2(b)** quality-matched (b* = 500) | persists iff p < 0.05 and CI lower > 0 | Δ_q = −0.100 (coh_T(500) 0.086 vs coh_P 0.186), p = 0.95, CI [−0.28, 0.09] | **explained by quality** |
| P3(a) R-error coherence | P1 tests | Δ_R = −0.004, p = 0.53, CI [−0.21, 0.18] | no curriculum difference |
| P3(b) operator family | arm difference in mix_rel at 2000 and at b* | 2000: +0.300, p = 0.004; b*: +0.367, p = 0.056 (b* mean inflated by a degenerate ceiling, bcT67_b500 ratio 2.02) | difference at 2000; **not significant at b*** |
| P3(c) common structured bias | mean T–P coefficient-vector correlation ≥ 0.9 | 0.872 (T–T 0.931, P–P 0.927) | **FAILED** (narrowly) |
| P4 input vs abstract | both CIs exclude 0 with same sign, or both include 0 | input CI [−0.047, 0.341]; Nabs Δ 0.230, CI [−0.21, 0.68] | **agreement** (both include 0) |
| P5(a) N vs R routes | one-sided Wilcoxon p < 0.05 | excess per model 0.17, 1.01, 0.05, 0.72, 0.53, 0.52 (T1), 0.12, 0.20, 0.04, −0.03, 0.04, 0.08 (P-count); p = 0.0005 | **PASSED** (routes differ, driven by T1 models) |
| P5(b) matched content | arm difference, two-sided | f_Cm T1 0.03–0.13, P-count 0.02–0.11; diff 0.017, p = 0.44 | no difference |
| P6 training stage (descriptive) | — | coh_T: 250 → 0.083, 500 → 0.086, 2000 → 0.335 | coherence appears only at the full budget |
| P1b one factor (secondary) | EXCESS_TT > 0, p < 0.05 | 2000: +0.135, p = 0.003; at b*: +0.038, p = 0.74 | T1-specific shared component at full budget only |

**Pre-registered reading: Outcome C (the primary replication criterion failed).** P2(b) independently says the T1 coherence is
explained by approximation quality. Outcome E's component (P3c) failed narrowly. H4 is not supported (P4 agreement).
P5(a) supports a route difference between null and relevant responses in T1 models.

Notes at evaluation:
- P3(b)'s statistic divides by a split-half ceiling; for one quality-matched model (bcT67_b500) and several 250-step models the
  ceiling is near zero and the ratio is meaningless (budget_b250 values of order 10⁴). This weakness of the frozen statistic was not
  anticipated.
- `scripts/p5b_matched_content.py` implements the frozen P5(b) estimand; it was written after the freeze because `cri_stats.py`
  did not compute it. `scripts/route_selectivity.py` is exploratory (post hoc).

Post-review notes (verdicts unchanged; REVIEWS_AND_ADJUDICATION.md):
- [R: ST2] Same-seed runs of the two arms share initial weights and transfer seed; pairing-respecting permutation for P1 p = 0.0625.
- [R: ST1] The pre-registered reading interprets P2(b) only if P1 replicates (it did not); P2(b)'s "explained by quality" is a
  low-power non-rejection and the manipulation changes training stage as well as KL.
- [R: ST4, ME] P5(a) is a pooled test; its route reading is withdrawn (blocking the readout route enlarges null responses).
- [R: CX7] P1b is invalid as a test (p < 0.05 in 200/200 simulated worlds without curriculum structure).
- [R: ST6, ST7] P2(a), P3(a), P4 are non-rejections, not demonstrations of equality.
- [R: ME] P3(b)'s split-half ceiling lacks a Spearman–Brown correction.

