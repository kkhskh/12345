# Downstream causal mechanisms (mandate §9)

Data: frozen path-patching arrays (`cri_run.py`: layer-2 input patched with counterfactual values only at, or everywhere except,
position sets Rq = {readout 36+q}, Qs = {question q's six source tokens}, Oth = the rest), `logs/conf/primary.json` (P5),
`logs/conf/route_selectivity.json` (exploratory, post hoc). Because layers 0–1 are frozen, patching every position reproduces the
natural counterfactual exactly, so these fractions decompose the natural input-level response by where it enters layer 2.

## 1. The comparison that avoids the site confound

Investigation IV's pathway claim compared sites *across* models and was withdrawn. Here the comparison is **within a model**,
between the Bayesian-null (N) and Bayesian-relevant (R) responses, so the site of the count variable is held fixed.

| model | N: share via Rq / Qs / Oth | R: share via Rq / Qs / Oth | N remaining with Rq blocked | R remaining with Rq blocked |
|---|---|---|---|---|
| cT61 | 0.09 / 0.85 / 0.06 | 0.03 / 0.96 / 0.01 | 0.90 | 0.97 |
| cT62 | 0.56 / 0.33 / 0.01 | 0.06 / 0.93 / 0.01 | **0.35** | 0.94 |
| cT63 | −0.04 / 1.02 / 0.00 | 0.00 / 1.00 / 0.00 | 1.03 | 1.00 |
| cT64 | 0.50 / 0.54 / −0.01 | 0.07 / 0.92 / 0.01 | **0.53** | 0.93 |
| cT65 | 0.26 / 0.55 / 0.10 | 0.05 / 0.91 / 0.04 | **0.66** | 0.95 |
| cT67 | 0.40 / 0.04 / 0.55 | 0.13 / 0.02 / 0.85 | **0.59** | 0.87 |
| cP61–cP67 | 0.74–0.93 / 0.04–0.19 / 0.03–0.14 | 0.82–0.93 / 0.04–0.17 / 0.03–0.07 | 0.07–0.26 | 0.10–0.20 |

**P5(a) passed** (one-sided Wilcoxon over the 12 models, p = 0.0005): null and relevant responses travel by different routes, and
the difference is carried by the T1 models.

## 2. The strongest causally supported explanation

In four of six T1 parents (cT62, cT64, cT65, cT67) the Bayes-relevant response enters layer 2 almost entirely at the source tokens
(or, in cT67, at other questions' tokens), while 26–56 % of the Bayes-null response enters at the **readout token itself**.
Holding the readout's layer-2 input at its base value removes 34–65 % of the null response while preserving 87–95 % of the relevant
response. This is a selective modification in the sense of §9.4 step 5: it removes much of the departure from Bayes and keeps most
of the useful Bayesian behaviour, without destroying accuracy (the R response is the Bayes-tracking part).

Reading: in these T1 models a second, readout-local copy of count-related information (Investigation IV found T1 readouts decode
n at 0.56–0.93) feeds the decision in a non-Bayesian way; the source-token route carries the Bayes-consistent computation. In
P-count models both responses use the same readout route; their errors are produced by the same operation that produces their
relevant responses.

Limits:
- Two T1 parents (cT61, cT63) show no route separation, so this is a frequent but not universal T1 property.
- The analysis identifies *where* the null response enters layer 2, not the operation inside layer 2 (attention vs MLP at the
  readout was not decomposed for N vs R; Investigation IV's attention/MLP split was shown to be confounded by site).
- The readout-route null responses are **not** what makes T1 models mutually coherent: the coherence of N responses carried by
  the Rq-only route is 0.11 within T1 vs 0.20 within P-count; with Rq blocked, 0.18 vs 0.09 (exploratory). The route difference
  and the coherence pattern are separate phenomena.
- The route test is exploratory beyond P5(a); its selectivity numbers were computed after the outcomes.

## 3. Mechanistic hypotheses (§9.1) and their status

| hypothesis | signature | status |
|---|---|---|
| direct count → configuration-posterior computation (Bayes mixture) | V-surface fitted by 𝔽_mix near the reliability ceiling | holds for T1 at full budget (relative R² 0.81–0.97); P-count 0.38–0.85 |
| additive source-evidence weighting (𝔽_vote) | 𝔽_vote fit near ceiling | rejected for all models (relative R² ≤ 0.36) |
| shortcut sensitive to absolute counts | null responses with "level" structure | present in all models (common bias, BAYES_NULL §4), small share of variance |
| redundant source and readout pathways | different N vs R routes | **present in 4 of 6 T1 parents**; absent in P-count |
| shared operation + curriculum-specific nuisance features | route separation plus common relevant operator | partially supported (T1 only) |

No curriculum-specific *operation* was causally identified as producing a curriculum-specific response signature, because the
signature itself (P1) did not replicate under the frozen criterion.
