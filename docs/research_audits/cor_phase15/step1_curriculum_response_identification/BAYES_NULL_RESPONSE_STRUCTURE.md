# Structure of Bayesian-null responses (mandate §5)

Development data: Investigation IV models on a new stream (`logs/dev/stats_dev6.json`, `stats_dev10.json`, `null_dev10.json`);
confirmation: the independent cohort (`logs/conf/primary.json`, `null_primary.json`). "N" = input-level Bayesian-null minimal edits
(exact Bayes response 0; NATURAL_INTERVENTIONS.md).

## 1. What the responses are (exact)

On every N pair, Δ_M = e_M(x′) − e_M(x) exactly (FORMAL_DECISION_EQUIVALENCE.md, Corollary 1): null responses are increments of
the model's own log-odds error. Their correlation across models measures how alike two models' error functions are along the
"level" direction (both non-colluder counts shifted, NATURAL_INTERVENTIONS.md §1).

## 2. Magnitude vs direction (§5.1)

| | T1 cohort (6) | P-count cohort (6) | development T1 (T2, T10, T11) | development P-count (P2–P4) |
|---|---|---|---|---|
| mean \|Δ\| on N (centred logits) | 0.10–0.19 | 0.22–0.34 | 0.14–0.21 | 0.27–0.33 |
| ratio to mean \|Bayes response\| on R | 0.30–0.56 | 0.65–0.99 | 0.40–0.60 | 0.77–0.93 |
| fraction of near-zero responses (max < 0.05) | 5–12 % | 1–5 % | 3–9 % | 1–2 % |

The responses are not small fluctuations: in P-count models they are as large as genuine Bayes-relevant responses; in T1 models
about half as large. Model-level magnitude tracks approximation error: across cohort models |N| correlates with the R-error
magnitude at r = 0.99 and with KL at r = 0.59 (development 0.97 and 0.95).

## 3. Output representation and metric (§5.2) — cohort Δ = coh_T − coh_P (95 % model-bootstrap CI)

| output space / metric | Δ |
|---|---|
| centred logits, Pearson (primary) | 0.149 [−0.047, 0.341] |
| margins, Pearson | 0.149 [−0.053, 0.341] |
| per-model z-scored (scale-free) | 0.149 [−0.056, 0.342] |
| cosine | 0.149 [−0.057, 0.344] |
| Spearman | 0.120 [−0.072, 0.308] |
| argmax-change agreement | 0.044 [−0.029, 0.121] |
| **probabilities**, Pearson | **−0.057** [−0.222, 0.104] |

Pearson on centred logits is invariant to per-model temperature (FORMAL §6), and the KL-optimal temperature is 1.00 for every cohort
model (grid step ≈ 2 %; 0.93–1.00 in development), so calibration does not move the result. In **probability space the contrast disappears** (cohort −0.06; development: 0.34 for the six Investigation IV models, 0.12 for all ten): the logit-space coherence is carried by episodes where the posterior is extreme, where small probability
changes are large logit changes. The contrast is not robust to this reasonable output representation.

## 4. Dependence on the count-equivalence class (§5.3)

Null edits exist only for |H| = 1 and change the level L = min(n_{g₁}, n_{g₂}) with d fixed. Regression of the confidence response
s = ⟨Δ, c*⟩/‖c*‖ on generator features (`null_structure.py`):

| | sign of coefficient (all 22 development + cohort models) | meaning |
|---|---|---|
| dir (counts up) | + | raising both non-colluder counts makes the model *more* confident in the Bayes direction |
| dir·L | − | the effect shrinks at higher levels |
| dir·\|d\| | − | and when one group is already clearly more reliable |
| dir·max p* | − | and when the Bayes posterior is already confident |
| colluder location h | adds ≤ 0.01 R² | no dependence on which group is the colluder |

This "more agreement → more confidence" bias is shared by both curricula: coefficient-vector correlations T–T 0.93, P–P 0.93, T–P
0.87 in the cohort (development 0.99 / 0.99 / 0.98). The frozen criterion for "common structured bias" (T–P ≥ 0.9) failed
narrowly (0.87). The features explain 2–36 % of each model's null-response variance (T1 0.02–0.36, P-count 0.07–0.19): most of the
null response is unstructured with respect to these features.

## 5. Two types of error (§5.4)

- Errors on Bayes-relevant changes (R-family response minus per-model-scaled exact response) and errors on Bayes-irrelevant changes
  (N responses) are tightly linked: model-level r = 0.99 (cohort), 0.97 (development); within models, across base episodes, the
  absolute N response and absolute R error correlate 0.19–0.42 (development).
- Coherence of R-errors does **not** differ by curriculum in the cohort (Δ_R = −0.004, CI [−0.21, 0.18]); in development it did
  (0.37). The raw R responses are more coherent within T1 (0.78) than within P-count (0.53) because T1 models track Bayes more
  closely, not because their errors are shared.

## 6. Summary

The null responses are approximation errors (Corollary 1), whose size follows the model's distance from Bayes, with a weak
structured component — a shared "more agreement, more confidence" bias — and a large model-specific remainder. The T1 coherence of
Investigation IV is smaller in independent runs, absent in probability space, and absent when T1 parents are transferred to
P-count-like quality.
