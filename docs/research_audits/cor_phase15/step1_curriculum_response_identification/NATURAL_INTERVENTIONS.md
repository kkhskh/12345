# Natural intervention families (mandate §6)

All families are built by `scripts/cri_run.py` from the exact generator (`gen_family.Fam()`: K = 3, Q = 6, α = 0.8, ρ = 1, pc = 1)
before any model output is computed; validity masks depend only on the generator. One scored decision q per base episode.
Support statistics: development stream `logs/dev/support_summary.json`; confirmation stream `logs/conf/support_summary.json`.

## 1. Exact class structure behind the families [P]

π(n) ∝ ρ^{n_g}[n_h = 5] (Investigation IV, A2). Let H(n) = {h : n_h = 5}.
- |H| = 1 (colluder candidate h; the other groups g₁, g₂): π depends only on d = n_{g₁} − n_{g₂}; the classes are the 9 values of
  d for each h (27 classes). **Minimal null edits exist exactly here**: shift n_{g₁}, n_{g₂} together by ±1 (d fixed, level
  L = min(n_{g₁}, n_{g₂}) changes), possible when both are in 0…3 (up) or 1…4 (down).
- |H| = 2 (third group g free): π depends on n_g (5 classes per pair, 15 classes); any change of n_g changes π, so no null edit exists.
- |H| = 3: one class.
Total 27 + 15 + 1 = 43 (matches Investigation IV's enumeration). Every null edit therefore changes the *level* of the two
non-colluder counts and nothing else that π uses; this is the nuisance feature a model must ignore.

## 2. Families and their support (development stream; every pair has X_q fixed)

| family | construction | valid | marginal support | same world | exact Bayes at q | reports changed (median) | agreement pattern changed |
|---|---|---|---|---|---|---|---|
| **N** Bayesian-null minimal | two non-colluder agreements toggled together | 82 % | 100 % | 100 % | **preserved exactly** | 2 | yes |
| N2 null resample | X_{-q} re-realised with another vector of the same π-class | 82 % | 100 % | 100 % | preserved | 20 | yes |
| **R** Bayesian-relevant minimal | one non-colluder agreement toggled; validity requires a Bayes change | 87 % | 100 % | 100 % | changes | 1 | yes |
| R2 relevant resample | random same-world vector with different π, re-realised | 100 % | 100 % | 100 % | changes in 86 % (constant where X_q makes all configs agree) | 20 | yes |
| **C1** relabel | other questions' reports cyclically relabelled | 100 % | 100 % | 100 % | preserved; V preserved | 30 | no |
| C2 swap | the two sources of every pair swapped (other questions) | 100 % | 100 % | 100 % | preserved; V preserved | 10 | no |
| C3 content resample | X_{-q} re-drawn with the same per-question agreement pattern | 100 % | 100 % | 100 % | preserved; V preserved | 20 | no |
| **Cm** matched content | the report tokens touched by N changed without changing agreement | 82 % | 100 % | 100 % | preserved; V preserved | 4 | no |

"Same world": the true colluder agrees in every question of x′, so x′ has positive probability under the latent configuration
that generated x (reliable and noise pairs can produce any report pattern). "Marginal support" is reported separately and is
never called same-world. N, R and the C families are single-world input interventions; N2/R2 additionally resample report
contents.

## 3. What each family isolates

- **N** varies only the level of the non-colluder counts (and the two report tokens needed to do so). By Corollary 1
  (FORMAL_DECISION_EQUIVALENCE.md) every model response on N is an increment of the model's own log-odds error.
- **Cm** touches the same report tokens as N but keeps V, so N − Cm estimates the part of the N response caused by the count change
  rather than by token contents.
- **R** gives a known non-zero Bayes response, so the *operator* (how count changes become candidate evidence) can be studied, not
  only its errors.
- **C1–C3** measure sensitivity to information outside V.

## 4. Abstract counterparts (secondary)

Nabs and Rabs apply exactly the count vectors of the N and R pairs at each model's Investigation IV site through
covariate-corrected encoders (own-question reports as covariates, fixing Investigation IV's ME1 leakage). They are activation
additions and are used only to check agreement with the input-level families (P4).

## 5. Input-level V-surface

For 300 base episodes, all 36 same-world count vectors (v_h = 5 for the true colluder h) are realised four times each with fresh
report contents and the model's centred logits averaged: g_M(x_q, v) at the input level. Split halves (2 + 2 realisations) give a
reliability ceiling for every fit to the surface.
