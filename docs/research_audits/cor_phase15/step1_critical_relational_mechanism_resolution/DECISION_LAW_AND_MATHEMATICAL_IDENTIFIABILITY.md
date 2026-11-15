# Decision law and mathematical identifiability (Workstream B)

COR Phase 1.5, Step 1 (provisional ≈ 25%). This document makes no phase closure, no thesis, and no claim about the O/D operators.

**Evidence labels:**
- **[proof]** mathematical proof;
- **[num]** numerical verification;
- **[dev]** development models (XIII-B seed units: uP161 s20–s23, uP160 s20/s21/s23, plus the first factorial cells);
- **[conf]** frozen confirmation (`FROZEN_CONFIRMATION_AND_STATISTICS.md`);
- **[expl]** exploratory.

Scripts:
- `scripts/math_identifiability.py` → `logs/math_identifiability.json`
- `scripts/fair_fits.py` → `logs/fair_fits_dev.json`, `logs/fair_fits_conf.json`
- `scripts/b4_pooling.py` → `logs/b4_pooling_dev.json`
- `scripts/b5_residual.py` → `logs/b5_residual_dev.json`
- `scripts/d_das.py` (behavioural part) → `logs/d_das_dev*.json`

Proof notes: `logs/NOTES_MATH_B1.md`.

## B1. What the power-mean exponent ρ can and cannot mean

Setting: the C = 1, β = 1 T4 generator (G = 3, K = 3, Q = 6, α = 0.8). At the target question q every group has exactly two sources. Write
b = (1 − α)/2, r = α/b = 8, and k_g(y) ∈ {0, 1, 2} for the number of group g's sources reporting y at q. Then the Bayes candidate evidence
is L_g(y) = (b²/noi) · r^{k_g(y)}.

The XIII-B nested family is PM(ρ, τ, w): log p(y) = (τ/ρ) log Σ_g w_g L_g(y)^ρ + c, with the weights w a softmax of
a·log W_g + k·log max(f_g, 1) + d·[f_g = 0].

**T1 (tempering identity) [proof; num 1.3e-15].** PM(ρ, τ, w) equals, as a function of the input:
1. raise the evidence to the power ρ;
2. mix it with the weights w;
3. raise the output to the power τ/ρ.

**T2 (target-reliability identity) [proof; num 1.8e-15].** For every ρ > 0, PM(ρ, τ, w) also equals:
1. the exact mixture computed with a misstated per-source reliability α′ at q, where α′/b′ = (α/b)^ρ;
2. followed by the output exponent τ/ρ.

*Proof.* L_g^ρ = (b²/noi)^ρ r^{ρ k_g} = κ · (b′²/noi) r′^{k_g} = κ L′_g, with r′ = r^ρ and κ independent of g and y, because every group has
two sources at q. Hence Σ w L^ρ = κ Σ w L′. ∎

Values: α′(0.6) = 0.635, α′(0.3) = 0.483, α′(0.9) = 0.765.

**Consequence.** The two descriptions are the same input–output function:
- "a different combination operator, ρ < 1";
- "the exact mixture operator applied to under-confident candidate evidence at the target question, followed by a sharpened output".

No observational experiment on inputs and outputs can tell them apart. We therefore do not claim an empirical distinction between them.
The equivalent description is specific: it concerns evidence miscalibration at q only, while the reliability weights W from the other
questions are untouched.

**T3 (where ρ is identified) [proof; num].** With all weight on one group, log p = τ log L_g, and ρ cancels exactly (2e-16). ρ is therefore
informed only by episodes with ≥ 2 effectively weighted groups whose L profiles differ. Under the exact weights:
- one weighted group: 6.3% of natural episodes;
- two: 88.7%;
- three: 5.0%.

**T4 (lapse confound) [num].** A Bayes posterior with a lapse floor, (1 − λ) p_B + λ/3, lies outside the family. Its KL-best PM fit nonetheless
has ρ < 1:

| λ | fitted ρ |
|---|---|
| 0.01 | 0.969 |
| 0.03 | 0.919 |
| 0.10 | 0.809 |

So ρ < 1 alone does not show a combination difference. Lapse must be fitted jointly.

**T5 (pure output temperature) [num].** p ∝ p_B^τ is in the family with ρ = 1, and fits recover ρ = 1.000 at τ = 0.7 and 1.5. Calibration
is not confounded with ρ.

**Misspecification [num, dev].** For a model the family does not contain, the best ρ depends on the scoring rule. uP161 s22 gives:
- ρ = 0.56 under XIII-B's least squares on centred log-probabilities;
- ρ = 0.71 under the proper KL objective (bootstrap 0.69–0.75);
- ρ = 0.79 when a lapse floor is fitted jointly (λ = 0.03).

**ρ is a property of (model, family, loss), not of the model alone.**

## B2. Fair nested fits: proper scoring, multi-start, held-out data, bootstrap [dev]

All families nest exact Bayes and share the same weight parametrisation. Each is fitted by minimising KL(model ‖ fit) on 3000 natural
episodes, with ≥ 2 starts (FULL also starts from the PML/HYB optimum; this was added after one non-converged dev fit, which is kept and
flagged). Evaluation uses 3000 held-out episodes.

The families:

| family | free parameters | structure |
|---|---|---|
| BAYES | 0 | — |
| TEMP | 1 | τ |
| MIX | 4 | τ, a, k, d |
| PM | 5 | MIX + ρ |
| MIXL | 5 | MIX + lapse |
| PML | 6 | PM + lapse |
| NPOE | 4 | ρ → 0 |
| HYB | 5 | MIX + ungated pooled evidence γ |
| FULL | 7 | all of the above |

Held-out results:

| unit (class) | KL(model‖Bayes) | KL MIX | KL PM | KL PML | KL FULL | ρ PM [bootstrap 5–95%] | ρ PML (λ) | KL NPOE |
|---|---|---|---|---|---|---|---|---|
| uP161 s22 (early) | 0.00781 | 0.00721 | 0.00674 | 0.00632 | 0.00630 | 0.714 [0.69, 0.75] | 0.791 (0.030) | 0.0158 |
| uP160 s21 (early) | 0.00456 | 0.00426 | 0.00402 | 0.00357 | 0.00352 | 0.731 [0.70, 0.76] | 0.801 (0.025) | 0.0140 |
| uP161 s23 (hybrid) | 0.00410 | 0.00398 | 0.00385 | 0.00376 | 0.00385* | 0.804 [0.79, 0.83] | 0.831 (0.010) | 0.0142 |
| uP161 s21 (mixture) | 0.00181 | 0.00180 | 0.00180 | 0.00178 | 0.00178 | 0.906 [0.89, 0.93] | 0.919 (0.005) | 0.0131 |
| uP160 s23 (mixture) | 0.00131 | 0.00127 | 0.00126 | 0.00125 | 0.00126 | 0.925 [0.90, 0.93] | 0.934 (0.003) | 0.0125 |

\* FULL did not converge on this unit: its KL is above PML's although FULL nests PML. Fixed by nested starts before confirmation.

Readings:
1. Normalised product-style pooling (NPOE) is decisively worse than every mixture-type family in every unit. The early class is not a
   product-of-experts.
2. Within the mixture-type families, ρ_KL separates the classes: early 0.71–0.73, hybrid 0.80, mixture 0.91–0.93. Bootstrap intervals do
   not overlap. With lapse fitted, the gap narrows to 0.79–0.80 versus 0.92–0.93.
3. Ungated pooling (HYB, γ) adds nothing (Shapley share ≈ 0) — see B3.
4. **The fits explain little.** The best 7-parameter family removes 19–23% of an early-class model's KL from Bayes, and only 2–6% in
   mixture and hybrid models. Most of every model's deviation lies outside any of these families (B5).

## B3. Why does the early class have ρ < 1? Hypotheses tested

**(i) Early ungated pooling — rejected [dev].**
- *Hypothesis:* the readout token pools candidate evidence over all six sources at layer 0, before relational gating is available, which
  forces log-linear (ρ → 0) combination.
- *Test:* on-support SWAP edits exchange the values of two self-agreeing groups at q. The multiset of source values at q is unchanged (any
  ungated pool is invariant: max diff 9e-16) and only the group assignment changes.
- *Result:* in every early unit, transplanting the readout token's layer-0 state from the SWAP donor transfers 0.99–1.00 of the model's own
  SWAP response (LIK edits: 0.87–0.88), and behaviour follows Bayes on SWAP (slope 0.93–0.94).
- The early readout state is **group-resolved**, not pooled, and the HYB pooling parameter is ≈ 0 in every fit.

**(ii) Routing organisation causes the exponent — tested causally by training interventions.**
- *Development:* a 600-step mask on layer-0 readout attention (ro0) turned the early-destined uP161 22/22/22 run into a mixture model. Its
  ρ_KL is in `logs/fair_fits_dev_ro0.json` if computed.
- *Confirmation:* prediction R3 tests whether intervention-induced organisation changes shift ρ_KL in the predicted direction on new runs.
  See FROZEN_CONFIRMATION §R3 and §B3-result below.

**(iii) Not tested:**
- MLP-capacity manipulation;
- architecture changes other than the attention masks (compute budget).

## B4. Distinguishable versus equivalent descriptions

| distinction | status |
|---|---|
| "combination exponent ρ" vs "target-question evidence tempering + output sharpening" | **identical functions** (T1, T2): not distinguishable by any input–output data |
| ρ vs pure output temperature | distinguishable (T5) |
| ρ vs lapse floor | distinguishable in principle; confounded under misspecification (T4). Report PML |
| mixture-type vs normalised product pooling | distinguishable; product pooling rejected in every unit |
| early ungated pooling | distinguishable internally (SWAP); rejected |
| the internal *factorisation* of the early class's law | **not identified**: the readout state is group-resolved and layer 2 combines it, but which internal step implements "tempering" is not determined by any experiment run here |

## B5. Where the class-specific error comes from [dev]

**(a) Parameter-group Shapley over the FULL family.** Shares of KL(model ‖ Bayes) removed by freeing each group:

| unit | CAL τ | COMB ρ | REL a | ELIG k, d | POOL γ | LAPSE λ | explained |
|---|---|---|---|---|---|---|---|
| uP161 s22 (early) | 0.024 | 0.038 | 0.073 | 0.008 | −0.005 | 0.055 | 0.19 |
| uP160 s21 (early) | 0.042 | 0.014 | 0.083 | 0.032 | −0.008 | 0.066 | 0.23 |
| uP161 s23 (hybrid) | 0.005 | 0.016 | 0.023 | 0.015 | −0.004 | 0.006 | 0.06 |
| uP161 s21 (mix) | 0.005 | 0.001 | 0.001 | −0.001 | 0.010 | 0.002 | 0.02 |
| uP160 s23 (mix) | 0.004 | 0.003 | 0.004 | 0.023 | 0.005 | 0.001 | 0.04 |

The combination exponent explains 1–4% of the early class's KL. That is less than reliability weighting (7–8%), the lapse floor (6–7%) or
calibration (2–4%).

**(b) Non-parametric decomposition** (`b5_residual.py`; 60,000 episodes canonicalised by the exact sufficient statistics):
- *structured* = KL(cell model ‖ Bayes): deviation that is a function of the sufficient statistics;
- *within-cell* = KL(model ‖ cell): dependence on information Bayes ignores, plus sampling noise of the cell means.

| unit | KL(model‖Bayes) | structured | within-cell | KL per invariance-violating edit: VPERM / MOVE / XVAL |
|---|---|---|---|---|
| uP161 s22 (early) | 0.00794 | 0.37 | 0.64 | 0.00037 / 0.00087 / 0.00056 |
| uP160 s21 (early) | 0.00450 | 0.45 | 0.56 | 0.00012 / 0.00031 / 0.00010 |
| uP161 s23 (hybrid) | 0.00400 | 0.29 | 0.74 | 0.00016 / 0.00054 / 0.00028 |
| uP161 s21 (mix) | 0.00177 | 0.25 | 0.79 | 0.00007 / 0.00036 / 0.00010 |
| uP160 s23 (mix) | 0.00134 | 0.30 | 0.72 | 0.00005 / 0.00019 / 0.00005 |

The fractions can sum to slightly more than 1, because the two KL terms are not exactly additive.

**Answer to Q6.**
- The early class's larger error, 2.5–6× the mixture class, splits about 40:60 into:
  - a structured miscomputation of the right sufficient statistics;
  - **non-invariance**: the model responds to information Bayes ignores. Moving an agreement between other questions (MOVE) costs the most.
- Of the structured part, the parametric families capture about half:
  - reliability-weighting exponent and lapse floor first;
  - then calibration and eligibility coefficients;
  - the combination exponent ρ least.
- In mixture models nearly all of the (much smaller) error is non-invariance.

## B6. E, A1 and A2 can be distinguished — and are

**Mathematics [num].** Four on-support edit families separate the three eligibility rules (T6; d_das families):

| family | edit | changes | E–A1 distance | E–A2 distance |
|---|---|---|---|---|
| E5 | n_h 5 ↔ 4 | E and A1, not A2 | 0.003 | 1.33 |
| E3 | n_h 4 ↔ 3 | A2 only | 0.002 | 1.44 |
| QF5 | agreement at q flips, n = 5 | E and A2, not A1 | 1.06 | 0.012 |
| QF4 | agreement at q flips, n = 4 | A2 only | 0.001 | 1.32 |

E is the only rule correct on all four. No single family separates all three rules.

**Behaviour [dev].** R² of each model's own response against each rule (E / A1 / A2):

| unit | E5 | E3 | QF5 | QF4 |
|---|---|---|---|---|
| uP161 s21 (mix) | 0.98 / 0.98 / 0.59 | 0.90 / 0.89 / 0.00 | 0.99 / 0.39 / 0.98 | 0.99 / 0.99 / 0.17 |
| uP161 s22 (early) | 0.93 / 0.93 / 0.69 | 0.71 / 0.71 / 0.00 | 0.95 / 0.33 / 0.94 | 0.97 / 0.97 / 0.21 |
| uP160 s21 (early) | 0.95 / 0.95 / 0.70 | 0.79 / 0.78 / 0.01 | 0.96 / 0.34 / 0.96 | 0.98 / 0.98 / 0.21 |
| uP161 s23 (hybrid) | 0.96 / 0.96 / 0.65 | 0.78 / 0.77 / 0.00 | 0.97 / 0.36 / 0.97 | 0.98 / 0.98 / 0.19 |

Every class follows E. A1 fails on QF5 and A2 fails on E5, E3 and QF4. The earlier label "agreement-at-q-gated" (XIII-B) can now be
sharpened to **E** at the behavioural level. Mediator-level identification is in CAUSAL_CONTROLS §D4.

## B-confirmation

See FROZEN_CONFIRMATION_AND_STATISTICS.md: R3 (routing → ρ, causal) and the ρ columns of R1/R2 units.
