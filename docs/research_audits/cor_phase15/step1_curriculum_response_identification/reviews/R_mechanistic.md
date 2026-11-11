# R_mechanistic — independent adversarial mechanistic review (mandate §13, "Mechanistic reviewer")

Scope: `src/cri.py`, `scripts/cri_run.py`, `route_selectivity.py`, `p5b_matched_content.py`, `phi_sign_posthoc.py`, forward pass
`fwd3.py`, `eqv.Model`, and DOWNSTREAM_CAUSAL_MECHANISMS / NATURAL_INTERVENTIONS / APPROXIMATION_VS_ALGORITHM / MAIN_RESULT.
Data: `logs/conf/` (frozen arrays) and checkpoints `ft_T1_s62`, `ft_T1_s64`, `ft_T1_s65`, `ft_T1_s61`, `ft_T1_s63`, `ft_P_s61`.
Checks: `reviews/checks/rv_mech_{routes,cancel,patchcheck,info,phi,surface}.py` -> `logs/review_checks/rv_mech_*.out` (+ `rv_mech_phi.json`).
"Norm" below means the mean per-pair L2 norm of the change in centred logits at the scored decision. Route models = cT62, cT64, cT65.

## Findings

**M1 — HIGH. The route "shares" are projections, and they hide large cancelling, non-additive route contributions.**
P5 and route_selectivity report `<Δ_set, Δ_full>/||Δ_full||²`. In the route models the Rq-only and Rq-blocked N responses
are each 2.5–3.9x the size of the natural N response, and they point in nearly opposite directions (`rv_mech_cancel.out`):

| model | ‖Δ_N full‖ | ‖Rq-only‖ | ‖Rq-blocked‖ | cos(Rq-only, blocked) | ‖full − Rq-only − blocked‖/‖full‖ |
|---|---|---|---|---|---|
| cT62 | 0.219 | 0.769 | 0.846 | −0.84 | 2.56 |
| cT64 | 0.342 | 0.716 | 0.777 | −0.70 | 1.54 |
| cT65 | 0.304 | 0.716 | 0.799 | −0.81 | 1.38 |
| cT63 (no route) | 0.306 | 0.081 | 0.322 | −0.26 | 0.06 |
| P-count | 0.43–0.65 | 0.40–0.67 | 0.19–0.51 | −0.02 to −0.32 | 0.20–0.47 |

So the natural N response is a small leftover from two large opposing effects that also interact non-linearly. The claim that
"26–56 % of the Bayes-null response enters at the readout token" is not a decomposition of the response. 0.56 is the length of
one large vector's projection onto a much shorter one. The interaction term is also above 0.2 in every P-count model, so the
P-count profiles are not additive either.
*Correction:* report norms and the interaction term next to every share. State that position-level patching is non-additive
in the T1 route models. Withdraw the "share via Rq" reading, or replace it with a symmetric (Shapley-type) attribution.

**M2 — HIGH. Blocking Rq is not a selective de-biasing. It increases the departure from Bayes several-fold.**
Rerun on the checkpoints (`rv_mech_patchcheck.out`, 600 valid pairs per family). At the scored decision, natural vs Rq-blocked:

| | cT62 N | cT62 R | cT64 N | cT64 R | cP61 N |
|---|---|---|---|---|---|
| KL(Bayes‖model) | 0.0081 → **0.0974** | 0.0092 → 0.0249 | 0.0133 → **0.0929** | 0.0138 → 0.0289 | 0.104 → 0.114 |
| mean \|clogit − Bayes\| | 0.160 → 0.479 | 0.161 → 0.257 | 0.215 → 0.456 | 0.204 → 0.270 | 0.542 → 0.576 |
| argmax = Bayes argmax | 0.887 → 0.802 | 0.898 → 0.855 | 0.897 → 0.798 | 0.902 → 0.875 | 0.800 → 0.807 |

On R pairs, the error left after removing β·ΔBayes grows from 0.18 to 0.42 (cT62), 0.27 to 0.45 (cT64) and 0.26 to 0.43 (cT65)
(`rv_mech_routes.out`). The "87–95 % of the relevant response preserved" figure is a projection onto Δ_R. The blocked response also
carries a large extra component that is orthogonal to it and not Bayesian (‖blocked‖ 0.69 vs ‖full‖ 0.63 in cT62). Two
sentences are therefore false: DOWNSTREAM §2 ("removes much of the departure from Bayes … keeps most of the useful Bayesian
behaviour, without destroying accuracy … a selective modification in the sense of §9.4 step 5") and MAIN_RESULT item 6
("selectively modifiable nuisance pathway").
*Correction:* withdraw the selective-modification claim. Blocking creates an internally inconsistent state (the readout says
"base", the sources say "counterfactual"), and the model's output degrades. This is the standard off-manifold ablation
artefact.

**M3 — HIGH. The N-vs-R route difference follows from where information sits, plus how the families are built.**
(a) Information location (`rv_mech_info.out`, ridge regression, held-out R²). The T1 readout input at layer 2 carries the
total count S = Σ_g n_g (R² 0.58–0.83). It carries **no** information about |d| = |n_g1 − n_g2|, the Bayes-relevant variable
(R² 0.00 in all five T1 models tested). The source tokens carry both (S 0.96–1.00, |d| 0.45–0.54). In P-count cP61 the readout
carries both (0.83 / 0.46). The route strength follows readout S decodability (cT62 0.83, cT64 0.80 vs cT61, cT63 0.58), but
not perfectly (cT65 0.63).
(b) The Rq route responds the same way to both families per unit change in S. Rq-only norm divided by |ΔS| is 0.385 vs 0.365
(cT62), 0.358 vs 0.341 (cT64) and 0.358 vs 0.331 (cT65) for N vs R. On R pairs it is not aligned with Bayes (R² on ΔBayes 0.00–0.02).
(c) The construction fixes |ΔS|: it is 2 for every N pair and 1 for every R pair (`rv_mech_routes.out`).
Together: the readout cannot carry d, so a Bayes-relevant response cannot enter there. "Blocking Rq preserves the relevant
response" is therefore implied by the representation, not discovered. The Rq share is small for R because the same
S-driven Rq vector is divided by R's large d-driven Qs response, and large for N because N's denominator is small (M1).
*Correction:* restate the result as "in T1 models the layer-2 readout input holds a total-count copy that layer 2 uses
against an opposing source-token contribution". Drop "N and R travel by different routes" as a mechanistic finding.

**M4 — MEDIUM. The readout route is not shown to be distinct from an approximation bias. The evidence points the other way.**
Given M1–M3, the natural null response in route models is the imperfect cancellation of a count-sum signal that is present both
at the readout and at the sources. A residual from incomplete cancellation is an approximation error of the count → evidence
map, and it should show up as level dependence in g_M. That is exactly the "common bias / level structure" the package already
assigns to approximation. No test separates "a second pathway with its own function" from "two copies of S whose downstream
weights cancel imperfectly". Grade MAIN_RESULT item 6 and DOWNSTREAM table row "redundant source and readout pathways: present"
as [E] "redundant S representation, cancelling contributions". Do not call it a nuisance pathway.

**M5 — MEDIUM. P5(a) does not test what the text says it shows.**
The statistic pools all 12 models (one-sided Wilcoxon, p = 0.00049, which is 11 of 12 positive). The excess is positive in
5 of 6 P-count models (cP61 0.12, cP62 0.20). The test contains no arm contrast, so "the difference is carried by the T1
models" is a post hoc reading. The excess is also built on the non-additive projections of M1. The split-half null controls
for sampling noise only. With about 1600 pairs, any systematic N/R difference passes it. There is no placebo route (for
example C-family pairs patched by route).
*Correction:* report P5(a) as "N and R projection profiles differ beyond sampling noise in 11/12 models, P-count included". Any
arm statement should be explicitly exploratory.

**M6 — MEDIUM. The phi_rel(5) − phi_rel(4) sign finding is robust for T1 but not meaningful for P-count.**
(`rv_mech_phi.out`.) Checked first: the frozen θ is reproduced exactly. Of 6 random starts, at most 3 land in worse local optima
(loss +0.5 % to +12 %), and the frozen start reached the best loss found in every model. The gauge constraints φ_rel(0) = 0 and
φ_col(5) = 0 remove the two exact symmetries, and no further exact symmetry remains. Episode bootstrap of s (15 resamples):
T1 SD 0.01–0.03, P-count SD 0.03–0.07, and the sign never flips in T1. So s is *stable*. But the profile test shows the P-count
sign is not *identified* behaviourally. Forcing s = 0 changes loss by +0.0 %, +0.1 %, +0.1 % and +0.2 % in cP61, cP62, cP64 and
cP66 (R²_within changes ≤ 0.004). Even forcing s = +|ŝ| costs ≤ 0.011 R² in those four. Only cP63 (+1.3 %) and cP67 (+3.2 %)
resist. In T1, forcing s = 0 costs +2.7 % to +17.7 %, so T1's s > 0 is real. Fits restricted to cells without a second count-5
group give s ≈ +7 to +11 in P-count models, meaning s is unconstrained there and set entirely by the 11/36 ambiguous cells.
*Correction:* restate the finding as "s > 0 identified in T1; in P-count, s is ≈ 0 within tolerance in 4/6 and negative in
2/6". The 13/13 sign count and the Fisher p = 0.0014 treat flat-direction point estimates as observations. Withdraw the
"perfect agreement is suspicious" interpretation for P-count. Also, the cohort-only Fisher table includes the failed runs
cT60, cT66, cP60 and cP65.

**M7 — MEDIUM. The split-half ceiling is biased low, so relative R² is inflated.**
`split_half_r2_within` is R² of one 2-realisation mean predicting the other. The target being fitted is the 4-realisation
mean, whose noise variance is half that of a 2-mean. A correct ceiling is 1 − (Σ(h_a − h_b)²/4)/SS_target (`rv_mech_phi.py`).
Corrected relative 𝔽_mix R²: cT67 0.81 → 0.61, cP61 0.74 → 0.58, cP62 0.85 → 0.73, cP66 0.38 → 0.30. High-reliability T1
models change only slightly (0.92–0.97 → 0.89–0.95). The T1 range "0.81–0.97" becomes 0.61–0.95, and the P-count range
"0.38–0.85" becomes 0.30–0.73. The ε-classification, and P3(b) if it uses these ratios, should be recomputed.

**M8 — LOW–MEDIUM. The matched-content control does not match the tokens, and the projection understates content sensitivity.**
For an agreeing pair, `matched_content_control` relabels *both* sources, while the N edit changed one. Cm touches a superset
of N's tokens in 100 % of pairs and exactly the same set in only 36 % (median 4 tokens vs 2). NATURAL_INTERVENTIONS ("the same
report tokens touched") is wrong. f_Cm = 0.02–0.13 is a projection. ‖ΔCm‖/‖ΔN‖ is 0.25–0.62, so content-only edits produce
responses a third to a half the size of N's, mostly orthogonal to N. The "N − Cm" estimand of §3 is never computed.
*Correction:* fix the description, report ‖ΔCm‖/‖ΔN‖, or build a one-token-matched control.

**M9 — LOW. The abstract interventions omit the readout copy in T1.**
For T1, Nabs writes only Qs. In route models the readout holds S (M3), so Nabs is not the abstract counterpart of N. P4's
"agreement" (both CIs include 0) is also uninformative as evidence of agreement.

**M10 — LOW. 𝔽_vote is a weak alternative.**
It has no colluder structure, and it fits exact Bayes at R²_within ≈ 0.09 (cT62 vote 0.09 vs mix 0.95). "No model is
𝔽_vote-representable" rules out only this strawman. It does not show that the curricula share one algorithm.

**M11 — LOW. The R family mixes regimes.**
15 % of valid R pairs already have |H| ≥ 2, and 3.7 % create a second count-5 group (`rv_mech_surface.out`). Each R edit also
changes S by 1, so R is not a purely relevant edit (see M3c).

**M12 — checked, no defect.**
Patching the layer-2 input at all positions reproduces the natural counterfactual (max diff ≤ 1.5e-6, 3 models × 2 families).
The reason is that resid_pre2 is the full residual state. "Layers 0–1 frozen" is irrelevant to this, so the doc's
justification should be reworded. Recomputed patch_Rq and block_Rq equal the frozen arrays bit for bit. Rq, Qs and Oth
partition all 43 positions, and block is the complement patch. Family N preserves π exactly, keeps the same world, and leaves
X_q fixed. V-surface realisation: the colluder stays in the same world (asserted), and cross-pair first-source agreement
matches natural data (0.335 vs 0.336). Report-truth coupling is not preserved (0.333 vs 0.489), but a model cannot observe it.
The covariate-corrected encoder's collinear one-hot blocks are harmless because only differences are used.

## Verdict table

| claim (document) | verdict |
|---|---|
| Patching all positions reproduces the natural counterfactual (DOWNSTREAM) | **Correct**; the justification ("frozen") should be reworded |
| N share via Rq 26–56 % in 4/6 T1 (DOWNSTREAM §1–2, MAIN 6) | **Not a valid decomposition** (M1): large cancelling, non-additive contributions |
| Blocking Rq removes 34–65 % of null response, keeps 87–95 % of relevant, selective, accuracy-preserving | **Refuted** (M2): KL to Bayes ×7–12 on N, ×2–3 on R; accuracy vs Bayes −0.03 to −0.10 |
| Readout-local nuisance route is a real pathway distinct from approximation bias | **Not supported** (M3, M4): follows from readout holding S but not d; reads as imperfect cancellation |
| "Blocking Rq preserves R" not an artefact of R construction | **Artefact** (M3): d is absent from the readout; S-route identical per unit ΔS |
| P5(a) passed, "carried by T1" | **Test valid as frozen; interpretation overstated** (M5) |
| P5(b) / matched-content control | **Description wrong, conclusion roughly holds as a projection** (M8) |
| 𝔽_mix representability, relative R² ranges | **Inflated** by the low ceiling (M7); qualitative T1 > P-count likely holds |
| No model 𝔽_vote-representable ⇒ no distinct algorithm | **Weak** (M10) |
| s = φ_rel(5) − φ_rel(4): T1 > 0 | **Robust** (multistart, bootstrap, profile) |
| s < 0 in 13/13 P-count, Fisher p = 0.0014, "perfect agreement suspicious" | **Not identified** in 4/6 cohort P-count models (M6); downgrade |
| L-BFGS local optima / gauge | **No defect** in the reported fits; local optima exist, but the frozen start found the best loss in all 12 |

Net: the package's input-level families and patching code are correct. The mechanistic interpretation of the T1 readout route
and the "selective modification" claim should be withdrawn. The φ-sign finding should be restricted to "T1 s > 0".
