# Adjudication of the independent reviews (atlas)

**The reviews.**
- **R1** (`reviews/R1_math_statistics_positive.md`): formal mathematics, statistical inference, positive-theory defence.
- **R2** (`reviews/R2_causal_transformer_representation.md`): causal identification, Transformer computation, representation theory.
- **R3** (`reviews/R3_novelty_counterexample_learning.md`): novelty, counterexamples, learning dynamics. Commissioned after the corrections below and the PP3 results, so it reviews the revised package.

**Provenance.** Where a disposition rests on a reviewer computation from their `/tmp` scripts (not preserved), it says so. Where I re-ran the check in-package, the log is cited.

## R1: dispositions

| # | finding | disposition | action |
|---|---|---|---|
| R1-a | Theorem S needs a second assumption: **at most one answer-dependent group per configuration**. With two reliable pairs, agreeing on the same value vs on different values gives likelihood 0.1366 vs 0.0043 | **Accepted** (major) | Theorem S restated with assumption (O) |
| R1-b | D1 "specialisation" is false as worded: wrong sign in 27,098/27,100 cases at (ρ .7, p_c .2, κ .3) and in 3.8% of T4 cases | **Accepted** (major) | specialisation withdrawn; only the exact identity (Theorem D) and the general expectation form are kept |
| R1-c | PP1 ran 500 base episodes, but the frozen text says 1500 (undisclosed) | **Accepted** (major) | **re-run at the frozen 1500** (`logs/pp1_sign_frozen1500.json`): passes 10/10, boundary 0.971–0.999, episode-cluster 95% CIs ≥ 0.96. The 500-episode run is kept and labelled a deviation |
| R1-d | the PP1 \|Δm\| > 0.5 filter raises agreement a lot; report unfiltered numbers and cluster CIs | **Accepted** | unfiltered boundary agreement (1500 run): 0.79–0.93 vs 0.20–0.39 for the monotone plug-in. Both are reported. The frozen filter decides pass/fail |
| R1-e | PP2's ρ = 0.9 and 0.7 shifts are non-discriminating: plug-in/true disagreements are near-ties (median log-gap 0.012–0.023) | **Accepted** (major) | "H1-strong falsified" now rests **only on the p_c = 0 shift**. There all 7 Transformers side with the plug-in in < 0.70 of cases (0.19–0.64), and all 3 DeepSets in ≥ 0.75. The ρ shifts are reported as uninformative |
| R1-f | the ρ = 0.999 plug-in is arbitrary off the support (0.9999 changes 378 answers) | **Accepted** | stated; the p_c = 0 conclusion does not depend on ρ_plug, because on T1 data the plug-in's error comes from the forced-colluder constraint, not from ρ. Fitting an implied ρ per model is listed as open |
| R1-g | the local/global axis is post hoc, has free choices, was scored on the same streams, and is a continuum (0.30–0.87), not two classes | **Accepted** (major) | all "global vs local" language is downgraded to a post hoc *continuous* index. OP-1 is restated |
| R1-h | I2 is undersold: unidentifiability also holds with κ = 0 at ρ* = 0.49. Combined with Theorem U this gives a new theorem: **if a_coll ≤ a_rel, no reversal is possible in any context, whatever p_c** | **Accepted** (positive-theory defence) | added as **Corollary U2**. Proof: if a_coll ≤ a_rel, the reliable role has the highest agreement probability, so by Theorem U its posterior is non-decreasing in its count |
| R1-i | the phase-diagram footnote is wrong: 29% of contexts at ρ = 1, p_c = 1 are undefined, not "pinned" | **Accepted** | footnote corrected |
| R1-j | "no estimator can succeed" under I2 is contradicted by Bayes 0.566 vs own-question 0.557 | **Accepted** | restated: no estimator can identify which group to trust; residual accuracy above own-question comes from averaging both groups |
| R1-k | the phase diagram counts drops of any size (mean drop 0.003 at ρ .5, p_c 1) | **Accepted** | note added: drop *magnitude* is not reported, only incidence |
| R1-l | minor items: "necessity of (U)" overstated; SR wording on p_c·ρ; PP2 falsifier code uses ≤ n/2 vs the frozen < n/2; Theorem U's interpretation conflates reliability with margin; I4 uncertainty | **Accepted** | each fixed in text. I4 CIs: T4 gap [−0.005, 0.006] (reviewer), T1 gap 0.026 [0.019, 0.034]. With the frozen < n/2, the p_c = 0 shift still falsifies (3/10 side with the plug-in; 7/10 do not) |

## R2: dispositions

| # | finding | disposition | action |
|---|---|---|---|
| R2-a | **reachability test invalid**: a pooled covariance is dominated by position/value differences. Per position, 39–61% of patched states lie beyond p99 | **Accepted** (major) | **re-run in-package** (`scripts/fixes_r2.py`, `logs/mech/fixr2_s*.json`): cond patch 45–60% beyond p99, held-out natural 1.3–1.6%, mismatch 67–75%, random 77–90%. **The claim that the patch is on-manifold is withdrawn.** Patched states are partly off-manifold, less so than the null or random displacements. Patch-based sufficiency results are now labelled "internal surgery, partly off-manifold" |
| R2-b | resample ablation from independent episodes mostly removes question q's *own* evidence | **Accepted** (major) | **re-run in-package**: partner resample (same X_q) gives a relation-specific drop of **0.20–0.22** in seeds 10–13. The independent-episode drops (0.38–0.41) are retained but relabelled "total" |
| R2-c | Shapley shares are not conserved under the exact head split (metrics are nonlinear) | **Accepted** (major) | the claim "Shapley aggregated by pattern class is the invariant replacement" is **withdrawn**. Shapley is reported descriptively. The invariant statements are only the head-free ones: the all-heads set and the behavioural law |
| R2-d | the "candidate-evidence site" (last-layer attention output at readout q) is the 2-D candidate-score code: the same subspace for context and own-report changes (cos 0.99–1.00) | **Accepted** (major) | E_q is merged into "answer representation". It is no longer a separate level of the abstraction |
| R2-e | I6 is not shown to hold with pre-LayerNorm (LN before Q/K makes scores nonlinear; final LN is affected) | **Accepted** (major) | I6 downgraded to a **conjecture**, with the obstacle stated |
| R2-f | the pattern-patch specificity test cannot fail on target edits (base values are kept) | **Accepted** (major) | the "target transfer ≤ 0.05" evidence is withdrawn as a specificity test. The pattern's relation-carrying role rests on cond IIA (0.84–0.85) and the replication R1 |
| R2-g | "path patching" is single-token activation patching that includes the final MLP | **Accepted** | renamed throughout |
| R2-h | per-head mediation Shapley values are cited but not computed | **Accepted** | the sentence is removed; only the computed ablation and path Shapley values remain |
| R2-i | the pairswap agreement 0.85–0.93 is inflated by unchanged decisions (Cohen's κ 0.79); in direction, the model picks Bayes's new answer in 81% of the cases where both change | **Accepted** | κ and the direction agreement are reported (reviewer computation) |
| R2-j | the pattern's invariance holds only for the listed exact symmetries; the residual symmetry group is larger (LN gains/biases fold into the following weights) | **Accepted** | stated |
| R2-k | `fwd.py` equivalence, the exact 5-head split, pairswap on-support and fairness, thresholded-readout additivity | — | supportive |

## Post hoc tests run while the reviews were in progress

**The unification hypothesis is rejected.** The hypothesis was that seed 12 reads the same source-token tag through values from other questions.
- All-token interchange (`logs/mech/alltok_s*.json`): patching the tag at other questions' tokens does nothing in tag-and-route seeds (IIA 0.01–0.02).
- In seed 12 it adds little (own-question tokens 0.37, all tokens 0.455).
- **Rejected.**

**Tag mean ablation (OP-5)** (`logs/mech/meanabl_s*.json`).

| seed(s) | mean ablation removes | conditional-mean ablation removes | random 2-D |
|---|---|---|---|
| 10, 11, 13 | 54–68% of the cross-question gain | 44–74% | ≈ 0 |
| 12 | 91% | 91% | ≈ 0 |

- In seed 12 the subspace is naturally necessary but not sufficient under interchange.
- Necessity without sufficiency fits seed 12 combining this subspace with other information. Which information is **open**.

## R3: dispositions (fresh review of the revised package and the post-review additions)

| # | finding | disposition | action |
|---|---|---|---|
| R3-1 | **E2 is implied by near-Bayes log-probabilities** (models are within 0.0007–0.0019 nats of Bayes NLL). A structureless noisy-Bayes null at the models' margin-error scale passes PP1. Rescaling Bayes log-probabilities by a margin-dependent factor keeps accuracy but drops sign agreement to 0.25. This is the known PFN result (Müller et al. 2022) | **Accepted** (major) | E2 is re-labelled **"a calibration consequence of KL-closeness to Bayes"**, not an independent law. My in-package check (`logs/pp1_reference_rules_posthoc.json`) agrees: accuracy alone does not imply E2 (the local plug-in has 0.733 accuracy but sign 0.84/0.85), while near-Bayes probabilities do (noisy Bayes with sd ≤ 1.0 gives ≥ 0.98). PP1 stays a passed prospective test. Its scientific content is "every implementation is KL-close to Bayes on the support, including at the non-monotone boundary". **Implementation-independence is a consequence of that, not an extra fact** |
| R3-2 | **E3 is known underspecification** (McCoy et al. 2020). The p_c = 0 shift is off support, so it does not count against H1 as an on-distribution account | **Accepted** (major) | E3 restated as an instance of underspecification. What remains (post hoc): its *direction* differs systematically by architecture (DeepSets global). H1-strong (follows the plug-in off support) is still falsified for the Transformers, but that is no longer framed as a failure of H1 proper |
| R3-3 | the novelty audit omits: Gunn et al. 2016 (paradox of unanimity: a small chance of systematic failure makes unanimity *lower* confidence; this is the qualitative content of the reversal and phase boundary); Pochampally et al. SIGMOD 2014 (fusing data with correlations); Makelov et al. ICLR 2024 (off-manifold subspace-patching illusions); Net2Net (function-preserving unit splitting, behind I5); Yosinski et al. 2014 (layer-transfer protocol behind PP3) | **Accepted** (major); citations as verified or flagged by the reviewer | novelty audit revised: phase-boundary novelty → "a quantitative instance of the known unanimity paradox"; I5 → "Net2Net-style split used against a preregistered criterion"; PP3 → "Yosinski-style transfer protocol, task-specific outcome" |
| R3-4 | U2 is a role-posterior statement. Inside its domain the group's own-candidate margin can still fall (0.3–0.5% of contexts) | **Accepted** (minor) | the atlas "help (monotone)" line is restated as "reliability posterior monotone"; the margin is governed by Theorem D |
| R3-5 | "The routing operation is the invariant object" names a modal implementation | **Accepted** (minor) | restated as "the most common relation-carrying operation (6/7 Transformer seeds); absent in seed 12 and in DeepSets" |
| R3-6 | the partner-resample "share ≈ 0.21" is 120–127% of the whole gain: it substitutes misleading context rather than removing it | **Accepted** (minor) | relabelled "substitution effect". The mean-ablation numbers (OP-5) are the removal measure |
| R3-7 | **PP3 arm (iv) is effectively from scratch** (the fresh layers moved only 8–12%). T0 and T5 are not matched controls for within-question pair parsing (a) or a cross-question scaffold (b). n = 1; arms unfinished; the [C] label is premature | **Accepted** (major) | PP3 is re-labelled **"partial, consistent with HC-relation but not discriminating it from (a)/(b)"**. **Post hoc follow-ups run:** (vii) freeze the T1 early layers and embeddings, reset the last layer; (viii) reset layer 1 only. Results in `GENERALIZATION_AND_EMERGENCE.md` §1a |
| R3-8 | optimiser state and data order are not differential confounds in PP3 | — | supportive |
| R3-9 | "the mechanism class survives fine-tuning" is near-trivial under lazy fine-tuning (weights move 8–16%) | **Accepted** (minor) | stated |
| R3-10 | **the T4-barrier hypothesis is contradicted** by in-package data: DeepSets learn T4 from scratch; `mlp_grouphist` fails from scratch but succeeds with curriculum; T1 with permuted sources fails although the monotone step is rewarded there | **Accepted** (major) | hypothesis withdrawn as the explanation. Restated: the barrier is **architecture-dependent** (absent for the pooled DeepSets; present for Transformer M and the per-group MLP). Its cause is open (OP-2) |
| R3-11 | **PP4's own-question reference is wrong** (it assumed a known pairing) | **Accepted** (major) | in-package (`logs/pp4_own_unknown.json`): own-question Bayes with unknown pairing is **0.634** (T1) and **0.559** (T4); majority vote 0.632 / 0.558. The model's 0.640 (T1) and 0.553 (T4) are **at the own-question level**: within-question evidence only, the same signature as the T4 barrier. No "regression below own" |
| R3-12, 13 | wording implies a preregistered success where PP5 failed; record inconsistencies | **Accepted** | fixed |

**Net effect of R3.** By the strict standard of mandate §14, the package's distinctive contribution beyond known results is **small**. What remains is:
1. A detailed, causally tested catalogue of implementations for this task, including the failures.
2. Exact restricted theorems: elementary, partly quantitative instances of known effects.
3. Methodological cautions demonstrated against preregistered criteria: off-support perm interventions, head-count criteria, off-manifold patching, the wrong own-question baselines.
4. A post hoc, architecture-linked direction of underspecified extrapolation (OP-1).

The atlas is revised to say this plainly.
