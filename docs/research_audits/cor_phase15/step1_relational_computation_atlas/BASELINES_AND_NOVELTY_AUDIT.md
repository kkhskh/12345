# Baselines and novelty audit (mandate §14)

**Method.** Each result is classified by the kind of contribution it makes. A result counts as "new" only if the cited prior work, read for what it actually says, does not already contain it.

**Literature checked in this step.**
- Dawid & Skene (1979), J. R. Stat. Soc. C 28(1):20–28. EM for observer error rates without ground truth.
- Dong, Berti-Equille & Srivastava (2009), "Integrating conflicting data: the role of source dependence", PVLDB 2(1):550–561. Bayesian detection of copying between sources, and iterative truth discovery that accounts for dependence.
- Xie, Raghunathan, Liang & Ma (ICLR 2022). In-context learning as implicit Bayesian inference over a latent concept.
- Geiger et al. (2023). Distributed alignment search (DAS), and causal abstraction as a framework for interchange interventions.
- arXiv:2507.08802. "Non-linear representation dilemma": unconstrained alignment maps make causal abstraction vacuous.
- Wang et al. (2022), IOI circuit in GPT-2 small: path patching, backup name mover heads.
- McGrath et al. (2023), "Hydra effect": self-repair under ablation.
- Zaheer et al. (2017), DeepSets (permutation-invariant pooling). Cited from memory, not re-verified in this step.

**Added after review R3** (reviewer-verified, or flagged as from memory):
- Gunn et al. 2016, "Too good to be true", Proc. R. Soc. A: the paradox of unanimity. A small probability of systematic failure makes unanimity *lower* confidence.
- Pochampally, Das Sarma, Dong, Meliou & Srivastava, SIGMOD 2014: fusing data with correlations.
- Makelov, Lange & Nanda, ICLR 2024: off-manifold subspace-patching illusions.
- Müller et al. 2022: Transformers can do Bayesian inference (PFNs).
- McCoy et al. 2020: underspecification and divergent generalisation of equally good models.
- From memory, not re-verified: Net2Net (Chen, Goodfellow & Shlens 2016), Yosinski et al. 2014 (layer transfer), Clemen & Winkler 1985 (dependent experts).

## 1. Contribution classification

| result | known mathematics in a Transformer? | new observation about how a Transformer implements it | new method | new theorem | new prospective prediction |
|---|---|---|---|---|---|
| Learned models approximate Bayes reliability inference | **known kind** (Xie 2022; Dawid–Skene for the target computation) | — | — | — | — |
| Agreement can signal copying rather than truth (T4 non-monotonicity) | **known** (Dong 2009 copy detection) | — | — | — | — |
| Theorem S (count sufficiency in the extended family) | elementary latent-class algebra; likely folklore | — | — | restated with the necessity counterexample | — |
| Theorem U (quasi-concavity of role posteriors in the agreement count; monotone extremes) | elementary (log-linear likelihood ratios ⇒ convex reciprocal) | — | — | **not found in the checked literature**; elementary. Value: makes the response's sign structure explicit | — |
| Phase boundary (p_c, ρ) for the reversal at full agreement | Bayes computation; **qualitatively the paradox of unanimity** (Gunn et al. 2016) | — | — | a quantitative instance for this family; computed exactly | — |
| Theorem D (margin-change identity) and sign rule D1 | elementary Bayes (a likelihood-ratio expectation) | — | — | restatement with a decision-level reading; no novelty claim | — |
| I2 (honest-copy vs colluder exchangeability) | standard non-identifiability argument (label symmetry) | — | — | **specific statement**; elementary | — |
| I5 (head split changes head-count criteria) | the split is Net2Net-style function-preserving widening; Hydra and backup heads show criteria depend on ablation choices | — | — | used exactly against a preregistered criterion. The construction is not new | — (the *near*-preserving prediction PP5 **failed**) |
| Tag-and-route (reliability tag on keys, then gated readout attention) | — | **new empirical observation** for this task; content-based attention gating is standard | — | — | (replicated, latent-reliability v2) |
| Decision-level sign rule holds across five implementations (PP1) | **implied by KL-closeness to Bayes** (PFN result; review R3-1) | a calibration check across implementations | — | — | frozen before testing, passed 10/10. Not distinctive |
| Off-support divergence between equally Bayes-optimal implementations, on the p_c = 0 shift (PP2 and post hoc) | **known underspecification** (McCoy et al. 2020) | the architecture-linked *direction* (DeepSets global) is a post hoc observation | — | — | PP2 (H1-strong) **falsified** as frozen |
| Seed-10 redundancy: additive margin contributions with super-additive accuracy necessity (thresholded readout) | related to backup heads and Hydra, but different (heads co-active, no downstream compensation) | **new small observation** | all-subset enumeration over heads instead of top-k. Shapley is *not* invariant under head splits (review R2-c) | — | — |
| Readout attention pattern as the most invariant relation-carrying object | — | **observation** (6/7 seeds) | — | — | — |
| On-support matched-aggregate intervention (`pairswap`) | — | — | **small methodological fix**: uses generator knowledge to stay on support | — | — |
| Curriculum controls (PP3) | Yosinski-style transfer protocol | task-specific outcome: the T1 last layer is unnecessary, the early layers necessary | — | — | PP3 frozen, partial (review R3-7) |

## 2. Distinctiveness, stated plainly

**What follows from Bayes (H1) alone:**
- the decision response law;
- its non-monotonicity;
- its phase boundaries;
- the I2 collapse.

None of these is distinctive of a "consistency theory".

**What follows from standard nonlinear computation plus attention:**
- the existence of tag-and-route, and of the other implementations;
- that a flexible count lookup explains the tag.

**What uses established interpretability tools:** all of the identification (interchange, path patching, DAS-like subspaces, Shapley over heads).

**What remains new, in the sense of this audit:**
**Revised after review R3: the remaining distinctive content is small.**

1. **The decision law (E2) is not distinctive.** It is a consequence of KL-closeness to Bayes, which is the expected outcome (PFN result). Its value is as a calibration check that holds at the non-monotone boundary in every implementation.
2. **The extrapolation divergence (E3) is known underspecification.** Its architecture-linked direction is the only regularity that Bayes does not predict. It is post hoc (OP-1).
3. **Head-criterion non-invariance (I5)** is a Net2Net-style construction used against a preregistered criterion. It is a methodological caution, not a theorem of new content.

**What would be needed to establish something distinctive beyond this:**
- (a) A predictor of off-support behaviour from internal structure, tested on held-out seeds (OP-1).
- (b) A relation family where a consistency-based description predicts a decision response that Bayes over the stated generator does *not*. By construction this cannot happen when the generator is known and the model is Bayes-optimal. It could only happen through model-specific biases, as in (a).

## 3. Strongest baseline per claim

| claim | strongest alternative explanation tested |
|---|---|
| models use relations, not aggregates | `mlp_hist` (aggregate-only) is capped at 0.638. G5 on-support relation-only edits move decisions. A flexible histogram-pattern regression explains the tag with R² ≤ 0.01 |
| the tag is a relation statistic | flexible lookup over (group, counts) explains R² 0.76–0.95. No claim of anything beyond counts |
| the decision response is Bayes | the monotone T1 plug-in fails at the boundary (13–17% sign agreement) |
| routing heads mediate | matched control heads (R3); all-subset Shapley; exact head split |
| on-distribution equivalence implies the same computation | **refuted** (PP2) |
