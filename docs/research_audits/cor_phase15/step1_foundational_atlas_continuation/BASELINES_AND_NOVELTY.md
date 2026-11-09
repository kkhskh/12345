# Baselines and novelty (mandate §15)

**Scope.** Each continuation result is checked against the closest existing work. The atlas novelty audit (`../step1_relational_computation_atlas/BASELINES_AND_NOVELTY_AUDIT.md`) still stands for the atlas results. The atlas's missing-literature list (Gunn 2016, Pochampally 2014, Makelov 2024, Net2Net, Yosinski 2014) is incorporated where it bears on a result.

Verdict scale:
- **known**: the result is in the literature in substance;
- **known in principle**: a standard tool applied to a new object;
- **instance**: a new instance of a known phenomenon;
- **possibly new**: not found in the searches made; limited search.

## Searches made (2026-11-09)

- In-context learning, task diversity and Bayesian estimators (Raventós et al. 2023, arXiv:2306.15063, and follow-ups).
- Head-level and Shapley attribution in Transformers; transformer parameter symmetries and quotients (da Silva, Dangel and Oore 2025, arXiv:2505.05409: the GL(d_h) value–output symmetry; Toward Manifest Relationality 2026, arXiv:2602.18948).
- Mean versus resample ablation versus patching (Heimersheim and Nanda 2024, arXiv:2404.15255; Zhang and Nanda 2024; Chan et al. 2022, causal scrubbing; the mechanistic-validity criteria pages).
- Truth discovery with copying (Dong, Berti-Equille and Srivastava 2009, VLDB "Integrating conflicting data: the role of source dependence"; Pochampally et al. 2014, "Fusing data with correlations").
- In-context reliability or annotator inference (Dawid and Skene 1979 and descendants).

The searches were limited. Absence of a hit is weak evidence.

## Result-by-result

| result (continuation) | closest existing work | verdict |
|---|---|---|
| C1–C3 pointwise certificates (calibration error versus true effect size) | elementary perturbation bounds; margin-based robustness certificates | **known in principle** |
| C4 density-ratio and Markov guarantee for intervention signs | covariate-shift bounds (importance weighting); Pinsker + Markov | **known in principle**, applied to intervention signs |
| C5 temperature or ECE irrelevance for signs; C6 accuracy-only counterexample | folklore that calibration ≠ ranking | **known in principle**. The explicit matched-accuracy demonstration on an intervention task is an *instance* |
| "PP1 sign agreement is a consequence of KL-closeness, not of mechanism" | atlas R3-1; prior-fitted networks (PFNs) approximate posterior predictive distributions | **known** |
| PP2: off-support divergence between Bayes-close models | underspecification (D'Amour et al. 2020; McCoy et al. 2020); low task diversity gives a Bayes estimator under the pretraining prior (Raventós et al. 2023) | **known phenomenon**. The *direction* (DeepSets follow the training plug-in, i.e. the Raventós-type "pretraining-prior Bayes"; Transformers mostly do not) is an instance. A frozen *internal* predictor of the direction is being tested (P-A) |
| PP2 internal signature (the colluder code picks the argmax-agreement group off support) | probing (Alain and Bengio 2016); no prior work found on this generator | **possibly new as an observation**; descriptive only |
| Seed 12: necessary-not-sufficient tag; additive head channels | AND/OR circuits under resample ablation versus patching (Nanda, "How to think about activation patching"); patching versus ablation disagreement (Heimersheim and Nanda 2024); subspace patching illusions (Makelov et al. 2024) | **instance** of known necessity/sufficiency dissociation. The specific reconstruction is new to this system |
| R1: quotient-game Shapley over exact pattern classes is invariant under exact head splits and gauge | Shapley values with coalition structures (Owen 1977); Net2Net (Chen et al. 2016) function-preserving widening; GL(d_h) symmetry (da Silva et al. 2025) | **known in principle**: Shapley of a quotient game is a standard construction. Applying it to fix the atlas I5 / R2-c non-invariance is an instance |
| relative-count abstraction test (uniform shift) | invariance tests for abstraction (causal abstraction literature: Geiger et al.) | **instance** |
| ρ* role-swap nonidentifiability line | atlas I2; label-switching nonidentifiability in latent-class models | **known in principle** |
| V1 "a copying pair is one vote" threshold | Dong et al. 2009: copiers' votes are discounted in the ideal vote count via the dependence graph; Pochampally et al. 2014 | **known**: the exact threshold for the K-ary symmetric case is a direct corollary |
| S needs (O); U fails for unions of roles | mixture models of binomials are not log-concave | **known in principle**; corrections to the atlas's own scope |
| ρ → 1 sensitivity and support change | standard: likelihood with a hard constraint at the boundary | **known in principle** |
| hidden-pairing ≤ 1/3 bound under T1 | atlas I1 | **known** (corollary) |
| overlap mis-explained under a one-block class | model misspecification (KL projection onto the model class; White 1982) | **known in principle** |
| OD1: discrepancy ranking = Bayes iff the semantics are trivial (the "if" direction proved) | majority vote is the MLE for i.i.d. symmetric annotators of equal accuracy; Dawid–Skene weighted vote otherwise | **known**: majority vote optimality for exchangeable equal-accuracy sources is textbook |
| P-local / P-count / P-group pretraining dissection of transferred features | Yosinski et al. 2014 (feature transferability); curriculum and transfer literature | method: **known**; results: pending |
| GATE / TRANS beyond-reliability tasks | multi-hop reasoning and compositional generalisation; induction and composition circuits (Elhage et al. 2021) | pending |

## Baselines used for each claim

| claim | baselines it was required to beat or match |
|---|---|
| P-A internal predictor | architecture indicator, T4 KL, accuracy, ECE, constant, random feature, ridge on all four (frozen) |
| PP1 sign theorems | constructed models matched on accuracy (TEMP, CX1) and on KL (CX2, CX3) |
| seed-12 operations | seed 10 run as a control under identical conditions; random 2-D subspace (atlas OP-5) |
| R1 | head-level Shapley, which is shown to move under the split |
| abstraction test | exact Bayes (invariance 0) and a single-group shift (positive control) |
| OD1 | the Bayes argmax under five generators that violate one assumption each |

## Net novelty statement

Nothing in this continuation is a new *mathematical* theory. The contributions are:
1. precise scope corrections to the atlas's own theorems;
2. a cleaner separation of *what produces* PP1-type agreement (KL-closeness with W ≈ 1) from *what would* discriminate implementations;
3. a reconciliation of the seed-12 necessity/sufficiency puzzle;
4. a frozen prospective test of an internal predictor of extrapolation direction (P-A);
5. a characterisation of the O/D bridge as holding exactly when source semantics are trivial.

Item 4 is the only candidate for a distinctive empirical prediction (failure F4). Its fate is recorded in PROSPECTIVE_PREDICTIONS.md.
