# Are curriculum-dependent relational decision responses real computational differences or structured approximation errors?

**Pre-review version.** Base ad4c629; cohort protocol 981af04; freeze db60086. Grades: [P] proved, [C] confirmed under a frozen
criterion, [F] frozen criterion failed, [E] exploratory / post hoc / descriptive. Investigation IV's frozen verdict (Outcome C,
equivalence fails) is unchanged.

## Direct answer

**The Investigation IV contrast did not replicate under the pre-registered test, and where it appears it is explained by
approximation quality. The evidence is best described as a shared response factorization with curriculum-dependent approximation
error, plus a curriculum-specific nuisance route in T1 models.**

1. **Replication [F: P1, Outcome C].** In 6 + 6 independently trained models (fresh seeds, original protocols), within-T1 null-response
   coherence 0.335 vs within-P-count 0.186: Δ = 0.149 (Investigation IV: 0.45), permutation p = 0.039, but the model-bootstrap interval
   [−0.047, 0.341] includes 0. With all 16 runs Δ = 0.094 (p = 0.13).
2. **Approximation quality explains what remains [C: P2(b)].** Transferring the same T1 parents for 500 steps (median KL 0.064, closer
   to P-count's 0.099) removes the coherence entirely (0.086 vs 0.186; Δ_q = −0.10, CI [−0.28, 0.09]). The KL-adjusted contrast over
   overlapping qualities is −0.004 (p = 0.53). T1 coherence appears only when T1 models are fully transferred and near the exact
   posterior.
3. **Not robust to the output space [E].** In probability space the cohort contrast is −0.06.
4. **What the responses are [P].** On natural Bayesian-null pairs a model's response is exactly the increment of its own log-odds error
   (corrected Theorem C3, Corollary 1), so null-response coherence measures similarity of approximation errors and cannot by itself
   identify an algorithm (counterexamples in FORMAL_DECISION_EQUIVALENCE.md §5).
5. **Shared operator form [E].** Every full-budget T1 model and the better P-count models are described by the same Bayes-mixture
   factorization (count → configuration evidence → posterior mixture); none by additive weighted voting. P-count models sit further
   from it and are under-scaled (τ ≈ 0.75). Bayesian-relevant response errors are not curriculum-specific [C: P3(a)].
6. **A curriculum-specific route [C: P5(a)].** In four of six T1 parents the Bayes-null response enters layer 2 partly at the readout
   token (26–56 %) while the Bayes-relevant response enters at the source tokens; blocking the readout route removes 34–65 % of the null
   response and preserves 87–95 % of the relevant one. P-count models use one route for both. This is a selectively modifiable nuisance
   pathway, not the source of the coherence pattern.
7. **A candidate stable approximation bias [E, post hoc].** Inside the shared factorization, P-count models give *less* reliability
   evidence to perfect agreement than to 4/5 agreement (13/13 models); full-budget T1 models give more (12/13); cohort Fisher p = 0.0014.
   Not prospectively tested.

## Answers to the final questions (§17)

1. **Replicate?** No under the frozen criterion (direction recurs; interval includes 0; effect ≈ ⅓ of the original).
2. **Robust to scaling/calibration?** Invariant to temperature by construction (all fitted T* = 1.00); **not** robust to probability
   space (Δ −0.06).
3. **Explained by Bayes approximation error?** The cohort contrast is fully removed by quality matching; null-response magnitude
   tracks R-error magnitude (r 0.99) and KL (0.59).
4. **Natural source interventions?** Yes — the primary test is input-level and support-checked; abstract interventions agree with it
   (both intervals include 0) [P4].
5. **Bayesian-relevant changes?** Their errors show no curriculum difference (Δ_R ≈ 0).
6. **Identifiable downstream pathways?** A readout-local route carries part of the null response in most T1 parents (selective
   blocking); the operation inside layer 2 was not decomposed.
7. **Curriculum or seed?** The route separation is in 4/6 T1 and 0/6 P-count models (curriculum-associated, seed-variable); the
   coherence pattern depends on training stage within T1.
8. **Common computation beneath the errors?** Yes, relative to declared classes: the Bayes-mixture factorization.
9. **Formally identifiable independently of the final decision?** The V-dependent output map on the 36-vector same-world surface and its
   class membership; the internal implementation is not identifiable from V-level experiments (Proposition 2; Investigation IV C4).
10. **Invariant under representation changes?** Pearson/centred-logit statistics are invariant to per-model scale; the coherence
    contrast is not invariant to the logit→probability change; class membership is a property of the output map, invariant to site.
11. **Population-level?** Only that a large curriculum effect (≥ the Investigation IV size) is unlikely for these protocols; 6 + 6 runs
    support nothing finer.
12. **What the corrected theorem implies.** Equal candidate-margin responses require exact posterior matching *and* pointwise
    interchange exactness on natural pairs; classification-optimality, small KL, or average interchange accuracy imply nothing about
    null responses.
13. **Bias vs algorithm.** Relative to operator classes: same class, different parameters = approximation bias; different best class at
    matched quality = different algorithm. The data give the former.
14. **Prospectively falsified.** P1 (replication); P2(b)'s "curriculum effect persists"; P3(c)'s common-bias criterion (narrowly);
    P3(b)'s quality-robust operator difference (not significant).
15. **Unidentifiable.** The internal implementation of the shared operator; whether the T1 route separation is an operation or a
    redundant representation; anything about P-count training stages (no schedule was affordable).
16. **Beyond standard results.** The exact error-increment identity for Bayesian-null responses and the counterexample package; a
    pre-registered, independently replicated demonstration that a curriculum-looking contrast in a trained-Transformer causal analysis
    is a training-stage/quality effect; an operator-class decomposition (shared factorization, model-specific residual); a selectively
    blockable nuisance route. All are restricted to this generator and these protocols.

## What was not done

P-count long-transfer arm and any P-count training-stage schedule (compute); re-examination of Investigation IV's necessity failures
(§12) with new operators; attention-vs-MLP decomposition of the readout route.
