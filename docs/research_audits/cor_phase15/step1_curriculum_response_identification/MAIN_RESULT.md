# Are curriculum-dependent relational decision responses real computational differences or structured approximation errors?

**Post-review version** (pre-review text: commit 44f17a6; corrections **[R: id]** per REVIEWS_AND_ADJUDICATION.md). Base ad4c629;
cohort protocol 981af04; freeze db60086. Grades: [P] proved, [C] confirmed under a frozen criterion, [F] frozen criterion failed,
[E] exploratory / post hoc / descriptive. Investigation IV's frozen verdict (Outcome C) is unchanged.

## Direct answer

**The Investigation IV contrast did not replicate under the pre-registered test (Outcome C). What remains of it is a property of
fully trained, near-Bayes T1 models that disappears when the same parents are transferred for fewer steps. No curriculum-specific
downstream computation was identified; the null responses behave as approximation errors.**

1. **Replication [F: P1].** 6 + 6 independently trained models (fresh seeds, original protocols): coherence of Bayesian-null
   responses within T1 0.335, within P-count 0.186, Δ = 0.149 (Investigation IV 0.45). Permutation p = 0.039, but the model-bootstrap
   interval [−0.047, 0.341] includes 0; respecting the seed pairing of the two arms (same-seed runs share initial weights and
   transfer seed — an undocumented design coupling **[R: ST2]**) p = 0.0625. All 16 runs: Δ = 0.094 (p = 0.13). The cohort excludes an
   effect of Investigation IV's size and neither confirms nor excludes one of ≈ 0.15 **[R: PT]**.
2. **Training stage / quality [C: P2(b) as frozen; reading corrected R: ST1, PT].** The same T1 parents transferred for 500 steps
   (median KL 0.064 vs P-count 0.099) have coherence 0.086; within parents, coherence rises from 500 to 2000 transfer steps by 0.248
   (CI [0.079, 0.423]). Quality and training stage cannot be separated here (only full-budget T1 models reach KL < 0.025), and the
   frozen "explained by quality" verdict rests on a low-power non-rejection. The supported statement is: **T1 null-response coherence
   emerges late in transfer, as models converge toward the posterior.**
3. **Output space [E].** In probability space the cohort contrast is −0.06.
4. **What null responses are [P, elementary — R: FO5].** On natural Bayesian-null pairs a model's response is exactly the increment of
   its own log-odds error; their correlation measures similarity of approximation errors and cannot identify an algorithm
   (counterexamples, including two exact Bayes-mixture models with null responses correlated at −0.985 **[R: CX4]**).
5. **Operator form [E, restricted R: FO1, FO2, CX1].** Full-budget T1 models are close to the Bayes-mixture class (they fit exact Bayes
   alone at 0.88–0.96) and prefer it over a log-opinion pool; P-count models are not characterised by the declared classes (linear and
   log pooling fit them equally). **No cross-curriculum "same algorithm" claim is made**: the class test mainly measures closeness to
   Bayes.
6. **Routes [withdrawn R: ME1–ME3, CX9].** The pre-review claim of a selectively blockable readout "nuisance route" in T1 models is
   withdrawn: blocking it makes null responses 2.3–3.9× larger and moves models away from Bayes. T1 models combine a source-token route
   and a readout total-count route that partly cancel; null responses are imperfect cancellation — an approximation bias.
7. **The strongest curriculum-associated finding [E, post hoc — R: PT, ME, CX2].** Within the fitted mixture class, the reliability
   evidence for perfect agreement relative to 4/5 agreement, s = φ_rel(5) − φ_rel(4), is higher in every T1-lineage model than in every
   P-count model, at full budget and at 500 steps (exact p = 0.0022 each). Its sign is not identified for P-count models. s does not
   produce Bayes-null responses (Proposition 3), so it is a decision-relevant bias. It has not been prospectively tested.

## Answers to the final questions (§17)

1. **Replicates?** No (frozen criterion failed; the effect is about a third of the original and not distinguishable from 0 at
   model level).
2. **Robust to scaling and calibration?** Invariant to temperature (all KL-optimal T = 1.00); not robust to probability space.
3. **Explained by approximation error?** Consistent with it: null magnitude tracks R-error magnitude (r 0.99), coherence appears with
   convergence; but KL alone does not determine coherence (CX5), and stage and KL are confounded.
4. **Natural source interventions?** The primary test is input-level and support-checked; abstract interventions give a compatible
   (also non-significant) result [P4: non-rejection, R: ST7].
5. **Bayesian-relevant changes?** No curriculum difference in their errors was detected (Δ_R ≈ 0; a non-rejection).
6. **Identifiable downstream pathways?** Only a non-additive two-route composition in T1 models; no pathway that selectively produces
   the null responses.
7. **Curriculum or seed?** The coherence depends on training stage within T1; the s-ordering is curriculum-associated (post hoc).
8. **Common computation?** Not established across curricula; T1 models are near the Bayes-mixture class.
9. **Identifiable independently of the decision?** The V-dependent output map on the same-world surface; not its implementation.
10. **Invariant under representation changes?** Pearson/centred logits invariant to scale; the contrast is not invariant to the
    logit → probability change; the s-ordering is invariant to the gate parameterisation, its sign is not.
11. **Population level?** Only that, for these protocols, an effect as large as Investigation IV's is unlikely (jackknife upper bound
    ≈ 0.40 **[R: ST3]**).
12. **The corrected theorem.** Exact posterior matching plus pointwise interchange exactness on natural pairs is *sufficient* for equal
    candidate-margin responses; classification optimality is not; with T4's posterior floor average KL bounds average null responses,
    but no average notion bounds them pointwise **[R: FO4]**.
13. **Bias vs algorithm.** Relative to declared operator classes: same class with different parameters = bias; different best class at
    matched quality = different algorithm. Here the classes do not discriminate for P-count models, so the question is open for them.
14. **Prospectively falsified.** P1; P2(b)'s "curriculum effect persists"; P3(c) (narrowly); P3(b) at b* (not significant). P1b was
    found invalid as a test **[R: CX7]**.
15. **Unidentifiable.** KL vs training stage as the cause of T1 coherence; the implementation of either curriculum's operator; whether
    P-count models pool linearly or logarithmically.
16. **Beyond standard results.** A pre-registered, independently trained replication showing that a curriculum-looking contrast in a
    trained-Transformer causal analysis is a late-training effect of one curriculum; a precise statement of when Bayes-mixture
    operators are null-invariant (Proposition 3); a counterexample package showing why null-response correlations cannot identify
    algorithms; and one robust curriculum-associated parameter ordering (s) as a candidate for prospective testing. All restricted to
    this generator and these protocols.

## Not done

A P-count long-transfer arm and any P-count stage schedule (compute); re-examination of Investigation IV's necessity failures (§12);
attention/MLP decomposition at the readout.
