# Review R_math_causal: adversarial mathematical and causal review

**Reviewer:** independent, adversarial. I regenerated `VERDICTS.md` with `scripts/aggregate.py`, which ran without errors.

**Scripts.** My own checks are in `/tmp/rev/{check,tagtest,ktest,lintest}.py`. They are outside the package and import the package's `src/` read-only. Each run used ≤ 4k episodes and took under 5 CPU-minutes.

**Severity scale.** fatal = it invalidates the headline; major = a headline claim must change or be qualified; minor = wording, scope or reporting.

**Bottom line.**
- No fatal flaw. The Level-1 mathematics is correct.
- The causal core for seeds 0 and 1 survives every new attack I ran:
  - the decoded variable is reliability, not the answer (#13);
  - the tag is count-sufficient (#7);
  - the tag is relational, not linear in the reports (#12).
- Four things are wrong or overstated:
  1. `THEOREM.md` generalises a 1-of-3-seed secondary result to "the T1 M models".
  2. The Level-2 "equals Bayes" derivation is wrong for leave-one-out tags. It is also only argmax-level.
  3. The IIA null is far weaker than the true chance baseline.
  4. Two content controls are broken by a feature-construction bug. The conclusions survive once it is fixed.

---

## A. Mathematical review

### 1. Level 1 numbers (Theorem S, Theorem M, λ, T4, E1): verified, no error

- **Claim attacked.** Every Level-1 number in `MATH_VALIDATION.md` §2, §5 and `THEOREM.md` Level 1.
- **Evidence** (`check.py`).
  - P(agree | reliable) = 0.66; LR_agree = 1.98; LR_disagree = 0.51; λ = log(1.98/0.51) = **1.35644**.
  - The T1 closed form softmax(λ·n) matches `episode_posterior` to 3.6e−15.
  - The T4 closed form matches to 1.8e−15 (3000 episodes).
  - T4 with (n1, n2) = (6, 2), n0 = 0…6 gives 0.062, 0.205, 0.500, 0.795, 0.938, 0.983, 0.498, matching `math_checks2.txt`.
  - E1, analytic and brute force: (0.42523, 0.42523, 0.14954) and (0.21782, 0.21782, 0.56436).
- **Proof of S.** Correct. The key step is that every role's likelihood is constant within the "agree" class and within the "disagree" class. So class probabilities can replace specific-value likelihoods.
- **Proof of M.** Correct. It follows from Y_q ⟂ X_{−q} | R, X_q.
- **Severity.** None.

### 2. Corollary T4 wording is false in general

- **Claim attacked.** `THEOREM.md`: "A group that agrees on every question is as likely to be the colluder as the reliable group."
- **Counterexamples.**
  - n = (6, 2, 2) gives P(G0 reliable) = **0**: G0 must be the colluder.
  - n = (6, 6, 6) gives 1/3.
  - The 50/50 holds only when exactly two groups have n = Q, and even then it is 0.4978 vs 0.5000.
- **Severity.** Minor.
- **Fix.** "…when two groups both agree on every question, each is about equally likely to be the colluder."

### 3. Theorem S: unstated scope assumptions

- **Claim attacked.** That Theorem S lists all its assumptions.
- **Unstated assumptions:**
  - groups have size exactly 2 (for larger groups the sufficient statistic is the per-question *partition pattern*, not a single agreement bit);
  - α is equal within the reliable group;
  - the colluder's c_q is uniform and independent of Y.
- **Not needed:** a uniform R prior. The function-of-counts property holds for any prior.
- **Severity.** Minor (scope).
- **Fix.** Add these to the assumption list.

### 4. Level 2: "the routing weights equal the Bayes group posterior" holds only for episode-level counts

- **Claim attacked.** Level 2 lets τ use "leave-one-out or episode-level" counts, then claims that βa = λ makes the routing weights equal P(R | X).
- **Why it fails for leave-one-out counts.**
  - Leave-one-out counts give softmax weights π_{−q}.
  - Bayes needs π_{−q}(R)·M_q(R). The own-question agreement factor is missing.
- **Simulation** (`check.py`, 4000 episodes, Bayes accuracy 0.7437):

| routed vote with βa = λ, using | argmax agrees with Bayes | accuracy |
|---|---|---|
| episode-level counts | 97.2% | 0.741 |
| leave-one-out counts | 93.6% | 0.7345 |
| leave-one-out π × exact P(y \| X_q, R) | 96.1% | 0.736 |

- **The learned tag's alignment.** It is slightly better aligned with the episode target:
  - probe R² 0.878 vs 0.861 (`T1_M_K3_s0_tag_conf.json`);
  - my regression on U_D2: 0.829 vs 0.795 (s0) and 0.691 vs 0.654 (s1).
- **Severity.** Major for the derivation; minor for the empirical story.
- **Fix.** State that the Bayes equality needs episode-level counts, or a key term in the own agreement A_q. Report which target the learned tag matches.

### 5. Level 2: "ℓ is the Bayes mixture" conflates a logit-linear vote with a probability mixture

- **Claim attacked.** Prediction step 3: "ℓ is then a P(R | X)-weighted vote. That is the Bayes mixture."
- **Why it is only argmax-level.** The model applies softmax(ℓ) with ℓ linear in votes. Bayes is linear in *probabilities* P(y | X_q, g). The map from a group's vote count to P(y | X_q, g) is not affine: vote = 0 maps to 0.015 or 0.059, depending on the partner pattern.
- **Further evidence:**
  - The routed vote has *exact* ties in 5.4% of question instances.
  - Accuracy does not depend on βa: βa ∈ {0.5, 0.9, 1.2, 1.357} gives accuracy 0.741–0.743 with episode counts. So "βa = λ" is not what makes the construction reach Bayes accuracy.
  - The model is calibrated (nll 0.589 vs Bayes 0.588), which Level 2 does not predict.
  - Model log-probs vs Bayes: R² 0.996. Model vs the routed vote: only R² 0.945 (`logs/posthoc/*.json`). So about 5% of logit variance comes from paths outside the Level-2 construction.
- **Severity.** Major (wording and scope of the Level-2 → Level-3 bridge).
- **Fix.** Say "same argmax as Bayes in about 97% of cases". Drop the calibration and "is the Bayes mixture" reading. Report the residual paths.

### 6. "βa = λ is not tested because tag units are arbitrary" is wrong

- **Claim attacked.** `THEOREM.md`, "What is not shown" item 1.
- **Why it is testable.** β is log-attention per tag unit, and a is tag units per count. Their product βa is log-attention per count: unit-free and directly comparable to λ.
- **Estimate from the logged numbers** (slope of the per-count tag means × head β):

| seed | βa, first route head | βa, second route head |
|---|---|---|
| s0 | ≈ 0.87 | ≈ 0.98 |
| s1 | ≈ 1.20 | ≈ 1.29 |
| s2 | ≈ 0.58 | ≈ 0.35 |

  All are below λ = 1.357. This is rough: the tag means by count are confounded by the other groups' counts.
- **Severity.** Minor.
- **Fix.** Regress the head's group log-attention difference directly on the count difference. Report it against λ, and note (#5) that accuracy is insensitive to it.

### 7. The target intervention is mislabeled as the count-preserving-X_{−q} test; I ran the real test

- **Claim attacked.**
  - Level 2: the patch "changes nothing if X_{−q} is changed in a way that preserves the counts. In particular … under the target intervention".
  - Level-3 table row "no effect under count-preserving target changes".
- **Why it is mislabeled.** `target` changes X_q, not X_{−q}. It is a value-content control, not the Theorem-S prediction.
- **My test** (`tagtest.py`). Resample X_{−q} while keeping every group's agree/disagree pattern, which leaves Bayes(q) exactly unchanged:

| seed | model argmax change rate | mean KL | tag RMS change (cond pairs = 100%) |
|---|---|---|---|
| s0 | 3.96% | ≤ 8e−4 | 26% (1.14 vs 4.32) |
| s1 | 3.94% | ≤ 8e−4 | 6% |
| s2 | 3.24% | ≤ 8e−4 | 6% |

  Cond pairs, by comparison, change the model's argmax in 26–27% of cases.
- **Reading.** Supportive. The tag and the model are close to count-sufficient, with a small non-count component in seed 0.
- **Severity.** Minor (mislabel); the evidence is missing from the package.
- **Fix.** Add this pair kind ("count-preserving") to the battery and relabel `target`.

### 8. Level 2 needs unstated attention assumptions

- **Claim attacked.** That Level 2 states every assumption it needs.
- **Unstated assumptions:**
  - The route head's softmax must put its mass on question q's 6 source tokens. In reality the softmax runs over all 43 tokens, so this needs a question-match QK term.
  - The c_s must be equal within a group.
- **Masking in the code.** `posthoc.py` lines 108–116 renormalise attention within q's sources. That hides any mass on other tokens.
- **Severity.** Minor.
- **Fix.** List the assumptions. Report the route heads' attention mass on q's sources vs elsewhere.

---

## B. Level separation and overclaims

### 9. Level 3 is generalised to "the T1 M models" against the package's own verdict

- **Claim attacked.** `THEOREM.md`: "What the learned result is. In the T1 M models the answer is produced by …"
- **Evidence from `VERDICTS.md`.** `TagRoute_D2` holds in 1 of 3 seeds. The secondary answer is **NO (scoped)**.
  - Seed 2 fails K2′, K3′, K5′ and R1′: tag D2 cond IIA 0.31, drop_frac 0.31, routing shift 0.34.
  - Seed 1 fails K4′: target change rate 0.104 against a threshold of 0.10.
  - The Level-3 table reports seed 0 only.
- **Broken references.** `THEOREM.md` cites `MAIN_ANSWER.md` and `REVIEWS/`, and neither exists.
- **The seed-2 readout pathway is post hoc.** The claim that seed 2 also carries reliability on the readout query comes from `posthoc_joint.py` (joint IIA 0.78). That analysis was not registered.
- **Severity.** Major.
- **Fix.** Report the result as "seed 0 meets the registered tag-and-route criteria; seed 1 misses one specificity threshold by 0.004 (SE ≈ 0.003); seed 2 uses a mixed key/query pathway (post hoc)". Give the table for all seeds.

### 10. "Reliability is decodable at the readout but causally inert" is trivial in seeds 0–1 and false in seed 2

- **Claim attacked.** `THEOREM.md`: "linearly decodable there (R² 0.65–0.81) but causally inert (IIA ≈ 0.01)".
- **Seeds 0 and 1.** s* = resid_post2 is the final residual. There, any direction orthogonal to the unembedding is inert by construction.
  - At the real input to the last layer (resid_pre2, readout), the probe R² is only 0.48 and 0.12.
  - Decodability at resid_post2 is plausibly a by-product of reliability-weighted routing.
- **Seed 2.** s* = resid_pre2 and the readout probe R² is 0.947. The readout is **not** causally inert:
  - full-readout IIA 0.48, kl_recovered 0.75;
  - query-side D2 patch (k = 2) IIA 0.47;
  - full-readout ablation drop_frac 0.55.
  Only the *probe* subspace is inert there (IIA 0.107).
- **Severity.** Major (interpretation).
- **Fix.** Restrict the claim. In seeds 0–1, the readout token at the last-layer input carries little causally used reliability (full-residual IIA 0.08 and 0.03). In seed 2 it does, in a direction other than the probe's.

### 11. Two further limits: cross-question reuse is architectural, and the tag's content differs by seed

- **Claim attacked.** K6′/P6 as evidence of reuse, and "the tag is the Bayes reliability logit".
- **Reuse across questions (K6′/P6).**
  - All token positions share weights, and question identity enters only as an additive embedding.
  - A direction fitted on questions 0–2 is therefore expected to transfer to questions 3–5.
  - This is not reuse in the mandate's sense: one variable serving several decisions or tasks.
- **The tag's content differs by seed** (`logs/posthoc`):

| seed | R², own-group count | R², Bayes reliability logit |
|---|---|---|
| s0 | 0.61 | 0.79 |
| s1 | 0.89 | 0.65 |

  Seed 1's tag is mainly the own-group count. Normalisation across groups then happens in the softmax, which is closer to the Level-2 construction. Seed 0's tag already includes some cross-group normalisation.
- **Severity.** Minor.
- **Fix.** Report both regressions per seed. Describe K6′ as a weight-sharing consistency check.

---

## C. Causal review

### 12. Content-control bug: features identical across the 6 source tokens

- **Claim attacked.**
  - "Linear in all raw reports (non-relational): R² 0.04";
  - "all_group_counts_onehot" R² 0.009;
  - K7's "other_question_histograms" R² ≈ 1e−4.
- **The bug.**
  - In `scripts/posthoc.py` lines 75 and 79, the features are `np.repeat(<per-episode features>, 6, 0)`. They are identical for all 6 source tokens of a question and carry no token or group identity.
  - The regression therefore cannot predict token-level tags that differ by group, so R² ≈ 0 is guaranteed.
  - `scripts/analyze_tag.py` line 143 has the same problem for the histogram control in K7. For histograms this happens to be harmless, because histograms are symmetric under group permutation and cannot carry group-specific reliability anyway.
- **Re-run with token-specific regressions** (`lintest.py`, one regression per (q, s), 3000 fit / 3000 test episodes):

| seed | linear in report one-hots | linear in pairwise equalities |
|---|---|---|
| s0 | 0.065 | **0.926** |
| s1 | −0.001 | **0.984** |

  The conclusion survives and gets stronger: the tag is almost exactly linear in pairwise-equality indicators, which are relational.
- **Severity.** Major (an invalid control as reported); the conclusion is unaffected.
- **Fix.** Use token-specific (or token-identity-interacted) features. Replace the reported numbers.

### 13. Does the decoder identify the target rather than reliability? No

- **Claim attacked.** That U_D2 is a reliability tag rather than an "agrees with the answer" feature. Such a feature would also be invariant under `target` shifts.
- **Evidence** (`tagtest.py`).
  - The tag regressed on Bayes P(Y_q = own report | X) gives R² 0.20, 0.19 and 0.10.
  - Adding it to the reliability logit raises R² by ≤ 0.008.
  - On disagreeing pairs, the within-pair tag difference is small: SD 0.21, 0.23 and 0.40 of the tag SD.
  - It is uncorrelated with the within-pair difference in answer mass: r = −0.07, 0.13, 0.04.
- **Severity.** None. The attack fails.
- **Fix.** Add this test to the package.

### 14. The IIA null is far too weak

- **Claim attacked.** The meaning of "IIA 0.83 ≥ random + 0.3", where random-direction IIA is about 0.03.
- **Mismatched-source control** (`ktest.py`). Patch the U_D2 tag from an *independent* cond draw E″, which shares X_q, and score it against E′.
  - IIA = **0.449** (s0) and 0.206 (s2).
  - On informative pairs with K = 3, given X_q, the alternative answer is often forced.
  - Chance-corrected IIA for seed 0 is (0.848 − 0.449)/(1 − 0.449) ≈ **0.72**.
  - kl_recovered discriminates much better: 0.16 mismatched vs 0.99 matched.
- **Severity.** Major for the interpretation of the thresholds; the s0 and s1 conclusions stand.
- **Fix.** Report the mismatched-source baseline and chance-corrected IIA for every criterion. Treat kl_recovered as primary.

### 15. Informative-pair selection: sound

- **Claim attacked.** That restricting IIA to informative pairs inflates it.
- **Evidence.**
  - About 27% of cond pairs are informative.
  - In seed 0, the patch changes the argmax in 24.2% of all pairs against 26.6% informative × 0.83. That bounds spurious flips on non-informative pairs at about 3%.
  - kl_recovered uses a median, which flatters: median 0.99 vs mean 0.92.
- **Severity.** Minor.
- **Fix.** Report the false-flip rate on non-informative pairs, and the mean kl_recovered next to the median.

### 16. Is the cond construction on-manifold?

- **Claim attacked.** That cond pairs, and the patched states built from them, are on-manifold.
- **The source episode is on-manifold.** R′ ~ P(R | X_q), then X_{−q} ~ P(X_{−q} | R′), samples P(X_{−q} | X_q) exactly, because X_q ⟂ X_{−q} | R (`analysis.py` lines 128–132). So E′ is a natural episode.
- **The patched hybrid is not.** Question q's tags come from E′ while every other token is from E.
  - The 1-D values lie in their natural range, so magnitude artifacts are unlikely.
  - In seed 2 the hybrid sets the query side (readout) and the key side (sources) against each other. That likely explains part of the low key-only IIA (0.33 key only vs 0.78 joint).
  - Perm and splice tag results are not split by on/off-manifold, although `analyze_model.py` does split them.
- **Severity.** Minor.
- **Fix.** Split the tag results by the on/off log P(E′) rule. For seed 2, report key-only patches with the query held from E′.

### 17. The target-pair specificity threshold sits near the floor

- **Claim attacked.** K4′ as a discriminating specificity test.
- **Evidence.** Random 1-D directions already give a target change rate of 0.048 (s0), 0.075 (s1) and 0.031 (s2).
  - D2's excess over random is 0.03.
  - The seed-1 failure (0.104) is within about 1.3 SE of the threshold.
  - The test is valid only as a "no own-value content" control. See #7 for the count-preserving test it is mistaken for.
- **Severity.** Minor.
- **Fix.** Report the excess over the random floor with CIs.

### 18. drop_frac > 1 is the expected effect of injecting wrong tags, not of removing information

- **Claim attacked.** That drop_frac 1.79 and 2.09 (K5′) measure the necessity fraction.
- **Wrong-tag floor** (`check.py`):
  - Bayes using an independent episode's reliability with own-question evidence: accuracy 0.607.
  - The routed vote with counts resampled from an independent episode: 0.578.
  - The model with U_D2 resample-ablated: 0.587 (s0) and 0.564 (s1).
  - The predicted drop_frac for a perfect, sole-carrier tag replaced by a coherent wrong one is (0.740 − 0.578)/(0.740 − 0.655) ≈ 1.9.
- **Reading.** The numbers are evidence that the model *reads* the tag, and possibly that the tag is the sole carrier. drop_frac is not interpretable as "fraction of the cross-question benefit that is necessary". The frozen D2_contrast drop_frac of 4.27 at resid_post2 is trivial, because that site is the output.
- **Severity.** Minor; the interpretation should change.
- **Fix.** Add mean ablation, or ablation to the tag's conditional mean given X_q. The expected accuracy is the own-question level, 0.655 (drop_frac ≈ 1). Report the wrong-tag floor beside the resample result.

### 19. D2 circularity: the data streams are disjoint, but the test is an in-distribution replication

- **Claim attacked.** That U_D2's cond transfer is independent evidence.
- **The streams are disjoint.** All streams use separate `default_rng` integer seeds.

| stream | seeds |
|---|---|
| analyze_tag conf: fit episodes | 827000 |
| analyze_tag conf: fit pairs | 827060–65 |
| analyze_tag conf: test episodes | 827002 |
| analyze_tag conf: test pairs | 827100–105 (cond) |
| analyze_tag conf: ablation episodes | 827005 |
| posthoc basis | dev 1,049,000 |
| posthoc tests | 555,000+ |
| training | 1,000,003·(s+1) + 7919·task + 31K, e.g. 1,008,015 |

- **The construction is the same.** Testing D2 on fresh cond pairs replicates its fitting procedure. Its first PC explains 81%, 85% and 57% of the cond-difference variance. D2 is therefore close to "the principal axis of whatever context does to q's source tokens". Given a working full-source patch (IIA 0.89 and 0.94), its transfer is close to guaranteed whenever that effect is low-rank: D2/full = 0.83/0.89 and 0.83/0.94.
- **Where the non-trivial evidence lies.**
  - content: #12 and #13;
  - count-sufficiency: #7;
  - generalisation to perm/splice pairs, held-out α, and questions 3–5.
- **A mixing issue.** The Level-3 table mixes a conf-refit basis (`analyze_tag`) with a dev-fit basis (`posthoc`).
- **Severity.** Minor.
- **Fix.** State this explicitly. Report the PC spectrum. Fit D2 on splice pairs and test on cond pairs (and the reverse).

### 20. The bypass test answers only whether the tag acts through attention scores (QK) or through values (OV)

- **Claim attacked.** That freezing the route-head patterns (IIA 0.83 → 0.07) settles bypass assumption (i).
- **What the test can show.** At resid_pre2, source tokens reach the readout output only through layer-2 attention. Freezing the patterns therefore only distinguishes the QK path from the OV path. It is not a test of redundant pathways upstream.
- **Upstream evidence elsewhere.** Readout full-residual IIA at resid_pre2 is 0.08, 0.03 and 0.48. The gap between q-token full and the tag patch (0.95 vs 0.83) leaves about 12% of informative-pair effect unexplained in seeds 0–1. Seed 2 has a substantial query-side pathway.
- **Selection.** The route heads are picked on the same data used to test them (`posthoc.py` line 49). This is minor with only 4 heads.
- **Severity.** Minor; the claim of conclusiveness should be narrowed.
- **Fix.** Freeze patterns while patching the *full* q-token residual. Run the converse: patch only the route-head QK input with the tag held fixed.

### 21. Seed 2's failure is not caused by the preregistered k_tag = 1

- **Claim attacked.** That seed 2 would pass with a 2-D tag. PC2 explains 30% of the variance there, so this is a fair question.
- **Evidence** (`ktest.py`, seed 2):

| patch | IIA |
|---|---|
| PC1 | 0.29 |
| PC2 only | 0.07 |
| PC1 + PC2 | 0.36 |
| PC1–4 | 0.38 |
| full residual | 0.46 |

- **Reading.** The failure reflects a different (query-side) pathway, not the choice of dimension. The k = 1 rule is not the cause.
- **Severity.** None.
