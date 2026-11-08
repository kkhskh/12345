# Replication results: T1→T4 curriculum mechanism (seeds 10–13)

**Protocol.** `REPLICATION_PROTOCOL_T4.md`, committed before training at c28e82a. The evaluation script was frozen at 13ef153 before its first run on any new seed.

**Raw outputs.**
- `logs/replication/eval_s1{0..3}.json`
- tally: `scripts/aggregate_replication.py` → `logs/replication/tally.json`

**Labels.**
- **[C]**: confirmed by the preregistered rule on new seeds.
- **[PH]**: post hoc diagnostic run after the confirmatory evaluation. These diagnostics explain failures; they do not change verdicts.

## 1. Preregistered verdict

| seed | R0 learn | R1 routing | R2 causal variable | R3 via route heads | R4 non-monotone | R5 (sec.) count-explained | localisation |
|---|---|---|---|---|---|---|---|
| 10 | pass | pass | pass | **FAIL** | pass | pass | key-localised |
| 11 | pass | pass | pass | pass | pass | pass | key-localised |
| 12 | pass | **FAIL** | **FAIL** | pass | **FAIL** | pass | none |
| 13 | pass | pass | pass | pass | pass | pass | key-localised |
| **k/4** | **4/4** | **3/4** | **3/4** | **3/4** | **3/4** | 4/4 | |

**By the preregistered rule** (R0–R4 each ≥ 3/4): **the causal account replicates [C].** Two caveats are part of the result:
- **Not every seed passes everything.** Only **2/4 seeds** (11 and 13) pass *every* primary criterion. Seeds 10 and 12 each fail at least one. The rule is per claim, and it was fixed in advance.
- **No new seed showed the distributed key/query implementation.** Three seeds are key-localised. Seed 12 is neither key- nor query-localised at the last-layer input (see §4). The distributed form remains a post hoc observation from seed 2 of the earlier curriculum models (and T1 seed 2).

## 2. What is confirmed [C], with numbers (seeds 10, 11 and 13 unless stated)

**Learnability under curriculum** (4/4).
- The T1 stage reaches 0.7438–0.7474, against Bayes 0.7470.
- The T4 stage reaches 0.7401–0.7427, against Bayes 0.7416 and own-question Bayes 0.5754.
- The model's argmax agrees with exact Bayes on 86–91% of evaluation questions.

**Reliability-gated routing** (3/4).
- The best last-layer head's group attention tracks the exact Bayes P(group reliable | X): R² 0.82, 0.88 and 0.96.
- Patching the last-layer readout attention row from cond pairs transfers the decision: IIA 0.84–0.86, KL recovered 0.98–0.99.

**A causal reliability variable at the input to the last layer** (3/4).
- The variable is a 2-D key subspace plus a 5-D query subspace, both identified from input contrast pairs only. Patched together (joint), they give:

| measure | value | comparison |
|---|---|---|
| cond IIA | 0.84–0.86 | mismatched-source null 0.44–0.45; random subspaces 0.06–0.07 |
| KL recovered | 0.99 | |
| Bayes-relative IIA (patched answer = exact Bayes answer on E′) | 0.84–0.86 | |
| perm IIA (aggregate votes preserved, relations changed) | 0.71–0.81 | |
| countpres change rate (relations preserved, aggregates changed) | 0.025–0.034 | |
| target change rate | 0.054–0.075 | |

- The key subspace alone carries it: IIA 0.76–0.84. The query subspace alone does not: 0.05–0.13.
- **Relations, not aggregates.** Under countpres, the model's decision changes only 3% of the time. Under perm, the patch transfers. So the model is sensitive to which sources agree and indifferent to vote totals.

**The effect runs through the reliability heads** (3/4).
- Freezing the two identified route heads during the joint patch cuts IIA to 0.14–0.29.
- Freezing the two matched control heads leaves it at 0.66–0.81, which is ≥ 0.7 × unfrozen.
- Freezing all heads gives 0.10–0.12.
- The patch reproduces 91–93% of the E→E′ shift in route-head group attention.

**Non-monotone use of agreement under misleading agreement** (3/4).
- The best route head's within-question log-attention to a source is grouped by the source's group leave-one-out agreement count c (0–5). It rises from c = 0 to c = 4 and **drops at c = 5**, when the group agreed on every other question, which is what colluders do.

| seed | m(0) … m(4), m(5) | drop m(4) − m(5) [95% CI] |
|---|---|---|
| 11 | −0.78 … 1.83, −1.24 | 3.07 [3.02, 3.13] |
| 13 | −1.16 … 1.92, −1.05 | 2.97 [2.92, 3.02] |
| 10 | −0.09 … 2.30, −2.18 | 4.48 [4.37, 4.59] |

- The exact Bayes leave-one-out log-odds show the same shape on the evaluation stream: −0.08, 1.44, 3.33, 5.50, 7.51, then −6.82 at c = 5.
- **Explanatory power.** Bayes reliability log-odds explain the attention better than any monotone function of the count:

| seed | R² Bayes log-odds | R² best monotone count function | R² flexible count lookup (upper bound, reported only) |
|---|---|---|---|
| 10 | 0.30 | 0.20 | 0.48 |
| 11 | 0.60 | 0.28 | 0.70 |
| 13 | 0.70 | 0.27 | 0.86 |

**Content: the tag is explained by agreement counts** (secondary, 4/4).
- A flexible lookup over (own group, the three leave-one-out counts) explains the 2-D key tag with R² 0.76–0.95.
- A flexible aggregate baseline (per-question histogram patterns × group) explains it with R² ≤ 0.01.
- **Interpretation, fixed in advance.** The tag is a *nonlinear function of agreement counts*. It is relational and not aggregate. Nothing here shows information beyond agreement counts, and no such claim is made. The flexible count lookup is the right null, and it succeeds.

**Baseline under the same curriculum.** `mlp_grouphist`, given the same T1→T4 curriculum, also learns most of T4: 0.697–0.729 vs Bayes 0.740 (from scratch it reached 0.556). The curriculum effect is therefore not specific to Transformers. The Transformer gets closer to Bayes.

## 3. Failures (preregistered; kept)

**Seed 10, R3.** Freezing the two route heads left IIA at 0.285, above the 0.25 limit.
- [PH] Seed 10 has **three** reliability heads, 0, 2 and 3, each with group-mass R² ≈ 0.80. Head 3 was assigned to the "control" set by the fixed top-2 rule.
- Freezing heads {0, 2, 3} gives IIA 0.105. Freezing the true non-reliability head {1} gives 0.865, i.e. no effect. Freezing {3} alone gives 0.671.
- The preregistered verdict stays **FAIL**. The diagnostic supports the routing account; the fixed head count did not fit this seed.

**Seed 12, R1, R2 and R4.** It reaches Bayes level (0.7401) through a different implementation.
- [PH] Two last-layer heads (0 and 1) read **other questions' tokens directly**, with 52% and 76% of their readout attention on them. The other two heads attend within question q, with weak reliability gating (group-mass R² 0.18 and 0.24).
- At the last-layer input, the full residual of question q's tokens carries only IIA 0.67. Part of the cross-question evidence is gathered *inside* the last layer rather than written into keys beforehand.
- The joint patch still has an effect (IIA 0.63). Freezing the within-question heads {2, 3} cuts it to 0.24; freezing the cross-question readers {0, 1} only to 0.54.
- Its routing head still shows the drop at c = 5 (1.71 [1.58, 1.83]). But the Bayes-vs-monotone R² margin is only 0.04, against the required 0.10.
- **Seed 12 is an existence proof** that the Bayes-level T4 solution does not *require* the tag-and-route localisation.

## 4. What remains post hoc or unconfirmed

- **The distributed key/query implementation** (old seed 2 and T1 seed 2): not observed in any new seed, so not confirmed. The joint patch was included so that it would have passed had it occurred.
- **Seed 12's cross-question-reader implementation:** observed once, post hoc.
- **Anything about T4 trained from scratch:** still 0/3 (preregistered failure, unchanged).

## 5. Answer to the replication question

Across independent seeds, Transformers trained with a T1→T4 curriculum learn reliability-sensitive candidate selection under misleading agreement. In 3 of 4 new seeds they implement it as follows:
- earlier layers write a low-dimensional reliability tag onto each source token;
- the tag is a nonlinear function of the group's agreement counts, rising with agreement except that perfect agreement (the colluder signature) is penalised;
- last-layer heads attend to each group in proportion to that tag;
- the attended votes determine the answer.

Each link is causally tested on new seeds:
- the tag patch transfers decisions toward the exact Bayes answer;
- it is insensitive to aggregate-preserving or relation-preserving changes, in the predicted directions;
- it stops working when the reliability heads' attention is frozen.

The fourth seed solves the task differently. So the account is a replicated *typical* mechanism, not a necessary one.

## 6. Affordable controls (main grid; frozen analyses, confirmation split; one seed each)

These were finished after the replication, as the protocol allowed. L-size training was not started.

- **T0 (reliability known; cross-question coherence has no Bayes value).**
  - Accuracy 0.7986, against Bayes = own-question 0.7992.
  - The decision barely depends on the other questions: only 8% of cond pairs change the argmax at all, essentially near-ties.
  - No reliability tag forms. D2 tag cond IIA 0.09, KL −0.01. Tag content: R² 0.08 for Bayes reliability against 0.39 for own values.
- **T5 (all sources exchangeable; the histogram is sufficient).**
  - Accuracy 0.6466, against Bayes 0.6464.
  - Only 6% of cond pairs change the argmax.
  - No tag (content R² 0.00).
  - **Reading T0 and T5 together:** the coherence computation appears only where it has Bayes value. It is driven by usefulness, not by an architectural bias.
- **T3 (R directly supervised; positive control).**
  - Accuracy 0.7389, against Bayes 0.7399.
  - The R head is decoded at the final episode token (R² 0.998).
  - It does **not** use the tag-and-route implementation: route-head group-mass R² ≤ 0.17 and tag IIA 0.01.
  - Cross-question evidence reaches readout q only *inside* the last layer: question-q tokens at its input carry IIA 0.06, while the last-layer readout pattern patch gives 0.76. This resembles replication seed 12.
  - **Conclusion:** supervising R changes *how* the task is solved. It is not evidence for or against the unsupervised mechanism (n = 1; T3 seed 1 was dropped for compute).
