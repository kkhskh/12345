# Preregistration: independent replication of the T1→T4 curriculum mechanism

**Status.** This document was written and committed **before**:
- any model with seeds 10–13 existed;
- any data from the replication streams were generated.

**What it replicates.** The **post hoc** discovery in `MAIN_ANSWER.md` §2 (T4 curriculum seeds 0–2), which had three parts:
- T4 models fine-tuned from T1 learn T4 to Bayes level;
- they route last-layer attention by a reliability variable that is non-monotone in agreement;
- that variable, localised on source-token keys *or* split across keys and the readout query, causally drives candidate selection.

The seed 0–2 results stay labelled post hoc. Only results from this protocol may be called *confirmed*.

**Original failures are kept.** Every earlier preregistered result is unchanged:
- PROTOCOL.md primary answer PARTIAL;
- addendum-2 secondary answer NO (scoped);
- T4 from scratch 0/3;
- the T1 seed-1 K4′ miss.

## 1. Models and training (frozen)

**Seeds.** 10, 11, 12 and 13. None of these seeds has been used for any purpose before.

**No replacement.** Every seed is trained and evaluated. A seed that fails a stage counts as a failure.

**Stage 1 (T1).** `scripts/train_run.py T1 M <seed> 6000 checkpoints/replication 3 tf`. This is identical to the main grid:
- architecture M (d 64, 3 layers, 4 heads);
- AdamW, lr 1e-3, warmup 200, cosine decay, weight decay 1e-4;
- batch 256;
- training data drawn from the fixed seed-indexed stream.

**Stage 2 (T4).** `scripts/train_curriculum.py <seed> 3000 checkpoints/replication <stage-1 checkpoint>`.
- Initialise from the stage-1 checkpoint.
- The R head (WR) is re-initialised; it is unused, because T4 trains no R supervision.
- Same optimiser and schedule, run over 3000 steps.

**Model selection.** None. There is no checkpoint selection, early stopping or hyperparameter search. The final parameters are evaluated.

**Baseline.** `mlp_grouphist` trained with the same curriculum on seeds 10–13: 6000 T1 steps, then 3000 T4 steps. It is reported, not a criterion.

## 2. Data streams (frozen)

| stream | base | use |
|---|---|---|
| identification | 2,100,000 | defines every subspace and head set below; uses inputs and Bayes quantities only, never model outputs on evaluation pairs |
| evaluation | 3,300,000 | every criterion |

## 3. Identification (frozen; all at the input of the last layer, `resid_pre2`, with no site selection)

- **Key subspace U_key** (k = 2):
  - take cond pairs (§4) on 1000 identification episodes × 6 questions;
  - compute the activation differences at the 6 source tokens of question q;
  - U_key is the top-2 PCA directions of those differences.
- **Query subspace U_qry** (k = 5 = |R| − 1): the top-5 PCA directions of the readout-token-q differences from the same pairs.
- **Joint mediator:** U_key at the source tokens of q *and* U_qry at readout q, patched together.
- **Route heads:** the 2 last-layer heads with the highest group-mass R², computed on 3000 identification episodes.
  - Group mass: the readout-q attention mass on each group's sources within question q, renormalised over the 3 groups.
  - R² is measured against the Bayes episode-level P(group reliable | X).
- **Control heads:** the other 2 last-layer heads.
- **Random control:** 8 random orthonormal bases, of dimension 2 (keys) and 5 (query).

## 4. Pairs (evaluation stream; 1800 base episodes × 6 target questions)

| kind | construction | purpose |
|---|---|---|
| cond | X′_q = X_q; X′_{−q} ~ P(X_{−q} \| X_q) | on-manifold change of the reliability evidence |
| perm | each other question's 6 reports permuted across sources | **aggregate votes preserved, group relationships changed** |
| countpres | each other question regenerated with the *same within-pair agreement pattern* and fresh values (agreeing pairs get a new common value; disagreeing pairs a new ordered pair of distinct values) | **relationships (all agreement counts) preserved, aggregates changed** |
| target | X′_q = X_q + shift (mod K) | own-value control |
| mismatch | source activations taken from the cond pair of a *different* base episode | null for IIA (review MC14) |

## 5. Metrics

- **IIA:** model-relative, on informative pairs, where the model's argmax differs between E and E′.
- **IIA_bayes:** the fraction of informative pairs whose patched argmax equals the **exact Bayes** argmax on E′.
- **kl_recovered:** the median of 1 − KL(m(E′) ‖ patched) / KL(m(E′) ‖ m(E)), over pairs with KL(m(E′) ‖ m(E)) > 1e−3.
- **change_rate:** the fraction of pairs whose patched argmax ≠ the base argmax.

Patching follows `src/analysis.py` and `scripts/posthoc_joint.py`. Head freezing replaces the frozen heads' last-layer attention pattern with the base episode's pattern for every query row.

## 6. Criteria (per new seed; evaluation stream)

| id | claim | pass if |
|---|---|---|
| R0 | curriculum learnability | stage-1 T1 acc ≥ own + 0.5·(Bayes − own) on T1, **and** stage-2 T4 acc ≥ own + 0.5·(Bayes − own) on T4 (20,000 evaluation episodes each) |
| R1 | reliability routing | max route-head group-mass R² ≥ 0.5 **and** last-layer all-heads readout-row pattern patch (cond) IIA ≥ 0.7 and kl_recovered ≥ 0.5 |
| R2 | causal reliability variable, localisation-agnostic | joint patch: cond IIA ≥ 0.7, kl_recovered ≥ 0.5, IIA ≥ mismatch IIA + 0.2, IIA ≥ random-joint IIA + 0.3; perm IIA ≥ 0.6; countpres change_rate ≤ 0.10; target change_rate ≤ 0.10 |
| R3 | effect runs through the routing heads | joint patch with **route heads frozen**: cond IIA ≤ 0.25; with **control heads frozen**: IIA ≥ 0.7 × unfrozen joint IIA; routing-shift fraction under the joint patch (the share of E→E′ movement in route-head group mass reproduced by the patch) ≥ 0.5 |
| R4 | non-monotone use of agreement | see below |
| R5 (secondary) | relational content, against nonlinear baselines | see below |

**R4 (non-monotone use of agreement).**
- For the best route head, take the within-question-centred log-attention on each source token of q, grouped by the token's group leave-one-out agreement count c ∈ {0, …, 5}. Call the group means m(c).
- Pass requires all of:
  - **(a)** m(4) − m(5) > 0;
  - **(b)** m(4) − m(0) > 0. For both (a) and (b) the episode-bootstrap (1000 resamples) 95% CI lower bound must be > 0.
  - **(c)** the R² of the log-attention from the exact Bayes leave-one-out log-odds of own-group reliability must exceed, by ≥ 0.10, the R² from the **best monotone function of the own count**. That is isotonic regression, increasing or decreasing, whichever is better. Both are fitted on half the evaluation episodes and scored on the other half.
- **Reported, not a criterion:** the R² from a **fully flexible lookup over (own group, LOO count triple)**. This is the upper bound for any count-based explanation.

**R5 (secondary: relational content, against nonlinear baselines).**
- Take the U_key coordinates at the source tokens of q, as a 2-D target.
- R² from the flexible lookup over (own group, LOO count triple) must be ≥ 0.7.
- That R² must also exceed, by ≥ 0.3, the R² from a flexible aggregate baseline: ridge on one-hot per-question global-histogram patterns of the other questions, interacted with the token's group.
- **Interpretation rule (fixed now).** A high R² from the count lookup means the tag is *explained by* agreement counts. Failure of a *linear* count predictor is **not** evidence of information beyond nonlinear functions of counts. No claim of "beyond counts" is made unless the flexible lookup fails while some other specified variable succeeds.

**Localisation class.** Reported, not a criterion; both classes pass R2.
- **key-localised:** key-only IIA ≥ 0.7.
- **query-localised:** query-only IIA ≥ 0.7.
- **distributed key/query:** joint IIA ≥ 0.7 and neither alone reaches it.

**Bayes comparison.** Reported for every seed:
- model vs exact Bayes accuracy;
- argmax agreement;
- IIA_bayes for the joint patch.

## 7. Verdicts

- **"Independently replicated causal account" (confirmed):** R0, R1, R2, R3 and R4 each pass in ≥ 3 of 4 new seeds.
- **Per-claim replication:** each R_i is reported as k/4. R5 is reported separately.
- **Fewer than 3/4 on any primary claim:** that claim is "not replicated". It is reported with the seed-level numbers.
- **Unaffected by this protocol:** the earlier post hoc results (seeds 0–2) and all earlier preregistered failures.

## 8. Code freeze

**Training.** The training code is the code at this commit, plus one change: `train_curriculum.py` gains an init-path argument. That change is behaviour-preserving for the existing seeds.

**Analysis.** The evaluation script `scripts/replicate_t4.py` is committed **before** its first run on any seed 10–13 model. It may be run beforehand on the old post hoc seed 0 model, using a separate code-check stream (base 4,400,000), only to debug it. Those numbers are not criteria.

**Fixes.** A bug found after the first new-seed evaluation is fixed and logged in `DEVIATIONS.md`, with both outputs shown.
