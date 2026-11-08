# Main answer: latent reliability (Phase 1.5 Step 1)

> **Did a learned Transformer develop a reusable, independently identified, consistency-related computation that causally contributes to candidate decisions?**

**Short answer.** It holds at the level of computation. It does not hold as an explicit readout variable. It holds for T1 from scratch, and for T4 only after a T1 curriculum (post hoc).

**Frozen primary answer** (`PROTOCOL.md` §6, applied literally): **PARTIAL**.
- The explicit readout-token mediator (Type A) fails for every model.
- The Type B evidence (reliability-gated attention routing) holds for T1/M (3/3 seeds) and T2/M.
- T4/M from scratch did not learn the task, 0/3 seeds.

**Secondary answer** (registered addendum 2, tag-and-route): **NO (scoped) for spontaneously trained models**.
- T1/M passes in 1/3 seeds. Seed 1 misses K4′ by 0.004. Seed 2 splits the computation (below).
- T4/M from scratch never learned the task.
- **Post hoc:** T4 models fine-tuned from T1 checkpoints pass every tag-and-route criterion in all analysed seeds; see the table.

**Evidence labels.**
- **[R]**: frozen or registered, confirmation split.
- **[PH]**: post hoc, exploratory. This includes everything about curriculum models.

## 1. What the task demands (Bayes level; proved)

The setup:
- six sources in three fixed pairs;
- one hidden reliability configuration R shared by six questions;
- T4 adds a colluding pair that always agrees.

What the exact Bayes model establishes (`MATH_VALIDATION.md`):
- The within-pair agreement counts of the groups across questions are an **exact sufficient statistic** for R (Theorem S, checked to 1e−14).
- One episode-level vector P(R | X) answers every question (Theorem M).
- Global vote histograms are insufficient: Bayes on histograms reaches 0.636, against 0.743 for full Bayes on T1.
- Insufficiency of histograms does *not* imply that a separate agreement variable is necessary.
- **In T4 the Bayes map from agreement to reliability is non-monotone.** The heuristic "trust the most coherent group" scores 0.359 there.

## 2. What the learned models do

### T1 (hidden reliability), M size, 3 seeds, from scratch

Accuracy is at Bayes: 0.740 vs 0.740, own-question level 0.650 [R]. Three findings follow.

**(a) The answer is produced by last-layer reliability-gated attention routing.** [R], 3/3 seeds.
- One or two last-layer heads put the readout-q attention on each group's sources *within question q*, in proportion to P(group reliable | X). The best head's R² is 0.93, 0.90 and 0.99 in seeds 0–2.
- Patching only the readout-q attention row of the last layer, using an episode with the same X_q and different other questions, transfers the decision: IIA 0.84–0.85, KL recovered 0.99.
- The routing is count-linear in log-attention (head log-attention vs agreement count: R² 0.79–0.88 [PH]), as Corollary T1 predicts.
- The model's logits are explained by the routed vote: R² 0.945–0.95 [PH].

**(b) The reliability variable that drives the routing is low-dimensional, output-free-identifiable and relational, but its location is seed-dependent.**
- **Seeds 0 and 1:** a 1-D direction on the source tokens at the input to the last layer, found from input contrast pairs alone [R]:
  - cond IIA 0.83 and 0.83, KL recovered 0.99;
  - matched histograms IIA 0.84 and 0.80;
  - target-only change 0.079 and 0.104;
  - transfer to unfitted questions 0.83 and 0.81;
  - content: Bayes reliability R² 0.79 and 0.65, versus own reports ≤ 0.002 and other-question histograms ≤ 0.0001;
  - patching the tag reproduces 93–94% of the routing shift;
  - freezing the route heads' patterns removes the tag's effect, IIA 0.83 → 0.07 [PH].
- **Seed 2** splits the same information between the source-token keys and the readout-token query [PH]:
  - key-only IIA 0.33;
  - query-only IIA 0.47 (its content is π_{−q} with R² 0.90);
  - joint IIA 0.78, KL recovered 0.97.
- The tag is a **group-indexed function of the agreement counts** [PH]:
  - linear in raw reports: R² ≤ 0.04;
  - linear in pairwise-equality indicators: R² 0.93–0.98 (reviewer computation);
  - the tag means rise monotonically with the group's leave-one-out count;
  - its exact encoding mixes in the other groups' counts and varies by seed.

**(c) Reliability is decodable at the readout token but causally inert there, where the frozen protocol looked.** [R]
- The readout probe has R² 0.65–0.95, yet patching its subspace gives IIA 0.01–0.11.
- In seeds 0 and 1 the frozen site rule picked the final residual, which is readout-equivalent. Its contrast subspace is the answer direction itself (target-only change 0.94). P4 rejects it correctly.

### T2 (with a confidence head)

- At Bayes accuracy. The confidence head matches the Bayes max posterior with MAE 0.010–0.017.
- Routing (Type B) holds in all analysed seeds [R].
- **Cross-head reuse** [R, reported-only metric]: patching the tag moves the *confidence* head toward the source episode's confidence.
  - Seed 1: corr 0.97, slope 0.93.
  - Seed 0 (a split seed, like T1 seed 2): corr 0.70, slope 0.47.

### T4 (colluders), M size

- **From scratch it is not learned.** 0/3 seeds at 6,000 steps; 1 post hoc seed at 24,000 steps. All reach the own-question level, 0.578.
- `mlp_full` (0.553) and `mlp_grouphist` (0.556) also fail.
- **DeepSets learns T4 to Bayes** in 3/3 seeds (0.743). Its pooling architecture *is* the Theorem-S factorisation (review A3).
- So the failure is an optimisation barrier at this budget and initialisation, not inexpressibility. A plausible cause, which is a hypothesis only: the T1-style first step "agreement ⇒ reliable" is actively wrong in T4, so there is no easy partial solution to climb from.
- **Post hoc T1 → T4 curriculum** (`scripts/train_curriculum.py`, 3,000 steps): Bayes level in 3/3 seeds (dev acc 0.7434, 0.7436, 0.7440 vs Bayes 0.7437).
  - Routing (Type B) holds in 3/3 seeds.
  - Tag-and-route (D2) holds in 2/3 seeds (s0, s1).
  - Seed 2, which was fine-tuned from the split T1 seed 2, again splits the information between keys and the readout query: key-only IIA 0.39, query-only 0.29, joint 0.73 / KL 0.90 [PH]. The split localisation is inherited from initialisation.
  - Seed 0: tag cond IIA 0.84 / KL 0.99, matched histograms 0.83, target-only change 0.05, reuse 0.85, content R² 0.93, routing shift 0.95.
- The tag implements the **non-monotone** Bayes map [PH, seed 0].
  - One tag dimension rises with the group's leave-one-out count from 0 to 4 and drops at 5, meaning the group agreed on every other question. Bayes P(reliable) by count is 0.03, 0.12, 0.34, 0.67, 0.89, then 0.11.
  - Route-head log-attention is explained by the 2-D tag (R² 0.75 and 0.59), *not* by the count (R² 0.01).
  - On held-out α = 0.7 and 0.9 it stays at Bayes (0.649 vs 0.647; 0.789 vs 0.790).

### Controls

- **DeepSets (method calibration).** Its explicit pool is found and is causal: probe IIA 0.94 on T1. So the identification method finds an explicit mediator when one exists architecturally. The Transformer null at the readout is not a failure of the method.
- **mlp_hist** is capped at its Bayes ceiling (0.638 vs 0.636).
- **T0/T5** (cross-question coherence has no Bayes value), **T3** (R supervised) and **S size**: see `VERDICTS.md` and the training records. Items not completed are listed in `logs/train/DROPPED.txt`.

## 3. Interpretation

**Is it reusable?** Yes, in a weak sense:
- the same tag direction serves every question (the D2 transfer to unfitted questions);
- the same tag drives both the answer and the confidence head (T2).
- Weight sharing across readout tokens makes cross-question reuse close to automatic (review A4). The non-trivial reuse is cross-head.

**Is it independently identified?**
- The routing heads are identified by Bayes P(group reliable), not by outputs.
- The tag direction is identified from input contrast pairs, then checked against the Bayes reliability (K7).
- Neither definition uses model outputs.

**Is it consistency-related?** Yes, precisely in this sense:
- the computed variable is a function of within-group agreement counts, which is the Bayes-sufficient consistency statistic;
- it is consumed as an attention key and query, not as a readout variable.
- In this generator, "consistency score" and "group reliability statistic" cannot be separated (review B3). The package does not claim a consistency variable distinct from reliability.

**Does it causally contribute to candidate decisions?** Yes, with these patch results:
- tag and routing patches transfer the decision, KL recovered about 0.99 [R];
- freezing the routing removes the tag's effect [PH];
- injecting wrong tags drives accuracy below the own-question level [R].

**Does the consistency interpretation add anything beyond known mechanisms?** Little.
- The mechanism is the classical reliability-from-agreement structure followed by a weighted vote (Dawid–Skene / truth discovery), implemented as attention gating.
- What is new for the COR program is the *located, causally verified learned instance*, and the **T4 result**: the same circuit encodes the non-monotone, Bayes-correct use of agreement. That result is post hoc, conditional on a curriculum, and still needs a registered replication.

**Scope.**
- One generator family, with fixed groups, K = 3, Q = 6, α = 0.8.
- Bidirectional Transformers of d = 64, 3 layers, 4 heads.
- No L-size models. The S size is limited to its own grid.
- Nothing here generalises to Transformers at large. The negative findings (the readout mediator is inert; T4 is not learned from scratch) are scoped to these models, budgets and protocols.

## 4. Where to look

| topic | file |
|---|---|
| exact generators and Bayes | `src/bayes.py`, `MATH_VALIDATION.md` |
| protocol, addenda, deviations | `PROTOCOL.md`, `PROTOCOL_ADDENDUM_TAG.md`, `DEVIATIONS.md` |
| verdicts and per-model numbers | `VERDICTS.md`, `logs/verdicts.json` |
| frozen analyses | `logs/analysis/main/*_conf.json` |
| curriculum analyses [PH] | `logs/analysis/curriculum/` |
| post hoc attacks and equation checks | `logs/posthoc/` |
| theorem (three levels) | `THEOREM.md` |
| reviews and dispositions | `reviews/`, `ADJUDICATION.md` |
| F-matrix update | `FAILURE_MATRIX_UPDATE.md` |
| checkpoints and training logs | `checkpoints/`, `logs/train/` |
