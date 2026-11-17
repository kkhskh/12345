# Learning transitions and frozen confirmation

**Commits.**
- FREEZE XV: commit `7957fee`. It precedes every confirmation run.
- Scoring commit: `180637c`.

**Scorers.** Both are fail-closed:
- `scripts/score_XV.py` → `logs/score_XV.json`;
- `scripts/score_T4.py` → `logs/score_T4.json`.

**Runs and data.**
- **GATE variants:** 16 confirmation models, four conditions (A–D) × seeds 201–204, each trained from scratch for 4000 steps with blind training.
- **DeepSets T4:** 4 models, seeds 301–304, 6000 steps.
- All 40 jobs (20 training, 20 measurement) returned rc = 0. There were no restarts and no replacements.
- Trajectories are in `confirm_outputs/traj_*.json`.

**Evidence labels.**
- **[frozen]** confirmation.
- **[expl]** post-scoring exploratory.
- **[dev]** development seed.

## 1. Per-model trajectories [frozen]

Columns:
- **t_S** is the source stage;
- **t_I** is the intermediate stage (A_p ≥ 0.5 with G ≤ 0.1);
- **t_G** is the gate (G ≥ 0.5), where "—" means not acquired within 4000 steps (censored);
- **width** is the step count from G ≤ 0.1 to G ≥ 0.9;
- **max A_p before gate** is the largest A_p while G ≤ 0.1.

| condition | seed | t_S | t_I | t_G | width | competent | max A_p before gate | final G / KL |
|---|---|---|---|---|---|---|---|---|
| A (a 2, gate gain 0.140) | 201 | 175 | 225 | 825 | 300 | yes | 0.91 | 0.99 / 0.000 |
| | 202 | 175 | 200 | 1500 | 175 | yes | 1.00 | 1.00 / 0.000 |
| | 203 | 150 | 275 | 900 | 225 | yes | 0.82 | 0.99 / 0.000 |
| | 204 | 175 | 225 | 1375 | 325 | yes | 0.92 | 0.98 / 0.000 |
| B (a 2, gate gain 0.277) | 201 | 175 | 350 | 500 | 300 | yes | 0.53 | 1.00 / 0.000 |
| | 202 | 150 | 250 | 575 | 175 | yes | 0.67 | 1.00 / 0.000 |
| | 203 | 150 | 450 | 625 | 200 | yes | 0.60 | 1.00 / 0.000 |
| | 204 | 150 | 475 | 675 | 325 | yes | 0.56 | 0.99 / 0.000 |
| C (a 3, gate gain 0.277) | 201 | 175 | 350 | — | — | no | 0.98 | 0.00 / 0.276 |
| | 202 | 150 | 250 | — | — | no | 1.02 | 0.00 / 0.276 |
| | 203 | 150 | 450 | — | — | no | 1.01 | 0.00 / 0.276 |
| | 204 | 150 | 475 | 2500 | 750 | yes | 0.93 | 0.97 / 0.001 |
| D (a 3, gate gain 0.139) | 201 | 175 | 225 | — | — | no | 1.03 | 0.00 / 0.138 |
| | 202 | 175 | 200 | — | — | no | 1.03 | 0.00 / 0.138 |
| | 203 | 150 | 275 | 2250 | 600 | yes | 0.95 | 0.99 / 0.000 |
| | 204 | 175 | 225 | — | — | no | 0.98 | 0.00 / 0.138 |

**Notes on the table.**
- **The censored models are not broken.** Each sits exactly at the best rule without the gate: its final KL equals the condition's gate gain, 0.276 or 0.138. They are fully competent at the source and agreement stages.
- **Competent models by condition:** A 4/4, B 4/4, C 1/4, D 1/4.
- **Source and intermediate times are identical by seed** across conditions that share ν: B and C both have t_I 350, 250, 450, 475; A and D both have 225, 200, 275, 225.
  - The reason is that arity-2 and arity-3 generators consume the random stream identically. A seed-paired B/C (or A/D) model sees the *same* labels, evidence and noise. Only the structure of the gate sources differs (a 2-source relation plus an irrelevant source, against a 3-source relation).
  - So before the gate is used, the paired models behave the same. This is a strong within-seed control for the complexity contrast: the divergence at the gate is caused by the gate's arity alone.

## 2. Frozen verdicts

**P1, complexity at matched usefulness: COMPLEXITY_DELAYS.**
- r ≥ ln 1.5 in **8/8** seed-paired comparisons. The median log-ratio is 1.44 (a ratio of about 4.2); the descriptive bootstrap interval is [0.99, 1.94].
- **Six of the eight arity-3 times are censored.** The correct reading is: *at matched decision value, the 3-source gate was acquired much later (2250 and 2500 steps against 500–1500), or not within the 4000-step budget (6/8).*
- This does not establish that those 6 models would acquire the gate later rather than never. The learning rate decays to 0 at step 4000 (steps 3000–4000 carry 2.7% of the cumulative rate), so censoring is at about 2700–3000 effective steps. P1 still holds in cumulative-learning-rate units (smallest pair 0.44 ≥ ln 1.5) (Review B F8).
- **Effective sample.** Seed-paired A/D and B/C models share their initialisation and their evidence and label streams, so the 8 pairs amount to about 4 independent replicates (Review B F4).

**P2, usefulness at fixed complexity: INCONCLUSIVE.**
- r ≥ ln 1.25 in 5/8 pairs.
- **At arity 2: 4/4 pairs** (a post-hoc subgroup, not a frozen rule; Review C F6). A (lower gate gain) acquired the gate later than B in every seed: 825–1500 against 500–675, ratios 1.4–2.6.
- **At arity 3:** the comparison is dominated by censoring. There are two ties at 4000, one in each direction, so it is uninformative.
- **Confound.** The single knob ν moves both gains, though not with a fixed sum (Review A F4). At ν = 0.05 the agreement cue also overlaps the gate more: it predicts g at 88%, against 74% (Review B F9). The result cannot separate "smaller gate gain" from "a stronger, more gate-overlapping agreement cue". A generator with separate pair and single-source reliabilities can decouple them (Review A F4).

**P3, intermediate-stage pattern: H1_COMPLEXITY_FIRST**, with the pattern (+, +, +, +).
- **What this verdict rests on (post-review; Reviews A F1, B F5).**
  - In C and D, "stage present" is close to entailed by censoring: the gate-free optimum has A_p = 1 and G = 0, and 6 of 8 models sit there.
  - P3 therefore discriminates between hypotheses mainly in B. There the stage lasts only **3 snapshots (50–75 steps) in 3 of 4 models** (B_s202: 12), re-checked from the trajectories.
  - The frozen verdict stands as returned, but **H1 is not robustly separated from RACE.**
- **Every one of the 16 models** passed through the pairwise-agreement stage (A_p ≥ 0.5 with G ≤ 0.1) before acquiring the gate, or without ever acquiring it.
- The RACE pattern (B absent), formed from development seed 100, is **falsified**. The H2 payoff pattern is **falsified**.
- **Threshold sensitivity** (`logs/xv_sensitivity.json`, exploratory):
  - **B is fragile.** In B the stage is weak and brief: A_p peaks at only 0.53–0.67. The criterion holds at 3 snapshots in 3 of 4 models; the earlier "150–325 steps" was t_G − t_I, not the stage's duration. At an A_p threshold of 0.6, B shows the stage in 1/4 models, and the pattern becomes RACE.
  - **A, C and D are robust** under every threshold combination tested.
  - **Competent-only sensitivity:** A and B PRESENT; C and D MIXED (1 competent each).

**P6, DeepSets T4 source stage first: CONFIRMED.**
- The trainability gate passed: 4/4 competent, final KL 0.0013–0.0014, G4 0.83–0.84.
- Every model held 7–10 consecutive source-stage snapshots (350–500 steps at the own-question rule, G4 ≈ 0) before acquiring the relational T4 rule.
- The acquisition was abrupt: t_G4 700–900, width 150–250 steps.

## 3. Answers to the learning questions

7. **Do models acquire source-wise information before relational information?**
   - **DeepSets T4:** yes, a real source plateau in all 4 [frozen].
   - **GATE variants:** order-1 accuracy is reached first (t_S 150–175, before t_I), but there is no separate source-only plateau. Gate-free second-order reliance (A_p) is already 0.11–0.28 at t_S, and accuracy cannot distinguish the order-1 rule from the agreement rule (Review B F6).
8. **When does an intermediate relational stage appear?**
   - In every GATE-variant condition here [frozen P3].
   - It is long and complete (A_p up to 1.0) when the gate is slow (C, D) or its competing cue is strong (A).
   - It is weak and brief when the gate follows quickly (B).
9. **When is it skipped?**
   - Not in this task family at the frozen threshold.
   - In XIV's original GATE it was skipped (A_p ≤ 0.27), but with a different agreement index, so the comparison is not like-for-like (Review B F10). This suggests that "skipping" depends on the task construction, not only on the payoff ratio: B and XIV's GATE have similar intermediate gains (0.025 and 0.031) [frozen, across investigations; the generators differ].
10. **Does decision benefit at fixed complexity change learning order?**
    - The *order* does not change: the stage appears in A and B.
    - The *timing* does change: at arity 2 the gate came later with half the gain in 4/4 pairs, confounded by the stronger agreement cue [frozen; confounded].
11. **Does complexity at matched benefit change learning order?**
    - The order of stages is unchanged.
    - The gate's acquisition is strongly delayed or not achieved within the budget (8/8) [frozen].
12. **What better predicts the gate's timing: complexity, usefulness, or learning signal?**
    - **Complexity** has the largest and most reliable effect.
    - **Usefulness** has a consistent but confounded effect at arity 2.
    - **Learning signal:** untested. The exploratory measure was at its noise floor and is withdrawn (OPTIMIZATION §1).
13. **Do abrupt transitions persist across independently initialised models?**
    - **At arity 2:** the gate transition takes 175–325 steps in all 8 models, against total training of 4000.
    - **At arity 3:** unresolved. After step 2000 the snapshots are 250 steps apart, and C_s204's jump falls within about one interval (Reviews B F7, C F4).
    - **DeepSets T4:** 150–250 steps [frozen].
14. **Are the jumps genuine, rather than calibration or sampling artefacts?**
    - **Not sampling, at arity 2 and in DeepSets:** 25-step snapshots resolve the arity-2 transitions into 7–13 intervals; DeepSets 50-step snapshots give 3–5. Not established at arity 3.
    - **Calibration:** a rule with no gate use has G = 0 at any calibration (exactly in the population, |G| ≲ 0.01 on the stream). But the *size* of G and of A_p scales with confidence (temperature λ gives A_p = λ), so the thresholds are partly confidence thresholds (Review A F3, F6).
    - **Accuracy confirms it** for B: 0.79 → 0.90.
    - Mechanism evidence is in OPTIMIZATION_AND_MECHANISM_ANALYSIS.md.
15. **Which prospective predictions were falsified?**
    - The development-derived RACE pattern (B would skip).
    - The H2 payoff pattern (B and C would skip).
    - **Not falsified:** H1's ordering claim and the complexity delay.
    - **Undecided:** H2's timing claim (P2 inconclusive).

## 4. Limits

- **Competence.** Only 2 of 8 arity-3 models became competent. Any stage conclusion for C and D rests on models that never acquired the gate, which the frozen rule counts as censored.
- **Sample size.** n = 4 per condition, and about 4 independent replicates overall, because seed pairs share their initialisation and their streams.
- **Scope.** One architecture and one optimiser recipe. The DeepSets T4 result uses a different architecture and task, so it is not a same-architecture comparison.
