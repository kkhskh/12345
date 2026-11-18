# Decision value and competing cues (Workstream B)

**Evidence labels:** [proof], [num], [dev], [frozen], [expl], [spec].

## 1. The XV confound and its repair

**XV.** XV varied the gate's value through a single knob, ν. That knob also strengthened the competing agreement cue: the cue's overlap with g went from 0.74 to 0.88, and the agreement benefit from 0.025 to 0.076. So XV's 4/4 arity-2 delays could not be attributed to value.

**The repair.** The XVI generator (`src/xvitask.py`) gives the agreeing pair and the single source separate reliabilities:
- b1, the pair's reliability when g = 1;
- b0, the single source's reliability when g = 0;
- ν, the noise-pair agreement, as before.

## 2. What each parameter changes [proof]

- The agreement indicator's law depends only on (b1, ν):
  - P(pair agrees | g = 1) = b1² + (1 − b1)²/(K − 1);
  - P(pair agrees | g = 0) = ν.
  - So the agreement cue's mutual information with g, and its best accuracy as a predictor of g (the "cue overlap"), depend on (b1, ν) only.
  - The single source's reliability b0 never enters these probabilities.
- Changing **b0 alone**, at fixed (b1, ν), therefore leaves the competing cue's law **exactly** unchanged [proof]. It changes:
  - how informative the single source is when it is trusted, and with it the gate's value;
  - the overall difficulty of the task.
- It does **not** automatically leave the agreement strategy's *benefit* unchanged: that benefit is an expected-NLL difference between two optimised rules. It has to be computed. It turns out to move only slightly (§3).

## 3. Search and selected contrast [num]

**Search.** An exact grid over b1 ∈ {0.8, 0.85, 0.9, 0.95}, ν ∈ {0.05, 0.15, 0.25, 1/3, 0.45} and b0 ∈ {0.40, 0.45, …, 0.95}: 240 conditions at arity 2, every reference computed by enumeration (`scripts/xvi_value_search.py`, `logs/xvi_value_grid.json`).

**Tolerances fixed before selection:**
- identical agreement law (same b1 and ν);
- agreement benefit within 2%;
- gate-value ratio ≥ 1.5.

**15 pairs** meet them.

**In every one of the 15, the high-value member is also easier overall.** Bayes accuracy is 0.075–0.25 higher and Bayes NLL 0.20–0.40 nats lower. With three parameters, the gate's value cannot be changed while the agreement law, the agreement benefit and the overall difficulty are all held fixed. **This residual confound is structural to the family, not a search failure** [num].

**Selected** (at XV's reliability level, where arity-2 acquisition is known to be reliable; refined over b0 ∈ [0.50, 0.60]). Verified independently by V4 in `logs/xvi_math_verified.json`:

| | low `2:0.9:0.55:0.05` | high `2:0.9:0.9:0.05` (= XV's A) | difference |
|---|---|---|---|
| gate value (order-2 optimum minus Bayes, nats) | 0.0802 | 0.1398 | ratio **1.74** |
| agreement benefit (order 1 minus order 2) | 0.0752 | 0.0757 | **0.66%** |
| P(agree \| g = 1), P(agree \| g = 0) | 0.815, 0.050 | 0.815, 0.050 | identical (< 1e-12) |
| MI(g; agree); cue overlap | 0.3453; 0.8825 | 0.3453; 0.8825 | identical |
| minimal decision order | 3 | 3 | same |
| MI of g with any single or proper gate subset | 0 | 0 | none |
| Bayes NLL / accuracy | 0.591 / 0.725 | 0.288 / 0.900 | **residual** |
| order-1 (source-wise) NLL / accuracy | 0.746 / 0.668 | 0.504 / 0.872 | **residual** |
| gate's accuracy gain | 0.0196 | 0.0283 | ratio 1.45 |

**What the contrast isolates (corrected after review; Review A finding 1).**
- The gate's value is varied while the competing agreement cue's **law** is held fixed. In fact the whole input distribution is identical between the conditions (Review A).
- The cue's **log-loss benefit** is matched to 0.66% against the order-2 rule, or 1.75% against the exact gate-free mixture.
- **At the decision level the competing cue is not matched:**
  - the agreement rule changes 18.9% of decisions and gains 0.037 accuracy in the low condition, against 0% and 0 in the high condition;
  - its benefit relative to the gate's is 0.94 against 0.54.

**What it does not isolate:**
- overall evidence strength (the residual rows above);
- the learning signal: no gradient-level matching was attempted. The mandate warns that a clean decision-value contrast is not automatically a clean gradient-level contrast.
- decision-level competition (above).
- **And the frozen P3 alternatives did not follow from the accounts** (Review A finding 2): decision-level competition, relative competition and the difficulty difference all predict that the low condition is slower, like the benefit account.

## 4. Experiment (frozen; FROZEN_PROTOCOL_XVI.md)

- **Models:** 4 seed pairs (501–504). Within a pair, the two conditions share the initialisation and every random draw. The generator's random consumption does not depend on b0 (verification V1), so gates, labels and noise are identical, and only the single source's reliability differs.
- **Recipe:** XV's 4000-step cosine schedule, with diagnostics every 25 steps.
- **Outcome:** t_A, sustained gate acquisition (F_G ≥ 0.5 and KL ≤ 0.75 · the gate-free KL, at 3 consecutive points).
- **Verdict P3:** BENEFIT_SPEEDS if ln(t_low / t_high) ≥ ln 1.25 in at least 3 of 4 pairs, with BENEFIT_SLOWS, NO_EFFECT and INCONCLUSIVE as alternatives. The expectation was BENEFIT_SPEEDS.

## 5. Results

**P3 [frozen]: INCONCLUSIVE.**

| seed pair | t_A, low value (0.080) | t_A, high value (0.140) | ratio low / high |
|---|---|---|---|
| 501 | 1175 | 725 | 1.62 |
| 502 | 1325 | 1875 | 0.71 |
| 503 | 1825 | 2075 | 0.88 |
| 504 | 1675 | 825 | 2.03 |

- All 8 models acquired the gate, became competent (final KL ≤ 0.0005, F_G = 1.00), and showed the floor-conditioned agreement stage.
- Transition widths were 75–250 steps.
- Means were 1500 against 1375. The median log ratio is 0.18 (1.19×). Exact Mann–Whitney p = 1.0; sign test p = 1.0.
- **My frozen expectation (BENEFIT_SPEEDS) was not supported.**

**Answers.**

- **Can gate value be changed while the competing cue is held fixed (Q9)?**
  - Yes for the cue's law, exactly [proof]; and for its log-loss benefit, to 0.66–1.75% [num].
  - **Not at the decision level** (Review A).
  - **But not without changing overall difficulty.** All 15 grid pairs within tolerance carry a 0.075–0.25 Bayes-accuracy difference [num].
- **Does gate value independently change acquisition timing (Q10)?**
  - Within this design no consistent effect was detected: 2 pairs faster with higher value, 2 slower or equal.
  - **With 4 pairs no effect size can be excluded.** The 95% t-interval on the mean pair ratio is 0.54–2.64 (Review B finding 7).
  - Only 1 of 4 pairs reached 2×, and one reversed [frozen data].
- **How much is attributable to optimisation signal (Q11)?** Not identifiable in this design.
  - The rebuilt estimator shows that the first-order gate signal is zero on the plateau and appears only about 150–250 steps before onset in every condition measured (LEARNING_SIGNAL_AND_STAGE_VALIDATION.md §6.2).
  - It was not measured in the WB models, and it could not separate value from signal in any case.
- **Remaining confounds (Q12):**
  - overall difficulty (Bayes accuracy 0.725 against 0.900; order-1 accuracy 0.668 against 0.872);
  - the gate's accuracy gain (ratio 1.45, against 1.74 in NLL);
  - no gradient-level matching;
  - n = 4 pairs.
  - Variation between seed pairs is large: within the high condition, 725–2075 (2.9×).

**Cross-investigation reading [expl]: withdrawn as an inference** (ADJUDICATION W4).

| condition | gate value | cue overlap / agreement benefit | t_A |
|---|---|---|---|
| XV B (ν = 1/3) | 0.277 | 0.74 / 0.025 | 475–625 |
| XV A (ν = 0.05) | 0.140 | 0.88 / 0.076 | 800–1500 |
| XVI high (= XV A condition, new seeds) | 0.140 | 0.88 / 0.076 | 725–2075 |
| XVI low | 0.080 | 0.88 / 0.075 | 1175–1825 |

- An earlier draft read this table as favouring a competition-driven account. **It does not support that reading:**
  - the XVI conditions also differ in decision-level competition and in difficulty;
  - the cross-study comparison mixes seeds and contrasts;
  - every account considered predicts the same direction.
- **What remains:** XV's consistent arity-2 delay was not reproduced by a contrast that holds the cue's law fixed. Its cause remains open.
- A clean test needs a generator in which gate value, cue law, decision-level competition and difficulty can be set separately: at least four free parameters [spec].
