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

**What the contrast isolates.** The gate's value is varied while the competing agreement strategy is held fixed, both in its law and, to 0.66%, in its benefit. That answers Review A's XV finding.

**What it does not isolate:**
- overall evidence strength (the residual rows above);
- the learning signal: no gradient-level matching was attempted. The mandate warns that a clean decision-value contrast is not automatically a clean gradient-level contrast.

## 4. Experiment (frozen; FROZEN_PROTOCOL_XVI.md)

- **Models:** 4 seed pairs (501–504). Within a pair, the two conditions share the initialisation and every random draw. The generator's random consumption does not depend on b0 (verification V1), so gates, labels and noise are identical, and only the single source's reliability differs.
- **Recipe:** XV's 4000-step cosine schedule, with diagnostics every 25 steps.
- **Outcome:** t_A, sustained gate acquisition (F_G ≥ 0.5 and KL ≤ 0.75 · the gate-free KL, at 3 consecutive points).
- **Verdict P3:** BENEFIT_SPEEDS if ln(t_low / t_high) ≥ ln 1.25 in at least 3 of 4 pairs, with BENEFIT_SLOWS, NO_EFFECT and INCONCLUSIVE as alternatives. The expectation was BENEFIT_SPEEDS.

## 5. Results

(Filled in after scoring.)
