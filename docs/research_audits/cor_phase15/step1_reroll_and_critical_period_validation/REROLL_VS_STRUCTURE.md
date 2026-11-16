# Re-roll or structure? (Workstream A)

COR Phase 1.5, Step 1, Investigation XIII-E. This is about how small transformers come to organise their computation. It does not validate COR and makes no claim about the O/D operators.

## Question and hypotheses

XIII-D falsified XIII-C's readout-map cause. One explanation was left open:

> Changing a seed simply re-draws a random initialisation, so the outcome is another sample from a distribution of trained solutions.

We test that re-roll explanation against three alternatives:

| | hypothesis | content |
|---|---|---|
| H0 | re-roll | Re-drawing W (or L) gives another draw from a distribution that may depend on the parent and on the components held fixed. The re-drawn tensor carries no property that matters beyond being a fresh draw. |
| H1 / H3 | structure | A property of a re-drawn tensor, alone or jointly with the fixed components, systematically influences the outcome. |
| H2 | asymmetric stability | Response to replacement differs between early- and mixture-producing starting points beyond what re-drawing explains. |

## A2. Are "matching seeds" special? No

`scripts/a2_init_audit.py` → `logs/a2_init_audit.json` (frozen in A-1):

- Within one seed's jax split, W_A uses sub-key 5 and the layer-2 tensors use sub-keys 20–25.
- No tensor is reused.
- The data order D drives a separate numpy generator.
- Over 200 seeds, every joint W–L statistic tested (alignment of W_A's column space with each layer-2 map: VO, Q, K, MLP) has the same distribution for matched pairs (s, s) as for mismatched pairs (s, s + 1000). KS p = 0.27–0.99, and both means equal the random-subspace value √(3/64) ≈ 0.216.

So a "diagonal" seed pair is not physically aligned. It is just a pair of independent draws with the same label.

## A3. Design

Frozen in A-1 (`2063085`) and A-2 (`a4f66b7`) before any measurement.

**Anchors.** Four diagonal configurations, chosen by their known outcomes; the selection is disclosed. Each parent contributes one early and one mixture anchor:

| anchor | parent | L / W / D | known outcome |
|---|---|---|---|
| E160 | uP160 | 21/21/21 | early |
| M160 | uP160 | 23/23/23 | mixture |
| E161 | uP161 | 22/22/22 | early |
| M161 | uP161 | 21/21/21 | mixture |

**Re-draws.** For each anchor:
- W re-drawn to 601…608, with L and D held fixed;
- L re-drawn to 701…708, with W and D held fixed.

That gives 64 runs to step 1211. The same new seeds are used for every anchor (a crossed design). Every new W (or L) is therefore observed in two contexts per parent: with the early anchor's fixed components and with the mixture anchor's.

**Measurement.** `unit_metrics.py`, stream 95.1 M. Classification at step 1211:
- EARLY: mix < 0.5 and lik ≥ 0.4;
- MIXTURE: mix ≥ 0.8 and lik ≤ 0.15;
- HYBRID: otherwise.

**Continuations.** 10 runs continued to step 4000:
- 8 preselected (W' = 601 and L' = 701 for every anchor);
- 2 borderline runs chosen by the frozen rule (mix@1211 closest to 0.5).

**Scoring.** `scripts/score_A.py` is fail-closed: it refuses to score unless every run, metric, configuration and continuation is present. The A-2 preflight checked P0, P1 and N1–N7.

## A6. Results

All 64 re-roll runs and the 10 continuations finished. 63 of 64 runs are competent (KL ≤ 0.05) at step 1211; the exception is one E161 W re-draw. The full table of 64 outcomes, with lik, mix, KL and class, is in `logs/score_A.json` → `table`.

### Probabilities (Clopper–Pearson 95% intervals; n = 8 per cell)

| anchor : re-draw | EARLY | MIXTURE | HYBRID | P(early) [95%] | mix_mass at 1211 (8 seeds, in seed order) |
|---|---|---|---|---|---|
| E160 : W | 5 | 1 | 2 | 0.62 [0.24, 0.91] | 0.93 0.20 0.17 0.66 0.20 0.94 0.17 0.17 |
| E160 : L | 4 | 0 | 4 | 0.50 [0.16, 0.84] | 0.93 0.17 0.16 0.57 0.93 0.16 0.83 0.45 |
| M160 : W | 4 | 4 | 0 | 0.50 [0.16, 0.84] | 0.94 0.94 0.31 0.16 0.95 0.93 0.16 0.16 |
| M160 : L | 2 | 4 | 2 | 0.25 [0.03, 0.65] | 0.17 0.93 0.93 0.93 0.94 0.94 0.94 0.37 |
| E161 : W | 0 | 4 | 4 | 0.00 [0, 0.37] | 0.91 0.96 0.93 0.80 0.96 0.93 0.93 0.85 |
| E161 : L | 0 | 7 | 1 | 0.00 [0, 0.37] | 0.96 0.97 0.97 0.95 0.96 0.90 0.96 0.96 |
| M161 : W | 0 | 7 | 1 | 0.00 [0, 0.37] | 0.95 0.96 0.94 0.90 0.95 0.95 0.95 0.93 |
| M161 : L | 0 | 8 | 0 | 0.00 [0, 0.37] | 0.95 0.95 0.97 0.96 0.97 0.97 0.97 0.97 |

Pooled by parent. These pools are descriptive: the runs share a parent and fixed components, so they are not independent pretrained models.

| parent | P(early) [95%] | P(mixture) | P(hybrid) |
|---|---|---|---|
| uP160 | 15/32 = 0.47 [0.29, 0.65] | 9/32 | 8/32 |
| uP161 | 0/32 = 0.00 [0, 0.11] | 26/32 | 6/32 |

**Parent baselines.** These are the natural diagonals from XIII-B, scored at step 4000:
- uP160: early s20 and s21; mixture s23.
- uP161: early s22; mixture s20 and s21; hybrid s23.

**Early versus mixture anchors** (A4; Newcombe 95% intervals on the difference in P(early)):

| parent : re-draw | difference | 95% interval |
|---|---|---|
| uP160 : W | +0.125 | [−0.30, +0.50] |
| uP160 : L | +0.25 | [−0.19, +0.59] |
| uP161 : W | 0 | [−0.32, +0.32] |
| uP161 : L | 0 | [−0.32, +0.32] |

No asymmetry is detected, but the intervals would allow differences up to about ±0.5. H2 is therefore not supported, and it is not excluded either.

**Homogeneity across the four anchors** (exact Monte Carlo): rejected for both re-draw types, with p = 0.003 for W and p = 0.0015 for L. Almost all of this comes from the parent difference: uP161 never gives early.

**Anchor-selection effect.** uP161's early anchor (22/22/22) is a rare configuration. None of its 16 re-draws, nor any of the other 16 uP161 re-draws, gives early. This is the regression toward the parent's base rate that the mandate warned about. In uP161, "re-drawing a component destroys early" says nothing about instability; it says the anchor was an unusual draw.

### R1: is there a transferable component? (frozen predictions)

**Method.** Each new seed is observed in two contexts per parent. We take the Spearman correlation of mix_mass at step 1211 across the two contexts, average it over the two parents, and test it against a stratified permutation null.

| | ρ (uP160 / uP161) | mean ρ | one-sided p | 95% bootstrap upper bound | outcome | verdict |
|---|---|---|---|---|---|---|
| R1-W (re-drawn W) | 0.55 / 0.76 | **0.65** | **0.006** | 0.86 | TRANSFER | **FALSIFIED** (re-roll null rejected for W) |
| R1-L (re-drawn L) | 0.07 / −0.12 | −0.02 | 0.54 | 0.52 | NO-TRANSFER | **CONFIRMED** (consistent with re-roll for L) |

**Class concordance** across the two contexts:

| parent : re-draw | same class | expected if independent |
|---|---|---|
| uP160 : W | 4 / 8 | 3.0 |
| uP161 : W | 5 / 8 | 4.0 |
| uP160 : L | 2 / 8 | 2.0 |
| uP161 : L | 7 / 8 | 7.0 |

The W signal is mostly in the continuous mixture mass:
- uP160: whether the run lands early or mixture;
- uP161: how hybrid a mixture solution is.

**Bounds** (power simulation frozen in `logs/A_power.json`):
- R1-L makes a cross-context latent correlation of about 0.6 or more unlikely for L **if transfer is similar in both parents**. If transfer exists in only one parent (c = 0.6–0.8), the frozen rule still returns NO-TRANSFER 27–35% of the time (`logs/A_power_reproduced.json`, post-review reconstruction; the original pre-freeze script was not preserved). The 'upper bound' is a decision statistic, not a confidence bound.
- R1-W detects a transferable W component with mean ρ = 0.65. *Post-review:* this is a pooled result; per parent p = 0.086 (uP160) and 0.018 (uP161), and uP160 shows no transfer in lik or class. Excluding the single incompetent run gives pooled p 0.010–0.014. The W − L contrast itself is untested (bootstrap CI for ρ_W − ρ_L [−0.18, 1.38]).

### Does the W effect generalise across parents? (post hoc, exploratory)

Source: `logs/A_posthoc_W_features.json`.

- The mean mix_mass per new W seed (averaged over its two contexts) correlates between the two parents at ρ = 0.02. A given W is "early-favouring" in its parent's contexts, but cross-parent transfer was **not detected** (CI about ±0.7; low power).
- No W-only spectral statistic (‖W_A‖, σ₁, σ₃, condition number) predicts the outcome with a consistent sign across parents. For example, ρ(‖W_A‖) = +0.17 in uP160 and −0.69 in uP161.
- The pre-declared exploratory step-0 joint statistic R_route gives AUC 0.59. Predictive value was not detected (AUC CI [0.41, 0.76]).

### A5. Does step 1211 predict step 4000?

**9 of 10 continuations keep their class.** The exception: uP160 L708/W21/D21 was EARLY at step 1211 (mix 0.45, near the boundary) and became hybrid at step 4000 (mix 0.84).

Two kinds of runs are reliable at step 1211:
- clear EARLY (mix < 0.25);
- clear MIXTURE (mix > 0.9).

Near the class boundaries, step 1211 can drift towards more mixture. All screen conclusions here depend mainly on clear cases. mix_mass is stable from step 1211 to 4000 in the 8 clear cases (|Δmix| ≤ 0.03). Both borderline runs drifted towards mixture: 0.45 → 0.84, and 0.57 → 0.92 while staying hybrid. The W transfer statistic uses mix_mass at step 1211, so its borderline values carry this drift uncertainty.

**Threshold sensitivity.** Shifting the class thresholds by ±0.05 changes at most one count per cell (E160:L 4 → 3).

## A8. Decision

**Outcome C: mixed.**

**Re-drawing the layer-2 initialisation L** behaves like a re-roll. Its outcome distribution depends on the parent and the fixed components, and no property of the new L transfers across contexts (R1-L).

**Re-drawing the readout map W** is not a pure re-roll. A new W carries a property that pushes the outcome the same way in two different L/D contexts of the same parent (R1-W, p = 0.006). *Post-review wording:* this shows a W component whose effect transfers within a parent; it does not by itself show an *identifiable* structure in the H1/H3 sense (a re-roll whose value matters would also do this). It is pooled over parents and carried mainly by uP161. Cross-parent transfer was not detected, and no W-only statistic found here captures it. That agrees with XIII-D: W's influence exists, interacts with the pretrained parent, and has no parent-independent law.

**What this does not show:**
- that training contains no deeper selection mechanism;
- what the W property is.

A held-out test of a candidate W–parent alignment statistic is the obvious next experiment (see COR_FOUNDATIONAL_NAVIGATION.md).

**Limits.**
- 2 parents and 4 anchors.
- 8 seeds per cell. Single-cell intervals are about ±0.35 wide.
- Outcomes for uP161 are almost all mixture, so its contribution to the W transfer comes from degrees of hybridness.
- The step-1211 screen was validated on 10 of the 64 runs. uP161's W contribution (graded hybridness at step 1211) may reflect convergence speed; those seeds were not continued.
- Intervals treat cells that share seeds as independent and are anti-conservative for pooled and difference quantities.
