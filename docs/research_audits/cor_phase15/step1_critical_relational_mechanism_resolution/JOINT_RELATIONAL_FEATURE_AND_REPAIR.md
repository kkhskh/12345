# Joint relational feature and repair (Workstream C)

COR Phase 1.5, Step 1 (provisional ≈ 25%). The setting is the specified G = 3, C = 2 shared-copying generator only.

**Evidence labels:**
- [proof] mathematical proof;
- [num] numerical verification;
- [dev] the XIII-B unit uP161 s21 (late readout) and uP161 s22 (early readout);
- [conf] see FROZEN_CONFIRMATION, R4 and R5.

## C1. Mathematics, re-verified before any new neural test [proof; num] (`scripts/c_math.py`, `logs/NOTES_MATH_C.md`)

| check | result |
|---|---|
| `posterior_shared` vs an independent direct enumeration from the sampler's semantics (shared copy value and shared copy event) | max diff 2.8e-15 (β = 0.6), 1.3e-16 (β = 1) |
| **Proposition S**: given per-group statistics and the values at q*, the posterior depends on the rest of X only through m = (m_01, m_02, m_12), m_H = #{q ≠ q* : the four sources of H all equal} | proof; 2495 m-preserving relabellings change it by exactly 0 |
| **Corollary P**: log p(g reliable \| X) is linear in m_{H(−g)}, the pair not containing g, with coefficient log(1 + βK³/(1 − β)) = 3.73 at β = 0.6. That a gating mixture *needs* this per-group form is an implementation hypothesis (tested in C5), not a consequence of the mathematics | proof |
| β = 1 local sufficiency (XIII-B proposition) | 3985 cells, within-cell spread 0.0 |
| β = 0.6 gap of the local observation | **3.6 ± 0.07 points** (exact, 100,000 episodes; review 1). The first estimate of 4.4 points came from an unconverged 48-draw Monte-Carlo estimator and is withdrawn |
| J-pairs | keep every per-group statistic and the values at q*; 88.9% change m (J_DM), and the rest have exact change 0 (J_D0) |
| decoy pairs (raw first-source cross-equality changes, m fixed) | exact change 0 by S |

The old independent-copying posterior is not used anywhere.

## C2. What does early-layer adaptation construct? [dev]

The probe is a ridge regression at the layer-2 input, residualised on per-group statistics, with held-out R²:

| site | native m | native decoy | adapted (early01) m | adapted decoy |
|---|---|---|---|---|
| readout of q | 0.05 | 0.03 | 0.74 | 0.60 |
| q source tokens (mean) | 0.00 | −0.00 | 0.72 | 0.66 |
| other source tokens (mean) | 0.08 | 0.08 | 0.81 | 0.78 |

Per-token probe at the q source tokens of the exact copying-aware responsibility p(g reliable): R² 0.52 (log r_g) and 0.58 (r_g).

Adaptation creates linearly decodable cross-group equality information everywhere. It does so not only for the sufficient statistic m but
also for the decoy statistic, which correlates with it. **Decodability does not discriminate between candidate statistics.**

Behaviourally the repair is m-specific, which the probes are not:
- the J response follows the exact change (slope 0.92, R² 0.85);
- relabellings that keep m produce 0.14 of the response size of those that change m (natural, unadapted model: 0.84);
- raw-equality decoys produce 0.08.

> **Review 3 (R5 caveat).** The probe directions are ridge *decoder* directions on pooled tokens. A decoder direction can be nearly
> orthogonal to the direction the network uses for encoding. The result shows that "the decodable linear decoder directions are not the
> used directions". Encoding-direction interchange, decoding m from the DAS subspace, DAS ablation and null-pair tests are open.

## C3. Is it *used*? Causal tests [dev; conf R5]

Interchange at the layer-2 input from the J donor (fidelity slope on the exact change):

| intervention | slope |
|---|---|
| full donor (all tokens) | 0.94 |
| readout token only / q source tokens only / other source tokens only | 0.34 / 0.40 / −0.00 |
| 3-dim **m-probe span** at RO / QS / OS | 0.002 / 0.008 / 0.000 (random 3-dim: ≤ 0.024) |
| 3-dim **responsibility-probe span** at q tokens | 0.015 (random: ≤ 0.015) |
| mean-ablating the m-probe span / the 1-dim responsibility direction: remaining effect | 0.94 / 0.93 |
| learned DAS subspace at q + readout tokens, held-out pairs: k = 1 / 3 / 8 | 0.64 / 0.88 / 0.92 |
| largest cosine between the DAS subspace and the probe spans (k = 3): responsibility / m(QS) / m(RO) / random baseline | 0.34 / 0.26 / 0.21 / 0.39 |

**Answers to Q7–Q8 (development).**
1. Successful early-layer adaptation writes the copying information into the q source tokens and the readout token of the target
   question. Neither the other-question tokens nor any single token set suffices; both are needed.
2. A **low-dimensional causal carrier exists**: three dimensions carry 0.88 of the effect.
3. That carrier is **not** the linearly decodable direction of m or of the responsibility. Interchanging along those directions transfers
   nothing, and removing them removes nothing. The causal subspace is no more aligned with them than a random subspace is.
4. The constructed feature is therefore not an explicit, linearly read statistic. It is a distributed code whose causally used part lies in
   directions that linear decoders of the "natural" statistics do not find.
5. This is a direct demonstration that **decodability ≠ use**.

## C4. How the frozen layer 2 uses the repaired representation [dev; conf R5c]

**Layer-2 Shapley** over {readout attention rows, value inputs at RO / q / other tokens, MLP2, skip} on J pairs in the early01-adapted
model:
- rows 0.81;
- MLP2 0.13;
- values at q tokens 0.06;
- all other components ≈ 0.

For each mixture head, the change in attention mass on each group's q tokens correlates **0.89** and **0.87** with the change of the exact
posterior probability that this group is the reliable one. The non-mixture heads give 0.26 and 0.05.

So the adapted early layers change *which group the frozen mixture heads attend to*, in proportion to the copying-aware reliability
posterior. The frozen layer's existing gating mechanism (rows → group responsibility) is driven by new inputs. This is evidence for
"supply new eligibility/responsibility weights to the frozen mixture". It is evidence against these alternatives:
- the readout MLP computes the correction (0.13);
- the candidate likelihood representation changes (values at q 0.06).

It is still not a claim that the final layer "reuses the same algorithm" in any stronger sense.

## C5. Information access vs optimisation vs capacity: matched arms [dev; conf R4]

β = 0.6, 1000 steps, identical data and learning-rate schedule for every arm, J slope on m-changing pairs. Bayes accuracy is 0.792;
native 0.469.

| arm | trainable params | acc | J slope (R²) | what it tests |
|---|---|---|---|---|
| L2 (layer 2 + readout) | 50,048 | 0.722 | 0.000 (0.00) | XIII-B failure, reproduced |
| L2plus (L2 + an extra block, zero-init) | 99,776 | 0.723 | 0.001 (0.00) | **capacity matched to early01: still fails** |
| early01 (layers 0 + 1) | 99,456 | 0.783 | 0.920 (0.85) | XIII-B success, reproduced |
| L1only (layer 1 only) | 49,728 | 0.701 | 0.000 (0.00) | one block, early, but downstream of layer 0: **fails** |
| **L0only (layer 0 only)** | 49,728 | **0.786** | **0.847 (0.80)** | one block, matched to L2: **succeeds** |
| preattn (one attention-only adapter before frozen L2) | 16,512 | 0.731 | 0.142 (0.16) | cross-token mixing at the layer-2 input: weak |
| pretok (token-wise MLP adapter before frozen L2) | 16,640 | 0.659 | −0.001 | no cross-token mixing: fails |
| supL2: exact m supplied group-agnostically at RO + q tokens; L2 trainable | 50,240 | 0.732 | 0.024 (0.09) | **information present at the L2 input: still fails** |
| **supG: same information in per-group form (m_{H(−g)} and g's pairs at g's q tokens); L2 trainable** | 50,240 | **0.787** | **0.802 (0.81)** | **format fixed: succeeds** |
| supGFrozen / supFrozen (only the input map trainable) | 192 | 0.548 / 0.504 | 0.144 / 0.075 | the frozen C = 1 readout cannot be steered by an input map alone |
| supDecoy (decoy statistic supplied) | 50,240 | 0.728 | 0.013 | control |

**Answer to Q9 (development; confirmation in R4).** The final-layer repair does not fail for lack of capacity: two trainable blocks above
layer 1 fail. It does not fail for lack of information either: m supplied at the layer-2 input fails.

It fails because the relation does not reach layer 2 **in the per-group form its gating can use**. Three results show this:
- the same information in per-group form makes the layer-2 repair reach Bayes;
- one trainable block at layer 0 also does — that is where cross-group comparisons at each question can be written into each group's
  tokens, ahead of the frozen layer-1 routing;
- one block at layer 1 does not.

Review 1 points out that supG and supL2 differ in the information each token receives (group-specific values at q tokens vs one vector at
the readout and q tokens), not in format alone. The contrast is therefore "per-group at the group's tokens" against "group-agnostic". This
is a *format/position accessibility* constraint, a refinement of XIII-B's "accessibility at the readout position". It is not a claim
of architectural impossibility: a learning-rate search was not run for the failing arms. ~~The strongest single-optimisation alternative is excluded, because supG and L2 use the same optimiser~~. Withdrawn after confirmation and review 3 (M5): supG failed at the frozen budget, so learning speed matters.

## C6. Across organisations [dev; partial]

- uP161 s22 (the early-readout sibling of s21, same parent): its L2 repair gives J slope 0.072 (acc 0.736). This is unlike XIII-B's early
  units (0.53, 0.63).
- The remaining arms on this unit are in `logs/adapt3_dev/`, exploratory, completed after the freeze if at all.
- Whether repair success tracks the readout organisation is tested prospectively in R4 (early arm) on a new early unit.
- The early-readout L2 advantage reported in XIII-B is therefore **not established as a class property**. It holds in some early units and
  not in the uP161 s22 sibling.

## Confirmation (frozen)

**R5 confirmed** on the new late unit uP161 L20/D23:
- a learned 3-dim carrier transfers 0.80 (8-dim: 0.88);
- the m and responsibility probe spans transfer 0.009;
- frozen rows carry 0.815;
- Δ mass vs Δ role posterior correlates 0.84 and 0.89.

**R4 falsified**, in two arms:
1. **supG at the frozen 1000 steps reached only 0.28.** The other five late-unit arms behaved as predicted:
   - L2 and L2plus (capacity-matched) fail;
   - L1only (one block) fails;
   - L0only (one block) succeeds at 0.91;
   - group-agnostic supply fails.

   *Exploratory, post-freeze:*
   - supG at 3000 steps reaches 0.895 (acc 0.790), and at lr 3e-3 with 1000 steps 0.934 (acc 0.791);
   - L2 stays at −0.002 and −0.001 under the same two settings.

   The plain layer-2 failure therefore survives a 3× step budget and a 3× learning rate, which excludes the simplest optimisation
   explanation. The supG failure at the frozen setting was optimisation speed.
2. **The early unit's layer-2 repair reached 0.378,** against a threshold of 0.4. That is partial use. The dev sibling uP161 s22 gave 0.072
   and XIII-B's early units 0.53/0.63.

**Post-review (review 3, M5; exploratory, `logs/expl_post_review.md`).** The group-agnostic supply supL2 also succeeds:
- 0.931 at 3000 steps;
- 0.902 at lr 3e-3.

The "per-group format" claim is **withdrawn**. Plain L2 fails under all three settings.

**Revised answer to Q9 (supersedes the text below where they differ).**
- Capacity is excluded.
- Simple optimisation (3× steps, 3× learning rate) is excluded for the plain layer-2 repair.
- **Availability of the joint relation at the layer-2 input decides.** It can be computed upstream (one block at layer 0 suffices; one at layer 1 does not) or supplied in any tested form, which a layer-2 repair can then learn to use, more slowly.

**Earlier answer (pre-review), kept for the record:**
- Capacity is excluded (two blocks above layer 1 fail) and information presence is excluded (m supplied at the layer-2 input fails).
- What makes a repair possible is **where and in which form** the joint relation enters:
  - a one-block repair at layer 0 suffices;
  - a layer-2 repair can use the relation if it is supplied per group, but learns it slowly (0.28 at 1000 steps, 0.90 at 3000);
  - group-agnostically it does not learn it at all within 3000 steps (exploratory for the 3000-step arm).
- "Per-group format" is therefore a learnability/accessibility advantage, not a strict necessity condition.
- **Q10 (does organisation change repair ability?):** early-readout units' layer-2 repairs use the relation partially and inconsistently
  (0.07–0.63). This is not established as a class property.
