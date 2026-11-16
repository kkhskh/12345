# Information access and repair (Workstream D)

COR Phase 1.5, Step 1, Investigation XIII-D. These are interpretability results on small trained transformers. They do not validate COR, and they make no claim about the O/D operators.

## The two-colluder mathematics (XIII-C, unchanged)

**Generator.**
- G = 3 groups, C = 2 sources per group, K = 3 values, Q = 6 questions, α = 0.8.
- In each episode one group is reliable. The other two collude: at each question, with probability β, all four of their sources copy one shared value.

**Posterior.** The exact posterior is `xii_bayes.posterior_shared`. It was verified against independent enumeration to 2.8e-15.

**Proposition S.** Hold fixed each group's self-agreement pattern and the values at the scored question q*. The posterior then depends on the rest of the episode only through m = (m_01, m_02, m_12), where m_H counts the other questions at which all four sources of the pair H agree. 2495 m-preserving relabellings change the posterior by exactly 0.

**XIII-C math-review corrections, carried over:**
- The exact LOCALQ gap is 3.6 ± 0.07 points on 100,000 episodes.
- That a gating mixture *needs* per-group format is an implementation hypothesis. It does not follow from the mathematics.
- At β = 0.6 the J-pair on-support check is vacuous.

XIII-D adds no new mathematics here and changes none of these statements. The math reviewer re-checked them (`reviews/`).

## Question

XIII-C reported that a frozen mixture model's failure on the two-colluder task is about **availability, not capacity**. Retraining layer 2 + readout fails, while supplying m succeeds. Two loose ends remained:

1. The successful supply arms used a stronger optimisation budget than the failing ones. Development had shown that supL2 fails at lr 1e-3 / 1000 steps but succeeds at lr 3e-3.
2. It was unknown whether the supply position matters, and whether layer 0 is special.

## Design (freeze 3a, commit `da9e569`)

- **Unit.** uP165 s20, a new mixture model never adapted in any earlier package.
- **Adaptation.** `adapt4.py`: β = 0.6, 1000 steps, identical training and evaluation episodes for every arm, AdamW with warmup-cosine.
- **Metric.** J_DM slope, i.e. observed versus exact posterior change on 1767 pairs that change m with the marginals fixed.

**Arms.**

| arm | trainable | supplied information |
|---|---|---|
| L2 | layer 2 + readout | none |
| supL2 | layer 2 + readout + E | m (group-agnostic) at the readout token and the 6 q-source tokens |
| supRO | same | m at the readout token only |
| supQS | same | m at the 6 q-source tokens only |
| L0only | layer 0 (one block) | none |
| L1only | layer 1 (one block; capacity-matched) | none |
| L0attn / L0mlp | only layer 0's attention / MLP | none (descriptive) |

## Results

| arm | lr | acc | KL | J_DM slope | J_D0/J_DM | DECOY/J_DM |
|---|---|---|---|---|---|---|
| native (no training) | — | 0.458 | — | −0.001 | — | — |
| L2 | 3e-3 | 0.729 | 0.178 | **−0.000** | 0.86 | 0.63 |
| supL2 | 3e-3 | 0.790 | 0.012 | **0.906** | 0.04 | 0.03 |
| supRO | 3e-3 | 0.791 | 0.013 | **0.894** | 0.04 | 0.03 |
| supQS | 3e-3 | 0.784 | 0.031 | **0.727** | 0.06 | 0.04 |
| supL2 | 1e-3 | 0.753 | 0.130 | **0.168** | 0.25 | 0.17 |
| L0only | 1e-3 | 0.786 | 0.030 | **0.832** | 0.18 | 0.07 |
| L1only | 1e-3 | 0.727 | 0.188 | **0.004** | 0.93 | 0.48 |
| L0attn (descriptive) | 1e-3 | 0.786 | 0.030 | **0.818** | 0.17 | 0.07 |
| L0mlp (descriptive) | 1e-3 | 0.721 | 0.207 | **0.008** | 0.82 | 0.29 |

Exact Bayes accuracy is 0.792.

**P5a (access, not capacity, and budget dependence): FALSIFIED on one arm.**
- At lr 3e-3, supplying m is sufficient (0.906), and the same trainable parameters without m achieve nothing (0.000). This is the access result, replicated in a new parent.
- The predicted budget dependence was "supL2 at lr 1e-3 < 0.1". It came out 0.168: still far below 0.906, but above the threshold.
- So supplied information is *partly* usable at the weaker budget in this model. The XIII-C statement "supply fails at lr 1e-3" (0.024–0.028 in two other models) does not hold as a categorical claim.

**P5b (supply position): CONFIRMED.**
- Readout-token supply gives 0.894.
- Source-token supply gives 0.727.
- The prediction was motivated by the carrier trace (P3e), which was known from an interim look at freeze-2 outputs (see FROZEN, disclosed events). It was not calibrated on a development run, because that run was stopped before producing an arm.
- **Interpretation limit (review 2, F10).** No trace was run on the supply arms. m added at the readout token enters the readout residual directly, so the trainable layer-2 MLP and readout could use it without m ever reaching the attention score. "Query side" and "key side" are therefore labels for *where m was supplied*, not demonstrated mechanisms.

**P5c (layer-0-specific repair): CONFIRMED.**
- At matched capacity (one block), the same budget (lr 1e-3) and the same episodes, retraining layer 0 repairs the relation (0.832). Retraining layer 1 does not (0.004).
- This replicates XIII-C (L0only 0.85 / 0.91, L1only 0.00) in a new model (uP165 s20; the same parent supplied C-C2).
- **Budget limit.** P5a shows that the learning rate changes outcomes qualitatively. A post-review exploratory arm, L1only at lr 3e-3, was run for this reason; its result is in "Post-review exploratory results" below.
- **Descriptive:**
  - layer 0's attention alone (16.5 k parameters) gives 0.818;
  - layer 0's MLP alone gives 0.008.

## Interpretation

**What a repair must do.** In the early01-repaired models (Workstream C), m reaches the decision through the layer-2 key–query score. Three interventions repair the relation in this model:
- supply m directly, at the readout token or at the source tokens (the pathway by which supplied m acts was not traced);
- retrain layer 0, which then writes m into the residual stream that layer 2 reads;
- retrain layers 0–1 (early01, Workstream C).

**What fails.** At lr 3e-3, layer 2 + readout cannot compute m from what they receive, even with full retraining. At lr 1e-3, layer 1 alone cannot either (for lr 3e-3, see the post-review result).

**Why layer 0 and not layer 1: two hypotheses, neither tested.**

- **(i) Access.** A retrained layer-0 *attention* is enough (0.818, while the MLP alone gives 0.008). That points to cross-token gathering as what is needed.
- **(ii) Depth (review 2, F10).** A retrained layer 0 is followed by a frozen layer 1 that already pools across questions. A retrained layer 1 would have to do both steps in one block. "Layer 0 has direct access to the sources" does not discriminate between the two, because layer 1's residual also contains the source embeddings.

The arm that would separate them, "layer 1 plus one extra fresh block", was not run.

**Repair versus implementation.** The same parent can be repaired along different routes: supply, layer 0, or layers 0–1. The repaired model then carries m in a 3-d subspace that layer 2 uses through keys and queries (Workstream C). Which route the repair takes, and where m ends up, depends on what is trained. It does not depend on the parent's original organisation: all repair units were mixture models.

**Same parent, different organisation (XIII-C data, re-read here).** Layer-2 retraining is not uniformly useless; it depends on the organisation:

| model | organisation | L2 slope | early01 slope |
|---|---|---|---|
| uP161 s21 | mixture | 0.000 | 0.92 |
| uP161 s22 | early | 0.072 | — |
| uP163 L22/W22/D21 | early | 0.378 | 0.833 |

So repairability by layer 2 depends on the implementation the model adopted. Plausibly this is because the early organisation's readout-token representation already contains part of what m needs, but that was not tested.

XIII-D ran no new adaptation of an early model. These comparisons come from XIII-C's frozen runs (`adapt3_conf`, `adapt3_dev`), not from a new confirmation.
