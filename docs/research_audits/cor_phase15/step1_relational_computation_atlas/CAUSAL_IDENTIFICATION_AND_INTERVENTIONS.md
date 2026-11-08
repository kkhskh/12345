# Causal identification and interventions

> **Revised after reviews R1 and R2.** See `ADJUDICATION.md`. Five claims below are corrected:
> - **reachability** (§2.5): patched states are partly off-manifold;
> - **natural necessity** (§2.3): the relation-specific share is about 0.21;
> - **Shapley invariance:** withdrawn;
> - **the 'candidate-evidence site'** is the answer code;
> - **the pattern 'target specificity' test:** withdrawn, because it cannot fail.
>
> The original numbers are kept where they are still correct, with labels.

This document separates what the T4 models do **naturally** from what internal surgery can **force** them to do. Every intervention family is classified by what it can establish.

**Raw records.**
- `logs/mech/*.json`: head subsets, Shapley values, reachability, input invariances, site specificity, on-support matched interventions.
- `../step1_latent_reliability/logs/replication/eval_s*.json`: the preregistered replication.

## 1. Intervention taxonomy and what each establishes

| # | family | construction | on support? | what it can establish | what it cannot |
|---|---|---|---|---|---|
| G1 | natural source edit | change one report in another question (`pp1_sign.py`) | yes for 69%; 31% break the colluder's perfect agreement (zero generator probability) | the model's **behavioural** response to a real input change; comparable to exact Bayes | which internal path carries it |
| G2 | conditional resampling (`cond`) | X′_q = X_q; X′_{−q} ~ P(· \| X_q) | yes, by construction | natural counterfactual of "different evidence about R, same target evidence" | mediation, unless combined with G6–G9 |
| G3 | count-preserving resample (`countpres`) | same agreement pattern, fresh values | yes | the natural **invariance**: Bayes changes its argmax in 0.5% of cases (ties) | — |
| G4 | histogram permutation (`perm`) | permute the 6 reports within each other question | **T4: no for 98.2% of hybrids**; T1: yes | in T4, only an **off-support stress test**. The earlier `perm` IIA results (replication R2, K3′) are not in-distribution counterfactual evidence in T4 | in-distribution claims in T4 |
| G5 | **on-support matched aggregate (`pairswap`)** (new) | swap the true reliable and true noise groups' report pairs in the other questions | yes (0% undefined) | **relation changes with every histogram fixed, on support** | — |
| G6 | activation interchange, subspace | patch the 2-D key and 5-D query contrast subspaces at the input to the last layer | internal; reachability measured (§3) | sufficiency of the subspace for transmitting the G2/G5 change | natural necessity |
| G7 | single-token activation patching of head outputs (called "path patching" earlier; review R2-g) | patch the chosen heads' outputs at readout q only; the final MLP still runs | internal | which heads *carry* the cross-question effect into the decision | upstream origin |
| G8 | attention-pattern patch | the readout-q row of the last layer | internal | whether the *routing operation* carries it | value-path contributions |
| G9 | freeze heads during G6 | the chosen heads' patterns are held at base values | internal | **mediation of the artificial patch** by those heads | natural necessity |
| G10 | resample ablation | replace the chosen heads' outputs or patterns at all readout tokens with those from an independent episode | internal, naturally reachable values (other episodes' activations) | **natural necessity**, measured by accuracy loss | identity of the replacement computation |
| G11 | mismatched source | the G6 source comes from a different base episode | internal | null baseline for G6 (review MC14) | — |
| G12 | random equal-norm displacement | — | internal, off-manifold | reachability reference | — |

## 2. Results by question

### 2.1 Does the model respond to relations, holding aggregate votes fixed, on support?

(Review R2-i, reviewer computation: Cohen's κ for change/no-change is 0.79. Where both the model and Bayes change, the model picks Bayes's new answer in 81% of cases.)

Yes. Under G5 (`pairswap`, on support, all histograms identical):

| model | model argmax change at q | Bayes change | model–Bayes agreement on whether to change |
|---|---|---|---|
| Transformer seed 10 | 0.297 | 0.227 | 0.898 |
| seed 11 | 0.278 | 0.227 | 0.921 |
| seed 12 | 0.345 | 0.227 | 0.852 |
| seed 13 | 0.274 | 0.227 | 0.926 |
| DeepSets seed 0 | 0.349 | 0.227 | 0.849 |

- **Relation sensitivity.** The relation-only change moves decisions in every model, in the direction Bayes does (85–93% agreement).
- **Over-reaction.** Every model reacts more often than Bayes. Its decisions are less stable than the posterior warrants under relation changes.
- **On-support versus off-support.** This is the on-support replacement for the G4 evidence.

### 2.2 Is the model invariant to aggregate changes that preserve every relation (natural)?

Mostly. Under G3, the argmax changes in 3.6–5.8% of questions across the T4 Transformers and DeepSets (the Bayes rate is 0.5%). Approximate count sufficiency holds; it is not exact.

### 2.3 Which components naturally matter? (G10, natural necessity)

**Correction (review R2-b).** Independent-episode resampling removes question q's *own* evidence as well as the relation. The relation-specific version (`logs/mech/fixr2_s*.json`) resamples all last-layer head outputs at readout q from a cond **partner** with the same X_q:
- accuracy falls from 0.745–0.749 to **0.531–0.544**, below own-question Bayes (0.578);
- the relation-specific drop is **0.20–0.22**;
- the independent-episode drop is 0.38–0.41.

The table below gives the *total* (independent-episode) necessity.

| seed | accuracy (clean) | heads ranked by single-head ablation drop | all last-layer heads ablated |
|---|---|---|---|
| 0 | 0.751 | h0 0.26, h1 0.06, h2/h3 ≈ 0 | 0.356 |
| 1 | 0.748 | h1 0.23, h3 0.07 | 0.372 |
| 2 | 0.755 | h1 0.40 | 0.368 |
| 10 | 0.747 | h0 0.11, h2 0.08, h3 0.06 | 0.338 |
| 11 | 0.749 | h2 0.12, h0 0.04, h3 0.01 | 0.362 |
| 12 | 0.744 | h3 0.11, h0 0.09 | 0.492 |
| 13 | 0.745 | h3 0.28, h0 0.03 | 0.375 |

**Redundancy in seed 10 (mandate A4).** Heads 0, 2 and 3 are partially redundant routing heads. Head 1 is inert: its Shapley share is ≤ 0.01 in path, ablation and pattern ablation alike.

The two measures combine differently:
- **Path sufficiency (G7) is additive.** Single heads give 0.36, 0.30 and 0.30. Pairs give 0.61–0.68. All three give 0.93.
- **Natural necessity (G10) is super-additive.** Ablating heads 0, 2 and 3 one at a time costs 0.11, 0.08 and 0.06. Pairs cost 0.25–0.28. All three cost 0.41.

Super-additive necessity together with additive sufficiency is what a **thresholded readout over summed evidence** produces. Removing one head lowers the margin, but the argmax often survives; removing several flips it.

This is **not** the dormant "backup head" self-repair reported for IOI (Wang et al. 2022) or the Hydra effect (McGrath et al. 2023). There is no downstream layer to compensate, and all three heads are active at once.

**Consequence for criteria.** Any fixed-k head criterion depends on the number of redundant heads, which is a seed-specific fact:
- the preregistered R3 failed on seed 10;
- the exact head split I5 shows such criteria are not function-invariant.

The invariant summary is:
- the "all last-layer heads" set, which in every seed carries 0.87–0.95 of the cross-question effect by single-token activation patching of head outputs.

Per-head Shapley shares are descriptive only. They are **not** conserved under the exact head split (review R2-c).

### 2.4 Mediation of the artificial patch versus natural necessity (mandate G)

Freezing heads during the tag patch (G9) measures whether the *patch's* effect runs through those heads. Resample ablation (G10) measures whether the heads matter *naturally*. In these models both point to the same heads:
- In seed 10, the heads whose freezing removes the patch effect (Shapley-largest) are the heads whose ablation costs accuracy.
- Correlation of per-head Shapley values across the two measures: every head with a ≥ 0.1 mediation share has a ≥ 0.03 natural-necessity drop, and vice versa (`logs/mech/s*.json`).

**Correction (review R2-h).** Per-head *mediation* Shapley values were not computed. Only path and ablation Shapley values exist. The correlation claim above rests on the head-subset table, not on mediation Shapley values.

**What the agreement shows and what it does not.** The patched mediator is not an artefact that the model never uses naturally. It does not show that the patched subspace *is* the natural variable. The patch is sufficient on 75–86% of informative pairs, so up to a quarter of the natural effect may run through other coordinates.

### 2.5 Are patched states internally reachable? (CORRECTED; review R2-a)

**Correction.** The pooled-covariance test below is **invalid**: the pooled covariance is dominated by position and value differences. The per-position re-run (`logs/mech/fixr2_s*.json`) gives:

| state | fraction beyond the per-position natural p99 |
|---|---|
| held-out natural | 1.3–1.6% |
| cond tag patch | **45–60%** |
| mismatched source | 67–75% |
| random equal-norm displacement | 77–90% |

In addition, 87.5% of patched episodes carry tags that disagree across questions, a state that never occurs naturally (reviewer computation).

**Conclusion.** Tag-patch states are **partly off-manifold**. They are closer to natural states than the null or random displacements are, but patch-sufficiency results are internal surgery, not natural counterfactuals.

The original (invalid) analysis follows.

Measured with the Mahalanobis distance of the patched source-token states under the natural covariance at the input to the last layer, against the natural 99th percentile:

| patch | fraction beyond the natural p99 |
|---|---|
| cond tag patch (G6) | 0.3–1.1% (natural: 1%) |
| mismatched source (G11) | 0.7–1.9% |
| random equal-norm displacement (G12) | 66–79% |

- ~~**The G6 patch states are on-manifold.** The patch effect is not an off-manifold artefact.~~ Withdrawn; see the correction above.

### 2.6 Does a downstream path bypass the mediator?

**Site specificity** (G6/G7/G8 under target pairs, `logs/mech/spec_s*.json`):

| site | cond (relation change) | target (own reports change) | reading |
|---|---|---|---|
| last-layer **attention output** at readout q, full | IIA 0.87–0.94 | 0.88–0.97 (seed 12: 0.34) | **the answer representation** (review R2-d): with K = 3 the 2-D subspace is the candidate-score code; context and own-report changes act through the same subspace (cos 0.99–1.00) |
| the same, 5-D cond-contrast subspace | 0.85–0.93 | 0.87–0.96 | same |
| last-layer readout **attention pattern** | 0.84–0.85 (seed 12: 0.64) | 0.01–0.05: **not informative**, since a pattern patch keeps base values and so cannot carry own-report changes (review R2-f) | carries the cond effect, so the routing operation transmits reliability |
| key tag subspace (replication R2) | 0.76–0.84 key-only | change rate 0.054–0.075 | relation-specific *representation* (seeds 10, 11, 13) |

**Bypass.**
- In seeds 10, 11 and 13 the relation reaches the decision through the routing pattern, and essentially nothing bypasses it. Freezing all last-layer patterns cuts the tag patch to IIA 0.10–0.12.
- In seed 12, part of the cross-question evidence reaches the readout through the **value path** of two heads that attend to other questions' tokens. Pattern patching transfers only 0.64, and the tag patch only 0.63. Freezing patterns removes most of it (0.19).

**Upstream reconstruction.** Can earlier layers reconstruct what a last-layer patch removes? Not in this architecture: the patched heads are in the last layer, and only the final MLP follows. A patch at the input to the last layer can be "undone" only by that MLP. The site × location sweeps in the latent-reliability package show no restoration at the final site.

## 3. Distributed and localised identification: what survives

1. **Behavioural (G1–G5).** Identified without any internal convention, and implementation-independent (PP1). This is the most robust level.
2. **Operation level** (invariance holds only for the listed exact symmetries; review R2-j). The last-layer readout attention pattern is the relation-carrying *operation* in 6 of 7 Transformer seeds:
   - pattern patch IIA ≥ 0.84 with target transfer ≤ 0.05, in all seeds except 12;
   - in seed 2 it holds even though the tag *representation* is split between keys and query.

   Identifying the routing operation does not require choosing a representation basis, because attention patterns are basis-free (a softmax over positions). This is the most invariant internal object found.
3. **Representation level.** The key-side tag (2-D, contrast-identified) is localised in seeds 0, 1, 10, 11 and 13; split between key and query in seed 2; and largely absent in seed 12. Its location and dimension depend on the seed. Under I6 (query/key placement), even its *side* is not a function-level property.
4. **Head level.** Not invariant (I5). Report Shapley shares, not head identities or counts.

**Dependence on the selection method.** Every subspace in this document is chosen from input-level contrast pairs (cond), never by maximising the evaluated logit response. Head sets come from all-subset enumeration, not top-k selection. The remaining convention is the linear-subspace class (I7).

## 4. Remaining causal problems (unresolved)

- **Natural necessity of the tag subspace itself.** Resample-ablating the tag injects wrong tags (drop_frac > 1; review MC18). A mean-ablation or conditional-mean ablation of the tag has not been run.
- **Seed 12's value-path variable.** The cross-question information that reaches the readout through values has not been localised to a low-dimensional variable. The 5-D contrast subspace of the attention output carries it (0.86), but that site is a candidate-evidence site.
- **A matched comparison of internal mediation against data-generating interventions.** G5 changes the latent assignment's evidence. The ideal comparison, "set R's posterior to a value while holding X fixed", has no input-level realisation, because the posterior is a function of X. No exact matched internal-versus-generative comparison exists for a *deterministic* function of the input. That is why mediation is compared against G2/G5 counterfactuals instead. (This is the obstacle the mandate asked to state rather than paper over.)
