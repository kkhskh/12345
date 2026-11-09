# Representation and causal identification (mandate §6, §7, §8 errata)

Status: post hoc and exploratory, except where a result is marked as theorem or construction.

Revision: revised after review round R2 (`reviews/ADJUDICATION_R2.md`); corrected claims are marked [R].

Scripts:
- `scripts/class_shapley.py`;
- `scripts/s12_operations.py`;
- `scripts/pp2_dev.py`;
- `scripts/pp2_mechanism.py`.

## 1. An identification framework robust to head splitting and gauge (§6)

**Problem (atlas I5, R2-c).**
- Head-level attributions are not properties of the computed function.
- An exact Net2Net split of a head changes head counts and head-level Shapley values while computing the same function.
- The atlas tried summing head-level Shapley shares by pattern class. It was withdrawn because the sums are not conserved.

**Definition (quotient game).**
- Fix a site: the outputs of the heads of one layer at a token set.
- Fix a value function v(S): the effect of patching the outputs of head set S from a counterfactual run. Here it is the mean KL recovered.
- Partition the heads into **pattern classes**: connected components of the relation "the two heads' attention patterns at the site are identical on the evaluation distribution".
- The **quotient game** has the classes as players: v̄(T) = v(∪_{C ∈ T} C).
- **Class attribution** is the Shapley value of the quotient game.

**Theorem R1 (split and gauge invariance).** Class attributions are unchanged under:
- (a) exact head splits (duplicate W_Q, W_K and W_V; split W_O as W_O = W_O′ + W_O″);
- (b) within-head gauge transforms W_V → W_V G, W_O → G⁻¹ W_O;
- (c) any composition of (a) and (b).

*Proof.*
1. (b) leaves every head output unchanged, so v is unchanged. [R] At this site (head outputs after W_O), the gauge part is therefore vacuous: head-level Shapley is also gauge-invariant (CA-F13). R1's content is the split invariance.
2. Under (a) both copies have the pattern of the original head, so they land in the same class.
3. A head's output is linear in W_O given its pattern. So patching the class in the split network gives exactly the same residual as patching the original head.
4. Hence v̄ is identical as a set function on identical players, and so is its Shapley value. ∎

**Check** (seed 10; split of head 0; random gauge with condition number about 4; `logs/class_shapley_s10.json`):

| network | head-level Shapley | class-level Shapley (τ = 0.99) |
|---|---|---|
| original (4 heads) | 0.354, −0.002, 0.322, 0.316 | 0.354, −0.002, 0.322, 0.316 |
| split (5 heads) | **0.177**, −0.002, 0.321, 0.316, **0.177** | **0.354**, −0.002, 0.322, 0.316 |
| gauge | unchanged | unchanged |
| split + gauge | as split | **0.354**, −0.002, 0.322, 0.316 |

Head-level shares of the *other* heads also move slightly under the split: 0.322 → 0.321. This is the R2-c non-conservation. Class shares do not move.

**Limits (stated, not hidden).**
- **L1.** R1 covers pattern-preserving reparametrisations only. A function-preserving change that moves attention mass between positions with equal value outputs (atlas review R2-j) changes patterns. It can move a head out of its class, and then R1 does not apply.
  - [R] The pre-review claim that no attribution can avoid this is tautological under the reading used (FM-F8).
  - Classes defined by the attention mass over *value-equivalence classes of positions* may be invariant to both kinds of change. This is untested (OPEN_FOUNDATIONAL_MAP OF-5).
- **Prior art.** R1 is the quotient-game step of the Owen value with exact clones merged (Owen 1977; Knudsen–Østerdal 2012). Here no two *original* heads share a pattern, so class Shapley equals head Shapley before the split.
- **L2.** Pattern equality must be exact for the theorem. Tolerance-based classes (τ < 1) are a modelling choice. At τ = 0.9 the seed-10 classes did not change, because no two original heads have pattern correlation above 0.9, even though heads 0, 2 and 3 all attend about 98% to the own question.
- **L3.** R1 is about *where* (which class of operation), not *what variable*. A variable-level identification still needs an alignment class (atlas I7). The atlas convention stands: linear subspaces of dimension ≤ 5–10 at named sites.
- **L4.** The site-level effect (all heads together) is trivially invariant. R1 is the finest invariant partition of that effect found here.

## 2. Intervention targets: natural necessity versus patchability (§7)

Each intervention type is listed with its target, its support class (PP1 §1) and what it can identify. The same internal site can be necessary but not sufficient (seed 12).

| intervention | support of the intervened state | identifies | does not identify |
|---|---|---|---|
| input edit inside S2 (natural counterfactual) | natural for the same world | the effect of a natural change in the evidence | the internal route |
| input edit in S1 \ S2 | possible under the generator, not for this world | the Bayes effect is defined; model sign agreement 0.968–1.0 | naturalness |
| input edit outside S1 | off-support | behaviour only (stress test); no Bayes target | any correctness |
| mean or conditional-mean ablation of a subspace (OP-5) | off-manifold activations (the means of a subspace) | **mean-replacement necessity** [R: renamed from "natural necessity"]: the effect of removing the subspace's variation. It is mediated partly by LayerNorm renormalisation (CA-F4), so it must be compared with a *variance-matched* control (CA-F3) | which values of the subspace are used; specificity by magnitude |
| resample ablation | another episode's activations; on-manifold per token, inconsistent jointly | necessity, *plus* the effect of injecting wrong values (drop_frac > 1) | clean necessity |
| interchange (cond pairs) of a subspace or head output | counterfactual activations, inconsistent with the rest of the base state | **patchability / sufficiency** of the site for carrying the counterfactual change | necessity; whether the site is the natural route |
| pattern freeze during a patch | base patterns with counterfactual values | mediation of the patched change by routing | anything about the *natural* role of routing |
| pattern patch (target pairs) | — | **uninformative** for target specificity: a pattern patch keeps base values, so it cannot carry own-report changes and cannot fail (atlas R2-f) | — |

**Mean-replacement necessity and patchability come apart** (seed 12, tag subspace; live LN):

| measure | seed 12 | seed 10 |
|---|---|---|
| mean-replacement necessity (drop fraction, OP-5; live LN) | 0.91 | 0.65 |
| same, with LN statistics frozen (reviewer-reported) | 0.63 | 0.71 |
| same, with last-layer patterns frozen | 0.40 | 0.02 |
| patchability (interchange of the tag at all 36 tokens) | 0.47 | 0.77 |

A site can be necessary without being sufficient when other channels carry the counterfactual change outside it. Seed 12 has at least three such channels (COMPUTATIONAL_CLASSES §1). The reverse also occurs: a site can be patchable without being the natural route. That is the atlas I6 construction, still a conjecture.

## 3. A causal graph from generative variables to logits (§7)

Generative level (exact): R = (reliable g, colluder h) → X ← Y, with the questions conditionally independent given R.

Learned level, class A′ (seeds 0, 1, 10, 11, 13). Every edge is supported by an intervention.

```
X_{q,s} ──► [L0–L1: within-pair agreement features]   pair-agreement probes (latent-reliability package); not re-tested here
   │
   ▼
t_g (2-D tag at every source token of group g)         contrast-identified; interchange 0.77 at own6 (s10);
   │                                                   mean-ablation removes 0.54–0.68 (s10, s11, s13)
   ▼
routing pattern at readout q over q's sources (L2 heads 0, 2, 3)   pattern patch 0.84–0.85; freezing removes the cond effect
   │        ▲
   │        └── values carry X_q's report identities
   ▼
attention output at readout q (2-D candidate-score code)   IIA 0.87–0.94; a candidate-evidence site, not a relation variable (R2-d)
   ▼
logits(q)
```

**Class C (seed 12).** Add a parallel edge X_{−q} (full residual) and t_g → head 0 values → readout q. Head 3 carries the A′ route at reduced strength. The readout's own residual from L0–L1 contributes as well (§COMPUTATIONAL_CLASSES 1).

**Class A (DeepSets).** X_q → φ → pool m (episode summary) → ρ(h_q, m, X_q) → logits.
- A probe on m decodes the colluder at 0.96–0.97.
- [R] On p_c = 0 the probe picks the top-agreement group in 100% of cases. A probe of the *same* form fitted on **raw graded agreement counts** also gives 1.00 (`logs/pp2_probe_control.json`). So this shows that m carries graded counts linearly. It does not show a colluder code with a configuration constraint.

## 4. The internal reason for off-support extrapolation (link to EXTRAPOLATION_AND_INTERNAL_STRUCTURE.md)

**Method.** A colluder probe was fitted on natural T4 representations and applied to p_c = 0 episodes.
- Every model commits to *some* colluder: mean max-probability 0.86–0.99.
- *Which* group the probe selects tracks the local index:

| model | probe picks the top-agreement group (unique top) | p_c = 0 local index |
|---|---|---|
| DeepSets s0, s1, s2 | 1.00, 1.00, 1.00 | 0.16, 0.05, 0.11 |
| TF s2, s12, s13 | 0.83, 0.79, 0.71 | 0.49, 0.31, 0.53 |
| TF s1, s11 | 0.61, 0.49 | 0.84, 0.70 |
| TF s0, s10 | 0.40, 0.40 | 0.59, 0.87 |

**Statistics.**
- Spearman with the local index: −0.89 over all 10 models, driven by the tied DeepSets cluster.
- Within Transformers: −0.71 (n = 7, p ≈ 0.07). These values were recomputed for this document; they are not in `pp2_mechanism.json`.

**Re-interpretation after review [R] (CA-F11; `scripts/pp2_probe_control.py`).** The same probe pipeline, fitted on raw input statistics instead of model states, gives:

| probe input | picks the top-agreement group |
|---|---|
| graded per-group agreement counts | **1.00** |
| per-question agreement indicators | 1.00 |
| the saturated "group agrees in all six questions" indicator | **0.39** |

The index therefore measures **which agreement statistic a representation carries linearly**. The pre-review reading, "the colluder code enforces 'the most-agreeing group must be the colluder' off support", is withdrawn. The defensible descriptive hypothesis is:
- **global** extrapolation goes with representations that carry *graded* agreement counts (DeepSets ≈ raw counts);
- **local** extrapolation goes with representations dominated by a *saturated full-agreement detector* (TF s0, s10 ≈ 0.40 ≈ the indicator's 0.39). Off support that detector is silent, so no colluder is discounted.

This is post hoc, measured on the shift distribution and statistically weak within Transformers. It is not a predictor. It suggests a probe-free test (OPEN_FOUNDATIONAL_MAP OF-4). Prior art for the architectural part: aggregation choice determines extrapolation (Xu et al. 2021).

## 5. Errata to the atlas CAUSAL / OPEN documents (§8). The originals are not edited.

1. **`CAUSAL_IDENTIFICATION_AND_INTERVENTIONS.md`, line 166.** It says "A mean-ablation or conditional-mean ablation of the tag has not been run." That is **stale**: it was run as OP-5 (`scripts/tag_mean_ablation.py`, `logs/mech/meanabl_s1{0,1,2,3}.json`).
   - Mean-ablation drop fractions: 0.65, 0.54, 0.91, 0.68.
   - Conditional-mean: 0.44, 0.66, 0.91, 0.74.
   - Random 2-D subspace: ≤ 0.03.
2. **`OPEN_FOUNDATIONAL_PROBLEMS.md`, OP-5.** It lists the mean-ablation as "Missing". **Stale**, for the same reason. OP-5's remaining open part is the joint tag+content interchange (COMPUTATIONAL_CLASSES §1).
3. **Pattern-patch target specificity.** Any use of the pattern-patch target-pair result (0.01–0.05) as evidence that routing is *relation-specific* is withdrawn. The test cannot fail (atlas R2-f; §2 above). The CAUSAL document already says so in its header. This erratum covers any remaining table entry read as a pass.
4. **"On-support" in PP1 (atlas).** It means S1, the marginal support. It does not mean natural reachability for the same world. 3,570 of the 93,570 S1 edits are S1 \ S2 (PP1 document §1).
