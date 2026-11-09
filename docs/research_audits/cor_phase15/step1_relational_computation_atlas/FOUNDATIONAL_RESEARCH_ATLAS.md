# Foundational research atlas: relational computation, causal decisions, identifiability and generality

> **Revised after three independent reviews** (`ADJUDICATION.md`). The most important revisions:
> - **E2 is a calibration consequence.** The cross-implementation decision law follows from every model being KL-close to Bayes on the support (the known PFN result). It is not an independent law.
> - **E3 is underspecification.** The off-support divergence is an instance of known underspecification (McCoy et al. 2020).
> - **The reversal is a known effect.** It is a quantitative instance of the "paradox of unanimity" (Gunn et al. 2016) and of copy detection (Dong et al. 2009).
> - **PP3 is partial.** It is consistent with early-layer transfer but does not separate the relation features from within-question parsing.
> - **The T4-barrier explanation is withdrawn.** The barrier is architecture-dependent.
> - **Tag patches are partly off-manifold.**
>
> By the mandate's distinctiveness standard, the package's **new** contribution is small (§7, "Distinctiveness").

**Phase.** Phase 1.5, Step 1. This is a research map, not a thesis. Historical verdicts are unchanged (see `ORIGINAL_COR_FAILURE_MATRIX.md` §0).

**Evidence labels.**

| label | meaning |
|---|---|
| **[T]** | theorem with proof (and numerical check) |
| **[C]** | prospective, frozen-before-test, confirmed |
| **[F]** | prospective and failed (kept) |
| **[E]** | measured, not prospective |
| **[PH]** | post hoc |
| **[—]** | not examined |

Every claim points to a supporting document and log. Reviews and dispositions are in `ADJUDICATION.md`.

## 0. The map in one page

```
LEVEL 1  GENERATIVE / BAYES  (what the data make decision-relevant)
  roles with agreement probabilities  ──Theorem S──▶ agreement COUNTS are sufficient for roles
        (needs (U) symmetric errors, (O) one answer-dependent group)
  counts ──Theorem U──▶ role posteriors are unimodal in a group's own count
        ├─ no higher-agreement role (a_coll ≤ a_rel): monotone (Corollary U2)    [T1 regime]
        ├─ a higher-agreement role with prior mass: reversal at high agreement    [T4 regime; phase diagram]
        └─ equal agreement rates: roles unidentifiable (I2)                       [no estimator can tell]
  counts + X_q ──Theorem D──▶ exact margin change = log E_{π^c}[r] − log E_{π^c'}[r]
        └─ no generator-free sign (D2): the same relational change has opposite optimal effects in T1 vs T4

LEVEL 2  BEHAVIOUR OF LEARNED MODELS   (the implementation-independent level)
  E1  Bayes-level accuracy (all T4-capable models)
  E2  the same margin-response law to natural relation edits:
      PP1 [C] 10/10 models, 5 implementations, non-monotone boundary included
      = a CONSEQUENCE of KL-closeness to Bayes (R3-1); accuracy alone does not imply it
        (the local plug-in: accuracy 0.733, sign agreement 0.84)
  E3  off-support extrapolation DIFFERS between E2-equivalent models (known underspecification):
      PP2 [F for H1-strong, p_c=0 shift only]; direction architecture-linked: DeepSets global,
      Transformers spread along a continuous index [PH]

LEVEL 3  INTERNAL COMPUTATION   (implementation-specific; identifiability-limited)
  relation-carrying operation: last-layer readout attention pattern (6/7 Transformer seeds)  [E]
  relation representation: 2-D key tag (5 seeds); key/query split (1); direct reread (1)    [E]
  heads: redundant and additive in margin (seed 10); counts not invariant (I5 [T])
  answer code: last-layer attention output = candidate scores (not a relation site)          [E]

LEVEL 4  LEARNING / EMERGENCE
  T4 from scratch: barrier for Transformer M and the per-group MLP, none for DeepSets (architecture-dependent) [E]
  curriculum: the T1 last layer is unnecessary; the T1 layer 0 (+ embeddings) is necessary and,
      FROZEN, sufficient for a single trained layer to learn T4 (PP3 6/6 frozen arms + 2 post hoc; n=1
      per arm); not yet separated from within-question parsing (R3-7)
  unknown groups: statistically learnable (I1, I4); the M Transformer stays at the
      own-question level (unknown pairing) (PP4 [F])
```

## 1. Mechanism classes (mandate §2), grounded in computational dependency

The provisional classes A–E were revised, because the evidence requires distinguishing **where the relation enters the decision** from **where it is represented**.

| class | computational dependency | members observed | causal signature (how it was revealed) | what it cannot explain / limits |
|---|---|---|---|---|
| **A. Explicit pooled relational mediator** | a single summary vector computed from all questions, read by every decision | DeepSets T4 | pool interchange IIA 0.94; probe R² 0.999 (T1) | it is architecturally forced, so its existence says nothing about Transformers |
| **D/A′. Key tag + reliability-gated routing** | early layers write a low-dimensional reliability tag on the target question's source tokens. Last-layer readout heads attend by it (the Q/K operation) and copy the attended votes (OV) | Transformer seeds 0, 1, 10, 11, 13 | key-only interchange 0.76–0.84 (partly off-manifold); routing pattern patch 0.84–0.86; freezing the route heads removes the tag effect; tag mean-ablation removes 54–68% of the cross-question gain | the tag's location and coordinates are not function-invariant. Under the (conjectural) I6, Q/K placement is not identifiable from behaviour |
| **B. Distributed (key + query) tag + routing** | as above, with the reliability information split between source-token keys and the readout query | Transformer seed 2 (and T1 seed 2) | key-only 0.39, query-only 0.29, joint 0.73 | — |
| **C. On-demand cross-question rereading** | last-layer heads read *other questions'* source tokens directly through values. Within-question routing is weakly gated | Transformer seed 12 | 2 heads put 52–76% of readout attention on other questions; pattern patch 0.64; tag sufficiency low (0.37–0.455), yet tag mean-ablation removes 91% | **no aligned low-dimensional relation variable at the last-layer input**, so the variable seed 12 combines with the tag is unidentified (OP-5) |
| **E. Non-relational / aggregate** | function of vote histograms only | `mlp_hist` | capped at 0.638; invariance profile opposite to the relational classes | cannot use relations by construction |
| **Redundant routing (a property, not a class)** | several co-active heads, each carrying part of the routed evidence; additive in margin, super-additive in accuracy | seed 10 (three heads), seed 11 | single-token head patching 0.30–0.36 each, 0.93 jointly; ablation 0.06–0.11 each, 0.41 jointly (total) | head-count criteria fail on these seeds (R3) and are not function-invariant (I5) |

**Why this classification is useful.** The classes are separated by decisive interventions:

| signature | A′ | B | C | A |
|---|---|---|---|---|
| key-only interchange works | yes | no | — | — |
| joint key + query works | yes | yes | — | — |
| routing pattern carries the relation | yes | yes | weakly | — |
| pool interchange | — | — | — | the whole effect |

The classes also differ in an external, **predictive** property:
- **On distribution** they all obey the same response law (E2).
- **Off distribution** A extrapolates "globally". A′ and B vary along a continuous local–global index. C is global-leaning (§4.3). The off-distribution part is post hoc.

## 2. Theorem families and their domains

Full statements are in `THEOREMS_BOUNDARIES_AND_IMPOSSIBILITIES.md`.

| result | statement (short) | domain | necessary? | inspectable without the model? |
|---|---|---|---|---|
| **S** [T] | agreement counts are sufficient for role posteriors | (U) symmetric errors, (O) one answer-dependent group | (O) necessary (two-reliable-pair counterexample). (U) is used, and its failure breaks sufficiency in the biased-noise construction | yes: generator only |
| **M** [T] | one episode-level P(R\|X) answers every question | conditional independence across questions given R | — | yes |
| **U** [T] | role posteriors are quasi-concave in a group's count; extreme roles are monotone | (U), (O); roles with constant agreement probabilities | the convexity argument needs the constant-agreement (binomial) form | yes |
| **U2** [T] | no reversal anywhere if no role agrees more than honest sources | as U | sufficient. Necessity of the converse is not claimed: reversal also needs enough prior mass | yes |
| phase boundary [T+N] | the drop-at-full-agreement condition; exact incidence over (ρ, p_c, α, K, Q, κ) | as U | — | yes |
| **D** [T] | exact margin change = log E_{π^c}[r] − log E_{π^{c′}}[r] | absolute continuity of the edit | the identity is exact. Its sign needs both generator factors | yes |
| **D2** [T] | no generator-free sign for agreement effects | — | — | — |
| **I1** [T] | noise re-pairing is unidentifiable and decision-irrelevant | — | — | yes |
| **I2** [T] | honest copying vs collusion is unidentifiable when agreement rates match | uniform Y; symmetric errors | — | yes |
| **I5** [T, exact construction] | head count and top-k criteria are not functional invariants | any attention layer | — | — |
| **I6** | Q/K placement is not identifiable from behaviour | **conjecture** (pre-LN obstacle) | — | — |
| **I7** [K] | unconstrained alignment maps are vacuous | — | — | — |

**Bridge from Bayes to network.** No theorem says that a network computes S, U or D. The bridge is **measured**: E2 holds prospectively, on the support (PP1). It is **not** a theorem off the support, and is in fact false there (PP2).

## 3. Boundaries: where relations help, hurt, or vanish

**Reliability posterior monotone** (margins follow Theorem D and can still fall in rare contexts; R3-4):
- independent errors, with one reliable group and noise (T1);
- colluders whose agreement rate is at most the honest rate (Corollary U2).

**Hurt at high agreement (reversal):**
- whenever a higher-agreement role (colluding, copying) has enough prior mass;
- at ρ ≥ 0.9, even p_c = 0.05 reverses at full agreement in about 99% of contexts;
- at ρ = 0.8 it needs p_c ≈ 0.2.

**Vanish:**
- reliability known (T0);
- exchangeable sources (T5);
- latent not shared across questions;
- agreement-indistinguishable roles (I2).

The learned T0 and T5 models confirm this behaviourally: no cross-question dependence, no tag.

**Sign depends on X_q.** The same relational change can raise or lower a candidate depending on whether that candidate is supported by the group whose credibility changed (Theorem D).

**Off-support stress tests are not counterfactuals:**
- 31% of natural single edits in T4 break the colluder constraint;
- 98% of histogram-permutation hybrids are off support;
- the on-support replacement is `pairswap` (G5).

## 4. Causal map (natural vs forced)

Details are in `CAUSAL_IDENTIFICATION_AND_INTERVENTIONS.md`.

### 4.1 What the models do naturally (input-level, on support)

- **Relation-only edits.** They change decisions in the Bayes direction: pairswap model–Bayes change agreement 0.85–0.93, κ 0.79, and the same new answer in 81% of the cases where both change.
- **Aggregate-only edits** that preserve relations change decisions in only 3.6–5.8% of questions (Bayes: 0.5%).
- **The margin-response law** matches Bayes, non-monotone boundary included (PP1).

### 4.2 What internal surgery can force (partly off-manifold)

- **Tag interchange.** Transfers 75–86% of informative decisions. 45–60% of the patched states lie beyond the natural per-position p99.
- **Route-head freezing.** Removes the tag-patch effect.
- **Natural necessity.**
  - The relation-specific share is about 0.21: partner-resample of all last-layer head outputs.
  - Tag mean-ablation removes 54–91% of the cross-question gain.

### 4.3 Identification hierarchy, most to least invariant

1. **Behavioural response law (E2).** Implementation-independent.
2. **Routing operation.** The readout attention pattern; basis-free. The *most common* relation-carrying operation (6/7 Transformer seeds), not an invariant: seed 12 lacks it, and DeepSets has no attention (R3-5).
3. **Representation site.** Seed-dependent.
4. **Heads.** Not invariant.

## 5. Learning and emergence

Details are in `GENERALIZATION_AND_EMERGENCE.md`.

- **T4 from scratch fails** for the Transformer (0/3 at 6k steps, 0/1 at 24k) and for the per-group MLP, but **not for DeepSets**. The barrier is architecture-dependent. The earlier "monotone-first-step" explanation is withdrawn (R3-10), and the cause is open.
- **The curriculum benefit lives in the early layers** (PP3, one seed per arm):

| starting point | result |
|---|---|
| T1 with the last layer reset | succeeds |
| T1 with layers 0–1 frozen, only the last layer trained (post hoc) | succeeds |
| T1 with layer 1 reset (post hoc) | succeeds |
| T1 with layers 0–1 reset | fails |
| T0 | fails |
| T5 | fails |
| T2 | succeeds |
| scratch | fails |

  **Reading.** The benefit is in **T1's layer 0 (with the embeddings)**, as *features*, not as an initialisation basin. What those features encode (pair parsing, a scaffold, or counts) is not separated. Each arm has n = 1.
- **Unknown groups.** Learnable in principle (Bayes 0.714 for T1). The M Transformer stays at the own-question level for unknown pairing (0.640 vs 0.634).
- **Mechanism class survives fine-tuning.** Seed 2: split → split. No predictor of the class exists yet.

## 6. Hypotheses H1–H7: which survive where

| hypothesis | status | evidence |
|---|---|---|
| **H1** ordinary Bayesian reliability inference implemented by attention | **survives on the support**: every implementation is KL-close to Bayes, and E2 follows from that. Off support, behaviour is underspecified (as expected). Its *direction* is architecture-linked (DeepSets follow the training plug-in; T4 Transformers mostly do not) | PP1 [C], PP2 [F for H1-strong] |
| **H2** a broader class of relation-conditioned decision computations | **survives in a restricted form**: five distinct implementations (A, A′, B, C, redundant) share the decision law. Beyond latent reliability: **not examined with learned models** in this step | PP1; OP-7 |
| **H3** an implementation-independent causal characterisation exists | **only in the weak sense** that every implementation is KL-close to the same Bayes posterior, so E2 adds nothing beyond H1. **Fails below the behavioural level**: no common internal alignment of the relation variable (seed 12 is the counterexample) | PP1; R3-1; equivalence §2 |
| **H4** a subset of mechanisms is identifiable from frozen computation | **partially**: the routing operation and key tags are identified from input contrasts (6/7 seeds). Head-level and coordinate-level identity is not (I5; I6 conjectural) | causal §4.3 |
| **H5** a consistency-oriented description predicts beyond Bayes / mechanistic accounts | **not supported**: every prediction that succeeded is a Bayes prediction. The only non-Bayes regularity (E3) is post hoc, is known underspecification, and is about implementation biases, not consistency | novelty audit; R3-2 |
| **H6** original O/D captures this structure under independently justified assumptions | **not supported**: decision-relevant content lies in the role semantics that O/D lacks (D2, F25 obstacle) | `ORIGINAL_COR_FAILURE_MATRIX.md` |
| **H7** isolated task-specific phenomenon | **partly**: the Bayes-level structure (S, U, D, U2, I2) holds across the role family. The *learned mechanism* is shown only for latent reliability in one architecture family, at one scale | OP-7, OP-9 |

## 7. Answers to the final-synthesis questions (§21)

### Computational structure

- **Classes identified:** A (pool), A′ (key tag + routing), B (split tag + routing), C (reread), E (aggregate), plus redundancy as a property.
- **Same class:** seeds 0, 1, 10, 11, 13 (A′); seed 2 (B); seed 12 (C); DeepSets (A).
- **What differentiates their pathways:** where the relation is represented (key, key + query, none localised at the last-layer input), and whether it enters through the routing pattern (A′, B) or through values (C).
- **A useful invariant:** the input-level margin-response law (E2). Inside the Transformers, the routing operation holds in 6/7 seeds.

### Mathematical characterisation

- **What can be proved:** S, M, U, U2, D, D2, I1, I2, I5, with stated domains.
- **What sets direction and magnitude:** the role agreement probabilities, the role priors (p_c), and the target question's evidence.
- **Necessary:** (O) for S; a higher-agreement role for reversal (U2).
- **Strongest counterexamples:**
  - seed 12 (no common internal alignment);
  - D2 (no generator-free sign);
  - E3 (behavioural equivalence does not fix extrapolation).
- **Impossibilities:** I1, I2, I5 (I6 is a conjecture), all under the stated restrictions.

### Generality

| setting | status |
|---|---|
| collusion, copying, correlated errors | **Bayes-level map complete** for the role family (phase diagram, U2, I2) |
| unknown groups | Bayes level done; learned level fails from scratch |
| candidate, source and context length | Bayes-level grid only |
| explicit vs implicit intermediate | explicit when the relation is reused across decisions (T1/T4) and absent when it is not (T0, T5, earlier REL-XOR3). This correlation is **not manipulated** (OP-3) |

### Causality and identification

- **Causally used:** the routing operation, key tags (5 seeds), and seed 12's value path.
- **Merely decodable:** the readout-token reliability probe in tag seeds (earlier package).
- **Survives reparameterisation:** the response law and the routing pattern under the exact symmetries.
- **Identified independently of outcomes:** all subspaces (input contrasts) and the head sets (enumeration).
- **Not uniquely identifiable:** head sets (I5); the tag's side (I6, conjecture); seed 12's variable.

### Learning

- **Why curriculum matters:** the T1 early layers are necessary and the T1 last layer is not (PP3, n = 1). Whether the transferred content is the count/tag, within-question pair parsing or a cross-question scaffold is **not yet separated** (R3-7; post hoc follow-ups in the generalisation document).
- **Why some seeds tag and others reread:** unknown. The class persists through fine-tuning.
- **Optimisation vs architecture:** T4 is representable and learnable via curriculum, so the from-scratch failure is optimisation. Whether it occurs is architecture-dependent (DeepSets do not show it).
- **Testable predictions:** classify the T1-stage mechanism and predict the T4-stage class (OP-1/OP-3); train with mixed prevalence (OP-2).

### Distinctiveness

| question | answer |
|---|---|
| what follows from Bayes | the response law, the non-monotonicity, the boundaries, I2 |
| what follows from standard nonlinear computation | the existence of each implementation |
| what uses established interpretability tools | all of the identification |
| what remains new (after review R3, small) | (1) a causally tested catalogue of distinct implementations for this task, including what failed; (2) restricted exact theorems: elementary, partly quantitative instances of known effects (unanimity paradox, copy detection); (3) methodological cautions demonstrated against preregistered criteria: off-support perm hybrids, head-count criteria (Net2Net-style split), off-manifold patching (Makelov et al. 2024), wrong own-question baselines; (4) a post hoc, architecture-linked direction of underspecified extrapolation |
| needed for more | a predictor of extrapolation from internals (OP-1), tested prospectively |

### Original COR

- **Connection to the discrepancy formalism:** only the trivial fact that disagreement is a coboundary-type edge function.
- **Remaining obstacles:** role semantics, the sign (D2), and non-invariance (I5).
- **Addressed:** F1 (restricted), F2 (characterised), F5 (scoped), F6, F7.
- **Still open:** F4 (no distinctive prediction), F29, F30.

## 8. Explicitly unexamined parts of the map

- Relation families beyond latent reliability, with learned models: constraint compatibility with reuse, multi-hop reconciliation, contradictions to reconcile, semantic and causal compatibility.
- Non-symmetric errors, adaptive collusion, reliability drift.
- Scale, depth, width, head-count and optimiser variation; L-size models.
- Learned models under p_c < 1, ρ < 1 or κ > 0 training.
- Groups of size ≥ 3; overlapping and hierarchical groups.
- A gradient-level account of the T4 barrier.
- A prospective predictor of mechanism class or extrapolation class.
