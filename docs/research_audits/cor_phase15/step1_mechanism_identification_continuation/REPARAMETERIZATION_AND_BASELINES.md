# Reparameterization and baselines

This document covers §10 (reparameterization), §15 (baselines) and §16 (transfer).

| topic | code | log |
|---|---|---|
| reparameterization | `src/reparam.py` | `logs/reparam.txt` |
| baselines | `src/learned.py`, `src/learned_posthoc.py` | `logs/learned_train_*.json`, `logs/learned_posthoc_*.txt` |
| transfer | `src/transfer.py` | `logs/transfer.txt` |

## 1. Reparameterization of learned checkpoints

The gauges were applied to the T1/TF1, T3/TF2 and T4/TF1 checkpoints (seed 0). Every gauge preserves the function: the largest logit change is ≤ 8.5e−5, which is float32 rounding.

| identification output | QK | VO | head perm. | unit perm. | unit split |
|---|---|---|---|---|---|
| margin, Δagree, flips | invariant | invariant | invariant | invariant | invariant |
| residual-probe R² (all sites) | invariant (≤ 0.004, solver noise) | invariant | invariant | invariant | invariant |
| probe-direction interchange effects | invariant (exact) | invariant | invariant | invariant | invariant |
| relation-unit **indices** | invariant | invariant | invariant | **changed** | **changed** |
| relation-unit **count** | invariant | invariant | invariant | invariant | **22 → 23** (T1), **34 → 36** (T3) |
| multiset of per-unit `I_agree` | invariant | invariant | invariant | invariant | **changed** (one unit appears twice) |
| set-level ablation effect on Δagree | invariant | invariant | invariant | invariant | **invariant** (both copies are flagged and ablated) |
| Q/K-space or V/O-space directions | **changed** | **changed** | permuted | — | — |

**What is intrinsic and what is not.**

- **Intrinsic up to these gauges:**
  - the input–output function;
  - residual-stream probe results and interchange effects, because the residual stream was not transformed;
  - the *set-level* causal effect of the relation-unit population.
- **Not intrinsic:**
  - unit indices and unit counts;
  - any per-unit share of the effect;
  - any coordinate inside a head's Q/K or V/O space.
- **The equivalence-class statement we can defend:** "the population of MLP hidden units at R whose activation has `I_agree ≥ 0.2` and `D_sum ≥ 0.2` carries a set-level ablation effect E." This statement is invariant under permutation and splitting.

**Gauges not applied.**

- *Residual-basis rotation.* The learned LN biases feed bias-free Q, K and V maps, so a rotation cannot be conjugated through LN without adding parameters.
- *Positive unit rescaling.* GELU is not positively homogeneous, so rescaling does not preserve the function.

The engineered-circuit gauges from the previous package still apply: head redistribution, QK, VO, ReLU rescale, neuron split, residual rotation fixing 1, LN gain folding.

## 2. Baselines

Every baseline uses the same data, optimizer, number of steps and validation split as the Transformers (§4 of the preregistration). B-add is exploratory.

**Validation accuracy, dev seeds 0–2 (mean, [min, max]):**

| task | TF1 | TF2 | TF2-AO | B-sum | B-lin | B-MLP | B-bil | **B-add** (relation-free) |
|---|---|---|---|---|---|---|---|---|
| T1 agreement (XNOR) | 1.000 | 1.000 | 1.000 | 0.599 | 0.479 | 0.999 | — | **0.482** |
| T2 Bayes integration | 1.000 | 0.999 | 0.999 | 1.000 | 1.000 | 0.999 | — | 1.000 |
| T3 conflict → prior | 1.000 | 1.000 | 1.000 | 0.859 | 0.859 | 0.998 | — | **0.997** |
| T4 coherent-but-wrong | 0.999 | 0.999 | 0.999 | 0.503 | 0.504 | 0.993 | — | 0.920 |
| T5a K = 3 Bayes | 0.999 | 0.999 | 0.999 | 1.000 | 1.000 | 0.997 | 1.000 | 0.998 |
| T5b K = 3 argmax-agree → prior | 0.984 | 0.999 | 0.993 | 0.813 | 0.813 | 0.984 | 0.855 | **0.991** |

The held-out seeds (3, 4) are reported in `LEARNED_SYNTHETIC_MECHANISMS.md` §2.

**Reading the baselines.**

1. **B-add is the decisive baseline.** It uses the same source information, but **no term can read both sources** (§15: "same source information but no explicit relation variable").
   - It matches the Transformers on T2, T3, T5a and T5b.
   - It fails only on T1, and partially on T4.
   - This matches T-RF: candidate-directed agreement rules are threshold logic.
   - T4 is LP-feasible as an additive rule on grids. B-add's 0.92 reflects the steep discontinuities that a smooth additive network needs near `b = 0`, not a representational barrier.
2. **B-MLP**, a matched-capacity unstructured nonlinear model, matches everything. Nothing the Transformers do exceeds generic nonlinear function approximation.
3. **B-bil vs B-add on T5b.** The bilinear source-interaction model (0.855) is worse than the relation-free additive model (0.991). The bilinear relation ⟨z_A, z_B⟩ is the wrong relation for candidate-relative agreement, as T-B-IMP predicts.
4. **TF2-AO**, an attention-only model, solves T1, which is XNOR, without any MLP. The required cross-source interaction can be produced by softmax/LN normalization across positions, not only by MLP gating. An "MLP agreement detector" is therefore one implementation among several.

## 3. Strong-baseline accounting (§15 checklist)

| baseline | result |
|---|---|
| net-evidence-only predictor | B-sum: fails T1, T3, T4 and T5b; ties on T2 and T5a |
| matched-capacity nonlinear predictor | B-MLP: ties on everything |
| ordinary MLP interaction explanation | the T1 relation units are ordinary GELU interaction units. Ablating them reduces Δagree by about half, and attention-side computation carries the rest |
| direct candidate-margin information | the readout direction is used only for the step-6 sign prediction |
| path-specific effects | C2: the query route carries 100% and the value route 1.8% (C′) |
| causal-abstraction account | C′ abstracts to "AND of per-source supports, defer otherwise". A relation-free model (T-RF construction) realises the **same high-level causal model at the decision level**. The relation variable is needed only to abstract C′'s *internal* routing variable |
| relation-blind computational model | the sum-only reader (X5) and the relation-free C′ twin (T-RF) |
| same information, no relation variable | B-add: see above |

**Conclusion.** No positive result in this package gives a *predictive* advantage over the matched baselines. The surviving contribution is a set of precise causal decompositions and identification criteria (the matched relation-blind contrast, C2 path shares, T-RF), not prediction.

## 4. Transfer (§16)

Transfer was tested on the engineered circuits and on the learned models.

| change | engineered circuits | learned models | classification (§16) |
|---|---|---|---|
| input magnitudes | C′ dead bands derived (C′1*); the A1* sign law is scale-free, with GELU cones scale-dependent | weak-evidence mixture in training; dead bands present | same mechanism; conditions derived |
| seeds | — | dev 0–2, held-out 3–4, transfer 5–6 | see `LEARNED_SYNTHETIC_MECHANISMS.md` |
| residual width / heads / MLP width | — | `logs/transfer.txt` | see `LEARNED_SYNTHETIC_MECHANISMS.md` §5 |
| source encodings | A: semantic drift (mix 0.6) gives a 0.895 sign law with unchanged weights | — | weaker behavioural correlation |
| candidate sets | the K = 3 bilinear gate tracks the dot product, not agreement | T5: the relation is not needed (B-add) | different mechanism, or none |
| priors | C′: π trades the agreement floor against the conflict band | T3: p enters additively | same; conditions derived |
| normalization | final LN with zero bias preserves sign; LN bias can flip decisions (B5) | — | conditions derived |
| bypasses | C4 bypass dominance (β = 6: conflict 0.978) | — | conditions derived |
| depth | C′2 (previous package): a single layer cannot route on a relation | TF1 vs TF2: both solve T1 | different mechanisms, same behaviour |
