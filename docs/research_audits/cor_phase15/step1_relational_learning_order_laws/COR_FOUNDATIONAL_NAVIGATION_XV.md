# COR foundational navigation after Investigation XV

**Status.** COR Phase 1.5, Step 1, provisional ≈ 25% (unchanged). No thesis is selected. The obstruction/discrepancy (O/D) framework is not established.

## New territory explored

**How small Transformers acquire relational trust.**
- Four GATE variants that cross the gate's arity (2 or 3) with the gate's decision value (about 0.14 or 0.28 nats). Sixteen fresh models.
- A cross-task positive control: DeepSets on T4, four models.

## Established mathematically [proof / num]

- **The trust relation needs all a sources.** The arity-3 gate has no pairwise or single-source shortcut (exact mutual information 0).
- **Decision order.** The Bayes decision has interaction order exactly a + 1. Every rule of order ≤ a has gate response exactly 0.
- **Value is matched across arity.** At fixed β and ν, the gate's decision value is identical for a = 2 and a = 3, to within 0.09–0.9%.
- **The usefulness knob is confounded.** Changing ν moves the gate gain and the intermediate gain in opposite directions. This is because ν is the only remaining knob, not because of a fixed sum; a two-reliability generator can decouple them (Review A F4).
- **Noisy gates.** They make the Bayes rule a full-order mixture and shrink the gate's value.

## Predicted prospectively and confirmed [frozen]

- **P1.** At matched value, the 3-source gate is acquired much later, or not within the budget (8/8 pairs; 6 censored; about 4 independent replicates; budget about 2700–3000 effective steps).
- **P3.** The frozen rule returned H1 (complexity first). In C and D the stage is close to entailed by censoring. In B it lasts 3 snapshots in 3 of 4 models. So H1 is **not robustly separated from RACE** (Reviews A F1, B F5).
- **P6.** In DeepSets on T4: a source stage first, then an abrupt relational acquisition (4/4 competent).
- **Source stage first.** This is a real plateau in DeepSets T4 (4/4). In the GATE variants, order-1 accuracy comes first but without a separate plateau (Review B F6).

## Predicted prospectively and falsified [frozen]

- **The RACE pattern** formed from the development seed, which predicted that B would skip the stage.
- **H2's decision-benefit patterns:**
  - its prediction of no arity effect;
  - its prediction that the stage would be absent in B and C.

## Inconclusive

- **P2, the effect of decision value at fixed complexity.**
  - At arity 2 it points one way in 4/4 pairs (a post-hoc subgroup).
  - It is confounded with intermediate-cue strength.
  - It cannot be judged at arity 3 because of censoring.

## Task-specific, or not yet general

- **XIV's skipped intermediate stage** does not reproduce in this task family. This suggests it was task-specific, though the agreement indices differ (Review B F10).
- **The DeepSets T4 result** is a different architecture.
- **One Transformer architecture and recipe** throughout.
- **The arity-3 competence rate** was 2/8 within 4000 steps.

## What may generalise [spec]

- Low-order structure first, then an abrupt relational acquisition: GATE variants at arity 2, DeepSets T4, and XIV. At arity 3, abruptness is unresolved because of snapshot spacing.
- Lower-decision-order cues are acquired before higher-order ones when both are available. At a = 2 both relations are pairwise (Review A F10).

## Withdrawn after review

- The learning-signal (gradient-SNR) analysis: the estimator was at its noise floor. P5 and H3 are untested.
- "Gate information before use" in D_s203: this was the agreement cue (baseline 0.88).
- "Less abrupt at arity 3": a snapshot-spacing artefact.

## Next decisive direction

Untangle complexity and value with a design that does not rely on ν:
- **(i)** Raise the arity to 4 at matched value, to test whether the delay grows with order.
- **(ii)** Give a longer budget or a curriculum to the censored arity-3 models, to tell "later" from "never".
- **(iii)** Vary the value with the agreement cue held fixed: through gate *relevance*, or through separate pair and single-source reliabilities (Review A F4).
- **(iv)** Add to stage rules a floor condition (error at or above the gate-free floor), and a cue-controlled probe baseline (Review B F2, F12).

## Broader map (unchanged, not resolved)

- **TRANS:** composition is not learned.
- **ADAPT:** position dependence.
- **Unknown or hidden groups:** pairing must be inferred.
- **Alternative architectures and objectives:** only DeepSets touched here.
- **Trust uncertainty:** mathematics only (Workstream F).
- **O/D framework:** untested.

All carry over from COR_FOUNDATIONAL_RESEARCH_MAP.md (XIV).
