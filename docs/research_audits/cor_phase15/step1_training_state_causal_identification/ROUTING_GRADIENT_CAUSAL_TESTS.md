# Routing-gradient causal tests (mandate Workstream C)

**Status.** COR Phase 1.5, Step 1 (≈ 25%, unchanged).

**Evidence labels.** [proof] / [num] / [dev] / [frozen].

## C1. Operational definition and what each intervention removes

The routing-learning contribution of route T is the part of the target layer's WQ/WK (and upstream) gradient that flows through the attention logits of T's edges, together with the softmax-coupled logits of the same rows.

**[proof, num] Softmax invariance.** For any row, d loss/d b_C = −d loss/d b_T exactly (verified to 1e-9).

| intervention | edge-local? | removes | leaves |
|---|---|---|---|
| gsg (XVIII) | yes | T-side logit gradient | complement-side gradient (+0.0079 on E, equal to plain); routing still moves T's share through the complement logits |
| gsv (XVIII) | yes | T-side logit and T value term | complement routing and upstream learning through other edges |
| frz (target-layer QK values **and** moments held) | no: all edges of that layer | all QK learning of the target layer | upstream routing, V/O. **Artefact:** the shared bias-correction counter makes the held moments under-scaled at release, giving 1.7–2.4× oversized steps over 111–200 (pre-review B1) |
| frzp (QK values held, moments update) | no | all QK learning of the target layer | upstream routing, V/O |
| transplant of the target-layer QK state (I←B L0_QK on E) | — | the learned routing *state* difference | everything else |

A partial gradient stop is not complete removal of route learning, and gsg in particular is not (`COMPUTATIONAL_GRAPH_AND_GRADIENT_AUDIT.md` C1).

## C2. Results

| arm | E φ (821 / 822 / 823) | M φ |
|---|---|---|
| frz | 0.01 / 0.01 / 0.50 [frozen, PARTIAL] | −0.05 / −0.02 / 0.04 [frozen, SUPPORTED nil] |
| frzp | 0.59 / 0.54 / 0.69 [frozen, SUPPORTED partial] | −0.03 / 0.06 / 0.05 [frozen, SUPPORTED nil] |
| I←B L0_QK (restore B's layer-0 QK state into I) | 0.98 / 0.96 / 0.96 [frozen, SUPPORTED: not necessary] | — |
| gsg (XVIII c18, other seeds) | MIXTURE ×2, HYBRID ×1 | HYBRID ×3 |

**[frozen] frz against frzp on E.**
- The two arms have **identical step-100 parameters** and differ only in the Adam moments. Yet frz gives 0.01 / 0.01 / 0.50 and frzp gives 0.54–0.69.
- So the frz nil is an optimizer-state artefact, as reviewer B predicted.
- This is also a natural-trajectory demonstration that optimizer moments can change the outcome (see `PARAMETER_AND_OPTIMIZER_STATE_TRANSPLANTS.md` A3).

## C3. Reading

**E.**
- Freezing all routing-parameter learning of layer 0 in the window (frzp) produces a stable **partial** switch: a hybrid organisation, competent, with both routes partly necessary.
- Routing learning of the target layer therefore contributes on E (X4a is **not** eliminated on E).
- It is not sufficient for a full switch.
- The routing *state* alone is not necessary at step 100: restoring B's layer-0 QK into I leaves the switch intact (X4b in its necessity form is eliminated on E).

**M.**
- Freezing the routing learning of the target layer (layer 2) does nothing, with or without the moment artefact.
- This eliminates "routing-parameter learning of the blocked layer" as a sufficient cause on M (X4a on M).
- It is consistent with the M switch running through value-vector learning (`FORWARD_CONTENT_VS_VALUE_LEARNING.md`).

**C3 specificity, as measured.**
- frz and frzp are **not** edge-local: they stop learning of WQ/WK for every edge of the layer.
- gsg is edge-local but incomplete.
- No intervention here isolates a single route's routing gradient. Isolating it would need a projection of the WQ/WK gradient onto the T-logit contribution of both softmax sides. That is the smallest additional intervention that would make the routing contribution identifiable (see `MAIN_RESEARCH_SYNTHESIS.md` Q31).
