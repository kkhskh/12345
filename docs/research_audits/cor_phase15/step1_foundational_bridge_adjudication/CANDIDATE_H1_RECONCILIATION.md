# Candidate H1: explicit reconciliation (softmax consensus)

**Final status: `REJECTED_CANDIDATE` as a consistency mechanism.** The identity it rests on (CI) is `PROVED` and is known mathematics.

**Phenomenon investigated.** A softmax head combines position values into a convex combination, a weighted consensus. Does the head *respond to a relation* between candidate-relevant signals in a way that constrains the decision?

**Model.** One head: `w_j = exp(s_rj)`, `v_j = ũ^T V_j W_O`, `M = Σ w_j v_j / Σ w_j`, `a = a_rest + M`, `m = a/σ + κ`.

**Relation.** The deviation of source `j` from the consensus, `v'_j − M`.

**Intervention.** Token substitution at an evidence slot, or any upstream site edit with recomputation.

**Theorem (proved).**
- CI: `ΔM = Z'^{-1} Σ_j w'_j(v'_j − M)`.
- CI.3*: for a layer-0 substitution, `ΔM = [w_j(v'_j − v_j) + (w'_j − w_j)(v'_j − M)]/Z'`.
- CI-5: exact last-layer margin decomposition (Rushing and Nanda 2024).

**Positive model.** Construction D. Designed salience: the neutral token has low weight. Weaker agreeing evidence lowers the margin.

**Nearby negative models.**
- Equal salience for the neutral and weak tokens: the crossover disappears (review C, probe a).
- A BOS sink with weight ≥ 0.8: it disappears (review D).
- Dedicated per-source heads: relation-blind (NEG).

**Separating assumption.** The substitution changes the source's attention weight, and the change is large relative to the source's own value change. Identifiable from the weights and activations. It is a fact about salience, not about agreement.

**Does it restate the outcome?** CI does not. The decision-level version needs A4, which does (review C, FATAL-4).

**Is it merely ordinary nonlinear gating?** Yes. It is softmax normalization. Every head satisfies CI by algebra, so CI cannot tell a reconciliation mechanism from a generic head (mandate H1.9; review C, FATAL-3). The convex-hull property (CI.4) holds for all heads.

**Where the causal information enters.** The sign is `sign(Δw)·sign(DLA_j − DLA_head)` plus a relation-free own-change term. Both pieces are readout-defined.

**Comparison.** Equal to head patching and to the attention-weight gradient `(v_j − M)/Z`. The mathematics is the weighted-mean update and the influence function of the mean (Hampel 1974).

**Applicability.** Exact for every head's direct numerator term. The decision sign can be reversed by the σ channel (591/4000 in random models) and by downstream re-readers (D′; self-repair).

**Prediction.** The earlier prediction, "sub-consensus support lowers the margin", is **withdrawn**: it is false for substitutions (review B, S2). Its corrected form is the head-patching DLA change. That is not a consistency prediction.

**What remains open.** A relation computed nonlinearly *in the agreement variable* (for example, an MLP neuron reading the product of two evidence features). It was not examined, and it would be a different hypothesis.
