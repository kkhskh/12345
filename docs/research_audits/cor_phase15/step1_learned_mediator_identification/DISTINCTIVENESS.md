# Is the mechanism distinctive? (mandate §F)

Each alternative explanation below is equally informed and evaluated on the same models and inputs.

| alternative explanation | what it predicts | result |
|---|---|---|
| **Aggregate-only nonlinear `m = h(z)`, or `h(z, p)`** | No decision difference on matched-aggregate pairs. Accuracy at most the aggregate-only ceiling: 0.754 (XNOR, MIS) and 0.877 (GATE). | **Rejected for REL tasks.** Accuracy is ≈ 1.0, and the model's decisions differ on 100% of target-differing matched pairs. **Confirmed for AGG tasks**, including AGG-NL, which has 423–450 order reversals out of 465 row pairs yet no decision difference on matched pairs. |
| **Matched-capacity MLP** | Solves every task (previous package: B-MLP accuracy ≈ 0.98–1.0). | Ties behaviourally. The Transformer's mechanism is one instance of an MLP-expressible interaction. |
| **Standard source interaction**, e.g. `f(a) + g(b) + c·φ(a, b)` | Any φ needed by the target. | Fits the mediator exactly. For XNOR, φ = `sign(ab)` (R² ≈ 1). For MIS, φ is a different, target-specific interaction. Consistency adds no restriction beyond "φ is what the target requires". |
| **Path-specific causal mediation** | Path shares. | Used directly. Path split for REL-MIS: direct path ≈ 1.0. For REL-GATE: MLP path 1.0. For XNOR: about 0.5 / 0.5. This is a standard analysis. |
| **Causal abstraction / DAS** | A high-level model (z, s, p) → y, aligned to a subspace by interchange. | S_das (target-trained) and S_rel (output-free) coincide: cosine 0.87–0.99, with equal IIA. Our result **is** a causal-abstraction result. The difference is only that the alignment is obtained in closed form from aggregate-matched pairs, without training on targets. |
| **Direct downstream sensitivity (S_grad)** | Directions of largest margin gradient at the site. | Weaker. Of the 22 relation-computing sites, S_grad's IIA is below 0.95 in 16 and at most 0.15 in 8, while S_rel's is at least 0.995. Gradient directions find what moves the margin locally, not what carries the relation between natural states. |
| **Orthogonal, partially decodable alternative** (S_alt) | Would mediate if "decodable implies used". | IIA ≤ 0.27 everywhere, and 0.000 at every relation-computing site. However, S_alt decodes agreement at only 0.52–0.73, against 0.999 for U (R_causal F4). So this is only weak evidence that decodability is not use. |

## Does the consistency concept give an independently testable restriction, a transfer prediction, or an identification advantage?

1. **Identification advantage: none established.**
   - In REL-XNOR and REL-MIS, agreement equals the target on matched pairs, so the "output-free" contrast is a target contrast.
   - A plain difference of means **without** aggregate matching gives the same subspace, with cosine 0.98–1.00 (R_counterexample F1).
   - The output-gradient subspace fails at TF2 `resid_mid1` because the final LN projects out the dominant direction there (R_causal F2). That is a property of LN geometry, not an advantage of the method.
2. **Restriction: consistent, but not preregistered here.**
   - The threshold-logic analysis of agreement rules (T-RF, previous package) implies that fixed-prior deferral rules need no relation computation.
   - The learned models agree: REL-GATE gives 0/12 relation-computing mediators and K3-GATE 0/12. Interactions appear where the rule is not threshold-representable: XNOR, and MIS, which is additive only on grids, with a steep discontinuity.
   - The prediction was available beforehand but was **not** stated as a hypothesis in `PROTOCOL.md`. It derives from representability theory, not from a concept of consistency.
3. **Transfer prediction: weak.**
   - The XNOR TF2 mediator survives the Gaussian evaluation inputs (5/6).
   - The TF2 MIS content verdict does not (1/6), because the criterion sits at a threshold.
   - Consistency gives no transfer prediction beyond "the mediator is whatever the target needs".

**Verdict.** The mechanisms are ordinary: attention-normalizer interactions and per-source threshold logic. In the one task where agreement is an intermediate distinct from the target (REL-XOR3), the models compute the target directly, or carry the per-source signs. No consistency mechanism was found.

One restricted positive result is real and should not be dismissed because its mathematics is familiar: a localised, causally verified **target** variable in XNOR models, created by the attention normalizer (T-ATT for TF1).
