# Forward content against value learning (mandate Workstream B)

**Status.** COR Phase 1.5, Step 1 (≈ 25%, unchanged).

**Evidence labels.** [proof] / [num] / [dev] (D801) / [frozen] (D821–823; `FROZEN_CONFIRMATION_AND_STATISTICS.md`).

## B1. The missing arm, and what it actually removes

XVIII's vrg rolled T's values across episodes *and* stopped their gradient. It therefore confounded forward-content loss with value learning.

XIX adds two synthetic gradient interventions. Both keep the forward computation identical; this was verified numerically to ≤ 1e-6 (`logs/graph_table19.csv`).

- **vsg** stops the gradient on the value *vectors* read through the target edges T.
  - [num] The routing gradient on T is exactly the plain one (d loss/d b_T unchanged).
  - WV and LN1 of the target layer, the embeddings and (on M) the upstream layers lose the T-term gradient.
  - **WO keeps learning from the T term** (pre-review B2).
- **vsgo** is vsg, and in addition the T term of the head output uses stop-gradient(WO). So V and O both lose the gradient from the T term.
  - [num] The forward pass and routing gradient are unchanged, and WO's gradient changes (0.26 relative).
  - **Limitation [proof]:** the downstream layers that read T's content (LN2, MLP, later layers) still learn from it. They cannot be stopped without changing the forward pass.

## B2. The reciprocal comparison

| arm | mandate B3 category | forward | value learning through T | E: φ (pairs 821/822/823) | M: φ |
|---|---|---|---|---|---|
| plain | — | real | full | 0 (anchor) | 0 |
| vrs | natural forward perturbation, natural gradients | **rolled** | gradient kept but uninformative | **1.00 / 1.00 / 0.99** [frozen] | **1.00 / 0.99 / 1.00** [frozen] |
| vrg | forward perturbation + synthetic gradient stop | rolled | stopped | 0.99 [dev] (XVIII: 5/5 flips) | 1.00 [dev] |
| vsg | synthetic gradient | real | V-side stopped, WO learns | **0.02 / 0.02 / 0.68** [frozen] | **0.97 / 0.97 / 0.99** [frozen] |
| vsgo | synthetic gradient | real | V and O stopped | **0.69 / 0.56 / 0.95** [frozen] | **0.99 / 0.98 / 0.98** [frozen] |
| block | forward block | none | none | 1 (anchor) | 1 |

**Frozen verdicts.**
- SUPPORTED: E_P4_vrs_switch, M_P4_vrs_switch, M_P4_vsgo_switch, E_P4_vsgo_exceeds_vsg.
- PARTIAL: E_P4_vsgo_switch.
- CONTRADICTED: E_P4_vsg_partial, M_P4_vsg_partial, M_P4_vsgo_exceeds_vsg.

The development pair had predicted hybrids for vsg on both bases (0.35 and 0.43). The fresh pairs show the opposite extremes on the two bases.

## B3. What this establishes, and what it does not

**Mixture base (M).**

[frozen] Removing only the gradient into the value vectors read through T — with the forward pass and the routing gradient exactly as in plain training — produces the full switch on all three pairs.

This eliminates the following for M, as *necessary* causes:
- forward-content loss;
- routing-gradient removal;
- attention-share reduction by forward masking (forward unchanged).

**XVIII confound 1 is therefore resolved on M:** value-vector learning through T, during the window, is sufficient.

It does not show that the effect is *only* value learning: the block also changes the forward pass. Nor does it show that the routing state is uninvolved. vsg still changes T's learned attention share at step 100 (pre-review A R3), because routing then adapts to values that learn differently.

**Early base (E).**
- [frozen] V-only removal does nothing on 2 of 3 pairs.
- V+O removal gives a large but mostly incomplete switch (0.56–0.95).
- Content loss gives a full switch.

On E, therefore, value-pathway learning through T accounts for most, but not all, of the effect. The remainder needs forward-content loss or learning outside V and O (downstream readers, or routing; see the frzp partial in `ROUTING_GRADIENT_CAUSAL_TESTS.md`).

**The two bases differ.** On M, the block of the late route acts through value-vector learning alone. On E, the block of the early route needs more.

**Natural against synthetic.** vrs is the only natural intervention: the forward is perturbed and its gradients are ordinary. vsg and vsgo are interventions on the optimisation algorithm (mandate B3); their results do not describe an ordinary network losing information. The M result is a statement about the sufficiency of a modified learning rule.

## B4. Measured quantities

[frozen; `logs/conf/s100diag_*.json`] For every window state:
- the step-100 T share;
- KL;
- route necessity;
- layer-0 displacement against the block's displacement.

[num] Per-component gradient, update and moment changes are in the graph audit.

The loss trajectories of the window are in each window state's `window_loss.json`.
