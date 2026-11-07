# Candidate mechanisms and causal graphs

## Identification procedure (frozen; `PROTOCOL.md`)

The procedure answers the six questions of mandate §C as follows:

1. **The relation sought.** D1 agreement, `s = 1[ab > 0]`, or argmax agreement for K = 3. It is defined by the generator.
2. **The site.** One frozen site per architecture, at a complete cut. Final residual sites are excluded because they are readout-trivial. The cut is checked by full-state interchange: IIA 1.000 at every primary R site in E1 and E2.
3. **The information used to select the mediator.** Only generator labels and matched pairs with identical aggregate `(z, p)`. U is the top-k singular subspace of `h(agree) − h(conflict)`. The aggregate cancels exactly in this difference. **No output information is used.**
4. **Decodability versus use.** U is tested by interchange, pair-mean ablation and complement interchange (sufficiency). These are compared with S_alt, an equally decodable subspace orthogonal to U. S_alt has IIA ≤ 0.27 in every model (the maximum is in a TF2-AO model where no mediator was identified), and 0.000 in most.
5. **Dependence on output information.** S_rel uses none. S_das, used as a comparison, is trained on task targets. S_grad, also a comparison, is output-derived.
6. **Reparameterisation ambiguity.** U is a subspace, not a set of coordinates.
   - Under the exact gauges of these models (QK, VO, head permutation, MLP permutation, and residual rotation fixing **1**, conjugated through LN), U transforms covariantly, so every IIA is invariant. This follows by construction; it was not re-run.
   - The content test (additivity of U's coordinates) is invariant to invertible **linear** maps of U, and to affine shifts. It is not invariant to nonlinear recoding.
   - The protocol therefore requires no particular signed encoding of the relation.

## Causal graphs

### TF1 (one layer)

```
a ─▶ LN(x_A) ─┐                          ┌──────────── direct residual path ─────────────┐
b ─▶ LN(x_B) ─┼─▶ keys/values ─▶ softmax pattern at R ─▶ h_R = resid_mid0[R] ──┤                                                 ├─▶ LN_f ─▶ margin
(p)─▶ LN(x_P) ┘        (q_R constant: R carries no source)          └─▶ LN2 ─▶ MLP0 ─▶ + ───────────────────────────────┘
```

- **Cut.** `h_R` is a complete cut. Patching the full state transfers 100% of decisions.
- **Mediator.** U, with k = 1–2, inside `h_R`.
- **Where cross-source interaction can arise:**
  - (i) the softmax normalizer, per head. With the pattern frozen, `h_R` is exactly source-additive (Theorem T-ATT). In every TF1 relation-computing model (XNOR s4; MIS s2–s4, s7–s9), the measured interaction share is 0.000.
  - (ii) the MLP0 nonlinearity acting on `h_R`;
  - (iii) the final LN.
- **Which path carries it.** The path split shows how U's effect reaches the output:

  | task | MLP path | direct residual path |
  |---|---|---|
  | REL-MIS | ≈ 0.00 | ≈ 1.00 |
  | REL-GATE | 1.00 | 0.00 |
  | REL-XNOR | about half | about half |

  So in REL-MIS the attention-computed interaction is read almost linearly by the readout. In REL-GATE, a source-additive U is combined nonlinearly by the MLP.
- **Source-additive TF1 XNOR models (5/6).** In these, U at `h_R` is causally identified but additive. XNOR, which needs an interaction, is therefore completed downstream.
  - X1 shows that no single downstream site is a cut: MLP-hidden IIA is 0.47–0.81, because the direct path carries the remainder.
  - The interaction is distributed across the MLP and direct paths, and is **not identified at a single site**.

### TF2 (two layers)

Layer 0 runs at all positions. Layer-1 attention at R reads all positions. `resid_mid1[R]` comes next, then MLP1 and the readout.

- **Cuts.** `resid_mid1[R]` and `resid_post0` at all positions are both complete cuts.
- **XNOR.** The interaction at `resid_mid1[R]` comes mostly from layer-0 attention:

  | seeds | interaction share: none frozen | layer-0 pattern frozen |
  |---|---|---|
  | s2, s4, s8, s9 | 0.51–0.99 | 0.015–0.24 |

  Seed s7 is redundant: freezing either attention layer alone leaves it (0.88 and 0.996), and freezing both removes most of it (0.075).
  - The remainder with all patterns frozen (≤ 0.08) comes from MLP0.
- **REL-MIS, seeds s3, s4 and s7–s9.** The interaction comes from **layer-1** attention. Freezing pattern 1 brings it to ≤ 0.009.
- **Layer-0 output across all positions.** This site often fails the necessity gate, which points to redundant or distributed routes.

### TF2-AO (attention only)

- The layer-0 output across positions is source-additive: additive R² ≈ 1.0 in XNOR.
- The interaction is created inside layer-1 attention.
- **No mediator was identified** at the tested cut. This is reported as a failure of single-cut localisation, **not** as evidence that the mechanism is absent: freezing the patterns drops TF2-AO to chance (previous package).

## Corrections after review

- **TF2 `resid_mid1@R` is readout-equivalent.** Deleting the final MLP leaves accuracy at 0.999–1.000, and U has cosine 0.79–0.97 with the readout direction (R_counterexample F1, R_causal F2). Verdicts at that site describe the decision variable itself.
- **The cut check is a code check.** It is guaranteed by the architecture.
- **S_alt is only partially decodable.** It decodes agreement at 0.52–0.73.
- **REL-XOR3.** See `XOR3_RESULTS.md`:
  - TF1 and TF2 R sites carry the **target**;
  - TF2-AO layer-0 outputs carry the **two source signs** separately, as source-additive content.
  - No computed agreement intermediate was found.

## Mechanisms by task

| task | mechanism |
|---|---|
| **REL-XNOR** | A target (logit) coordinate `≈ c·sign(ab)` plus additive terms. In TF1 it is created by the attention normalizer: LN-saturated per-source keys and values make each source sign-like, and the shared softmax denominator couples them. Because the target here **is** agreement, this is a learnability control. |
| **REL-MIS** | A cross-source interaction that is **not** `sign(ab)`-shaped. Created by layer-0 attention (TF1) or layer-1 attention (TF2), and read mainly through the direct residual path. |
| **REL-GATE and K3-GATE** | Source-additive mediator. The decision is threshold logic on per-source support and the prior. **No relation is computed.** |
| **Aggregate tasks** | No relation-sensitive behaviour. |
