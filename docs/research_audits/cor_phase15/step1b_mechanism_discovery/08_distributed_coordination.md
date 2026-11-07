# 08 — Family B: distributed coordination

Meanings, not slogans:

| Word | Linear residual readout `L=w·Σ p_i` | What it is not |
|---|---|---|
| Independent support | `w·p_i` nonzero, `w·p_j` nonzero, mixed Hessian 0 | Vector inner product `p_i·p_j` |
| Redundant support | `p_i` and `p_j` have the same projection on `w` | `p_i ≈ p_j` as vectors |
| Cancellation | `w·(p_i+p_j)=0` with each term large | Decision failure |
| Compensation after ablation | `w·p_i ≈ 0` | The other path "agreeing" |
| Synergy | mixed derivative of a later `σ` | A coboundary |

**Signature.** Ablate path `i`, ablate `j`, ablate both. Linear readout: the both-term is the sum of the singles (`I=0`, randomized test). Nonlinear shared bottleneck: `I ≠ 0`.

**Comparison.** Path patching and mediation already measure `w·p_i` or the finite analogue. The additional content of a "coordination score" would have to predict those four numbers without computing them. Architecture predicts the vanishing or not of `I` when the next map's linearity is known. That prediction is real and is ordinary calculus.

**Hypothesis H-COORD** (retained as reclassification, not as COR): residual addends are complementary linear contributions until they hit a nonlinearity. `REDUNDANT_WITH_EXISTING_THEORY`.

No scalar `||p_i - p_j||` appears in the linear ablation formula.
