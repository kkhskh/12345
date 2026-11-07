# 12 — Baseline equivalence

The audited historical comparison, not refit here, is attribution repair AUC about `0.979`, an obstruction-using score about `0.955`, obstruction-only Spearman with margin change about `0.095`, and attribution Spearman about `0.905`. The prompt's repeated IOI pair `0.936` versus `0.141` was already reported as not found in the Phase 1 sources. It is not used.

## 1. What each baseline needs

| Predictor of the sign of `R` along `δ` | Information it uses | Uses `D`? |
|---|---|---|
| Output gradient | `∇L · δ` | No |
| Jacobian-vector product | `w·J δ` after recomputation of a linear block | No |
| Activation difference | a norm or a direction of `Δφ` | No |
| Attribution | `∇L` at the edited site | No |
| Path attribution | the same product along a named path | No |
| Linear response `J_m δ` | the manuscript's own equation (20) | No |
| Hessian / interaction `I` | four intervened outputs, or `∇²L` | No |
| Margin itself | `m_c` | No |
| Graph residual `\|\|ds\|\|` | `P, ρ, s` | This is `O` |
| Transport approximation error | `P_v f - ρ P_u` | Object 1 of file 06 |
| Task feature | a symbolic coordinate | No |

## 2. Reductions

**Sign along a proposed repair step.** Theorem Descent in file 09: the sign of `L` along `-∇D` is the sign of `-∇L·∇D`. The obstruction gradient supplies a direction. The output gradient supplies the sign of that direction. CE-17 is a case where the two disagree, and the output gradient has the correct first-order sign by construction. There is no remainder that improves the first-order term.

**Magnitude.** Theorem Lip is `K` times distance to `ker d`. The constant `K` is a Lipschitz constant of `L`, which is a property of `∇L` on a region. The bound is not stronger than the corresponding bound written in activation coordinates. It does not rank two interventions of equal activation size when their inner products with `∇L` differ.

**Exact transport.** When `ρ` matches `f`, the residual is `0` on the recomputed path and the path effect is still `w·Jδ`, which the residual does not contain.

**Misspecified transport.** On a linear map the residual equals `(ρ-J)` applied to the parent. That is the definition of surrogate-Jacobian error. The true path effect remains `w·Jδ`. The residual is not a second prediction.

**Update Proposition 2.** A pointwise Jacobian does not bound a finite increment, so the printed threshold is not a proved stability test. Where an increment really is bounded by `S`, the bound without `τ O` is the stronger sufficient condition.

**Repair objective `O - β m`.** The closed form `-A^{-1}(∇O - β ∇m)` is a linear combination of the two gradients. As `β` grows it approaches the attribution step. The Phase 1 empirical statement that mixing the two did not beat attribution is consistent with this algebra and is not re-estimated here.

## 3. A prediction that would have been distinctive

The assignment's examples are the right tests. None is met.

- Ranking interventions of similar `||∇L||` differently from `∇L`: would require a term orthogonal to `∇L` that still changes `L`. At first order that term changes `L` by `0`.
- Predicting failure of state transfer from a frozen residual: the residual on a recomputed exact trace is `0`, including traces whose transfer changes `L` (CE-19).
- Predicting interference of two interventions: that is `I(a,b)`, which uses the intervened outputs or the Hessian, not `D`.
- Predicting rank stability: Update Proposition 2 does this from the margin and `J`, with `O` as slack.
- Predicting which perturbations the downstream map suppresses: that is a singular-value statement about `J`, not about `D(s)`.

## 4. Conclusion

Every signed prediction that survives is a prediction of `∇L`, of `J`, or of an assumption that `L` is monotone in `D`. The consistency scalar does not add a signed term that those comparators lack. This is the F29 finding, now attached to each family rather than to one score.
