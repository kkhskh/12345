# 08 — Computational consistency

## C1. The question, without the original conclusion

The original construction picks maps `\rho_{u\to v}` and a residual `ds=\rho s_u-s_v`, then a quadratic `O` or `D`. The question that survives Phase 1 is narrower. Can a relation that is fixed by the computation, rather than by the desired margin, have a discrepancy whose variation tracks a causal property?

The property need not be correctness and need not be a larger logit. It could be stability, approximation error, or an interaction. It still has to be implied by the relation.

## C2. Exact structural equations

On a recomputed trace, `h_v=f_v(h_{pa(v)})` by assignment. The residual of that equation is zero for every intervention that recomputes descendants. A quantity that is identically zero on a family cannot track a response that varies on the family. `\tanh` shifted by `+0.5` and by `-0.5` has structural residual zero in both cases and different responses.

Under a simultaneous clamp, the descendant is set to a value the equation did not produce. The residual then equals the difference between the imposed value and `f_v` of the parents. It measures the imposition. It does not measure a conflict in the unclamped model. Using it as a property of the model confuses the intervention with the mechanism.

## C3. Projected relations

Let `z_u=P_u(h_u)` and `z_v=P_v(h_v)`, and suppose the full equation holds. The projected residual `z_v-\rho(z_u)` can still be nonzero because:

- `P_v` discards coordinates `h_v` still has;
- `\rho` was chosen for a different operating point;
- a parent other than `h_u` enters `f_v`;
- `f_v` is nonlinear and `\rho` is linear;
- the analyst's `\rho` is a candidate-specific matrix that the block does not apply.

The two-token attention example has a projection that matches and a full attention equation that holds on each state separately, while any single linear `\rho` fit to one state's projected coordinates need not describe the other. The failure mode in that example is loss of the key coordinate, not a fight between two pathways. Calling every nonzero projected residual a conflict merges these sources. They are not separated by the sign of the residual.

## C4. Local linearization

If `f` is differentiable at `h_0`,

\[
f(h_0+\delta)=f(h_0)+J_f(h_0)\delta+r(\delta),\qquad r(\delta)=o(\lVert\delta\rVert)
\]

when the derivative exists in the usual sense. The transport of a displacement is `J_f(h_0)`, applied to `\delta`, added to `f(h_0)`. The map `\delta\mapsto J_f(h_0)\delta` is not a transport of the absolute state `h_0+\delta`. For `f(h)=5h+3`, the derivative applied to the absolute state drops the constant `3`. For `f(h)=h^2+3` at `h_0=2`, the first-order formula at step `1` is short by `1`, which is the second-order term of the square.

A projected Jacobian condition `P_v J = \rho P_u` constrains derivatives. Together with a matching offset `P_v f(h_0)=\rho P_u h_0`, it constrains the first-order projected image. It does not constrain the remainder, and it does not constrain directions in `\ker P_u` unless those directions are killed by `P_v J`. That last clause is Error 4.

## C5. What a nonzero discrepancy can be

| source | what is true in the computation | what the discrepancy sees |
|---|---|---|
| surrogate error | the block equals `f`, the surrogate equals `\rho\circ P` | the gap between them |
| representation insufficiency | `f` uses coordinates `P` drops | a gap even if `\rho` is a left inverse on the kept coordinates |
| computational interaction | `I_F\neq 0` for the intervention | nothing, unless the discrepancy was constructed to equal `I_F` |
| counterfactual incompatibility | two clamped states violate an equation the unclamped model satisfies | the clamp |
| response-relevant interaction | `I` of the margin is nonzero | the margin's own factorial contrast, not an upstream residual, unless a proved equality connects them |

These are separable in examples. `u^2+v^2` has interaction zero and can still have a large residual against a bad surrogate. `(u+v)^2` has a nonzero interaction and a zero residual against the surrogate `f` itself. The residual does not order the interactions.

They are not separable from the residual alone. The corrected N3 statement is: given only `r(h)`, the source is not identified. A direction that preserves `P_u` need not preserve `r`, so one cannot even test "the leftover is unchanged, hence the source is in the kernel" without specifying the measurement that was actually held fixed.

If the available measurements do not include the parents `f` uses, the split between information loss and surrogate error is nonidentifiable from those measurements. Adding a name does not remove the nonidentifiability.

## C6. Beyond the ordinary gaps

The candidates for "extra" information were: distance to a surrogate, projection error, activation magnitude, gradient sensitivity, and ordinary approximation error.

In the families checked here, a consistency residual is one of the first two, or a clamp artifact. It does not determine interaction (`u^2+v^2` against `(u+v)^2`, both with structural residual zero under recomputation). It does not determine sign (N1). It does not become response-relevant by being quadratic. `O` and `D` remain different quadratics, and neither is the margin.

What would make a compatibility residual causal rather than descriptive is a proof that, under hypotheses that do not mention the margin, the residual determines a named property of `R`. The fiber theorem says this happens only when that property is constant on the residual's level sets. The examples were built so that the property is not constant there. No alternative residual in this step meets the condition.

## C7. Scope

Computational consistency with the structural equation is meaningful as a debugging statement about recomputation: the residual is zero when recomputation happened, and nonzero when a node was overridden or a parent was omitted from `f`. It is vacuous as a predictor on the recomputed family. A projected version is a measurement of information loss or of surrogate error, and those two are not identified by the measurement alone. No consistency-like quantity examined here supports a nontrivial causal hypothesis beyond that measurement statement.
