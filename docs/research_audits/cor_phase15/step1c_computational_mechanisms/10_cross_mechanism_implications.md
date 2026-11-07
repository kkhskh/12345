# 10 — Cross-mechanism implications

The four investigations use one block and one intervention vocabulary. The connections below are the ones that follow from `06`–`09`. Where no implication follows, the row says so.

## Insufficiency and coordination

A projected disagreement is a statement about `(P(h),P(h'))`. If `P` drops a coordinate that `Q` or `K` reads, two residuals can project to the same point while attention recomputes to different outputs (`06`, the pair `X_a`, `X_b`). Apparent agreement is then missing context, not evidence that two pathways carry the same content. Apparent disagreement, for a different `P`, can be the complementary failure: `P` keeps a coordinate the readout ignores. Coordination measured after a lossy `P` is coordination of the measurements, not of the paths.

No general repair converts an insufficient coordinate into a sufficient one by subtracting a transport. Subtraction is a function of the tuple. If the tuple is insufficient, every function of it is insufficient.

## Missing context and disagreement

A nonzero `z_v-\rho z_u` is compatible with an exact structural equation, because the equation constrains `h_v` and the residual constrains the projections (`08`). The sources that produce the same symptom are omitted parents, a nonlinear map paired with a linear `\rho`, a projection kernel, a one-point `\rho` evaluated off its pair, and an actual nonzero interaction of the downstream map. The symptom does not say which source is present. That is the corrected nonidentification claim. It is not the withdrawn claim that every kernel vector of `P_u` preserves the residual.

## Jacobians decide whether coordination reaches the margin

Path contributions affect a differentiable margin through `\nabla m` and, at finite size, through the remainder. If `\nabla m` is orthogonal to a coordinated subspace, first-order movement inside that subspace does not move the margin. LayerNorm's Jacobian already has a kernel of dimension at least 2, so there are coordinated displacements of the pre-normalization residual that do not move `LN(h)` at first order. A relation that is constant on those displacements cannot be credited with stabilizing the normalized margin. The margin was locally invariant for a more elementary reason.

If the intervention recomputes attention, the total derivative already sums the pattern path and the value path (`04`). A second score that collapses those paths is informative only when it is sufficient for that sum. The relations examined here are not.

## Consistency as a description of compensation

Compensation, as defined in `07`, is a change in one structural equation after an intervention on another. A consistency residual on the natural trace does not contain that change. On the recomputed trace the exact residual is zero whether or not the margin was restored. A residual can be computed after the fact and compared with the restoration. That comparison describes the pair of numbers. It does not show that the residual caused the restoration. The causal object is the recomputed equation.

## A sufficient representation and a consistency score

If `P` is sufficient for the named family, `R=g(P(h),\delta)`. A consistency score is then either a function of arguments `g` already has, in which case any extra predictive value has to come from a restriction on `g` that the score enforces, or a function of arguments `g` does not have, in which case the score is using information outside the sufficient representation and the sufficiency claim was about a different family.

In the first branch, the examples do not give a restriction that improves on `g`. The score does not tighten the value-only identity, which is already an equality. In the second branch, the score is a different measurement and has to meet the fiber theorem on its own. The scores checked here do not.

## Can a relation help when one projection is insufficient?

Only by using more information than that projection. The joint tuple can be sufficient when each factor is not. The relation, a scalar function of the tuple, cannot separate fibers that it collapses. `z_v-z_u` collapses `(0,0)` and `(1,1)`. There is no theorem in this step of the form "the scalar succeeds on a family where the tuple fails," because the scalar is computed from the tuple.

A system in which the additional information changes a prediction would need a quantity that is not a function of the insufficient projection and that is constant exactly on the fibers of `R`. The Jacobian-vector product is such a quantity. It is not a consistency residual. Constructing a residual that equals the Jacobian-vector product assumes the gradient, which is the baseline the residual was supposed to improve on.

## Equivalences

Under a clamped pattern, `Y=AVW_O` is linear in `V`, and that linearity implies zero interaction of value edits. Zero interaction does not imply the map is linear: `u^2+v^2` is a counterexample. Under `C^2` maps on a rectangle, identically zero interaction and separability are the same statement. One pair with zero interaction is not that statement: `(u+v)^3` has interaction zero at `(1,-1)` and is not separable. Under exact sufficiency, "a function of `P`" and "constant on the fibers of `P`" are the same statement. The quantifiers are equal projections imply equal responses, on the named set and the named family.

A consistency residual is not equivalent to any of these unless its level sets are proved to match. They do not match in the counterexamples of `14`.

## What does not follow

- A small residual does not imply redundant support, compensation, or a preserved sign.
- Redundant support does not imply a small residual.
- Nonzero interaction does not imply an upstream inconsistency.
- Interaction zero at one pair does not imply a linear block and does not imply separability. Identically zero interaction on a `C^2` rectangle does imply separability, and a separable map can be nonlinear.
- Sufficiency of `P` for one intervention family does not imply sufficiency for residual edits.
- None of the four investigations restores `D\downarrow\Rightarrow L\uparrow`.
