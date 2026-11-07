# 03 — Exact computation, approximation error, and stitched states

This is the preliminary attack required before any bridge. The question is when a nonzero value of `d s` is a disagreement inside the model's computation, and when it is an artifact of the surrogate.

## 1. Theorem Z — exact commutation kills obstruction on real traces

**Statement.** Let one site be a function of one parent, `φ_v = f(φ_u)`, with `f` any map. Let `P_u`, `P_v`, and `ρ` be linear and satisfy the identity of maps

```text
P_v ∘ f = ρ ∘ P_u.
```

Then for every parent activation `h` and every intervention `δ`, with the child recomputed as `f(h)` and `f(h+δ)`,

```text
ρ P_u h - P_v f(h) = 0,
ρ P_u (h+δ) - P_v f(h+δ) = 0.
```

So `O = D = 0` on naturally generated trajectories and on trajectories obtained by one intervention followed by the model's own downstream map. On that class, `O` does not distinguish correct from incorrect decisions, easy from difficult decisions, or stable from unstable decisions.

**Proof.** Substitute `φ_v = f(φ_u)` into the commutation identity. The intervention clause uses the same identity at the point `h+δ`, which is legal because the identity was assumed for every parent activation, not only for a sample. QED.

**Multi-parent form.** If the true aggregation is `φ_v = f(φ_u, φ_w)` and the joint linear identity `P_v f(h_u, h_w) = ρ_u P_u h_u + ρ_w P_w h_w` holds for all inputs, the joint residual is zero on every recomputed trace. A coboundary that keeps only one parent is a different operator. Its residual can be nonzero while the joint residual is zero (`test_ce12_omitted_parent_creates_discrepancy`: `1+1-2=0`, while `1-2=-1`).

**Residual-stream form.** The map `h_v = h_u + U(h_u)` with `U(h)=h` is exactly `h_v = 2 h_u`. The skip-only choice `ρ = I` has residual `||h||`. The choice `ρ = 2I` has residual `0` (`test_residual_skip_omission_is_not_a_conflict`). The nonzero score is the omitted branch. It is not a candidate conflict. Including the branch returns the example to Theorem Z.

## 2. What the manuscript's Jacobian formula already drops

PDF equation (11) sets

```text
ρ ≈ P_v J (P_u)^†,    J = ∂φ_v/∂φ_u.
```

Then `ρ P_u = P_v J Π`, where `Π = (P_u)^† P_u` projects onto the row space of `P_u`. This equals `P_v J` for every tangent vector if and only if `P_v J` vanishes on `ker P_u`. When the child stalk depends on a direction the parent stalk threw away, equation (11) is not exact, and the leftover is that discarded direction (`test_ce06_nonzero_residual_is_lost_coordinate`). The leftover does not refer to a candidate logit.

## 3. Theorem App — a nonzero residual of an approximate map

**Statement.** Suppose no linear `ρ` satisfies `P_v ∘ f = ρ ∘ P_u`. Let `ρ̂` minimize the mean squared residual of `ρ P_u φ_u - P_v φ_v` on a set of naturally generated pairs. The minimized residual is zero if and only if `P_v φ_v` is a linear function of `P_u φ_u` on that set. Otherwise the residual is the part of the projected child that the projected parent does not determine.

**Consequence.** On natural trajectories, after `ρ̂` has been fit to those trajectories, a remaining residual measures the failure of the stalk to be a sufficient statistic for the next stalk. It measures surrogate error. Calling it a decision conflict adds a claim that is not in the residual.

The same holds inside one linear piece of a piecewise-linear map and fails across pieces. For `f(h)=max(h,0)` on `{-2, 2}`, one global slope is `1/2` and both residuals equal `-1`. On the positive piece alone the slope is `1` and the residual is `0` (`test_relu_global_linear_map_residual_is_the_kink`). A single `ρ` for a ReLU network therefore mixes distinct linear pieces. The mixture residual is the kink, not a candidate-specific disagreement.

## 4. Stitched states are not recomputed traces

Assigning `s_u` from one forward pass and `s_v` from another can make `ρ s_u - s_v ≠ 0` even when every individual pass satisfies Theorem Z. The exact example is `ρ=2`, parent `1`, child taken from the zero pass: residual `2` (`test_ce05_exact_commutation_is_zero_after_recomputation`). That number is a property of the stitch. It is not a property of either forward pass, and it is not the effect of `do(h_u ← h_u+δ)` followed by `f`.

## 5. Domain table

Theorem Z applies when the child is the value of one fixed map `f` on the argument that was intervened on. A pre-norm Transformer block is not such a map of one token. Its next residual depends on the whole sequence through attention, and LayerNorm, softmax, and the MLP are nonlinear, so no constant matrix intertwines the block on an open set of sequences. A nonzero linear residual on a natural Transformer trace is therefore the failure of a constant linear `ρ` to represent that block (Theorem App, or the ReLU and square-map tests). It is not yet evidence that a parent was omitted, and it is not a candidate conflict.

| Domain | What is computed | `ds` when a constant linear intertwiner exists | If `ds ≠ 0` |
|---|---|---|---|
| A. Natural pass of a fixed `f` | `φ_v=f(φ_u)` | 0 | `ρ` is not an intertwiner, or a parent was omitted |
| B. One intervention of the argument of that `f`, then `f` | `φ_v=f(argument+δ)` | 0 | Same. A one-token edit through attention is not this row |
| C. Several sites set at once | Not one application of `f` | Can be nonzero | The imposed tuple |
| D. Stitch across inputs | Not one application of `f` | Can be nonzero | The stitch |
| E. Algebraic vector in `ker d` | No activation required | 0 by construction | Need not be attainable |
| F. Perturbation without recomputation | Analyst-chosen coordinates | Not Theorem Z | Not a file-04 effect |

## 6. Separation that is not available

There is no second functional of `(P, ρ, s)` which, on these definitions, marks one nonzero residual as "computational disagreement about candidate `c`" and another as "surrogate error." Both are `ρ s_u - s_v`. The label comes from an extra story about why `ρ` failed. The story is not identified by the score.

Changing `ρ` after seeing which label is convenient is the failure already numbered F30. This step does not do that.

## WHERE DOES THE CAUSAL CONTENT ENTER?

It does not. Theorem Z says the score is identically zero when the representation really is the computation. Theorem App says the score is the representation's approximation error when it is not. Neither statement mentions `L_c`.
