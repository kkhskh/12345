# 05 — Competing bridge families

Each family is a proposed way to put a causal restriction back into Theorem NFL. The restriction is stated, the implication that actually follows is proved, and a counterexample shows what the implication does not say. No family is allowed to borrow a success from another.

Notation is the finite-dimensional one in file 02. `Π` is the `M`-orthogonal projection onto `ker d`.

## Family A — Exact computational commutation

**Definition.** A commutation error for a single parent is the map `E = P_v ∘ f - ρ ∘ P_u`.

**What exact commutation implies.** `E = 0` implies Theorem Z: `ds = 0` on every natural trace and every one-site intervention followed by `f`. The representation transport agrees with the projected computation. It does not mention `L_c`.

**What it does not imply.** It does not imply that the stalk is read by the output. Let `f` be the identity on `R^2`, `P` the first coordinate, `ρ = 1`. Then `E = 0` for every vector, while the logit `h ↦ h_2` takes different values on vectors with the same stalk (`test_ce13_exact_abstraction_can_ignore_the_decision`). Commutation is internal coherence of the abstraction. Coherence is not use.

**The converse fails too.** A feature can be the logit itself while a proposed `ρ` is wrong. Copy the scalar and set `ρ = 0`. The feature determines `L` and the proposed compatibility residual is nonzero (`test_ce14_useful_feature_with_a_wrong_map`). Failure of a chosen `ρ` is not evidence that the feature is unused.

**Approximate commutation.** A small `E` bounds how far the projected child is from `ρ` applied to the projected parent. That is a bound on representational approximation. It becomes a bound on `|ΔL|` only after a Lipschitz or factorization assumption is added. Those assumptions are Families B and C, not consequences of `E`.

**Relation to causal abstraction.** Geiger et al. (2025) define faithfulness by agreement of intervened mechanisms. CE-13 shows that a commuting linear transport can hold while the logit reads a different coordinate. That blocks one implication. It does not place linear transport inside interchange invariance, and an interchange-accurate alignment of a real block does not produce a constant linear intertwiner. The two claims are incomparable. A vanishing residual is also not necessary for an exact nonlinear abstraction: `f(h)=h^2` has least-squares slope `0` and residual norm `√2` on `{-1,0,1}`, and it is still an exact map. Distributed alignment search (Geiger et al., 2024) estimates an alignment by optimizing an intervention loss. That estimator is not a `ρ` fit by ridge on clean activations.

**WHERE DOES THE CAUSAL CONTENT ENTER?** It does not, unless an output-factorization or intervention-agreement assumption is added. Those additions are named below and in the literature file. They are not commutation.

## Family B — Transport error and Lipschitz propagation

**Definition.** Assume `|L(a)-L(b)| ≤ K ||a-b||_M`.

**Theorem Lip.** `|L(s) - L(Π s)| ≤ K √D(s)`.

**Proof.** `D(s) = ||s-Π s||_M^2` by the projection theorem in finite dimension with `M ≻ 0`, which is the MVP argument. Apply the Lipschitz inequality to the pair `(s, Π s)`. QED.

**What the bound is.** An upper bound on the absolute gap between `L(s)` and `L` at one particular consistent vector, the nearest one. It is not a lower bound. It is not a sign. It does not say that a different consistent vector has a nearby logit. It does not say that moving toward `Π s` increases `L`.

**Explicit failure of the sign.** Identity edge, `L(s)=s_1`, `s=(1,0)`, `Π s=(0.5, 0.5)`, `D=0.5`. The gap is `0.5 ≤ √0.5`. The same distance at `(-1,0)` has gap `0.5` with the opposite sign (`test_ce15_bound_does_not_determine_sign`).

**Comparison with ordinary sensitivity.** If the cochain is a linear image `s = P φ` and `||P φ||_M ≤ c ||φ||`, then `√D(s) ≤ c ||φ-φ'||` whenever `Π s` is the image of some `φ'`. The resulting bound is at most a constant times the ordinary Lipschitz bound in activation space. Equality of constants is not the point. The point is that the bound uses the metric distance to a subspace and nothing else. It does not use a candidate-specific mechanism beyond the choice of the subspace.

**Multi-path and residual cases.** If `d` omits a parent, `D` measures that omission (Family G). A Lipschitz bound on that `D` then bounds distance to the wrong constraint set. The bound remains valid as mathematics and refers to the wrong set.

**WHERE DOES THE CAUSAL CONTENT ENTER?** In the Lipschitz constant of `L`, which is ordinary sensitivity of the readout, and in the choice of `M`. No directional response is derived.

## Family C — Factorization through the consistent subspace

**Definition.** `L(s) = q(Π s)` for some function `q`.

**Theorem Idle.** Under that identity:

1. `L(s) = L(Π s)`, so the move from `s` to the nearest consistent state does not change `L`.
2. Along any `M`-normal direction `r`, `L(Π s + t r) = L(Π s)` while `D(Π s + t r) = t^2 ||r||_M^2`.
3. States with equal `D` and different projections can have different `q(Π s)`.

**Proof.** `Π(Π s + t r) = Π s` because `r` is `M`-orthogonal to `ker d` and `Π` is the `M`-orthogonal projection. Substitute into `q`. The distance formula is the definition of `D`. QED.

**Numerical instance.** `s=(1,0)`, `Π s=(0.5,0.5)`, `L=s_1+s_2`. Then `L(s)=L(Π s)=1`, and `D` falls from `0.5` to `0`. Adding `(0.7,-0.7)` raises `D` and leaves `L` at `1` (`test_factorization_through_kernel_makes_discrepancy_idle`).

**What people might have expected.** That "the decision depends only on the consistent part" would make low obstruction good. The algebra says the opposite about the repair: the component `D` measures is the component `L` ignores. Factorization makes obstruction causally idle. It does not make obstruction a cause.

**Uniqueness.** `q` is unique on the set of projections that occur, and arbitrary on consistent states that do not occur. Many projections realize the same `D`, so `D` still does not determine `q`.

**Identifiability.** Declaring that the orthogonal complement of `ker d` is irrelevant is a causal-sufficiency claim. It is not implied by the existence of `d`. It is the definition of a sufficient statistic, and it is testable by intervening in that complement. The test uses the intervention response. It cannot be replaced by a small value of `D` on natural data: natural data can lie near `ker d` because `ρ` was fit to them (Theorem App), while the complement remains causally live (CE-13).

**WHERE DOES THE CAUSAL CONTENT ENTER?** In the assumption `L = q ∘ Π`. Under that assumption the content is that obstruction does not affect `L`.

## Family D — An energy that the model is supposed to descend

Four claims, kept separate:

1. An analyst writes down a scalar `E(s)`.
2. The network's forward pass implements descent on `E`.
3. The argmin of `E` is the candidate decision.
4. Interventions that decrease `E` increase `L` in the network.

Claim 1 is always available. Set `E = D`. Nothing about the Transformer follows.

Claim 2 is false of the computation as defined. A block computes a map `f`. It does not iterate `s ← s - ∇D(s)` until convergence. No COR source supplies a theorem that the residual stream is that iteration.

Claim 3, written as `L = -α D + B` with `α>0`, makes claim 4 true by substitution. **Theorem (assumption-encoded).** If `L(s) = -α D(s) + B(s)` and `B` is held fixed, then a decrease in `D` increases `L` by `α` times that decrease. **The hypothesis is the result.** Replacing `-α` by `+α` reverses the prediction and is equally easy to write (`test_ce16_monotonic_assumption_is_the_whole_implication`).

A decomposition `L = -α D + B` could be an empirical regression. Fitting `α` on the responses later used as evidence is the circularity numbered F7. Fitting `α` on held-out responses estimates a correlation between `D` and `L`. It does not show that the network descends on `D`. The Phase 1 record already reports an obstruction-only Spearman with margin change of about `0.095` against about `0.905` for attribution, on the audited 2D comparison. This step does not refit those numbers and does not treat them as a reason to choose the sign of `α`.

**WHERE DOES THE CAUSAL CONTENT ENTER?** In the identification of `L` with a decreasing function of `D`, or in the unsupported claim that the forward pass is descent on `D`.

## Family E — Path-specific influence

**Definition.** For a linear map `φ_v = J φ_u`, the first-order effect of `do(φ_u ← φ_u+δ)` on a linear readout `w·φ_v` is `w·J δ`, after recomputation. That product is a Jacobian-vector product along the unique path.

**Relation to `ρ`.** If `ρ` is set to the manuscript's equation (11) and `P = I`, then `ρ = J` and Theorem Z says the residual on the recomputed pair is zero. The path effect `w·J δ` is not a function of that residual. The residual is zero while the path effect is whatever `w·J δ` is.

If `ρ` is not `J`, the residual on a natural linear pair is `(ρ-J)φ_u`, which is the error of the surrogate Jacobian. Substituting the surrogate into `w·ρ δ` computes the wrong path effect. The size of `(ρ-J)` bounds that error only after a norm on `w` is chosen. Again a Lipschitz statement. It does not create a new path.

**Mediation, scrubbing, interchange.** Each of those methods compares intervened outputs. Their numbers are values of `R_c` or of an interchange score. A compatibility residual that is not computed from those interventions is not those numbers. A compatibility residual that is computed from them is not a prediction of them.

**Nontrivial distinction.** None survives inside this family. The linear path effect is the Jacobian product. The nonlinear path effect is the finite intervention `F(do(·)) - F`, which is the definition of `R_c`, not a function of `D`.

**WHERE DOES THE CAUSAL CONTENT ENTER?** In `J` and `w`, which are the linearization of the model and the readout. Those are the attribution baseline.

## Family F — Stability

**Objects that can be defined without a fake iteration.** The Jacobian of `L` at a state, its singular values, and the first-order change of `L` under a specified set of `δ`. Contraction and attraction require a map that is reapplied. A single forward pass does not provide that map. This family therefore refuses fixed-point theorems for the Transformer block.

**Separations.**

- Low obstruction, large sensitivity: put the Jacobian of `L` on `ker d` and set it to zero on the complement. Then `D` can be increased with no first-order change in `L`, and a unit change inside `ker d`, where `D` stays zero, changes `L` arbitrarily. This is Theorem Idle with an extra statement about the derivative.
- High obstruction, zero sensitivity: put the Jacobian of `L` at zero everywhere. `D` is free and `R=0`.
- Small Jacobian, wrong decision: `L=-1` constantly, `D=0`. Locally insensitive and incorrect (CE-04 plus a zero derivative).

**Update Proposition 2 is not a proved bridge, and the pointwise Jacobian does not play the role the formula suggests.**

The printed hypothesis says the margin response is locally Lipschitz and then writes `sup_{||δ||≤r} |J_m δ|`. A Jacobian at the base point does not bound a finite increment. Counterexample: `L(t)=1-2t^2` at `0`, radius `1`. Then `L'(0)=0`, so the displayed Jacobian budget is `0`, and `|L(0)|=1` clears it. `L(1)=-1`, so the sign flips (`test_pointwise_jacobian_does_not_bound_a_finite_step`). Adding `τ O(0)=0.5` still leaves `1 > 0.5` and the sign still flips. The obstruction term does not patch a missing Lipschitz bound.

A correct sufficient condition is different and still does not use obstruction. If `|L(h+δ)-L(h)| ≤ S` for every `||δ||≤r` and `|L(h)| > S`, the triangle inequality preserves the sign. For `τ≥0` and `O≥0`, replacing `S` by `S + τ O` only makes the hypothesis harder. Deleting `τ O` weakens it. On that corrected statement, obstruction is slack beside a genuine increment bound. The manuscript's mixture of the word "Lipschitz" with a pointwise `J` does not establish the printed formula. Phase 1 was right that the repository contains no proof. This step does not supply one that makes `O` causal.

**WHERE DOES THE CAUSAL CONTENT ENTER?** In a genuine bound on the increment of `L`. The pointwise Jacobian does not provide that bound. Obstruction does not either.

## Family G — Several parents, residual branches, attention-like mixing

**Joint residual.** For an additive aggregation, define `(d_joint s) = ρ_u s_u + ρ_w s_w - s_v`. Exact aggregation gives `d_joint s = 0` on recomputed states, which is Theorem Z again.

**Pairwise residual.** `s_u - s_v` at `(1,1,2)` equals `-1` while the joint residual is `0`. An equal-weight mix `v = (u_1+u_2)/2` at `(1,-1,0)` has joint residual `0` and pairwise residuals `+1` and `-1` (`test_attention_mix_pairwise_residual_cancels_in_the_aggregate`). The pairwise score reports a conflict the aggregation does not have. Paths can also agree and the aggregate can still be wrong if both carry the same error: `u_1=u_2=1`, `v=1`, pairwise residuals `0` against `ρ=1`, while a target value `0` is missed. Agreement of the transports is not correctness.

**Edge duplication.** Two copies of one mismatched edge, equal weights, double `O` (`test_ce12`). The computation is unchanged. The score is a choice of how many times to write the edge.

**Node splitting.** Replacing one node by two nodes tied by `ρ=I` adds a constraint. If the split copies are equal, `O` gains nothing. If a later gauge changes one copy and not the metric, `O` moves. The underlying map need not move (file 08).

Adding edges does not add a decision mechanism. It adds rows to `d`.

**Scope, after review.** The zero-residual claim is for a fixed linear aggregation, as in `v = u + w` or a constant-weight average of two scalars. A softmax weight that depends on the inputs is not a constant `ρ`. No constant pair intertwines that mix on an open set, so the multi-parent clause of Theorem Z does not apply to attention. The residual of a shared linear map on a softmax or GELU edge is approximation error with the parent set held fixed. Including a branch returns the residual to zero only when the branch is linear, as in `h_v = h + h`. It does not do so for `h + tanh(h)`.

**WHERE DOES THE CAUSAL CONTENT ENTER?** It does not. A joint residual that matches a fixed linear aggregation is zero on recomputed states of that aggregation. A residual against any other graph measures that graph.

## Family H — Counterfactual composition

**Definition.** For two interventions whose targets are not nested, clamp both, recompute every non-clamped node in topological order, and set

```text
I(a,b) = F(do(a,b)) - F(do(a)) - F(do(b)) + F(∅).
```

If one target is a descendant of the other, this expression is not yet defined: clamping the descendant and also recomputing it are different policies, and the two policies need not agree. Overlap of descendant sets is not the same condition. Two incomparable parents of one child have a well-defined joint clamp.

`I` is not the mixed Hessian of a single untruncated map. The four terms can come from different truncated graphs. A finite contrast can also have the opposite sign from the mixed second derivative.

**What addition does not imply.** Residual addition does not imply `I=0` once the readout is nonlinear. For `v = u + w` and `y = v^2`, at the origin, each single clamp of size `1` gives `y=1`, the joint clamp gives `y=4`, and `I=2`. The aggregation residual is `0` on all four recomputed worlds (`test_nonlinear_head_has_interaction_while_recomputed_residual_is_zero`). So `D=0` while `I=2`. The interaction is not a function of `D`.

**Circular use.** If all four terms are measured, `I` is an estimand. Using it as a formula for the joint term `F(do(a,b))` is circular because that term is already in `I`. Rejecting that misuse does not reject the contrast as a measurement.

**Non-circular use.** A hypothesis stated before the measurement, for example that two already materialized addends contribute linearly and the readout is linear, can imply `I=0`. That hypothesis is ordinary. It is false for a nonlinear head on a residual sum, and it is not Definition 3.

The visible/null factorial letter `TARGET_SPECIFIC_VISIBLE_NULL_INTERACTION_NOT_SUPPORTED` is a Direction 1 outcome about a different intervention design. This step does not reinterpret it as a COR result and does not fit `ρ` to it. It also does not treat a near-zero interaction as evidence for a consistency energy.

**WHERE DOES THE CAUSAL CONTENT ENTER?** When `I` is computed from the four intervened outputs, the joint output is an input to the score, not a prediction. When `I=0` is assumed in advance, the assumption is ordinary linearity of the readout on already materialized addends, and the square-readout example shows it is false for a residual sum. Neither version is `D`.

## Summary of what each family actually yields

| Family | Valid statement | Directional prediction of `R_c` | Status as a COR bridge |
|---|---|---|---|
| A | `E=0` implies `ds=0` on recomputed traces | No | Fails. Coherence is not use |
| B | Absolute gap to `Π s` is at most `K√D` | No | Generic Lipschitz bound |
| C | `L∘Π` ignores the component `D` measures | The repair move has `ΔL=0` | Idle, not a cause |
| D | `L=-αD+B` implies the desired sign | Only by assumption | Rejected as an explanation |
| E | Path effect is `w·Jδ` or the finite `R_c` | Yes, and it is attribution | Not distinct from the baseline |
| F | A genuine increment bound preserves sign; a pointwise Jacobian does not; `τO` repairs neither | No signed role for `O` | Obstruction is slack or unused |
| G | Fixed linear joint residual vanishes; attention is not that map | No | Theorem Z only for fixed linear aggregations; otherwise approximation error |
| H | On non-nested clamps, `I` need not match `D`; nesting is undefined until truncation is specified | `I` is not a function of `D` | Not Definition 3 |
