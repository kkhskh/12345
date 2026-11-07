# 10 — Counterexample catalog

Each row is a different failure. Calculations are in `tests/test_causal_bridge_math.py`. The identity-edge examples use `d=[1,-1]`, `M=W=I`, and `D(s_1,s_2)=(s_1-s_2)^2/2`. They are not twenty copies of one trick: the later rows change the graph, the map class, or the intervention semantics. A repair is listed only when a mathematical restriction blocks the example. "Blocks it" does not mean the restriction is independently justified.

## CE-01 — Equal obstruction, opposite decisions

System: `s=(1, 0.5)`, `O=0.25`, `D=0.125`. Heads `w=(1,0)` and `w=(-1,0)`.

Claim attacked: `O` or `D` selects the candidate.

Intervention: none required. The two heads are two systems with the same cochain.

Result: `L=1` and `L=-1`.

Assumptions satisfied: linear coboundary, `M≻0`, `W≻0`.

Defeats: F1, F3, F25, and any universal reading of `D_c < D_{c'} ⇒ L_c > L_{c'}`.

Repair: fix the head. That repair is the output gradient. It is the attribution baseline, not a property of `D`.

## CE-02 — Reduced obstruction, decreased logit

`s=(1,0)`, `D=0.5`, `L=s_1=1`. Move to `(0,0)`, `D=0`, `L=0`.

Claim attacked: `ΔD<0 ⇒ ΔL>0`.

Defeats: F2.

Repair: assume `L` is a decreasing function of `D`. CE-16 says that assumption is the claim.

## CE-03 — Increased obstruction, increased logit

`(0,0) → (4,0)`. `D` increases and `L` goes from `0` to `4`.

Claim attacked: higher obstruction harms the target.

Defeats: the converse of F2.

Repair: the same monotone assumption as CE-02.

## CE-04 — Perfect consistency, incorrect decision

`s=(1,1)`, `D=0`, `L=-s_1=-1`.

Claim attacked: `D=0` means the decision is correct.

Defeats: F1 on the kernel itself.

Repair: identify `ker d` with the correct-answer set. That is a new empirical claim, not a consequence of `dg=0`.

## CE-05 — Zero discrepancy on every exact natural computation

`f(h)=2h`, `ρ=2`. Residual `0` at `h∈{-1.5,0,0.75}` and after every `h←h+δ` with recomputation. Stitching child `0` to parent `1` gives residual `2`.

Claim attacked: obstruction varies across natural decisions.

Defeats: the hope that exact computational `ρ` still ranks decisions. It ranks nothing on Domains A and B.

Repair: none inside exact commutation. Leaving exact commutation means the residual is no longer the computation.

## CE-06 — Nonzero discrepancy from a dropped coordinate

`f=I` on `R^2`, parent stalk keeps coordinate 1, three inputs `(1,0),(1,1),(1,-1)`. Least squares from the parent stalk predicts child coordinate 2 as `0`. Residual norm `√2`.

Claim attacked: nonzero `ds` is a candidate conflict.

Defeats: that reading of Theorem App. The residual is the discarded coordinate.

Repair: enlarge `P` until commutation holds. If the enlargement includes the logit, the stalk has been given the answer.

## CE-07 — Ranking reversal under a metric change

Two identity edges. `s_1=(1,0,0,0)`, `s_2=(0,0,1,0)`.

`M=diag(10,10,1,1)` gives `D(s_1)>D(s_2)`. `M=diag(1,1,10,10)` reverses it. `W=diag(10,1)` versus `diag(1,10)` reverses `O`.

Claim attacked: candidate ranking by `D` is intrinsic.

Defeats: F5 and F30 for metrics.

Repair: freeze one `M`. Two frozen choices remain legal and disagree.

## CE-08 — Trivial projection

`P=0`, `s=0`, `O=D=0`.

Claim attacked: the definitions force a nontrivial decision state.

Defeats: F5.

Repair: a nondegeneracy constraint on `P`. Convention, not a cause.

## CE-09 — Input-specific map

On each example, `ρ = s_v s_u^T` for a basis vector `s_u` satisfies `ρ s_u = s_v`. One shared zero map leaves residual above `1` on the constructed pairs.

Claim attacked: agreement is a property of the computation rather than of `ρ`.

Defeats: F5 and F30.

Repair: one shared `ρ`. Necessary, and CE-06 still applies.

## CE-10 — Equivalent rescaling, different Euclidean obstruction

`O=1` at `ρ=1`, `s=(1,0)`. After `ρ'=2` and unchanged Euclidean `W`, `O=4`. `W'=1/4` restores `O=1`.

Claim attacked: Euclidean `O` is representation-invariant.

Defeats: F29's geometric reading of the coded score.

Repair: transform `W` with the stalk. The estimator has to implement that transform. Algebraic invariance does not do it automatically.

## CE-11 — Low obstruction off the natural set

Natural image `{(1,1)}`. The vector `(2,2)` has `D=0` and is not natural. The Euclidean repair of `(1,0)` is `(0.5,0.5)`, which is also not natural.

Claim attacked: the nearest global section is an attainable model state.

Defeats: reading `ĝ` as the network's intended activation.

Repair: restrict the minimization to the image of `φ`. That is a different optimization. It needs the image, which the quadratic formula does not use.

## CE-12 — Redundant paths and duplicated edges

Joint rule `s_v=s_u+s_w` at `(1,1,2)` has residual `0`. The single-parent residual `s_u-s_v` equals `-1`. Writing that mismatched edge twice doubles `O` from `1` to `2`.

Claim attacked: the numerical residual is fixed by the computation.

Defeats: F30 for graphs, and the multi-path reading of a pairwise `d`.

Repair: use the true parent set. The exact true parent set returns the residual to `0` on recomputed states (CE-05).

## CE-13 — Exact abstraction with no decision content

`f=I` on `R^2`, stalk = first coordinate, `ρ=1`, logit = second coordinate. Commutation error `0`. Vectors `(1,3)` and `(1,-4)` share the stalk and not the logit.

Claim attacked: commutation implies causal relevance.

Defeats: Family A as a bridge.

Repair: require the logit to factor through the stalk. That is Family C, which makes the inconsistent component idle, or it is an interchange claim in the sense of Geiger et al. (2025).

## CE-14 — Causally used feature, failed proposed map

The scalar is copied and the logit equals the scalar (`L=1.5`). The proposed `ρ=0` has nonzero residual.

Claim attacked: failure of compatibility implies the feature is unused.

Defeats: the converse of Family A.

Repair: set `ρ` to the true copy. The residual becomes `0` and CE-05 applies: the score no longer moves with the decision.

## CE-15 — A correct bound with no direction

`|s_1 - 0.5| = 0.5 ≤ √D` at `s=(1,0)` and at `s=(-1,0)`, with opposite signs, and `D` equal.

Claim attacked: Theorem Lip predicts the sign of the repair.

Defeats: Family B as a directional bridge. The bound itself stands.

Repair: add `∇L`. That is file 09.

## CE-16 — The assumption is the result

On `D∈{0, 0.25, 1}`, the function `-D` decreases and the function `+D` increases.

Claim attacked: "if `L` decreases with `D`, then reducing `D` increases `L`" is an explanation.

Defeats: Family D as anything but a reparameterization of the desired monotone relationship.

Repair: none that stays noncircular. An independent derivation that the network computes descent on `D` would be a different theorem. It is not in the sources.

## CE-17 — Descent on obstruction opposes the logit gradient

`∇O=(2,-2)`, `∇L=(1,0)`, inner product `+2`. Descent direction `(-2,2)` has derivative `-2` for `L`. The attribution direction `(1,0)` has derivative `+1` and is not parallel to the descent direction.

Claim attacked: an obstruction score predicts intervention sign beyond the output gradient.

Defeats: F29 and the alignment wording that treats a positive inner product as a repair condition.

Repair: take the step `∇L` instead. The consistency term is then unnecessary for the sign.

## CE-18 — A fitted alignment of an unrelated algorithm

Activations `0.2` and `0.9`, network outputs `0` and `1`, high-level algorithm the constant `1`. A lookup that ignores the activation matches the algorithm on both inputs and does not match the network.

Claim attacked: a perfect alignment residual recovers the computation.

Defeats: F5 in the form proved generally by Sutter et al. (2025) for unrestricted alignment classes.

Repair: restrict the class tightly enough that a lookup is excluded, and then show the restricted class still carries a causal prediction. The restriction that returns `ρ` to a shared linear map returns the problem to CE-05 and CE-06.

## CE-19 — Recomputation keeps consistency and changes the decision

`f(h)=2h`, `ρ=2`, logit read at the parent. `do(h←h+1)` followed by `f` leaves residual `0` and changes the logit by `+1`.

Claim attacked: preserving computational consistency preserves the decision.

Defeats: any identification of "consistent" with "decision-invariant" on Domain B.

Repair: none under Theorem Z. Consistency was preserved because the computation was exact.

## CE-20 — Stitch with a large residual and no output effect

Parent `0`, stitched child `100`, residual `100`. Logit defined as the constant `0.1`, so the stitch changes `L` by `0`.

Claim attacked: large discrepancy implies a large causal effect.

Defeats: reading a Domain D assignment as a Domain B intervention.

Repair: recompute the child. Under an exact `ρ` the residual returns to `0` (CE-05) or, if `ρ` is wrong, becomes the surrogate error (CE-06). Neither case makes the stitch itself a causal effect.

## Class coverage

| Class in the assignment | Example |
|---|---|
| Exact linear | CE-01, CE-05, CE-19 |
| Rank-deficient | CE-08, CE-06 |
| Several parents | CE-12, attention-mix test |
| Piecewise linear | ReLU global-slope test in file 03 |
| Smooth nonlinear | `f(h)=h^2` on `{-1,0,1}` has least-squares slope `0` and residual norm `√2` |
| Residual | residual-skip test, file 03 |
| Attention-like mix | equal-weight cancellation test |
| Intervened graph | CE-19 versus CE-20 |
| Reparameterization | CE-10 |
| Approximate abstraction | CE-06, CE-18 |
