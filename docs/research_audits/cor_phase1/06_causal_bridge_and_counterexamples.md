# 06 — Causal bridge and counterexamples

## 1. Decision function

Let `F_θ(x)` be the frozen model output distribution / logits. Candidate score `L_c(x)=m_c(x)=z_a-z_b` in the manuscripts. Intervention `do(h_u ← h_u+δ)`. Response `ΔL_c = L_c(do(·))-L_c(x)`.

These are not interchangeable with JS, KL, legal-move mass, accuracy, or obstruction reduction. The manuscripts mix them in `S_u=-ΔO+β Δm-γ Cost`.

## 2. Does low obstruction imply candidate selection?

The manuscripts do **not** prove `D_c<D_{c'} ⇒ L_c>L_{c'}`. The thesis says evidence “must become sufficiently compatible” to support commitment. That implication is not a theorem.

Counterexample (Attack A): two-node identity sheaf, same `s`, `ρ`, `O`, `D`; two linear heads `w` and `-w`. Opposite selected candidates. Verified: `test_attack_a_same_obstruction_opposite_decisions`.

Status of the implication: PROVEN FALSE BY COUNTEREXAMPLE without extra assumptions linking the head to `ker d`.

## 3. Does reducing obstruction improve behavior?

Does `D(s+δ)<D(s)` imply `L(s+δ)>L(s)`? NO.

Attack B: `s=(1,0)`, Euclidean identity edge, `D>0`; `s'=(0,0)` has `D=0` but smaller target logit if the head reads the first coordinate. Verified: `test_attack_b_lower_obstruction_worse_target_logit`.

Reverse: `s=0` has `D=0`; `s''=(4,0)` has larger `D` and larger target logit. Verified: `test_attack_b_higher_obstruction_better_target_logit`.

Obstruction reduction is not sufficient for behavioral repair. It is not necessary either.

Proposition 1 / original Proposition 2 (minimum-energy repair) **assumes** an objective that already contains `-β m_c`. The closed form then trades `∇O` against `∇m`. That is an optimization identity, not a derivation that `O` causes `m`. If `β=0`, the “repair” minimizes obstruction regardless of the decision. If `β` is large, the score is attribution.

Status: PROVEN UNDER STATED ASSUMPTIONS as a quadratic-program solution; does not license “obstruction reduction repairs decisions.”

## 4. The missing causal bridge

Search result: **no manuscript theorem** derives `D_c` from `F_θ` or conversely.

Possible additional assumptions, labeled as such (not original results):

| ID | Additional assumption | Sufficient for some implication? | Necessary? | Independently measurable? | Falsifiable? | Tautological? | Supported empirically? | Gauge-stable? | Applies to Transformers? |
|---|---|---|---|---|---|---|---|---|---|
| B1 | Downstream logits factor through a global section `g∈ker d` | YES, then `D>0` states are off the decision manifold | NO | Hard | YES, if specified | NO | NO. Othello factorial negative for visible/null match | Only with transformed metrics | Unknown |
| B2 | Model energy includes `O` | YES | NO | NO in frozen LMs | YES if specified | Near-yes if `O` defined from `m` | NO | NO | NO evidence |
| B3 | `m` Lipschitz and `|m|>Lip+τ O` (Update Prop. 2) | Stability of **already positive** margin | For that proposition | `τ` free | Conditionally | Can be tuned | NOT PROVED, not tested as stated | Metric-dependent | Local only |
| B4 | `∇O` aligned with `∇m` | Then reducing `O` increases `m` locally | NO | YES (`corr(∇O,∇m)`) | YES | NO | 2D: obstruction-only poor for `Δm`; attribution excellent | NO | Empirically false as a general rule |
| B5 | `ρ` equals true Jacobian of computation | Then `O` is linearization residual of assigned germs | For “computational consistency” reading | YES in principle | YES | NO | Ridge `ρ` is not shown equal to Jacobian | Local | Surrogate only |
| B6 | `S=-ΔO+βΔm` predicts repair | By construction if `β` large | — | — | — | YES if `β` selected on `Δm` | 2D selected `β=2`; repair AUC worse than attribution | — | Observed |

B6 is the actual experimental object. It is not a derivation from sheaf mathematics.

## 5. Tautologies to reject

“The decision score decreases whenever obstruction increases” assumes the conclusion.

“Obstruction predicts interventions that change obstruction” is a Taylor identity for `O` (2D `pred_ΔO` AUC 0.943). It does not mean `O` is the decision mechanism.

“Compatible germs glue into a stable global commitment” is the thesis, not an independent assumption that can be used to prove the thesis.

## 6. Attack L — off-manifold repairs

The nearest point of `ker d` need not be a naturally attainable activation. Verified: `test_attack_l_consistency_projection_can_leave_natural_set`. Projecting onto a consistency subspace can leave the image of `φ`. Causal repair claims that treat `g_*` as “the model’s intended state” are then undefined as interventions on the model’s attainable manifold.

## 7. What would make a noncircular bridge

An independently frozen map family whose kernel is shown, **without using evaluation `Δm`**, to be the only channels that affect `L_c` in a specified intervention class. That is a causal-abstraction / interchange claim. It is not implied by defining `d`. Othello work tested related interchange and got mixed/negative official letters (`0A C/PARTIAL`, validation NEGATIVE, factorial NOT_SUPPORTED). Those letters are not COR obstruction tests (`OBSTRUCTION_READY=NO`).
