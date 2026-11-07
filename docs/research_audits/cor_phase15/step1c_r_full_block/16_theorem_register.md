# 16 — Theorem register

Statuses name the mathematical claim. A numerical check is not given `PROVED`. Novelty is separate from truth. `REDUNDANT_WITH_KNOWN_RESULT` means the identity is correct and is an application of a known calculation, not a COR mechanism.

The model class, unless a row says otherwise, is the block in `04`: one pre-norm residual block, per-token LayerNorm, causal mask, two heads, `tanh` MLP, no dropout, no Q/K/V bias. The intervention class is `H ↦ H+E_{t,δ}` with the block recomputed, unless the row names a clamp.

| id | statement | status | novelty |
|---|---|---|---|
| R1 | The Jacobian of `LN_ε` is `diag(γ)[C/s − zz^T/(d s^3)]`, with `s=sqrt(v+ε)`. Bias does not enter. | `PROVED` | Calculus of the stated formula. SymPy checks the bracket at `d=3`. |
| R2 | For `ε>0`, `ker` of the bracket is `span{1}`. If every `γ_i≠0`, `rank J=d−1`. | `PROVED` | Corrects the Step 1C rank sentence. Standard for positive-ε LayerNorm. |
| R3 | For `ε=0` and `v>0`, if `γ` is invertible then `ker J=span{1,z}` and `rank J=d−2`. | `PROVED` | Ordinary kernel of a map invariant under shift and positive scale. The Step 1C sentence omitted the `ε=0` restriction. |
| R4 | For `ε>0`, exactly one zero gain and the others nonzero leaves `rank J=d−1`. Exactly two zero gains and the others nonzero leaves `rank J=d−2`. | `PROVED` | The image of the bracket is `1^⊥`. |
| R5 | `h ↦ h+a1` leaves `LN_ε` unchanged for every `ε≥0` and every gain. For `d≥2`, some gain nonzero, and `α>0` with `α≠1`, the identity `LN_ε(αh)=LN_ε(h)` for every admissible `h` holds if and only if `ε=0`. If every gain is zero the map is the bias, and the scale identity holds for every `ε`. The scale `α=1` holds for every `ε`. | `PROVED` | Ba, Kiros, Hinton define the map. The `ε` split is the correction. |
| R6 | On this block, `H_2(H+E+E_{t,a1})=H_2(H+E)+E_{t,a1}` for every residual edit `E`, every weight setting, and every `ε≥0`. Thus the constant changes `R` by `a 1^T w` if `t` is read and by `0` otherwise. The interaction of the constant with `E` is zero. | `PROVED` | Pre-norm plus addition, applied at an arbitrary base point. Not a new mechanism. The pure-constant case is the special case `E=0`. |
| R7 | If `m=c_1+c_2>0`, at most one deletion-to-zero is necessary for the positive sign. | `PROVED` | Repairs Step 1C. The old example is `DISPROVED` as a both-necessary example. |
| R8 | There exist a baseline and two contributions such that both deletions are necessary and the four-term interaction is identically zero. The pair `b=−1.5`, `c_1=c_2=1` is one. | `PROVED` | Exact arithmetic. Necessity is not interaction. |
| R9 | The directional derivative of causal attention splits into `dQ`, `dK`, `dV`, the masked softmax differential, and `W_O`. An edit at token `t` does not change attention outputs at `i<t`. | `PROVED` | Chain rule on Vaswani attention with a fixed causal mask. |
| R10 | At one `X`, attention directional derivatives add across heads. | `PROVED` | Linearity of the sum. Elhage et al. already add head outputs. |
| R11 | Finite responses of two leave-one-out networks, each with one head's `W_O` rows zeroed, need not sum to the joint response. | `COUNTEREXAMPLE_FOUND` | Seed 4. Not a theorem about every seed. |
| R12 | The value-only directional derivative can have the opposite sign from the total directional derivative. | `COUNTEREXAMPLE_FOUND` | Seed 16. The value-only map is the frozen-pattern OV term, not the residual intervention. |
| R13 | `Dm·E=0` does not imply `R=0`. | `COUNTEREXAMPLE_FOUND` | Seed 0, direction orthogonal to the token-1 gradient, step `0.2`. Taylor, not a new remainder theory. |
| R14 | A coordinate projection can agree on two states whose recomputed responses have opposite signs. No function of that projection returns both signs. | `PROVED` for the logical step. `COUNTEREXAMPLE_FOUND` for this block, seed 16, feature 3. | The logical step is the definition of sufficiency. Geiger et al. (JMLR 2025) already require the high-level variable to carry the intervened behavior. |
| R15 | On the constant-shift family, `R` does not depend on `H`. Every function of `H` is sufficient for that family. | `PROVED` | Immediate from R6. Vacuous as a representation theorem. |
| R16 | The residual of either structural assignment is zero on every recomputed trace. It does not order `m`. | `PROVED` | By construction. Using it as an obstruction is `CAUSAL_INTERPRETATION_UNSUPPORTED`. |
| R17 | A rank-one map solved from one token pair can change its residual under a constant shift that does not change `m`. | `COUNTEREXAMPLE_FOUND` | Seed 1. Approximation error, not a decision conflict. |
| R18 | Two token edits can have nonzero four-term interaction. | `COUNTEREXAMPLE_FOUND` | Seed 4, `I≈−0.062808`. Does not say every pair interacts. |
| R19 | If the skip, attention, and MLP contributions to `Dm` sum to zero, then `Dm=0`. | `REDUNDANT_WITH_KNOWN_RESULT` | The sentence is the definition of the sum. |
| R20 | `|Dm·E| ≤ \|\|g_t\|\| \|\|δ\|\|`. This does not bound the finite remainder. | `PROVED` for the inequality. `COUNTEREXAMPLE_FOUND` for promoting it to a uniform finite bound. | Cauchy–Schwarz and R13. |
| R21 | No score examined here, short of `m` or a derivative of `m` or the constant-shift evaluation of `1^T w`, determines `sign(R)` for a general `δ`. | `IDENTIFICATION_UNRESOLVED` outside the examined scores. Inside that list, the projection and the structural residual fail, and the derivative succeeds by being the derivative. | Not a proof that every future score fails. |

Quantifiers that are easy to drop, and are not dropped here: R6 is every weight vector of this architecture, not every Transformer. R12 and R13 are existence. R2 needs `ε>0` and, for the rank, nonzero gains. R4's two-zero clause needs the other gains nonzero, and its one-zero clause needs the other gains nonzero. R5's scale biconditional excludes `α=1` and `d=1`, and it requires some gain to be nonzero. If every gain is zero the map is the bias. R10 is the derivative at one point, not the leave-one-out finite responses in R11. The skip is not one of the head increments in that derivative.

Where the causal information enters is recorded in `13` and `15`. Rows that determine a sign use `m`, `Dm`, or `1^T w` together with the architecture. None of them uses a consistency residual.
