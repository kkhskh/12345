# Workstream 6. Adversarial audit

Statuses used in `PROOF_REGISTER.csv`, and nowhere else as claim labels: `PROVED`, `CONDITIONAL`, `COUNTEREXAMPLE_FOUND`, `DISPROVED`, `REDUNDANT_WITH_KNOWN_RESULT`, `IDENTIFICATION_UNRESOLVED`, `CAUSAL_INTERPRETATION_UNSUPPORTED`, `NOT_EVALUATED`, `CONJECTURE`.

Reviews A–H are passes in this same cloud environment. They are not external referees. Each dropped assumption below was checked by an analytic identity and by a recomputed quantity that does not call that identity.

| claim | assumption dropped | what happens |
|---|---|---|
| Finite level set equals the constants | A second gain is set to zero | The algebraic pair `h=(1,-1,0,0)`, `h'=(√2,-√2,√2,-√2)` at `ε=1`, `γ=(1,1,0,0)` preserves LayerNorm and is not constant. |
| Finite level set equals the constants | `ε=0` and `v>0` | Positive scale joins the level set. Already proved in the `ε` split. Not re-used as a positive-`ε` fact. |
| `|R|` unbounded in budget B0 | The edit is constant | The scaling family leaves `X` fixed, the MLP stays zero, and `R` sticks at `κ`. The hypothesis "non-constant" is necessary. |
| Lipschitz constant `≥ \|\|g\|\|` | Differentiability at `H` | The inequality is the definition of the derivative along lines. A kink is outside this `tanh` block and was not given a separate bound. |
| Theorem K | `ΔX` is replaced by `J δ`, one step `0.8` | Linear skip matches `Dm`; finite `R` differs by `0.00741`, and the gap grows with the step. |
| Theorem K adds non-constant edits | Stacked `QKV` is injective and gains are full | Kernel of the stack is `{0}`, so Theorem F forces constants. Seeds 0–11 are injective. |
| `O=D` | `W` is Euclidean rather than `(d M^{-1} d^T)^+` | `O=1`, `D=1/2` on the identity edge. |
| `D ↓ ⇒ L ↑` | No extra link from the head to `ker d` | Zero cochain lowers `D` and lowers the first-coordinate readout. |
| Both deletions necessary for `c1=1`, `c2=-0.5` | The sum is positive | Deleting the negative summand leaves a positive margin. |
| Value-only sign equals the total sign | The pattern is not clamped | Seed 16: total `+0.02019538349844552`, value-only `-0.008580691798845158`. |
| `Dm=0 ⇒ R=0` | The step is finite | Seed 0, direction orthogonal to `g`, step `0.2`, finite response about `-0.629`. |
| A dropped coordinate determines `sign(R)` | The states differ in that coordinate | Seed 16, feature 3, opposite signs. |
| Raw constant-shift identity for a final LN readout | The readout is `w·LN(H2)` | The constant changes `H2` and not `LN(H2)`, so that response is `0` while the raw response is `a 1^T w`. |
| Cohomology class constrains `R` | The cochain is global | `[ds]=0`, so the class is the zero functional. |

Rows that remain conditional rather than false: Theorem K's equality under its kernel hypothesis; the explicit `Λ` as a bound, not as a sharp constant; local isolation of exact zeros where the branch linear form is nonzero. The exotic large-amplitude zero is a `CONJECTURE` in the register only as an open existence question, not as a claimed mechanism.

Two routes, not one function calling itself:

1. Closed forms: the scale identity `R = α ||w||^2 / T`, the eigenvalue list of the LayerNorm bracket, the SymPy factorization `eps(1-lam^2)`, the skip dot under Theorem K, Popoviciu's `1/2` bound.
2. Recomputation: `response` through `margin` and `block`, central-difference gradients, SVD of the numeric Jacobian, the algebraic LayerNorm pair evaluated by the numeric map.

The Jacobian-vector product is a third check against central differences on the preserved seed-0 direction. It is a derivative check. It is not counted as the finite response.
