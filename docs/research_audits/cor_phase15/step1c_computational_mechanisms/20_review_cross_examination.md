# 20 — Cross-examination

The four review passes were separate subagents in this session. Their ids are in `reviews/`. They are not external referees. After they returned, the sentences they broke were rewritten and the new finite checks were run. `31` tests passed. A passing test is still a finite check.

| objection | response | where |
|---|---|---|
| One zero of `I_F` is necessary for separability, not unnecessary | The row now says necessary and not sufficient. `(u+v)^3` at `(1,-1)` remains the sufficiency counterexample | `11`, `10` |
| Zero interaction is not equivalent to `Y=AVW_O` | The clamped-pattern row is one direction. `u^2+v^2` is the converse counterexample | `11`, `10`, `18` |
| Fiber quantifiers in `03` were reversed | The row now says equal projections imply equal responses | `03` |
| Residual zero is not equivalent to "the child was assigned by `f`" | The converse was dropped. A copying clamp is the counterexample | `11` |
| `w·h=0.5` in the RMSNorm comment | The comment now matches `h=(1,1)`, `w=(1,-0.2)`, value `0.8`, flipped value `-1.2` | the test |
| Interaction facts were attributed to the pre-norm block | They are attributed to `Y=AVW_O` or to `Q=XW_Q` with no bias and no LN. The pre-norm edit goes through `LN(H+\delta)` | `04`, `proofs/attention_value_linearity.md`, `03` |
| `W_Q=0` fails once `Q` has a bias, and inside one token once the value path is LN | Both boundaries are stated and tested. Cross-token edits at a constant pattern stay additive and are tested, so the correction does not say LN always interacts | new tests |
| Softmax does not raise every other weight just because the row sums to one | The claim is now the two-point identity, plus `\partial a_i/\partial z_k=-a_i a_k` for softmax | `04` |
| The second attention output is `1/(1+e^{16})`, not `0` | `06`, `09`, `14`, and `proofs/fiber_and_trace.md` use the exact value | those files |
| Rank-one `\rho` zeros a larger set than one pair | `12` states the ray and the orthogonal complement | `12` |
| ReLU rescaling was unquantified and ignored bias | Inactive units do not move. The identity is bias-free. The MLP in `04` has biases and is not included | `13` |
| "Coordinates the family reads" was called fiber constancy | The architectural hypothesis and the property of `R` are split | `12` |
| "The ancestor effect double-counts a screened path" | The path is absent. The table no longer says the clamped node is recomputed | `05` |
| "No extra hypothesis saves the LayerNorm sign for all `h`" | Too broad. `1^\top w=0`, `\gamma=1`, `\beta=0` does save it. The universal claim over `w` stays false and is tested both ways | `11`, `proofs/layernorm_readout.md` |
| The tuple-determines-a-function-of-the-tuple arrow was labeled a proved implication | It is now a definitional identity | `03` |

No objection was answered by adding an assumption that the margin improves when a residual falls. No historical letter was edited. The four Step 1B corrections in `01` still stand: the review narrowed later sentences, and it did not put `M` back into `O`, did not restore a universal LayerNorm sign claim, did not put `ker P_u` back inside every residual, and did not restore "nonlinear hence interacting."

Reviewer 5's reduction still applies after the repairs. Reviewer 7 found one over-closed sentence and no reason to reopen `O` or `D`. Reviewer 8's condition for a necessary-conditions verdict was that the necessity claim stay inside the classes actually proved and that the stacks stay unresolved. That is the scope of `23`.
