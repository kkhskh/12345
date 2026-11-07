# 21 — Corrected theorem register

The statements below are the ones that remain after `20`. Each line is one direction unless it says "iff". A finite test is named only as a check of an example. The general claim is the proof.

| id | statement | hypotheses | source |
|---|---|---|---|
| T1 | `F=g(u)+h(v)` implies `I_F\equiv 0` | product domain | `proofs/interaction_conditions.md` |
| T2 | On a `C^2` open rectangle about `0`, `I_F\equiv 0` iff the mixed partial is identically zero iff `F` is separable | `C^2`, rectangle, identical quantifier | same |
| T3 | `I_F(u_0,v_0)=0` is necessary for separability and not sufficient | one pair | `(u+v)^3` at `(1,-1)` |
| T4 | Nonzero mixed partial at `0` implies `I_F(tu,tv)\neq 0` for all small `t\neq 0` with `uv\neq 0`. It does not constrain a named finite pair | `C^2` | `uv-u^2v-uv^2` |
| T5 | Nonzero `I` at a pair implies the map is not affine. Nonlinearity does not imply nonzero `I` | affine maps are separable | `u^2+v^2` |
| T6 | `O=(ds)^\top W(ds)` does not depend on `M`. `D` can | `d`, `W`, `s` fixed | `proofs/metric_split.md`, `D=10/101` |
| T7 | A fixed `w` need not keep its sign under plain LN | existence | `h=(1,0)`, `w=(1,2)` |
| T8 | If `1^\top w=0`, `\gamma=1`, `\beta=0`, `\sigma>0`, then `w·LN(h)=(w·h)/\sigma` | `eps=0` | `proofs/layernorm_readout.md` |
| T9 | Plain RMSNorm without `\gamma` preserves linear signs. A diagonal `\gamma` need not | `rms>0` | the test |
| T10 | LN at `eps=0` is not injective, and its derivative has rank at most `d-2` | non-constant `h` | same proof |
| T11 | `\delta` leaves the linear residual invariant iff `\delta\in\ker(P_v A-\rho P_u)` | linear `A`, `\rho` | `proofs/measurement_kernel.md` |
| T12 | N1, both orientations in the domain | linear `Q`, linear `L` | unchanged from Step 1B, rechecked |
| T13 | `R=g(P(h),\delta)` on a set and a family iff equal projections imply equal responses there | exact, not approximate | `proofs/fiber_and_trace.md` |
| T14 | If `Y=AVW_O` and `A` is clamped, value-edit interaction is zero and the response equals the linear map. The converse is false | the clamp | `proofs/attention_value_linearity.md` |
| T15 | On `Q=XW_Q` with `W_Q=0`, no bias, and no LN, residual-edit interaction is zero. A query bias removes this. LayerNorm of two coordinates inside one token, the value path when `A` is constant and `W_V=I`, need not be additive. Edits of two different tokens in front of a constant pattern stay additive | those maps only | the three tests |
| T16 | There exist bias-free residual edits with nonzero interaction. Token-1 value computed by the harness: `-0.611855656607887` | that one map and baseline | existence |
| T17 | The projection that drops the key coordinate does not determine the two attention outputs `1/2` and `1/(1+e^{16})` | that pair of states | `06` |
| T18 | A scalar of a tuple does not determine every response the tuple determines | the pairs `(0,0)` and `(1,1)` | `proofs/fiber_and_trace.md` |
| T19 | If the child is assigned by the structural equation, the residual is zero, and that zero does not constrain a non-constant response. A copying clamp also has residual zero | the intervention row | `08`, `11` |
| T20 | Rank-one `\rho` zeros a ray and an orthogonal family of sources, not a discovered transport | `z_u\neq 0` | `12` |
| T21 | Sign survival after deleting one addend of `c_1+c_2` is the sign of the remaining addend | the other addend not recomputed | `09` |
| T22 | First-order expansion keeps `f(h_0)`. Applying `J` to the absolute state drops the offset | differentiability | the affine test |

Not in this register, because they are false: the original signed bridge; nonlinearity implies nonzero finite interaction; changing only `M` changes `O`; every fixed readout survives LN; `ker P_u` preserves every residual; zero interaction implies a linear attention map; `W_Q=0` implies additive pre-norm edits.

Not in this register, because they were not proved: multi-layer sufficiency, parallel branches, post-norm path decompositions, rotary scores, a full RMSNorm block, and any relational tightening of a Jacobian bound.
