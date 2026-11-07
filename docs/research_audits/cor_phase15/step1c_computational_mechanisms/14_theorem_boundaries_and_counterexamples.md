# 14 — Theorem boundaries and counterexamples

The counterexamples are the smallest maps that hit a stated hypothesis. They are not trained models. Each one names the assumption it removes.

## Algebra

| map | what is true | what fails |
|---|---|---|
| `u^2+v^2` | nonlinear, `I\equiv 0` | "nonlinearity implies nonzero interaction" |
| `(u+v)^3` at `(1,-1)` and `(1,1)` | `I=0` and `I=6` on one nonseparable map | "the interaction is a property of the formula alone, independent of the pair" |
| `uv-u^2v-uv^2` | mixed partial at `0` equals `1`, `I(2,-1)=0` | "nonzero baseline Hessian implies every finite interaction is nonzero" |
| `5h+3` | affine offset | "apply `J` to the absolute state" |
| `h^2+3` at `2`, step `1` | remainder `1` | "first order matches the finite step" |
| `O` at fixed `d,W,s` across two `M` | `O=1` both times, `D=1/2` and `D=10/101` | "changing `M` changes `O`" |

## Representation

| map | what is true | what fails |
|---|---|---|
| `r=P_v h` with `\delta\in\ker P_u\setminus\ker P_v` | `P_u` is invariant, `r` moves by `1` | "kernel of one factor, kernel of the residual" |
| N1 pair `Q=(1\ 0)`, `\delta=(0,1)`, heads `\pm e_2` | same `Q`, opposite signs | sign from `Q` on a two-sign domain |
| `(z_v-z_u)` on `(0,0)` and `(1,1)` | same relation, different `z_v` | "a relation adds the tuple's information" |
| rank-one `\rho` fit to one pair | residual `0` on that pair | "zero residual means the transport is the computation" |
| ReLU width rescaling | same function, different hidden vector | "the score is intrinsic to `F`" |

## Normalization

| map | what is true | what fails |
|---|---|---|
| LN, `h=(1,0)`, `w=(1,2)` | scalars `1` and `-1` | "a fixed readout keeps its sign" |
| LN Jacobian | kills `1` and `h-\mu 1` | "LN is a local coordinate chart" |
| plain RMSNorm | positive scale, signs agree | treating RMSNorm as if it centered |
| LN with `\gamma=(-2,1)`, `w=(1,0)`, `h=(1,0)` | plain LN keeps the sign, `\gamma` reverses it | dropping `\gamma,\beta` from the block |

## Transformer-shaped

| map | semantics | what is true | what fails |
|---|---|---|---|
| `Y=AVW_O` | `A` clamped | `I=0` for value edits, response exact | using this as a theorem about `do(X)` |
| scalar causal QKV, all weights `1`, baseline `(1,1)` | total residual edit | token-1 interaction `-0.611855656607887` | "softmax implies every family interacts" is already refused by the next row |
| `W_Q=0`, no query bias, no LayerNorm | total residual edit of `Q=XW_Q` | `I=0` | "a nonlinear attention formula has nonzero `I`" on that map. Not a pre-norm theorem |
| nonzero query bias, `W_Q=0` | the same two-token family with `Q=b_Q` | interaction nonzero | extending the previous row to a biased `Q` |
| `P` drops the key coordinate | total residual edit of the bias-free map | same `P`, token-1 outputs `1/2` and `1/(1+e^{16})` | sufficiency of that `P` |
| `relu(u+v)` | two scalar edits | `I(1,1)=0`, `I(1,-2)=-1` | "a shared ReLU is an interaction at every operating point" |
| diagonal ReLU | two scalar edits | `I=0` | "an MLP nonlinearity mixes every pair of coordinates" |
| three meanings of a residual edit | total, branch clamped, descendant clamped | three different numbers for `\tanh` | reading a clamp as a total effect |

## Boundaries that are theorems rather than examples

Identically zero interaction on a `C^2` rectangle is separability. Value edits at a fixed pattern are exactly linear. A statistic determines a response exactly when the response is constant on its fibers. A recomputed structural residual is identically zero and therefore not a non-constant predictor. `O` does not depend on `M`.

## What a nearby system does

Drop the clamp on `A` and the value-linearity theorem stops applying; the bias-free QKV numerical example is a nearby system with nonzero interaction. Restrict to `W_Q=0` with no bias and no LayerNorm and the interaction on that map drops back to zero. Add a query bias and it does not. The restriction is checkable from the formula without looking at `R`. It is ordinary attention algebra, and it is not a theorem about `H+\mathrm{Attn}(\mathrm{LN}(H))`.

Add the assumption "response sign agrees with a decrease in `O`" and the original bridge becomes true by assumption. The assumption is not checkable without the sign it restates, unless a separate argument derives it. No such argument is in this directory.

## Not shown

These examples do not show that every projection fails, that every interaction in a trained model is nonzero, or that every LayerNorm application flips a sign. They show that the universal sentences are false, and that the surviving positive sentences are the ones with the hypotheses in `11`.
