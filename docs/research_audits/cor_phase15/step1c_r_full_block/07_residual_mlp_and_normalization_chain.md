# 07 — Residual, second norm, and MLP

The attention derivative `dA` is a function of `dX`, which is a function of `dH`. The first residual update is

\[
dH_1 = dH + dA.
\]

The skip is the first term. It is present even when `dA=0`. For a constant `dH_t = a 1` and `dH_s=0` otherwise, `dX=0` and `dA=0`, so `dH_1` equals that same sparse constant. This is the differential form of the finite identity in `02`. For a non-constant edit, `dA` is generally nonzero at every legal query that can see token `t`, and `dH_1` is nonzero at those rows as well as at the skip row.

The second normalization is again per token:

\[
dU_s = J_{\operatorname{LN}_ε}(H_{1,s})\, dH_{1,s}.
\]

A constant sitting in `dH_1` is killed here, just as it was killed in the first norm. A non-constant vector is not killed when `ε>0` and every gain is nonzero. It is killed when it lies in a larger kernel: at `ε=0` the centered direction is such a vector, and at `ε>0` two zero gains produce a kernel vector that is not constant. The product of the two Jacobians is not the identity, and neither factor is invertible.

The MLP derivative uses `d\tanh(pre) = (1-\tanh(pre)^2) \odot d\,pre`. With `hidden = tanh(U W_1^T + b_1)`,

\begin{align*}
d\,\mathrm{pre} &= dU\, W_1^T, \\
d\,\mathrm{hidden} &= (1-\mathrm{hidden}^{\odot 2}) \odot d\,\mathrm{pre}, \\
dM &= d\,\mathrm{hidden}\, W_2^T.
\end{align*}

Then `dH_2 = dH_1 + dM`, and the readout derivative is `w^T dH_2[2]`.

## The full first-order response

Collecting the maps, `Dm(H)·E_{t,δ}` is the composition

\[
E_{t,δ}
\xrightarrow{J_{LN1}}
dX
\xrightarrow{dA}
dH_1 = E_{t,δ}+dA
\xrightarrow{J_{LN2}}
dU
\xrightarrow{dM}
dH_2
\xrightarrow{w^T \text{ at token } 2}
Dm.
\]

Every arrow is computed in `proofs/block_math.py` by `margin_jvp`. On seed 0, token 1, and `δ=(0.2,-0.1,0.05,0.3)`, this value and the central difference quotient at step `10^{-6}` agree to relative error about `3×10^{-6}`.

Nothing in the composition is optional. Dropping `dA`'s pattern term is the value-only sign error. Dropping the skip is the mistake of treating a normalization kernel vector as invisible to `H_2`. Dropping `J_{LN2}` would differentiate a block that does not normalize `H_1`.

## Where a nonsmooth activation would change the argument

`tanh` is `C^∞`. The directional derivative exists everywhere. For a ReLU MLP the same formal expression holds on the open set where no preactivation is zero, and it fails on the kink arrangement, which has measure zero but is reachable by an intervention. This block does not cross into that case. A claim about ReLU regions remains the Step 1C statement: inside one linear piece the four-term interaction of increments that stay in the piece is zero. It is not re-proved on this `tanh` block, because the activation is not piecewise linear.
