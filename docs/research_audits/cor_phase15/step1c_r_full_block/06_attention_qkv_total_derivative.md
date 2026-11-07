# 06 — Attention derivative under a residual edit

Let `X = LN_ε(H)`. A residual direction `dH` produces, at each token,

\[
dX_s = J_{\operatorname{LN}_ε}(H_s)\, dH_s.
\]

Tokens do not share normalization statistics. An edit at token `t` produces `dX_s=0` for `s≠t` at this stage. It does not stay that way after attention, because later tokens can attend to `t`.

## Per head

Drop the head index. With `dX` known,

\begin{align*}
dQ &= dX\, W_Q, &
dK &= dX\, W_K, &
dV &= dX\, W_V, \\
dS_{ij} &= (dQ_i · K_j + Q_i · dK_j)/\sqrt{d_k}
\end{align*}

on pairs that the mask allows, and `dS_{ij}=0` on pairs the mask forbids. The row-wise softmax differential at a probability vector `p` is

\[
dp = p \odot ds - p\, (p^T ds).
\]

Illegal positions have `p_j=0`, so they contribute nothing even if a floating-point score were finite. Then

\[
d(PV) = dp\, V + P\, dV,
\]

and the concatenated head derivatives are multiplied by `W_O`. The output bias has derivative zero.

An edit at token `t` changes `Q` only in row `t`, `K` only in row `t`, and `V` only in row `t`. Consequently:

- scores `S_{t j}` change for legal keys `j≤t`, through the query;
- scores `S_{i t}` change for queries `i≥t`, through the key;
- the value row `t` changes the outputs of those same queries `i≥t` even if the pattern were held fixed;
- value rows `i≠t` stay fixed because `dX` is zero there, whether or not a mask is present;
- attention outputs at `i<t` stay fixed because those rows do not read key `t` or value `t`.

The mask is what keeps token `t` out of earlier outputs. The untouched value rows are the per-token normalization, not the mask.

## Value-only is the second term

The value-only directional derivative sets `dQ=0` and `dK=0`, hence `dS=0` and `dp=0`, and keeps `d(PV)=P dV`. That is the OV term in the usual decomposition, transported through `W_O` and then through the rest of the block. It is not the residual intervention.

On seed 16, at token 1, with

\[
δ = (0.01206307677054339,\ -0.8987843131414323,\ 0.00171581962876215,\ -0.22913906964580683),
\]

the full directional derivative of `m` is `+0.02019538349844552` and the value-only derivative is `-0.008580691798845158`. They have opposite signs. A value-path account of this intervention predicts the wrong direction. The missing piece is `dp`, hence `(dp) V`, then `W_O`. That increment enters `dH_1` and is carried both by the residual add into `dH_2` and by the second normalization and the MLP. It is not only a branch term. Because the chain rule is linear, the difference of the two directional derivatives is exactly that pattern contribution at this operating point.

## Heads add in the attention derivative

At a fixed `X`, `concat @ W_O` is the sum of the two head-row pieces. The bias `b_O` is added once, not once per head. Its derivative is zero, so the directional derivatives of the two head pieces add to the directional derivative of attention. Zeroing one head's rows of `W_O` and adding those two directional derivatives recovers the joint attention derivative within `10^{-10}` on seed 4. Adding two forward passes that each keep `b_O` would count the bias twice. That is a different sum. Neither sum is a claim that the finite block response is the sum of two single-head block responses. The MLP is applied to the sum. Evaluating the MLP at each head separately and adding is a different function. The first-order block derivative does add across heads, because it is a derivative at one shared point. The finite four-term interaction of two token edits need not be zero. On seed 4 it is larger than `10^{-4}` in absolute value.

Heads are not causally independent. They share `X`. A change in token `t` enters every head. The sum is where their first-order effects meet. There is no further product between head outputs before the residual add.
