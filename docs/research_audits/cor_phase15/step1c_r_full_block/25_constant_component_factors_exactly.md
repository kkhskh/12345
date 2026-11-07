# 25 — The constant component of any finite edit

R6 was stated for a pure constant. The same proof applies at an arbitrary base point, so the constant piece factors out of every finite edit.

Let `E` be any residual increment, on any tokens. Let `C` add `a 1` at one token `t` and add nothing elsewhere. For every weight setting of this block and every `ε≥0`,

\[
H_2(H+E+C) = H_2(H+E) + C.
\]

The normalized input of token `t` is unchanged by `C`, because both LayerNorms kill constants. Every other token is unchanged by `C`. Attention and the MLP therefore take the same inputs they take under `E` alone. The only change in `H_2` is the skip `C`. The readout sees that skip only when `t` is the read token:

\[
R(H, E+C) = R(H, E) + \begin{cases} a\, 1^T w & \text{if } t \text{ is read}, \\ 0 & \text{otherwise.} \end{cases}
\]

The four-term interaction of `C` with `E` is zero. This holds at finite amplitude. Softmax and `tanh` do not create an interaction with a constant, because the constant never enters them.

The same statement is false for two non-constant pieces on one token. On seed 4, two centered vectors at token 1 have interaction about `0.198`.

## Maximality

A single-token direction that is not constant does not have a response determined by `w` and the read index alone.

Set every attention and MLP weight and bias to zero. Then `H_2=H`, so `R=w·δ` when the edited token is read and `R=0` otherwise. For a constant, the live block agrees with that value. For a centered direction it does not have to. On seed 7, a centered edit at token 1 has live response about `−0.032` and dead-branch response `0`. The same centered shape at the read token has live response about `0.094` and dead-branch response `w·v`, about `−0.039`.

So the constant line is the whole family of single-token edits whose finite response is independent of the branch weights. The identity does not determine the seed-16 sign or the remainder along `u`. It removes the constant component of those interventions exactly, and leaves the centered component to be computed.
