# 05 — The residual intervention

The intervention replaces the input residual and recomputes the block. For a token `t` and a vector `δ ∈ R^d`,

\[
(E_{t,δ})_{s} = \begin{cases} δ & s=t, \\ 0 & s\neq t, \end{cases}
\qquad
R(H,t,δ) = m(H+E_{t,δ}) - m(H).
\]

Descendants are recomputed. Attention probabilities are not clamped. The value-only directional derivative in `06` is a different linear map: it sets `dQ=0` and `dK=0`, so the pattern increment is zero, and then it still carries the residual skip and the MLP. Only the attention increment is restricted to `P dV`. When that map is used, the text says so. It is not the derivative of the recomputed block.

A simultaneous clamp of an intermediate activation is not this intervention. If `H_1` were overwritten after `H` had been edited, the attention equation would not be the one that produced the output. The structural residual `H_1 - H - A(LN(H))` would then measure the clamp. On the intervention defined above that residual is zero, because `H_1` is assigned by the equation.

The first-order comparison is the directional derivative of `m` at `H` in the direction `E_{t,δ}`, denoted `Dm(H)·E_{t,δ}`. It is not `R`. The tests compare it with the difference quotient at step `10^{-6}`.

Reading a token other than `t` does not, by itself, make `R` zero. Causal attention lets token `t` affect later positions through keys and values. The readout is token 2, so an edit at token 0 or 1 can change `m`. An edit at token 2 cannot change tokens 0 or 1 through the mask. The skip of a non-constant `δ` still enters token 2's own MLP.

The constant direction is the exception that is exact rather than causal-mask-limited in its branch effect. If `δ = a 1`, both norms ignore it, every branch is unchanged, and

\[
R(H,t,a 1) = \begin{cases} a\, 1^T w & \text{if } t=2, \\ 0 & \text{if } t\neq 2. \end{cases}
\]

No weight of `A` or `M` is consulted. This is checked for `a=0.7` on seed 1. It is the finite response, not a linearization. The right-hand side does not depend on `H`. Every pair of residuals therefore has the same response to a given constant intervention, and every function of `H`, including the function that discards `H`, is sufficient for this family. Sufficiency here is the symmetry. It is not evidence that the discarded coordinates are causally idle for any other family.
