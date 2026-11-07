# 04 — The block that is actually differentiated

The object is one sequential pre-norm block. It is not a stack, not a parallel residual, and not a trained network. Dropout is absent. Token count `T=3`, width `d=4`, two heads, head dimension `2`, MLP width `5`. The normalization offset is `ε=10^{-5}` unless a test says otherwise. The activation is `tanh`, which is smooth, so the chain rule below applies at every input. A ReLU block would need a separate almost-everywhere argument. This one does not use ReLU.

For a residual `H ∈ R^{T×d}`,

\begin{align*}
X &= \operatorname{LN}_ε(H), \\
H_1 &= H + A(X), \\
U &= \operatorname{LN}_ε(H_1), \\
H_2 &= H_1 + M(U).
\end{align*}

Each LayerNorm is per token, with its own gain and bias, using the formula in `02`. The same `ε` is used in both norms.

`A` is causal multi-head attention. Head `h` has matrices `W_Q^h, W_K^h, W_V^h ∈ R^{d×2}` and

\begin{align*}
Q^h &= X W_Q^h, &
K^h &= X W_K^h, &
V^h &= X W_V^h, \\
S^h &= Q^h (K^h)^T / \sqrt{2} + \mathcal{M}, &
P^h &= \operatorname{softmax}_{\mathrm{row}}(S^h),
\end{align*}

where `\mathcal{M}_{ij} = -∞` for `j>i` and `0` otherwise. The head output is `P^h V^h ∈ R^{T×2}`. The two outputs are concatenated and mapped by `W_O ∈ R^{4×4}` plus a bias `b_O`. There is no query, key, or value bias in this specification. Adding those biases after the products `X W` would not change the shift theorem, because a constant shift never changes `X`. It would change the forward scores. The derivative checks are for the maps written here.

The MLP is

\[
M(U) = \tanh(U W_1^T + b_1) W_2^T + b_2,
\]

with `W_1 ∈ R^{5×4}` and `W_2 ∈ R^{4×5}`, applied per token.

The scalar response reads the last token:

\[
m(H) = w^T H_2[2].
\]

`w` is a fixed vector of width 4. The arrays and the initial residual are drawn from `numpy.random.default_rng(seed)` in the order written in `make_params`, scaled by `0.4`. The recorded floats are outputs of that stream on NumPy 1.26.4. The weights are not fit to a response.

## What this is not

Post-norm would normalize `H+A(H)` rather than normalize before the branch. A parallel block would add attention and MLP of the same residual. RMSNorm would not subtract the mean. GPT-2 uses pre-norm LayerNorm, and its attention projection has biases this block does not. None of those variants is given the rank and shift proofs again. The shift proof uses per-token mean subtraction and a residual add. It does not transfer to a normalization that keeps the mean.

The numerical weights are small, of order `0.4`. They are a specified function, not a sample from a trained distribution. Statements that hold for every weight configuration are marked as such. Statements that use seed 0 or seed 16 are existence statements about this family of blocks.
