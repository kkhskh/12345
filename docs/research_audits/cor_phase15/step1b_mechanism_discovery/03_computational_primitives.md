# 03 — Computational primitives

Standard pre-norm block, one layer, residual stream `h ∈ ℝ^{T×d}`:

```text
a = Attn(LN(h))
h' = h + a
m = MLP(LN(h'))
h_out = h' + m
```

Attention at token `t` is a softmax-weighted mix of value vectors, with weights a function of all tokens' queries and keys. MLP is an elementwise nonlinearity composed with linear maps. LN subtracts a mean and rescales by a standard deviation, so Euclidean inner products in `h` are not the inner products in `LN(h)`.

Facts that are architectural, not empirical claims about a particular checkpoint:

1. Residual addition is linear in its addends. If a later map is linear in `h_out`, contributions of skip, attention, and MLP add in the readout. If a later map is nonlinear in `h_out`, they share a bottleneck and mixed derivatives need not vanish.
2. Attention mixing is not a constant linear `ρ` on an open set of sequences (`test_attention_weights_are_input_dependent`).
3. The block as a whole is not a constant linear map of one token (`test_residual_block_is_not_a_constant_linear_map`).
4. A one-site activation clamp changes the inputs to every later block, including other tokens' attention weights.

These are accounting facts about the written stencil, not theorems about a trained checkpoint, RMSNorm, post-norm, parallel residuals, or causal masks. Tests in this directory are small maps (a softmax mix with logits supplied, a tanh-plus-ReLU step on one vector, 3-d mean-std normalize). They check the toys. They do not establish block-level theorems.

A linear readout of a frozen sum `h+a+m` is not the next Transformer map: the next sublayer reads `LN(h_out)` or RMSNorm, and `m` depends on `a` in sequential pre-norm. Direct-effect `w·a` freezes `m`. Causal masks: a clamp at token `t` does not update tokens `< t` in a decoder.

These constraints still kill a constant linear `ρ` as "the block." They do not produce a signed decision rule.

Recurrent iteration until a consistency fixed point is not a primitive of this architecture. It is `OUTSIDE_SCOPE` as an implementation claim unless a later paper identifies such a loop in a specific model.
