# Attention is linear in `V` when the pattern is held fixed

Let `A` be a row-stochastic matrix that does not depend on `V`, and let `W_O` be constant. The head output

\[
Y(V)=A V W_O
\]

is linear in `V`. For any value displacements `\Delta_1,\Delta_2`,

\begin{align*}
&Y(V+\Delta_1+\Delta_2)-Y(V+\Delta_1)-Y(V+\Delta_2)+Y(V)\\
&\quad=A(\Delta_1+\Delta_2-\Delta_1-\Delta_2)W_O=0.
\end{align*}

The intervention is `do(V \leftarrow V+\Delta)` with `A` clamped. It is not `do(X \leftarrow X+\delta)` on the residual that produces `Q`, `K`, and `V`.

Checked for two numerical pairs in `test_value_only_attention_interaction_is_exactly_zero`.

## A residual edit is a different intervention

With `Q=XW_Q`, `K=XW_K`, `V=XW_V`, a displacement of `X` moves the scores whenever `W_Q` and `W_K` let it, and moves `V` whenever `W_V` lets it. The softmax of a changed score multiplies a changed value. The composition need not be additive.

Existence, not universality: scalar causal attention, all four maps equal to `(1)`, baseline `X=(1,1)`, displacements of the two tokens by `1`. The token-1 outputs are

| edit | token-1 output |
|---|---|
| none | `1` |
| token 0 only | `1.7310585786300048` |
| token 1 only | `1.8807970779778822` |
| both | `2` |

The four-term interaction equals `-0.611855656607887`. Checked by `test_residual_intervention_moves_qkv_together_and_can_interact`.

## A nonlinear formula can still give zero interaction

If `W_Q=0`, there is no query bias, and `Q=XW_Q` with no LayerNorm, every unmasked score is zero, the causal pattern does not depend on `X`, and `Y` is linear in `X` through `V`. The four-term interaction of residual edits of this map is zero. A nonzero query bias is a different map: scores still depend on `K`. A pre-norm value path is also a different map, because `LN` is not linear inside a token. Softmax is a nonlinear function on its own domain. The bias-free, normalization-free family does not exercise that nonlinearity.

Checked by `test_zero_query_map_makes_residual_edits_additive`, `test_query_bias_keeps_scores_dependent_on_keys_when_W_Q_is_zero`, and `test_constant_pattern_through_layernorm_is_not_additive_inside_one_token`.

## Softmax redistribution is the partition constraint

For two legal positions, `a_1=1-a_0`. Sending one logit to `-\infty` moves that position's weight onto the other because the weights sum to one. That identity is the definition of the row. It is not a second computational mechanism sitting beside attention.
