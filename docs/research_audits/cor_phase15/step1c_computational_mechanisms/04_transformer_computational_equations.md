# 04 — Transformer computational equations

The equations below are the standard pre-norm decoder block. Othello-GPT is a pre-norm GPT-2-style network. This file does not load that network. It writes the maps a residual intervention actually passes through, and it records where other published blocks differ.

## 4.1 Residual stream

For one head, or a sum of heads already projected by `W_O`,

\[
H' = H + \mathrm{Attn}(\mathrm{LN}(H)),\qquad
H'' = H' + \mathrm{MLP}(\mathrm{LN}(H')).
\]

The skip adds the incoming residual to the branch output. Earlier coordinates remain present unless a later map erases them. Addition can cancel in a readout: `w·(a+m)=0` is compatible with both `a` and `m` being large. That cancellation is the readout. It is not a separate conflict variable.

The branch output is not an independently intervenable cause in the same sense as the residual. To change `Attn(LN(H))` while leaving `H` fixed, the intervention has to clamp the branch. To change `H`, the intervention changes the skip and the input of LN together, so `Q`, `K`, and `V` can all move. The sum `H+\mathrm{Attn}(LN(H))` does not mean the two summands are two parents with separate structural equations unless the analyst adds those clamps.

Post-norm blocks apply LN after the addition. Parallel attention/MLP blocks add both branches to the same residual. Those are different equations. The value-linearity identity below is the map `Y=AVW_O` with `A` clamped. It applies inside a pre-norm block only when the intervention clamps the pattern after LN has already been evaluated. It was not re-proved for a parallel block. The numerical residual-edit interaction later in this section is the bias-free map applied to `X` itself, not `H+Attn(LN(H))`.

## 4.2 Attention

\[
Q=XW_Q,\quad K=XW_K,\quad V=XW_V,
\]
\[
A=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+B\right),\qquad
Y=AVW_O.
\]

`B` is the mask, including a causal upper triangle of `-\infty` in a decoder. Softmax is row-wise.

**Value edit at a clamped pattern.** `Y` is linear in `V`. The four-term interaction of any two value edits is zero. Proof in `proofs/attention_value_linearity.md`.

**Residual edit of the bias-free map.** Replacing `X` by `X+\delta` in `Q=XW_Q`, `K=XW_K`, `V=XW_V` replaces `Q`, `K`, and `V` by the linear images of `X+\delta`. There is no LayerNorm in this sentence and no bias. Scores move when the edited coordinates are read by `W_Q` or `W_K`. At scalar causal weights equal to `1`, baseline `(1,1)`, and unit edits of each token, the token-1 interaction computed by the harness is `-0.611855656607887`. That number is an existence result for this map. A pre-norm residual edit is a different map: the skip changes by `\delta`, and `Q`, `K`, and `V` change through `LN(H+\delta)`, which is not the linear image of `H+\delta`. An edit in a direction that `W_Q` and `W_K` do not read can still move the scores after centering and scaling.

**When that bias-free formula does not interact.** If `W_Q=0` and there is no query bias and no LayerNorm, every unmasked score is zero, the mask alone sets `A`, and residual edits are additive. A nonzero query bias removes the conclusion: `Q=b_Q` still multiplies `K`, so the scores depend on `X`. Checked by `test_query_bias_keeps_scores_dependent_on_keys_when_W_Q_is_zero`. Softmax being a nonlinear function is not sufficient for nonzero interaction of a named family.

If the pattern is constant and the value path is per-token LayerNorm, edits of two different tokens are additive, because each token's LN is a function of that token only. Edits of two coordinates inside one token need not be. Checked by `test_constant_pattern_layernorm_is_additive_across_tokens` and `test_constant_pattern_through_layernorm_is_not_additive_inside_one_token`.

**Cross-token.** Under a causal mask, token `t` does not depend on tokens after `t`. An edit at `t+1` does not change the output at `t`. An edit at `s<t` can change token `t` through `K_s` and `V_s`, and through `Q_t` if the edited residual is at `t` itself. A token that other tokens attend to is a parent of those tokens' outputs. It is not a parent of earlier tokens in a causal decoder.

**Peaked patterns.** If one weight is close to `1`, the output is close to that value vector, transformed by `W_O`. "Close" is a numerical approximation. It is not an identity, and it is not a license to drop the other parents from the structural equation.

**What not to call a conflict.** For two legal positions, `a_1=1-a_0`. Sending one logit to `-\infty` moves that position's weight onto the other. That is the two-point identity. For a longer row it is not true that every other weight increases merely because the weights sum to one: a masked position with weight zero stays zero, and a redistribution that is not softmax can lower a second weight while keeping the sum equal to one. For softmax itself, `\partial a_i/\partial z_k=-a_i a_k` for `i\neq k`, so lowering one finite logit does not lower another positive weight. That derivative is the reason, not the sum constraint alone. Path competition in the sense of a dedicated suppression circuit is a further hypothesis. Softmax does not establish it.

## 4.3 MLP

\[
M(h)=W_2\,\sigma(W_1 h+b_1)+b_2.
\]

With `\sigma=\mathrm{relu}`, any region on which the active set is constant is affine. Increments that keep all four evaluation points in one affine piece have interaction zero.

If `W_1` and `W_2` are diagonal, each output coordinate is a function of one input coordinate. The map can be nonlinear and still have zero cross-interaction. A nonzero interaction requires the two intervened directions to mix, and even then only says that the interaction function is not identically zero when the mixed partial is not identically zero. `relu(u+v)` mixes and still has interaction `0` at `(1,1)`.

An apparent disagreement between two features can be this mixing. It does not require a consistency variable upstream of the MLP.

## 4.4 LayerNorm

\[
\mathrm{LN}(h)=\frac{h-\mu 1}{\sigma},\qquad
\mathrm{LN}_{\gamma,\beta}(h)=\gamma\odot\mathrm{LN}(h)+\beta.
\]

`\sigma` uses the mean of squares of the centered vector. The map is invariant to `h\mapsto ah+b1` for `a>0`. The Jacobian has rank at most `d-2` wherever it exists. Euclidean distance in `h` is not Euclidean distance in `LN(h)`.

A fixed `w` need not satisfy `\mathrm{sign}(w·h)=\mathrm{sign}(w·LN(h))`. The counterexample `h=(1,0)`, `w=(1,2)` is exact at `eps=0`. The learned `\gamma` is part of the map when the block has it, and can reverse a further sign.

Plain RMSNorm, `h/\mathrm{rms}(h)`, preserves linear signs because it is a positive scale. That is not LayerNorm. GPT-2-style blocks use LayerNorm with `\gamma` and `\beta`, not plain RMSNorm.

The operating point matters: the Jacobian depends on `h`. A perturbation that is large in a coordinate with small residual variance can dominate `LN(h)` after centering and scaling. A perturbation along `1` is killed by plain LN.

## 4.5 Readout

A standard tied or untied decoder produces logits by normalizing the final residual and multiplying by an unembedding matrix:

\[
\mathrm{logits}=W_U\,\mathrm{LN}_{\gamma,\beta}(H_L).
\]

A candidate margin is a linear form in those logits, or a difference of two coordinates. It is a function of the full final residual through LN and `W_U`. It is a function of a projection `P(H_L)` only under the fiber condition in `proofs/fiber_and_trace.md`.

Locally, the derivative of the margin with respect to `H_L` is the unembedding form composed with the LayerNorm Jacobian at that residual. The Jacobian is singular. Directions in its kernel do not move the first-order margin. A probe that keeps a different subspace can move, or fail to move, with the margin. Nothing here identifies the probe subspace with the unembedding.

## 4.6 What the composition already contains

The total derivative of `Y` with respect to `X` sums the path through `A` and the path through `V`. Holding `A` fixed drops the first path. That partial map is the value-only intervention. An ordinary derivative of the full block already includes both. Reducing the pair of paths to a scalar residual of `P(X)` discards the path decomposition unless a theorem says the scalar determines it. No such theorem holds for the projections checked here.

## 4.7 Variants not derived

RMSNorm blocks, parallel attention and MLP, post-norm residuals, rotary or relative position biases beyond an additive `B`, multi-query and grouped-query layouts, mixture-of-experts routers, query and key and value biases, and a stack of heads with unclamped patterns are not given separate proofs. The bias-free numerical interaction is not those cases. Where a variant changes a hypothesis used above — in particular, where it removes mean subtraction, adds a query bias, or stops the branch from depending on the skipped residual — the corresponding sentence does not automatically transfer. Those cases are marked `NOT INVESTIGATED` in the coverage map, except for the two explicit boundaries that were checked: a nonzero query bias with `W_Q=0`, and per-token LayerNorm in front of a constant pattern.
