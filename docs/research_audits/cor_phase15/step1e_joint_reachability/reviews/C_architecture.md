# Reviewer C — Transformer architecture

Same-context pass, not an external referee.

The block has the pre-norm residual, causal mask, two heads, positive `ε`, and a tanh MLP. The last-token edit leaves earlier rows fixed, which the test checks, and that is the right causal-mask fact. The earlier-token edit is included in the theorems rather than asserted by symmetry.

The fixed pattern is a real restriction. `W_Q = W_K = 0` is an intervention on the score mechanism, labeled as such. A moving-pattern draw no longer matches `μ Δλ`. I would reject any sentence that drops that hypothesis.

Width 2 makes the mean-zero complement one-dimensional, so the invariant line is almost forced. The width-3 singular map shows the graph lemma needs an invertible value map. Both points are in the notes. Final LayerNorm is handled by counterexample rather than by a second theorem, which is the honest choice: at step `0.2` the raw response is near `0.019` and the normalized response is near `1.375`.

I do not think this is evidence about a trained multi-layer model. The notes do not say that it is.
