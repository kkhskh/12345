# 13 — Representation, metrics, and reparameterization

## The operations that get conflated

| operation | what moves | what is invariant |
|---|---|---|
| invertible linear change of a representation, with the dual change of every linear form and the congruent change of every quadratic form | coordinates of `h`, of `w`, of `W`, of `M` | the scalar `w·h` and the quadratic form, when all of them are transformed |
| edit of `M` only | the quadratic form inside `D` | `O`, because `O` does not read `M` |
| edit of `W` only | `O` | `D` |
| LayerNorm as computation, readout fixed | the vector that `w` is applied to | not the sign of `w·h` |
| plain RMSNorm, no `\gamma` | scale of `h` | the sign of every linear readout |
| positive rescaling of a ReLU hidden layer, with the inverse rescaling of the outgoing weight | the hidden activation | the composed function, for the positive-homogeneous activation |

`test_function_preserving_rescale_changes_an_internal_score` records the last row for a bias-free ReLU. `W_1` is replaced by `2W_1` and `W_2` by `W_2/2`. Positive homogeneity holds on inactive units as well, because both sides of `relu(c z)=c\,relu(z)` are zero when `z<0` and `c>0`. The composed function therefore matches at every input of this bias-free map. The hidden vector changes at inputs whose preactivation is positive. The tested input `(0.4,-0.2)` has preactivation `0.65`. On the inactive set both hidden vectors are zero, so a score of the hidden vector does not change there. The identity also uses `b_1=0`. Scaling `W_1` and `W_2` while leaving a nonzero `b_1` unscaled changes the function. The MLP formula in `04` has biases. This paragraph does not absorb them.

## Consequences

A theorem can be invariant under a change of coordinates while a numerical table of `O` is not, if the table forgets to transform `W`. The invariant object is the quadratic form, not the array of entries in one basis.

A causal prediction about an activation intervention is a prediction about a specific node in a specific parameterization. It is not required to be invariant under a reparameterization that moves that node. It is required to say which node. If two functionally equivalent parameterizations assign different relational scores to corresponding inputs, the score is not a property of the function. The ReLU rescaling is a case. Any interpretation that calls the score an intrinsic conflict of the function is then unjustified. An interpretation that calls it a property of one parameterization can be correct and still be useless for comparing models that are free to rescale hidden units.

LayerNorm is not a license to treat the residual stream up to `ah+b1` as the same intervention. Adding `b1` before LN does nothing to plain LN, and it does something to a readout that has not yet been normalized. The intervention `do(h\leftarrow h+\delta)` is defined in the coordinates of `h`. If `\delta` is a constant vector, plain LN kills it. The same numerical `\delta` added after LN is a different intervention and need not be killed.

## Metrics

`D` ranks states by distance to `\ker d` in the metric `M`. Two metrics can reverse that ranking. `O` ranks states by `W` on the coboundary. Changing `M` does not perform that reversal. A result that says "the ranking is metric-dependent" is true of `D` and of `O` separately. A result that says "therefore changing `M` changes `O`" is false.

Privileging the Euclidean metric because activations are stored as tuples is a coordinate convention of the implementation. It is a reason to report Euclidean quantities as Euclidean. It is not a reason to treat them as the unique geometric content of the computation. LN is already a non-Euclidean change of the vector inside the block, with a state-dependent Jacobian.

## Identifiability

The objects that are defined once the block is written down, without an analyst choosing a subspace, are `X`, `Q`, `K`, `V`, `A`, `Y`, the MLP activations, the LN outputs, the logits, and the responses under a stated `do`. Subspaces, transports, `W`, and `M` are defined once the analyst chooses them. Intervention effects on the implemented nodes do not depend on that choice. `O` and `D` do.

An arbitrary `\rho` manufactures a zero residual at one pair whenever the source vector is nonzero, by the rank-one formula in `proofs/fiber_and_trace.md`. What blocks that manufacture is a restriction that fixes `\rho` without using the target of the pair: a Jacobian taken from the weights, a transport shared across a family and then tested on held-out pairs, or no transport at all. The restriction makes the residual a definite number. It does not make the number a cause. Causal relevance still requires a proved link to `R` under the fiber theorem, which is a link these restrictions do not provide by themselves.

## Functionally equivalent models

The ReLU pair implements one function and two internal scores. Therefore a relational score is not determined by the input-output function. Two models can agree on every input and disagree on the score. Conversely, matching scores do not imply matching functions: `u^2+v^2` and `(u+v)^2` can be given the same structural residual under recomputation, namely zero, and they differ as functions. The score does not separate them. The evaluation does.
