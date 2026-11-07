# 12. The strict gap

Under Theorem S3, for every `ρ > 0`, both intervention sites, the zero base, and the special gains,

```text
max_{t ∈ [0, ρ]} (f(t) + g(t)) = 0 < f(ρ) = max f + max g.
```

At `ρ = 1` the numerical gap is `0.7070997102248`. The attention and MLP endpoint readouts are `0.7070997102248` and about `-0.88838`. Neither path is a zero branch. The full read-token response satisfies `R(t) ≤ t ≤ ρ`, while the three-way product is `ρ + f(ρ)`. At `ρ = 1` that product is `1.7070997102248`, and the true maximum of `R` on a log-spaced grid is the endpoint value `0.818715171479619`. The proved joint upper bound used in the comparison is `ρ`, not the tighter endpoint value. It is still strictly below the product.

Under Theorem S2, with no special-gain restriction, the same strict inequality holds, but the joint maximum need not be zero. For `μ = 2`, `s_gain = 2`, `α = 1`, `ρ = 2`, a uniform grid of 4001 points gives a joint branch maximum `1.051649186242511`, equal to the endpoint, against a marginal sum `2.828420053704895`. The gap on that grid is `1.776770867462384`, which equals `-g(2)`. The existence of a strict gap is proved. The identification of the maximum with the endpoint value is numerical on this gain. It is not needed for Theorem S2.

The source of the improvement is the empty corner: the step that maximizes the attention readout minimizes the MLP readout. A separate accounting adds those two numbers. The block cannot.

This does not beat the exact one-dimensional function. Once the scalar reduction is in hand, evaluating it is cheap and returns the exact maximum. The theorem explains why the product is loose and, for the special gain, names the maximum without a line search. It is not a cheaper algorithm for a generic high-dimensional intervention set. That cost is stated in note 15.
