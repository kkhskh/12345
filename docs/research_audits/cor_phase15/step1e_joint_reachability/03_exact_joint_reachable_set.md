# 3. The jointly reachable set

For an intervention family `U` and a fixed base state, write

```text
a(δ) = A(LN1(H + E_δ)) - A(LN1(H)),
b(δ) = M(LN2(H1 + E_δ + a(δ))) - M(LN2(H1)).
```

The jointly reachable set is

```text
S_{H,U} = { (a(δ), b(δ)) : δ ∈ U }.
```

The marginals are the projections `S_A` and `S_M`. Always `S_{H,U} ⊆ S_A × S_M`. The raw branch readout is `w · (a_r + b_r)`. When the edit is at the read token the full raw response adds `w · δ`. That skip depends on the same `δ` and is not maximized on its own.

These identities are the block equations. They are not a new mechanism. They are the dependence the rest of the package uses.

## Graph lemma

Fix the base state. Assume `W_Q = W_K = 0`, the value map is `μ I` with `μ ≠ 0`, and the MLP is any map of the form `M ∘ LN2`. A single-token edit changes only one LayerNorm row. The attention increment at a downstream token is `μ` times that row's LayerNorm change, because every causal weight on the edited token equals `1/T` after the `T` in the value map cancels it.

LayerNorm is constant on translates by a multiple of the all-ones vector, and its value on a fiber determines the vector up to that translate. The same translate is invisible to `LN2`. Therefore `b` is a function of `a`: the joint set is a graph over the attention increment.

The constant-fiber test uses a width-2 MLP that is not the rank-one structural MLP. Adding `1.3` times the all-ones vector to an edit leaves both increments unchanged through `1e-10`. The graph property is not an artifact of the opposing MLP.

The graph property does not by itself make the support maximum smaller than the sum of the marginal maxima. A graph can still pass through the corner of its bounding box. The negative-gain example does that.
