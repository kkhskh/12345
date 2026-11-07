# Workstream 3. Compatibility, as joint parents

A pairwise equality `ρ s_u = s_v` is the wrong aggregator for attention. Token `t` has every earlier token as a parent, through a row-stochastic pattern that depends on queries and keys. The structural equation is the whole row, not one edge. Forcing one edge to match and ignoring the others manufactures a residual equal to the omitted parents.

On a recomputed trace the joint equation has residual zero whether the margin is positive or negative. Zero is not a decision. A nonzero number has more than one source, and the sources below can share one vector.

## Taxonomy

Let the observed residual be a vector `r` in the child stalk.

| source | what is true | what the scalar `||r||` says |
|---|---|---|
| Surrogate error | `ρ` is not the structural map. A rank-one `ρ` from one pair leaves `r` equal to the missed component on another pair. | Only the size. |
| Missing parents | The child is `ρ s_u + p` and the score forgets `p`. Then `r = -p` if the score uses that `ρ`. | Only the size. |
| Off-manifold stitching | `H1` is clamped off `H+Attn(LN(H))`. The structural residual is the clamp error. This intervention is not `do(H_t <- H_t+δ)`. | Only the size. |
| Incomplete abstraction | A coordinate deletion. The seed-16 states agree on the kept coordinates and their responses do not. The residual of the kept coordinates can be zero while `R` changes sign. | The kept residual can be zero. |
| Genuine interaction | The four-term contrast of the downstream map is nonzero. An affine map has interaction zero, so a nonzero residual is not required for interaction and a zero interaction is not a linear attention pattern. | Not a function of `||r||`. |
| Researcher inconsistency | Two different graphs, or a `ρ` refit on the evaluation intervention. The number changes because the definition changed. | Indistinguishable from a real residual of the same size. |
| Numerics | A recomputed structural residual on this block is below `10^{-12}`. A report of `10^{-8}` on an identity that should vanish is a tolerance, not a conflict. | A threshold can relabel it. |

## When they cannot be distinguished

From `||r||` alone, or from `r` after an unknown choice of `ρ`, the first two rows are the same vector up to sign: the worked pair has missing-parent residual `-p` and surrogate residual `+p` with the same Euclidean norm. Adding the scalar `O = ||r||_W^2` does not split them.

A functional of `(P, ρ, s)` that does not retain the omitted parent, the clamp, or the intervention response cannot label the source. Step 1C's corrected statement stands: the kernel that must be held fixed is the kernel of the statistic actually used, and naming that kernel does not name the source.

Joint sufficiency is the tuple of parents the structural equation names, or a proved invariant of that equation such as the constant shift. A list of pairwise equalities is not that tuple.

Arrow: a nonzero residual —unsupported→ a unique source among the seven rows.
The split requires the structural equation and the intervention definition —exact equation→ which row is being computed, and then the residual is either identically zero (recomputed trace) or the definition of the surrogate error (any other `ρ`).
