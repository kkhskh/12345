# 16 — Family J: invariants

| Object | Survives invertible stalk maps `A`? | Survives LayerNorm? | Survives changing `M` with `s` fixed? |
|---|---|---|---|
| Euclidean `O` | No, unless `W` is fully congruent | No | No (CE-07) |
| `D` with transformed `M` | Yes | Not automatically | That operation is not a reparameterization |
| `ker P`, `im P` as subspaces | Yes as subspaces, not as Euclidean bases | The subspace in residual coordinates is moved by LN | Yes |
| Path effect `w·δ` | Yes if `w` and `δ` are the same physical intervention | The same clamp after LN is a different Euclidean vector | Yes |
| Sign of `L` | Yes | Yes | Yes |

Usefulness does not require gauge invariance. It does require that the interpretation match the status. A Euclidean residual is a coordinate-bound descriptive statistic. A path effect is an invariant of the intervention and the readout.

Rank, kernel, and image of a frozen linear `P` are intrinsic to that `P`, not to the network, until `P` is identified with a causal variable.

**Status.** Step 1A's congruence law, corrected for cross terms, still stands. LN is an additional, architectural source of metric change.
