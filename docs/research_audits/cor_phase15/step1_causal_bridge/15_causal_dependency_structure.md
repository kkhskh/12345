# 15 — Where causal content would have to enter

No positive bridge remains, so this is not a dependency graph of a surviving prediction. It is the graph of the claims that were examined, with the node at which each one either stops or imports the readout.

```text
Definitions 2–4
    P, ρ, graph, M, W
        │
        ├─ exact intertwiner on a fixed map ──► ds = 0 on recomputed traces
        │                                         (no L in this node)
        ├─ no intertwiner ──► residual = approximation or omission
        │                         (no L in this node)
        ├─ squared residual D, O
        │       ├─ Lipschitz of L ──► |gap| ≤ K √D
        │       │                      causal content: the Lipschitz constant of L
        │       ├─ L = q(Π·) ──► repair has ΔL = 0
        │       │                 causal content: the factorization, and it idles D
        │       └─ L = -α D + B ──► desired sign
        │                            causal content: the displayed formula
        └─ output derivative ∇L or J
                ├─ path effect w·Jδ
                ├─ sign along -∇D equals -∇L·∇D
                └─ sign stability if an increment of L is bounded
                   (τ O is not this bound)
Interaction I(a,b)
    requires the four intervened outputs, or an assumption that is not D
```

Every arrow that reaches a signed statement about `R_c` passes through `L` or through an assumption that already writes the sign. The nodes built only from `(P, ρ, d, M, W)` stop at zero residual, approximation error, or an unsigned distance.

## Assumption types

| Node | Type |
|---|---|
| `M ≻ 0`, finite dimension, `Π` the orthogonal projection | Mathematical regularity |
| A block is a particular function of its inputs | Architectural fact, and a constant linear `ρ` generally does not intertwine it |
| `ρ` fit by shared least squares | Learned approximation |
| `L` Lipschitz, or `∇L` known | The readout, already used by attribution |
| `L = q(Π s)` or `L = -α D + B` | Causal assumption, and the second is the conclusion |
| `β`, `τ`, edge multiplicity, Euclidean `W` | Researcher convention |

No node marked "learned approximation" or "researcher convention" determines the sign of `R_c`. The nodes that do are marked as the readout or as the conclusion.
