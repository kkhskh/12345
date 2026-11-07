# 25 — Noncircularity

| Object | Architecture | Weights | Natural `h` | Labels | Outputs | Evaluation `R` | Logical independence from the claimed `R` |
|---|---|---|---|---|---|---|---|
| Residual add | yes | — | — | no | no | no | yes: `I=0` for a later linear map |
| Softmax attention | yes | yes if using actual `W_QK` | no | no | no | no | the mix is not `R` |
| `P` probe | no | fit | often | often | sometimes | leak if fit to `R` | no, if `P` is built from `L` |
| Shared `ρ` | no | optional Jacobian | yes | via `P` | no | leak if per-example or outcome-tuned | residual still not `R` |
| `w = ∇L` | yes through the graph | yes | at a point | no | **yes** | using `R`'s derivative | **no**: this is the outcome sensitivity |
| `O,D` | no | no | via `s` | via `P` | no | leak through `M,W,β` | not `R`, and N1 says it may miss `R` |
| Majority decoder | only if present | — | — | no | no | no | yes as a map; not present as a primitive |

Statistical split: a train-only `P` can avoid evaluation-label leakage and still be logically the wrong object (CE-13). Logical independence: `L = -α D + B` is circular even with a split.

`I=0` and `I=2` are both properties of `L` (or of a named next map of the clamped objects). Residual addition does not imply `I=0`. Calling vanishing `I` architectural and `I=2` circular was an inconsistent split; both are deductions from a known `L`. For a checkpoint, later LN, softmax, and MLP do not identify `I=0` for `R` without querying `L`.

`w=∇L` is outcome sensitivity. N1 does not use `w`.
