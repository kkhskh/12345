# 26 — Hypothesis registry

Machine-readable copy: `hypothesis_registry.json`.

| ID | Target | Family | Quantity | Assumptions | Identifiable without eval `R`? | Signature | Status | Keep? |
|---|---|---|---|---|---|---|---|---|
| H-SUF | sufficiency of `P` or of a tuple | A | `z=P(h)` or `(P_i h)` | `L=q(z)` on a family | only if `P` is frozen independently | kernel edit does not move `L` | redundant with causal abstraction; N2 proved | reclassification, not COR |
| H-COORD | path roles | B | `w·p_i`, `I` | linear or named `σ` | `w` is `∇L` | ablation table | redundant with mediation | keep as vocabulary |
| H-CORR | correction | C | majority / checksum | redundant discrete code | architecture would have to contain it | Hamming-1 absorbed, Hamming-2 not | empirically untested | open, not claimed |
| H-MULTI | true aggregator | D | `f_v`, not pairwise `ρ` | the structural equation | yes if `f_v` is the block | pairwise residual can be fake | proved as a warning | keep as constraint |
| H-STAB | stability | E | increment bound `S` | `|L|>S` | `S` needs `L` | sign preserved in a ball | generic | reject as COR |
| H-COMPOSE | interaction | F | truncated `I` | SCM truncation | vanishing vs not: architecture; value: curvature of `L` | `I=0` iff next map linear in addends, in the tests | proved in toys; redundant as COR | keep as reclassification |
| H-EVID | competing parents | G | mediation of two sites | both enter `L` | needs `∇L` or ablations | opposite path effects | unidentified as energy | reject as `O` |
| H-BOTT | minimal causal state | H | `z=T(h)` | `L` factors through `T` | field-open | interchange | redundant / unidentified | open identification, not COR |
| H-ABS | faithfulness | I | interchange accuracy | restricted alignment class | if class frozen | agreement under `do` | established | not COR |
| H-INV | coordinate status | J | subspaces, path effects | declared `P` or `do` | yes for `do` | Euclidean `O` moves | proved | constraint |
| H-REP | repair | K | `δ` in `ker P` with `ΔL≠0` | frozen `P` | `P` yes, `ΔL` is the outcome | probe-preserving effect | Direction 1 | independent of COR |
| H-BLIND | necessity | — | any `Q` | unrestricted `L` | yes as a negative theorem | `Q` constant, `L` not | proved N1 | **central** |
| H-OD | original `O,D` bridge | original | `||ds||^2` | Definitions 2–4 | score yes, bridge no | lower `D` raises `L` | disproved universally | rejected |

Revisions: H-OD entered rejected from Step 1A and stays rejected. H-COMPOSE was narrowed from "undefined nested do" to "defined under truncation, still not `D`." H-CORR was added from the correction-versus-gating split; it is not a success of COR.
