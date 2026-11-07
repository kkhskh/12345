# Reviewer 6 — Counterexamples

Provenance: the counterexample half of [mathematical review](3f3cfe6e-8f2a-41c3-aedf-115b3659d77e), plus the architectural counterexamples from [architecture review](bfdad80b-da28-4c94-9abc-2f7ca708fb69).

The maps that were checked against the draft and that still kill the universal sentences:

| target | map |
|---|---|
| nonlinearity implies nonzero `I` | `u^2+v^2` |
| one zero implies separability | `(u+v)^3` at `(1,-1)`, with `I(1,1)=6` |
| nonzero baseline mixed partial implies every finite `I` is nonzero | `uv-u^2v-uv^2` at `(2,-1)` |
| changing `M` changes `O` | `O=1` at both `M=I` and `M=\mathrm{diag}(10,0.1)` |
| fixed readout sign survives LN | `h=(1,0)`, `w=(1,2)` |
| `ker P_u` preserves `r` | `\delta=(0,1)`, `P_u=(1\ 0)`, `P_v=(0\ 1)` |
| both signs from a blind `Q` | N1 |
| zero interaction implies `Y=AVW_O` | `v^2+w^2` |
| second attention output is exactly `0` | `1/(1+e^{16})` |
| `W_Q=0` survives a query bias | `Q=b_Q` on the two-token family |
| constant pattern implies every LayerNorm edit is additive | two coordinates inside one token |

No further counterexample was found against the rectangle equivalence, against clamped-pattern linearity as a one-direction implication, or against `D=10/101`.

A clamp that copies `f(h_u)` kills the "iff" on structural residual zero. That is a semantics counterexample, not a new map of the block. `11` no longer states the converse.
