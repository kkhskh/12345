# Counterexamples

Each row is recomputed in `tests/test_reconstruction.py` unless it is marked as cited from the earlier block and only rechecked.

| id | statement that fails | witness |
|---|---|---|
| CE-F | Finite LayerNorm level set is only the constants, with no gain hypothesis | `γ=(1,1,0,0)`, `ε=1`, `h=(1,-1,0,0)`, `δ` to `(√2,-√2,√2,-√2)` |
| CE-U | A weight-free finite bound exists for this non-constant `δ` | `R = α · 0.0466...` at `α=1,10,100` |
| CE-L | The explicit Lipschitz certificate beats Cauchy–Schwarz | Seed 0, `Λ=2894603.0377523545`, `\|\|g\|\|=3.548311334245126` |
| CE-J | Jacobian kernel membership gives the finite skip identity | Step `0.8`, skip `-0.6853485283715309`, `R=-0.6779358472770767`; gap increases at `1.5, 2.5, 4` |
| CE-K-scale | The manufactured non-constant zero is a ray | Scales `0.25, 0.5, 1.5, 2` have nonzero branch norm; scale `1` has branch norm `<10^{-15}` |
| CE-OD | `O=D` in general | Identity edge, `O=1`, `D=1/2` |
| CE-M | Changing only `M` changes `O` | `O` fixed, `D` moves when `M=diag(4,4,1,1)` |
| CE-sign | Same obstruction, one decision | Readout `w` and `-w` |
| CE-repair | Lower `D` raises the margin | Zero cochain versus `(1,0)` |
| CE-both | `c1=1`, `c2=-0.5` makes both deletions necessary | Deleting `c2` leaves `1` |
| CE-I | Nonlinearity implies a nonzero four-term interaction | `u^2+v^2` has interaction `0`; `(u+v)^2` has `2uv` |
| CE-value | Value-only sign equals the total sign | Seed 16, `+0.02019538349844552` versus `-0.008580691798845158` |
| CE-rem | `Dm=0` implies `R=0` | Seed 0, step `0.2`, finite response about `-0.629` |
| CE-fiber | Dropping feature 3 preserves `sign(R)` | Seed 16, `H_{0,3}` increased by `2`, opposite signs |
| CE-final | Raw constant-shift response equals a final-LN readout response | Raw `a 1^T w`, final LN difference `0` |
| CE-source | One residual norm has one source | Missing parent `-p` and surrogate `+p`, same norm |

The constant-shift identity is not a counterexample. It is an equality on its family. Using it as a bound on centered edits is the move Theorem U closes.
