# 15. Adversarial register

Each row states the hypothesis that was dropped or the claim that was attacked, and a witness that meets the remaining hypotheses. Details and floats are in `counterexamples/register.md`. The witnesses were recomputed in this package.

| id | claim attacked | witness | survives? |
|---|---|---|---|
| CE-U-scope | Theorem U applies to every architecture, including a fixed final LayerNorm | Diameter bound `2 ||w|| R_f = 3.505739880594811` on seed 0 with `γ=1`, `β=0` | U survives only for the raw unbounded-weight class |
| CE-L-scope | Every finite certificate is at least `||∇m|| ||δ||` | `α=10^{-6}`, `B_dir=0.004051`, `||∇m||=0.876435`, injective QKV | L survives for isotropic constants only |
| CE-linear | `|∇m·δ|` bounds `|R|` | same point, `|∇m·δ|=5.19e-8 < |R|=1.33e-7` | the linear term is not a finite bound |
| CE-vacuous | A valid bound is a useful comparison at unit scale | `α=1`, `B_dir=1.629250e6`, `|R|=0.0108` | the bound stays true and stops being informative |
| CE-kernel | Output kernels are an open phenomenon, or require tiny weights | `σ(W_O)` and `σ(W2)` stay near 1, `||ΔA||>0.1`, exact raw identity, positive codimension | exactness is real and not open |
| CE-final-orth | `w·δ=0` and zero branches imply a zero normalized response | final-LN response `-0.32024254867419905` | raw orthogonality does not transfer |
| CE-final-kernel | Theorem Kernel-out survives an outer LayerNorm | raw gap `0`, final-LN response `-0.13186230895395856` | it does not |
| CE-cancel | `J_A + J_M = 0` implies a small finite response | step `2`, `|R|=0.16901883579206622`, linear term `0`, both path derivatives nonzero | finite cancellation fails |
| CE-flip | Equal separate bounds imply equal finite responses | Householder reflection of `W2`, same `B_align`, `R` changes from `-0.0108` to `-0.0564` | the norm bound is blind; the derivative sees the change |
| CE-trace | Exact structural equations determine the sign of `m` | seed-0 weights, margins `+0.4644721976523223` and `-0.4644721976523225`, residuals zero | they do not |
| CE-const-final | The raw constant-shift value equals the final-LN value | constant `1`, raw `1.5010589548912336`, final `0` | the transfer fails in this direction too |

No witness was discarded because its weights are small or scientifically dull. CE-L-scope is valid and is the small-rate regime. CE-kernel is the order-1 regime and is not open. Both are reported. Validity and relevance are separate columns in the scientific verdict, not a filter on the register.
