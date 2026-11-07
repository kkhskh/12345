# 14 — Bounds that follow from the block

## Exact on a symmetry

For every weight setting of this architecture, every `ε≥0`, every gain, and every bias,

\[
H_2(H+E_{t,a1}) = H_2(H) + E_{t,a1}.
\]

Hence `|R|` equals `|a 1^T w|` when `t` is the read token and equals `0` otherwise. The bound is an equality. It is tighter than a Lipschitz bound that uses `||Dm||\,||δ||`, which is zero when `t` is not the read token and `δ` is constant, and which only approximates the read-token case. The equality does not need `Dm`. It is the invariance.

The same equality does not hold for a non-constant `δ`. Doubling token 1 on seed 1 moves `H_2` by something other than the skip, with a discrepancy larger than `10^{-2}`.

What does survive for a non-constant edit is the split in `25`. Write `δ=a1+v` with `v` centered. Then `R(H,t,δ)=R(H,t,v)+κ`, where `κ` is `a 1^T w` on the read token and `0` otherwise, for every finite `v` and every weight setting. The constant component is exact. The centered component is not given by `w` alone. On seed 7 a centered edit has a different response when the branch weights are zero than when they are not.

## First-order bound

Where the derivative exists, which is everywhere for this `tanh` block,

\[
R(H,t,δ) = Dm(H)·E_{t,δ} + r_2(δ),
\]

and `r_2(δ)/||δ|| → 0` as `δ→0` along a line, by differentiability. Cauchy–Schwarz gives `|Dm·E| ≤ ||g_t||\,||δ||`, where `g_t` is the gradient of `m` with respect to token `t`. This uses `g_t`. A smaller number computed without `g_t` is not available for general `δ`.

The remainder is not controlled by that inequality. Along the seed-0 direction `u`, the linear term is zero and `|R|` at step `0.2` is about `0.629`. Any theorem that promotes the Cauchy bound into a finite-response bound needs a remainder hypothesis this block does not satisfy uniformly in `δ`.

## Path cancellation is the derivative

If the skip contribution, the attention contribution, and the MLP contribution to `w^T dH_2` sum to zero, then `Dm·E=0`. The sentence is the definition of the sum. It does not constrain `r_2`. Treating it as a discovered cancellation mechanism double-counts the chain rule.

## What was not obtained

No inequality of the form `|R| ≤ f(ρ, P, H)`, with `f` free of the branch weights and valid for all `δ`, was proved. The feature-3 pair has identical projected coordinates and responses of opposite sign, so no function of that projection bounds the responses by a common interval around zero tighter than the trivial interval that contains both numbers. A bound that contains both `-0.12` and `+0.11` does not determine the sign.
