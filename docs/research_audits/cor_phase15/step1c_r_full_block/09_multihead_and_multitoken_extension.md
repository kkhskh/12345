# 09 — Two heads and three tokens

The block already has two heads and three tokens. This note records what that changes relative to one head and one token, and what it does not.

## Mask

Token `t` does not affect attention outputs at positions `i<t`. Combined with the skip, `H_1` at `i<t` is unchanged by `E_{t,δ}` unless `i=t`. For `t=2`, tokens 0 and 1 are unchanged at `H_1` and therefore at `H_2`. The readout, which sits at token 2, still moves through token 2's query, key, value, skip, and MLP.

For `t=0`, every later token may change through keys and values. The numerical opposite-sign pair in `10` is an edit of this kind: the state change is at token 0, the intervention is at token 1, and the readout is at token 2. Token 0 is a legal key for token 1. That is why a coordinate of token 0 can change `R`.

## Heads

The attention derivative adds across heads at one fixed `X`. On seed 4 the test zeros one head's rows of `W_O` at a time, differentiates attention at the original `X`, and the two pieces sum to the joint attention derivative within `10^{-10}`. That sum is linearity of `W_O` at one point. It is not a sum of two block derivatives taken at two different operating points.

The downstream maps are linear in their increments at the shared `H_1`. The first-order increment is the image of `dH + dA^0 + dA^1`, with both head increments pushed through the same `J_{LN2}` and the same MLP derivative. The skip `dH` is not a head increment. A constant edit has `dA^0=dA^1=0`. The response is `a 1^T w` when the edited token is the read token and `0` otherwise, and in both cases the only carrier is the skip. If `1^T w=0`, the read-token response is `0` as well. Evaluating `margin_jvp` on a network whose `W_O` has a head removed is a different calculation: that network has a different `H_1`, so the second Jacobian is not the one at the original point.

Finite responses of those leave-one-out networks need not add. On seed 4, token 1, `δ=(0.2,-0.4,0.1,0.3)`, the joint response is `0.162853943` and the two single-head responses are `0.059199942` and `0.111275110`. The singles sum to `0.007621109` more than the joint response. This gap is not the four-term interaction of two increments inside one MLP. Each leave-one-out network is a different function, and each response subtracts that network's own baseline. What the gap does show is that finite responses do not add across heads. Patterns do not depend on `W_O`, so the discrepancy is produced by the output projection together with the second normalization and the MLP. The number is an existence check on seed 4. It is not a formula for the gap.

Head parameters being distinct does not make the heads independent given `H`. The shared normalized residual is a common parent. An intervention on `H` is a common cause of both heads. Summing their outputs is not a control for that common cause.

## What does not extend

The value-only sign error was found on this two-head block. It is an existence result, not a statement that every two-head block has such a direction. The shift identity does extend to any head count, because it never enters the heads. The mask statement extends to any causal pattern of this form. A bidirectional mask would let token `t` affect earlier positions. That architecture is not this block.
