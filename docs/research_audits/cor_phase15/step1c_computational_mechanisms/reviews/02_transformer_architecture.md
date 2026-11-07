# Reviewer 2 — Transformer architecture

Provenance: subagent [architecture review](bfdad80b-da28-4c94-9abc-2f7ca708fb69), this session. Not an external architect.

## What held

The pre-norm skeleton `H'=H+Attn(LN(H))`, `H''=H'+MLP(LN(H'))`. The separation between a clamped pattern and `do(X)`. The causal mask. The digits of the bias-free token-1 interaction. Population LayerNorm, the sign flip, and the statement that plain RMSNorm is a positive scale and is not LayerNorm.

## Defects

1. `04` said the interaction facts were proved for the pre-norm form. The proofs apply `Q=XW_Q` to `X`, not to `LN(H)`. A residual edit of `H` changes the skip by `\delta` and changes `Q,K,V` through `LN(H+\delta)`.
2. `W_Q=0` does not make pre-norm residual edits additive. A query bias `Q=b_Q` leaves scores dependent on `K`. Inside one token, LayerNorm of two coordinates need not be additive even if the pattern is constant.
3. "If one logit falls, the other weights increase because they sum to one" is false for a masked third position, and the sum constraint is the wrong reason when the softmax Jacobian is what raises the other positive weights.
4. The two-token counterexample's second output was written as `0`. It is `1/(1+e^{16})`.
5. The RMSNorm comment arithmetic, same defect as Reviewer 1.
6. Biases, `eps>0`, and multi-head were not listed next to the maps that omit them. Value-linearity extends to a sum of heads only if every pattern is clamped.

## Disposition

`04`, `proofs/attention_value_linearity.md`, `06`, `09`, `14`, and `proofs/fiber_and_trace.md` now scope the numerical interaction and the `W_Q=0` sentence to the bias-free, LayerNorm-free map. Two new tests record the query-bias boundary and the within-token LayerNorm boundary. Cross-token edits in front of a constant pattern stay additive; that test was added so the boundary would not be over-corrected into "LayerNorm always interacts." The softmax paragraph is now the two-point identity plus the softmax Jacobian. Section 4.7 lists biases and unclamped multi-head as not investigated. The exact kernel and scale identities are labeled `eps=0`.
