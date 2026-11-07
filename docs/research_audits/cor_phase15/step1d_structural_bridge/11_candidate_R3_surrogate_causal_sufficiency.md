# 11. Candidate R3 — surrogate error and causal sufficiency

## The only surrogate that dropped out of the algebra

Let `s(H) = 1_{t=r} w · H_t`, the skip readout, and let `m` be the full raw margin. For an edit at token `t`,

```text
R_m - R_s = w·ΔA_r + w·ΔM_r,
```

and Theorem S bounds the right-hand side by the branch piece `B_branch`. So

```text
|R_m - R_s| ≤ B_branch.
```

If in addition `W_O w = 0` and `W2.T w = 0`, the inequality is the identity `R_m = R_s`.

## What the residual is

The difference `m - s` on a recomputed trace is `w·(A + M)` at the read token. It is the branch contribution the surrogate omitted. On every recomputed trace the structural assignments `H1 = H + A` and `H2 = H1 + M` have residual zero; the seed-0 pair with opposite margins in `01` has residuals `(0, 0)` and margins of opposite sign. A nonzero gap between `m` and `s` is approximation error of the skip surrogate. It is not evidence that the block's equations failed, and it is not a conflict between two causal parents.

The same numeric gap can be produced by an off-manifold stitch that sets `H1` to something other than `H + A`. The scalar `|m - s|` does not say which production it came from. Distinguishing them requires the parent vectors `A` and `M`, not a compatibility norm.

## Relation to abstraction

If a surrogate agrees with `m` up to `ε_s` on a set that contains both `H` and the edited state, then the intervention responses differ by at most `2 ε_s`. That is the standard approximation lemma behind exact and approximate causal abstraction: a high-level variable tracks a low-level variable only as well as the implementation map allows. Geiger, Ibeling, Zur, Chaudhary, Chauhan, Huang, Arora, Wu, Goodman, Potts, and Icard, JMLR 26(83):1–64, 2025, formalize exact transformation and constructive abstraction for interventions. The bound above is the elementary numerical form of that idea for one scalar surrogate. It is not a new abstraction theorem, and it does not use an interchange-intervention accuracy.

The bound does not need a learned alignment, a partition of inputs, or a map `ρ` chosen after seeing `R`. The surrogate is the skip term of the block, read off the architecture. Building a different surrogate by fitting `R` would leave Regime 1 and would be circular for prediction. That construction was not used.

## Final LayerNorm

The skip surrogate is not causally sufficient for `w · LN(H2[r])`. With zero branch outputs and `w·δ = 0`, the raw responses agree at `0` and the normalized responses differ by `-0.32024254867419905`. Controlling the raw residual does not control the normalized intervention. Any surrogate claim has to name the readout.

## Verdict on R3

`|R_m - R_s| ≤ B_branch` is true and useful as an approximate-implementation bound. It is the branch half of Theorem S. It does not undercut `B_align`, because it is that estimate. It does not say that a compatibility residual is an obstruction to a decision. R3 does not clear Level 3, and it does not revive `O` or `D`.
