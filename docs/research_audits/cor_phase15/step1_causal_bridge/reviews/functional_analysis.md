# Reviewer A — functional analysis

Independent pass. The reviewer read files 02, 03, 05, 08, 09 and the synthetic tests, and did not edit them. The objections below are the reviewer's. File 14 records which ones were accepted and what the text now says.

## Objections

1. **Theorem NFL, as first written, did not prove its quantifiers.** "Pick any `δ` and correct orthogonally" fails when `δ` is parallel to `s_0`. A constant readout has derivative zero, so it does not realize an arbitrary `β` at the zero cochain. The closing pair of linear heads shows opposite values, not arbitrary magnitudes, and a cochain displacement is not automatically a recomputed intervention.

2. **Theorem Descent used the Euclidean gradient.** For `D = ||s - Π_M s||_M^2`, the step toward `Π_M s` is `-(s-Π s)`. Euclidean steepest descent is parallel to `-M(s-Π s)`. They agree for every residual only when `M` is a multiple of `I`. Explicit split: `M=diag(1,100)`, `s=(1,0)`, `∇L=(1,10)`. The repair direction and the Euclidean descent direction have derivatives of opposite sign.

3. **The displayed block formula for `W'` dropped cross terms.** Full congruence `W' = A_head^{-T} W A_head^{-1}` preserves `O`. Zeroing the off-diagonal of a `2×2` weight changes `O` from `3` to `2` in the reviewer's example.

4. **CE-07's two metrics are stalkwise congruent.** `A=diag(√10,√10,1/√10,1/√10)` carries one to the other. The ranking reversal keeps `s` fixed while `M` changes. That is real, and it is not a pair of inequivalent metrics.

5. **Minimum-norm least squares does not commute with a coordinate change** unless the tie-break is transformed. One sample `s_u=(1,0)`, `s_v=1`, minimum-norm `ρ=(1,0)`, becomes a different minimum-norm solution after a shear.

6. **The test rejected every singular Gram**, including an exact duplicate edge whose pseudoinverse formula is correct and whose distance is `1/2` rather than `2`. Structural rank deficiency is not numerical ill-conditioning.

7. **Theorem Idle, item 3, had no quantifiers.** It is false if `d` is injective or if `q` is constant. Items 1 and 2 were not attacked.

No error was found in Theorem Lip, Theorem Sign, Theorem Z on a single map, Theorem App on a finite sample, the Jacobian pseudoinverse identity, or the algebraic fact that a nonnegative `τ O` can be deleted from a hypothesis that already bounds the increment.

The reviewer judged that the negative conclusions survive these repairs. This review did not propose a replacement bridge.
