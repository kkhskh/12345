# 7. Baseline hierarchy

Every row is for the same block and the same intervention family. "Beats" is used only when both objects are finite-response upper bounds under a named regime.

| predictor | regime | returns | finite? | uniform in `H`? |
|---|---|---|---|---|
| A. Diameter of a final LayerNorm readout, `2 ||w|| R_f` | 1, and only for that readout | certified bound | yes | yes, and independent of `δ` |
| B. Isotropic `B_iso = ||w|| (1+L_A L_1)(1+L_M L_2) ||δ||` | 1 | certified bound | yes | yes |
| C. Separate branch triangle `B_dir` | 1 | certified bound | yes | yes |
| C′. Aligned separate triangle `B_align` | 1 | certified bound | yes | yes |
| D. `B_dir` on the sub-family `w·δ = 0` | 1, using the scalar `w·δ` | certified bound | yes | yes |
| E. A joint-path bound strictly below `B_align` | 1 | not obtained | — | — |
| F. `∇m · δ` | 2 | linear prediction | no | no, one JVP per `H` |
| G. `∇m · δ` plus a global second-derivative remainder | 3 | certified bound if the remainder is proved | yes, and loose | no |
| H. Recomputed `R` | 4 | exact value | yes | no, one forward per pair |

There is no Regime 1 joint row. That absence is a result of `09`–`12`, not a missing entry.

## Fair comparisons

**C against B.** Proved in Theorem S-iso. `B_dir ≤ B_iso`, gap `||w|| ||δ|| - |1_{t=r} w·δ|`. At `α = 10^{-6}` on the unit orthogonal edit, `B_iso - B_dir = 0.8764349701487027`, which equals `||w||`. The entire gap is skip orthogonality. The branch piece that remains is `B_dir` itself, `0.004051326443971923`.

**C′ against C.** Proved by `||W_O w|| ≤ σ(W_O)||w||` and `||W2.T w|| ≤ σ(W2)||w||`. At the same point `B_align = 0.0028008328328102084`. The further drop is the alignment of each output map with `w`, still one path at a time.

**C against F.** Not a finite-response comparison. At the same point `|∇m · δ| = 5.19×10^{-8} < |R| = 1.33×10^{-7} < B_dir`. The linear term is below the response, so it is not a bound. The Cauchy number `||∇m|| ||δ|| = 0.876435` is above `B_dir` and is still only a bound on the linear term. Reporting `B_dir / ||∇m|| ||δ|| ≈ 0.0046` as a superiority ratio would compare a finite bound with a first-order product.

**C against H.** `H` is exact and costs a forward. `C` is a bound, reusable across `H`, and at unit scale it exceeds `|R|` by five orders (`1.63×10^6` against `1.08×10^{-2}`). Reuse is the advantage. Tightness is not.

**C against G.** A remainder whose LayerNorm factor alone is `ε^{-3/2} ≈ 3.16×10^7` at a constant token does not undercut Theorem S at unit step. No global Hessian certificate was promoted.

## The non-relational baseline, fixed before the candidates

The response splits exactly as

```text
R = 1_{t=r} w·δ + Δconcat_r · (W_O w) + Δhidden_r · (W2.T w).
```

The separate baseline bounds each inner product by a product of norms and sums the three terms. Any candidate that claims a relational improvement has to undercut that sum, or undercut `B_align` if it is not allowed to use the realized increments `Δconcat` and `Δhidden`. Using the realized increments is a partial perturbed forward, which is not Regime 1.

## When the branch bound becomes vacuous

`B_dir` is a valid upper bound at every scale. It is numerically useless when it exceeds both the scale of `|R|` one actually sees and any other valid bound one is willing to use. On this block that happens as soon as `L_1 ≈ 320` multiplies an order-1 attention constant. At `α = 1`, `B_dir ≈ 1.63×10^6` while `|R| ≈ 0.011` and `||∇m|| ≈ 0.96`. The bound has not become false. It has stopped distinguishing interventions. The final-LayerNorm diameter `3.51` is the tighter Regime 1 statement for that other readout, and it likewise does not distinguish small edits.

Vacuous here means "larger than every quantity one would use to rank these edits." It does not mean "false."
