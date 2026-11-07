# The strongest decision-related result

## Theorem T-ATT (structural; exact)

**Class.** Let the model be a one-layer, pre-norm, causal Transformer with:

- tokens `[A, B, (P), R]`;
- source tokens `x_A = a·w + E_A` and `x_B = b·w + E_B`, so each depends on one source;
- a prior token that depends only on p;
- a readout token `x_R = E_R` that carries no source content;
- H heads, followed by an MLP at R, a final LN and a linear readout.

**Statement.** The R residual after attention is

```
h_R(a, b, p) = E_R + Σ_h [ Σ_j ω_j^h(x_j) · v^h(x_j) ] / [ Σ_j ω_j^h(x_j) ] · W_O^h ,
ω_j^h(x_j) = exp(⟨q^h(LN E_R), k^h(LN x_j)⟩ / √d_h)
```

Here j ranges over the tokens A, B, (P) and R. In each head, the numerator and the denominator are sums of terms that each depend on one token only.

**Consequences.**

1. If the attention patterns are held fixed, `h_R` is exactly source-additive.
2. So every cross-source interaction in `h_R` is created by the per-head normalizers `Σ_j ω_j^h`.
3. Downstream of `h_R`, cross-source interaction can be added only by the R-MLP and the final LN, both of which act on `h_R`.

**Proof.**

- LN acts per position. So `LN(x_A)` depends on a only, and `LN(x_B)` on b only.
- The query at R depends only on `E_R`.
- So each score `s_j` depends on its own token only, and each value `v(x_j)` likewise.
- Substituting the softmax gives the displayed form. With the weights α fixed, the sum is additive. ∎

**Check.** Freezing the layer-0 pattern makes the interaction share of U exactly 0.000 in all 7 TF1 relation-computing models (`logs/locus_E1_E2.json`).

## Theorem T-MED (conditional decision theorem for an identified mediator)

**Class.** A frozen learned model F, together with:

- a site h that is a **complete cut** for the readout-position decision, verified by full-state interchange IIA = 1;
- a k-dimensional subspace U identified output-free (S_rel);
- the matched-pair family `M = {(x, x′) : z(x) = z(x′), p(x) = p(x′), s(x) ≠ s(x′)}`, drawn from the stated generator;
- `δ(·)`, the candidate decision of the **total** margin. This includes every residual, bypass, MLP and LN contribution, because each intervention is evaluated on F's full output.

**Hypotheses.** Each is measured on held-out pairs from M on which `δ(x) ≠ δ(x′)`:

- **(H1) interchange:** `δ(F[h_x + UUᵀ(h_x′ − h_x)]) = δ(x′)` on a fraction ≥ 1 − ε₁ of pairs.
- **(H2) sufficiency:** `δ(F[h_x′ + UUᵀ(h_x − h_x′)]) = δ(x)` on a fraction ≥ 1 − ε₂.
- **(H3) necessity:** after pair-mean ablation of U, `δ` differs within the pair on a fraction ≤ ε₃.
- **(H4) content:** U's coordinates satisfy `Uᵀh_x = f(a, p) + g(b, p) + c·σ(a, b) + r(a, b)`. Here σ is the independently defined relation and the residual satisfies `‖r‖²/‖Uᵀh − f − g‖² ≤ ε₄`.

**Conclusion.** On M, up to the stated ε's:

1. The candidate decision difference between matched-aggregate inputs is determined by the k coordinates `Uᵀh`.
   - (H1) and (H2): exchanging `Uᵀh` exchanges the decision, and keeping it keeps the decision, whatever the rest of the state is.
   - (H3): removing its within-pair variation removes the decision difference.
2. With (H4), the part of `Uᵀh` that differs within a matched pair **beyond the per-source terms** is `c·[σ(x) − σ(x′)]`, plus `r`.
3. The relation therefore causally determines the candidate decision on M through the identified computation (T-ATT for TF1), and **total**-margin effects are accounted for. This is not limited to a branch sign, which is how T-MED corrects the earlier A1* limitation.

**Instantiation.** For the TF2 XNOR models, the measured values on held-out pairs are:

| quantity | value |
|---|---|
| ε₁ | 0 |
| ε₂ | 0 |
| ε₃ | ≤ 0.152 |
| ε₄ | ≤ 0.003 (R² by `sign(ab)` of 0.997–1.000), with σ = `sign(ab)` |

**What T-MED does not give, and the missing condition.**

- **It is an ε-statement on the sampled support M, not a bound.** A uniform decision-sign theorem needs a guarantee that the downstream map `Φ(U-coordinate, complement)` has the measured sign behaviour for **all** inputs in the domain. That would require certifying the downstream map, for example by interval or Lipschitz bounds over the reachable state set. This was not done.
- **The relation is not separable from the spread.** At fixed z, `σ = sign(ab)` is a deterministic threshold function of `|a − b|`. T-MED cannot distinguish "relation" from "thresholded spread".
- **(H1) and (H2) are evaluated on off-manifold patched states** (nearest-neighbour ratio 18–189 for the headline models). The conclusion is about the frozen computation's response to the U-coordinate, including at unnatural values.

## Comparison with the earlier engineered theorems

- **A1\*** (previous package) gave the sign of the MLP *branch* only. Here the total margin is the measured quantity.
- **T-ATT** is the learned-model counterpart of C′'s query-composition analysis. There is no designed gate: the interaction is the softmax normalizer, which is ordinary attention arithmetic.
