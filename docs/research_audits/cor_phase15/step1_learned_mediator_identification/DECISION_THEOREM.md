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
h_R(a, b, p) = E_R + b_O + Σ_h [ Σ_j ω_j^h(x_j) · v^h(x_j) ] / [ Σ_j ω_j^h(x_j) ] · W_O^h ,
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

The `b_O` term was missing in the reviewed version (R_math). With it added, the formula matches the forward pass to 3e−7.

**Check.**
- Freezing the layer-0 pattern makes the interaction share of U exactly 0.000 in all 7 TF1 relation-computing models (`logs/locus_E1_E2.json`).
- Freezing only the normalizers gives ≤ 1e−28 (R_math).
- T-ATT applies to **one-layer** models only. In two layers, layer-1 attention reads mixed states, so the theorem's Consequence 3 does not hold.

## Theorem T-MED (conditional; corrected after review)

**Class.** A frozen model F, together with:

- a complete cut h. This holds by architecture; the full-state interchange is a code check, not evidence;
- a k-dimensional subspace U;
- the matched family M, as before;
- `δ`, the decision of the total margin.

**Hypotheses** (measured on held-out pairs from M on which the model's decisions differ):

- **(H1) interchange in both directions:** patching U from the partner makes `δ` follow the partner, on a fraction ≥ 1 − ε₁ of pairs.
  - The "sufficiency" test of the reviewed version builds exactly the same patched states as the reverse-direction interchange. It is equal in 150 of 150 logged rows, so it is **not** an extra hypothesis (R_math 1, R_causal F3).
- **(H3) necessity:** after pair-mean ablation of U, `δ` differs within the pair on a fraction ≤ ε₃.
- **(H4′) content, on pairs:**
  - On held-out pairs, the additive surrogate of U's coordinates leaves `δ` different within the pair on a fraction ≤ ε₄. That is, the per-source part of U alone does not separate the decisions.
  - The reviewed version measured content on the grid only, which is insufficient (R_math 2).

**Conclusion** (on M, up to the ε's). The within-pair decision difference is mediated by the **non-additive** component of U's coordinates.

- (H1) and (H3) show that U carries the difference.
- (H4′) shows that the per-source part of U does not.

This holds for the **total** margin, because every intervention is evaluated on F's output.

**What T-MED cannot conclude.**

- **It cannot say which variable the non-additive component is.** On REL-XNOR and REL-MIS pairs, the relation and the target are confounded, so "mediated by agreement" and "mediated by the target" are the same claim there.
- **On REL-XOR3, where they differ, the non-additive component is the target** (`XOR3_RESULTS.md`).
- **It does not cover states other than the matched partner's.** "Whatever the rest of the state is" in the reviewed version overclaimed (R_math 3), and the patched states are off-manifold.
- **It is not a uniform decision-sign theorem.** That would additionally need a certificate on the downstream map over the reachable set (missing condition).

**Which models satisfy (H4′).**

- TF2 XNOR seeds 2, 7 and 9: ε₄ = 0.010–0.020.
- **Not** TF2 XNOR seeds 4 and 8: the additive part alone keeps 34–37% of the decision differences.
- TF1 XNOR seed 4: no (58%).

## Comparison with the earlier engineered theorems

- **A1\*** (previous package) gave the sign of the MLP *branch* only. Here the total margin is the measured quantity.
- **T-ATT** is the learned-model counterpart of C′'s query-composition analysis. There is no designed gate: the interaction is the softmax normalizer, which is ordinary attention arithmetic.
