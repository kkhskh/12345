# 2. Theorem U, restricted

## Statement

Let the readout be raw, `m(H) = w · H2[r]`. Fix `ε>0`, the LayerNorm gains and biases, `w`, a base state, and a nonconstant single-token edit `δ`. Let the branch weights vary without a bound, including the attention output map. Then no finite number computed from the fixed data alone upper-bounds `|R|` for every such weight tuple.

The overnight witness still stands: zero the score maps and the MLP, put the value map at a split identity, and set the output map proportional to `α`. The read-token attention increment is `(α / T) w` in that construction, so `R` grows like `α ||w||^2 / T`. The recomputation is in the overnight tests. It is not re-derived here as a new mechanism.

## What the witness uses

The witness needs a raw linear skip. `H2` contains `H` additively, so enlarging the branch enlarges `m` without a saturating nonlinearity after the sum. It also needs permission to send `||W_O||` to infinity while keeping the weight-free data fixed. That is a statement about a class of models, not about one frozen model.

## Final LayerNorm blocks the same scaling

Let `m_f(H) = w · LN_ε(H2[r])` with the same `ε>0`, fixed gain `γ` and bias `β`. For every vector `h`,

```text
||LN(h)||_2 ≤ ||γ||_∞ √d + ||β||_2 =: R_f.
```

Proof: the centered unit `z / s` satisfies `||z||^2 / s^2 = ||z||^2 / (||z||^2/d + ε) ≤ d`, so `||z/s||_2 ≤ √d`. Multiplying by `γ` coordinatewise costs at most `||γ||_∞`, and `β` adds its Euclidean norm.

Therefore `|m_f| ≤ ||w|| R_f` and

```text
|R_f| ≤ 2 ||w|| R_f
```

for every intervention and every branch weight. With `γ = 1`, `β = 0`, `d = 4`, one has `R_f = 2` and the bound is `2 ||w|| R_f`. On seed 0, `||w|| = 0.8764349701487028`, so the bound is `3.505739880594811`.

## Assumptions, stated one at a time

- `ε>0` keeps LayerNorm defined and the radius finite. At `ε = 0` the map is undefined on constant tokens.
- `γ`, `β`, and `w` are fixed. If any of them may be rescaled without bound, the diameter bound is not uniform over that larger class.
- If the scalar readout is a difference of two unembedding rows, replace `w` by that difference. The form of the bound does not change.
- The bound does not depend on `δ` or on `H`. For a small intervention it is a diameter statement, not a local sensitivity. In the zero-branch orthogonal example of `05`, `|R_f| = 0.32024254867419905`, which is inside the diameter bound and far larger than the raw response `0`.
- A model class that can rescale the final gain, or delete the final LayerNorm and return to the raw readout, recovers Theorem U. The architecture has to be part of the quantifier.

## What this does not say

The diameter bound is not a causal mechanism. It does not rank interventions. It exists to stop the inference "Theorem U, therefore every Transformer readout is unboundable."
