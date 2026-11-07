# 2. The frozen block and the intervention families

The block is the sequential pre-norm map

```text
X  = LN1(H)
H1 = H + A(X)
U  = LN2(H1)
H2 = H1 + M(U)
```

with causal scaled dot-product attention, two heads, and a tanh MLP. Both LayerNorms use `ε = 10^{-5}`. Dropout is off. The raw readout is `m_raw = w · H2_r` at `r = T - 1`. The normalized readout is `m_norm = w · LN_f(H2_r)`, with the same `ε` and with `γ_f = 1`, `β_f = 0` unless a note says otherwise. A conclusion for one readout is not copied onto the other.

An intervention replaces `H_t` by `H_t + δ` and recomputes the block. Attention weights are not held fixed except in the structural model, where `W_Q = W_K = 0` makes the causal pattern uniform. That case is labeled. A separate example sets those weights nonzero and shows that the scalar formula stops matching.

## Structural weights

Width `d = 2`, length `T = 3`, head width `2`, MLP width `1`. The mean-zero unit axis is

```text
n = (1, -1) / √2.
```

The readout of the structural theorems is `w = n`. The value map satisfies `W_Vcat W_O = μ T I`, split evenly across the two heads, so a one-row LayerNorm change is multiplied by the causal weight `1/T` and becomes `μ` times that change. The MLP row is `-α w` and the MLP column is `s_gain w`. Gains used in the exact bound are `μ = 1/2`, `α = 1`, `s_gain = 1`. The sign theorem allows any positive gains.

The zero matrix is a point of the invariant line: every token equal to `y n` stays in `span{n}` under LayerNorm, under this attention map, and under this MLP. The coordinate `y` is the object of the scalar reduction in `proofs/line_model.py`.

## Families

The proved family is the segment

```text
U(ρ) = { t n : t ∈ [0, ρ] },    ρ > 0,
```

applied either at the read token or at token `0`. A single fixed `t` is not treated as a reachability problem. The negative ray is the same argument run backward. Edits off the line, constant shifts, and a width-3 singular value map are counterexamples, not extensions of the bound.

The direct skip `w · δ` is part of the raw response when the edit is at the read token. It is identically zero at the read token when the edit is at token `0`, because the mask stops token `0` from writing the residual at `r` except through attention.
