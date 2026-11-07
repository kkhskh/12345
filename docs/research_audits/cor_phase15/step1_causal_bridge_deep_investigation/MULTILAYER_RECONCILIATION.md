# Mechanism C: two-block reconciliation

Theorems C1–C3 are in `THEOREMS_AND_COUNTEREXAMPLES.md` §4. The construction is `src/cbmath.py::reconciliation`, with parameters committed before any run.

## Construction

`d = 8`. Tokens: A = `a e_1 + κ·anchor`, B = `b e_2 + κ·anchor`, read = `ρ·anchor_R`.

**Block 1 (forms the relation).**
- A uniform head copies `x e_1 + y e_2` to the read token.
- A GELU MLP with the Mechanism-A pairing writes the agreement signal `λ_1 q e_3`. The readout `Δu = f_U` does not read `e_3`.

**Block 2 (rereads and reconciles).** Attention is off. A ReLU MLP reads `Ŝ = ⟨e_1+e_2, LN h⟩` (net evidence) and `q̂ = ⟨e_3, LN h⟩` (agreement):

```
ReLU(βŜ + γq̂ − θ) → +λ_2 f_U
ReLU(−βŜ + γq̂ − θ) → −λ_2 f_U
ReLU(±εŜ)          → ±f_U          (direct net-evidence vote)
```

An optional unembedding bias `−κ_0` encodes a prior toward `c'`.

## What block 2 does with the earlier relation

- **Candidate-relative amplification (C1.1–C1.2).** Agreement multiplies the vote of the candidate favored by net evidence. It never reverses that direction: `sign m = sign Ŝ` on all 156 nonzero grid cells (PC1).
- **Decision consequence at fixed net evidence (C1.3, PC2).** Pairs matched exactly on `Ŝ`, by inverting the source LN in input space:

| `Ŝ` | agreeing pair `m` | disagreeing pair `m` |
|---|---|---|
| 0.985 / 0.974 | 4.187 | 0.243 |
| 0.716 / 0.647 | 2.601 | 0.162 |

With any prior `κ_0` between the two values, agreement decides the candidate. With `κ_0 = 2.215` the decision is `c` (+1.97) under agreement and `c'` (−1.97) under disagreement. This value of `κ_0` was chosen to *demonstrate* the interval claim, not predicted.

## Necessary, sufficient, merely correlated

| Variable | Role | Evidence |
|---|---|---|
| `q̂` (block-1 MLP output) | **necessary** for the agreement effect at fixed `Ŝ` | ablating block 1's MLP shrinks the gap from 3.944 to 0.182 |
| `q̂` | **sufficient, up to LN coupling** | interchange from the agreeing run into the disagreeing run: `m` goes 0.243 → 3.928 (agreeing run: 4.187); the remaining difference is block 2's LN scale |
| `Ŝ` | sets the decision sign | `sign m = sign Ŝ` |
| an unread residual direction | causally irrelevant except through the LN scale | no sign flips in 169 cells (PC5) |
| raw source magnitudes `|a|`, `|b|` | correlated with `q`; act only through `Ŝ` and `q̂` | by construction of the block-2 read; not separately intervened |

## Single-block and bottleneck comparisons

- **Bottleneck (C2, PC4).** If block 1 writes both sources onto one direction, source identity is lost. Every matched-sum pair then gives identical margins (`1.8e−15`). This is screening theorem SS* acting on a scalar sum.
- **Single block (C3, PC6).** One wide ReLU layer on `(x, y)`, fitted by least squares, approximates the same margin function to held-out relative RMSE 0.032 (sign agreement 0.923), with no `q` variable. Exact representation by one ReLU layer fails (kink-set sketch), but behavior on the tested family does not reveal the two-block structure.

**Does the extra computation contribute meaningful causal structure?**
- **Inside this implementation, yes.** `q̂` is an interchange-identifiable mediator, necessary and nearly sufficient.
- **At the level of the function, no.** A shallow implementation that approximates the same behavior lacks it.

## What this does not establish

- That trained models reconcile evidence this way.
- That "reconciliation" is more than gated amplification: a multiplicative gate on a vote, which is standard.
- That `O`/`D` plays any role.
