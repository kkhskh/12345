# Mechanism B: query–key agreement gate

The theorem is B1 in `THEOREMS_AND_COUNTEREXAMPLES.md` §3. The construction is `src/cbmath.py::qk_gate`, whose default parameters were committed before any run (`b85222b`).

## Construction (dynamic attention; `W_Q, W_K ≠ 0`)

`d = 8`, with seven mean-zero orthonormal directions `f_A, f_B, f_Q, f_U, anchor_S, anchor_R, marker_A`. Positions: A, B, read.

| | Head | Query reads | Key reads | Value | Writes |
|---|---|---|---|---|---|
| **Layer 1** | H1 | `anchor_R` | `marker_A` | reads `f_A` | `2f_Q` (a later query feature) and `0.6 f_U` (A's direct vote) |
| **Layer 2** | H2 | `f_Q`, gain λ = 4 | `f_B` | reads `f_B` | `1.5 f_U` (B's vote) |

- Layer 1: the read token attends to A with weight ≈ 1, but not exactly 1.
- Layer 2: `score(read → B) = λ x̂ ŷ/√2`, where `x̂` is the LN-coordinate of the copied `a` and `ŷ` is B's LN-coordinate. The other scores are 0.
- There are no MLPs. Readout: `m = 2 f_U·h_read`.

The attention pattern is genuinely dynamic. B's weight runs from 0.002 to 0.99 across the grid as A and B agree or disagree.

## Results (all from fully recomputed input interventions unless labeled)

**Score, gate and decisions.**
- The score is exactly bilinear, and B's weight is monotone in `x̂ŷ` (PB1).
- The margin formula `m = v_A + 2kŷw` holds to `1e−12`.

| (a, b) | `w_B` | `m` |
|---|---|---|
| (+1, +1) | 0.99 | +5.19 |
| (+1, −1) | 0.002 | +0.96: A wins, B suppressed |
| (−1, +1) | 0.002 | −0.96 |
| (+0.5, +0.5) | 0.76 | +2.46 |

- **Unanimity (PB2):** in agreeing cells the decision has the shared sign.
- **Preference reversal relative to fixed routing (PB5):** in 242 of the conflict cells on a 31×31 grid, uniform routing (`W_Q = 0`) follows B while the gate follows A. Theorem B1.4 gives the band exactly.
- **Competition (PB8):** with two B sources of opposite sign, attention goes 0.99 against 0.00 to the source that agrees with A, giving `m = +5.19` (against +0.91 with fixed routing).

**Mechanism-dissection interventions (labeled; not ordinary recomputed interventions).**

| Intervention | Spread of B's vote over `a ∈ {−1, −.5, .5, 1}` (b = +1) |
|---|---|
| full recomputation | 4.238 |
| layer-2 pattern clamped to its `a = 0` value (routing fixed) | 0.104 |
| layer-2 value vectors clamped (values fixed, routing free) | 4.651 |

Routing carries the effect. The remaining 0.104 under fixed routing is LN coupling: layer 1 also writes A into B's residual, which moves B's LN scale and so `ŷ`. **This was checked directly:** `ŷ` varies across `a` by exactly 0.10407, both in the gated model and with `W_Q = 0`. The same coupling explains PB6's additivity failure. The idealized class B1 has no such coupling.

**Query-only, key-only, and both (PB4), from `(a,b) = (1,1)`.**

| Substitution | Margin change |
|---|---|
| query from `a = −1` | −4.212 |
| key from `b = −1` | −4.212 |
| both | 0.000 |

The non-additivity is +8.42. The gate responds to the *product* of query and key features, not to either one.

**Negative controls.**
- `W_Q = 0` in layer 2 is relation-blind by the ordinal LP test. But PB6 also predicted the margin would be additive, and it is not (residual 0.208, from LN coupling). **Rescored per the addendum, PB6 FAILED.** The first run scored it PASS on the relation-blindness part alone (R1 S4).
- Putting B after the read position removes every `b` dependence (PB7, causal mask).

**Full-margin claim (mandate §5B).** In the class B1 the head-level identity *is* the full margin, because (B4) excludes downstream readers. Add one layer-2 MLP that re-reads `f_U` and subtracts above a threshold, and 26 cells show the gated head's direct change and the total margin change with opposite signs. For example, `a: −1 → 0.5` at `b = 0.5` gives a head change of +1.905 and a margin change of −7.315 (PB9). A final LN with bias produced no such cell on the tested grid. So the head-level identity does not determine the decision when a downstream component re-reads the head's output.

**Scope of Theorem B1 (corrected).** (B4) must exclude *other* writers of source-dependent content along `Δu`. One extra head with `W_Q = 0` breaks unanimity in 348 of 450 agreeing cells (R1 S3). The reversal band in B1.4 assumes the other scores share the read row's offset. With an offset (R2 S3), the reversal can go toward B instead.

**Invisible to 2×2 sign-flip designs (R2).** At the input level, every sign-symmetric square `(±a, ±b)` has exactly zero four-term interaction in this class. Only designs with unequal levels detect the gate.

**Perturbations (supplementary).** Unanimity holds in 20 of 20 seeds at noise 0.03 and in 18 of 20 at 0.1. The reversal band stays non-empty in 20 of 20. R1 reports similar robustness.

**More than two candidates (R1 S2).** The gate's relation is the inner product of feature vectors. It is not agreement about the preferred candidate. Both sources can prefer `c_1` while `⟨x,y⟩ = −1.70`, and the gate then cuts the agreeing source's weight from 0.333 to 0.003. R1 gives a full-block case where `c_2` wins. Candidate-relativity is established only for a binary pair with scalar evidence.

## Source consistency, or an attention-score comparison?

The operation is a **content-based query–key match** (K-composition in the sense of Elhage et al. 2021). Its relation is the bilinear product of a feature copied from A and B's feature. "Agreement" here means "the query feature and the key feature have the same sign", which is the same pairing issue as in Mechanism A. The pairing is fixed by `W_Q2^T W_K2` composed with H1's OV map.

The effect *is* candidate-relative: it gates B's vote toward B's own candidate. The score itself is candidate-symmetric (`x̂ŷ` is unchanged under a joint flip). The candidate enters through the value path, not through the relation.

## What this does not establish

- That trained models gate votes by agreement.
- That the score's pairing has task meaning in any real model.
- That the decision follows the head when downstream re-readers exist.
- Novelty: this is QK composition used as a gate.
