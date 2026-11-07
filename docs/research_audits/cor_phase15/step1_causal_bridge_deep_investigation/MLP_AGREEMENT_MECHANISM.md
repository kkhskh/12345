# Mechanism A: agreement-sensitive MLP

Theorems A1–A4 are in `THEOREMS_AND_COUNTEREXAMPLES.md` §2. Numbers come from `logs/experiments.txt` and `logs/positive_control_matrices.txt`.

## 1. Positive control (mandate §3), reproduced

The construction is `src/cbmath.py::positive_control`, and the exact matrices are in `logs/positive_control_matrices.txt`.

- `d = 4`. `e_1, e_2, e_3` are orthonormal and mean-zero, obtained by QR of the centering matrix.
- Tokens: `[a e_1, b e_2, 0]`, where the last token is the read token.
- `W_Q = W_K = 0`, so attention is uniform over positions 0–2.
- `W_V = I` and `W_O = 3√(1/4+ε)·I`. The read token then receives exactly `a e_1 + b e_2`. The audit's construction, which uses `√(1/d+ε)` on the source tokens, is the same map.
- MLP: four GELU units (tanh form). `W_in = √(1/2+ε)[e_1+e_2, −(e_1+e_2), e_1−e_2, −(e_1−e_2)]`, giving preactivations `a+b, −(a+b), a−b, −(a−b)` exactly. `W_out` has rows `e_3, e_3, −e_3, −e_3`.
- Readout: final LN (`γ = 1`, `β = 0`) and `W_U = [e_3, −e_3]`.

| a | b | agree | margin (final LN) | margin (no final LN) | margin before the MLP | MLP write `q` |
|---|---|---|---|---|---|---|
| −1 | −1 | yes | +3.2142216080 | +3.8183907764 | ~1e−17 | +1.9091953882 |
| −1 | +1 | no | −3.2142216080 | −3.8183907764 | ~1e−17 | −1.9091953882 |
| +1 | −1 | no | −3.2142216080 | −3.8183907764 | ~1e−17 | −1.9091953882 |
| +1 | +1 | yes | +3.2142216080 | +3.8183907764 | ~1e−17 | +1.9091953882 |

**Interventions on input tokens, recomputed.**
- Flipping A with B fixed reverses the sign. So does flipping B with A fixed. Flipping both leaves the margin unchanged (PA4).
- Ablating the MLP output gives margin 0 in every cell.
- The causal readout decomposition is: residual before the MLP, 0; MLP write, all of the margin.

**Attention is not responsible.** Clamping the read row of the attention pattern to 50 random positive weightings leaves every sign unchanged (PA3). This is a *mechanism-dissection* intervention. It is Corollary A1.3: separate positive rescalings of `x` and `y` preserve `sign(xy)`. That holds as long as each reader coordinate depends on one source only (source separation, A0). With two heads whose routing mixes the sources, the law fails: the circuit then follows A alone (R1 S1).

**Independent implementations.**
1. `cbmath.py`, in numpy.
2. `positive_control_ref.py`, in pure Python with no numpy, written from the specification. It agrees to `1.8e−15`.
3. The audit's standalone script, which shares no code. It gives the same margins to 10 decimal places (`sources/independent_audit_handoff/independent_standalone.log`).

## 2. How the MLP computes agreement

Let `h(t) = φ(t) + φ(−t)`. For tanh-GELU, `h(t) = t·tanh(√(2/π)(t + 0.044715t³))`: even, zero at 0, and strictly increasing in `|t|`. The MLP writes `q = h(x+y) − h(x−y)`. Since `|x+y|² − |x−y|² = 4xy`, we get `sign q = sign(xy)` for **every** real `x, y` (Theorem A1).

- The `±` pairing makes each pair of units an even function: an "absolute value" of `x + y` and of `x − y`.
- The sign of the output weights subtracts the two.

The unit pairs are exactly a (smoothed) `|x+y| − |x−y|` circuit, the standard ReLU construction of XNOR or multiplication, as noted in `BASELINES_AND_NOVELTY.md`.

## 3. Which quantity explains the behavior (mandate §4)

The quantities were defined from the source semantics and the computational equations *before* evaluation. Values on the binary cells are in the log.

| Candidate quantity | Explains the MLP? | Reason |
|---|---|---|
| Binary sign agreement `[sign a = sign b]` | **yes, for the sign**, in all regimes | A1.1: `sign q = sign(xy)` |
| Product `xy` | **magnitude in the small-signal regime** only: `q ≈ 4√(2/π)xy` | A1.3 |
| `2 sgn(xy) min(|x|,|y|)` | **magnitude in the large-signal regime**; exact for ReLU | A1.2 |
| Cosine of the two source writes | **no**: identically 0 (orthogonal writes) | log: cos = 0.000 in every cell |
| Euclidean distance `‖a e_1 − b e_2‖` | **no**: `√2` in every binary cell | log |
| Difference magnitude `|x − y|` under the pairing `e_1 ↔ e_2` | informative on binary cells, but `q` is not a function of it | `q = h(x+y) − h(x−y)` uses both |
| Agreement relative to a candidate | **no**: `q(x,y) = q(−x,−y)` | A1.2, PA13. Two sources agreeing on `c'` push toward `c` |
| Agreement under a specified transformation | **yes, and this is the honest description.** In the residual the sources are orthogonal. "Agreement" exists only relative to the pairing that the reader's directions `e_1 ± e_2` impose: same polarity means the same sign of the coordinate the reader takes for "the same thing". The input semantics fix it (positive `a` and `b` encode the same polarity), and `W_in` implements it | — |
| Consistency with a structural equation | **no** | no equation is violated in any cell |
| Consistency of an abstract causal model | the circuit is an exact abstraction of `Y = [sign A = sign B]` | interchange on the signs is exact |

**What the two sources represent.**
- *In the positive control*, `a` and `b` are polarities. The candidate pair is "consistent" against "inconsistent". The model computes a consistency judgment, not an evidence-weighted choice between two answers.
- Under an *evidence* reading ("`a > 0` supports `c`"), the circuit makes the wrong choice whenever both sources support `c'`.

This distinction is why Mechanisms B and C, which are candidate-relative, are needed.

## 4. Continuous family and interaction

`agreement_family` uses `d = 6` and adds token anchors, so that LN1 does not erase the magnitude of the evidence. The read coordinate is `x = a/√((a²+κ²)/d+ε)`, which is odd and monotone.

- The sign law holds on 144-cell grids, fully recomputed, for GELU (both forms), ReLU, SiLU and softplus. It holds with final LN as well (PA5, PA12).
- With ReLU the margin equals `2 sgn(xy) min(|x|,|y|)` to `1e−15`.
- Odd activations (tanh) and activations with a constant even part (sigmoid) give 0 (PA6).
- **Strength versus relation.** The *sign* of the response depends only on the relation. The *magnitude* is product-like for weak evidence and min-like for strong evidence: the weaker source caps the effect. Saturation enters only through `h`.

**The interaction `I(a,b) = m(a,b) − m(a,0) − m(0,b) + m(0,0)`** measures non-additivity of `m` on the chosen scale.
- Here `m(a,0) = m(0,b) = 0`, so `I = m`.
- A nonzero `I` does not show that a consistency mechanism exists. The sum-only reader `h(a+b)` has an interaction with the same sign in the small-signal regime, and an odd–odd component of exactly `q/2` (Theorem A3). Any even nonlinearity of net evidence produces an "agreement-signed" interaction.

## 5. What separates relation computation from a nonlinearity of net evidence

- **Binary data cannot.** A sum-only reader with an offset reproduces the positive control's margins exactly on the four cells, to `3.1e−15` (Theorem A2). This holds without final LN; with final LN even the decisions differ (R2). At the weight level, "reads `x − y`" is not invariant: cancelling linear units can add such a reading without changing the function (R3).
- **Matched post-normalization sum can** (Theorem A4). At fixed reader coordinate `Ŝ`, the sum-only reader is constant: `1.270372` three times. The designed reader goes `1.270 → 0 → −1.675` as `|D̂|` grows. Those values use residual interventions, some of them on states no input reaches. **With input substitution only**, on reachable inputs: designed `0.307 → 0 → −0.720`, sum-only `0.307412` three times.

**Preregistration failure PA9, kept.** Matching the *raw* sum `x + y` in the residual does not work. The reader's LN divides by `‖h‖`, which depends on `x² + y²`, so the sum-only reader varies (`1.270, 1.122, 0.830`). Pre-norm LN is itself a non-sum coupling of the sources. Matched-statistic controls must be matched at the reader's input.

## 6. Boundaries: removing each condition of A1

Witnesses were found by search. Details are in the theorem file. In short:
- skewed read directions, a read bias, a direct readout path, and a final-LN bias each break the law on 4–8% of the grid;
- an odd activation zeroes it;
- unequal output weights break it only near the axes. On the preregistered grid PA10 failed to show this; post hoc, there are 60 errors at `|b| = 0.05`.

**Vector evidence (supplementary).** With three-dimensional sources and coordinatewise pairing, `sign q = sign⟨u,v⟩` in 100% of draws in the small-signal regime, 95% at scale 1, and 92% at scale 5. Agreement is an inner product only for weak signals. For strong signals it is coordinatewise and depends on the basis.

**Supplementary variants, not preregistered.**

| Variant | Grid error rate |
|---|---|
| 4 / 16 / 64 extra random units, output scale 0.05 | 0 / 0.19 / 0.06 |
| same, output scale 1 | 0.27 / 0.50 / 0.42 |
| B encoded at 60° to A (non-orthogonal) | 0.08 |
| B encoded at 30° to A | 0.19 |
| B's write 3× larger than A's | 0 (as A1.3 predicts) |
| noise on an unread direction | 0 |
| noise on a read direction | 0.17 |

- **Perturbations.** PA11 is reclassified as non-discriminating: its test could not fail (R1 S5). Descriptively, the binary cells survive relative noise of 0.03 in all 20 seeds. Grid errors grow to 24% at 0.1. Errors concentrate where the weaker source is small.
- **Generic MLPs.** Of 400 random 4-unit GELU readers of `(x, y)`, 10 reproduce the agreement sign pattern on the binary cells, and none reproduce the sign law on the continuous grid. The agreement law is a special, designed property. An earlier "400/400 depend on more than the sum" statistic used raw-sum matching, which has no power (R1 S6). It is withdrawn.

## 7. What this mechanism establishes, and what it does not

- **Establishes.** Existence: a standard pre-norm block can compute an independently defined relation (sign agreement under the reader's pairing) and let it alone decide the candidate. There is a class theorem (A1) with exact boundaries.
- **Does not establish.**
  - That trained models use this.
  - That the rule is candidate-relative (it is not).
  - That it is novel: it is an `|x+y| − |x−y|` XNOR circuit.
  - That `O`/`D` causes anything.
  - That the binary positive control alone identifies relation computation (A2).
