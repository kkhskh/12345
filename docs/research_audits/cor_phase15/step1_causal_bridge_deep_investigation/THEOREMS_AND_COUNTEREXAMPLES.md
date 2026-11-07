# Theorems and counterexamples

Implementation: `src/cbmath.py`. Experiments: `src/experiments.py` (log in `logs/experiments.txt`). Tests: `tests/test_deep.py`. Each theorem is stated with every hypothesis. Proved theorems are labeled **PROVED**. Statements checked only on constructions are labeled **VERIFIED (construction)**. Numerical tests are not offered as proofs of universal claims.

**Common setting.** Pre-norm residual Transformer layers: LN, multi-head causal attention, residual add, LN, MLP, residual add. Optional final LN with `ε > 0`, then a linear unembedding. Candidate margin: `m = L_c − L_{c'} = Δu^T ĥ + Δb`, where `ĥ` is the read-position final residual, after final LN if present. An intervention replaces the value of a named site (an input token, `Q`, `K`, `V`, scores, pattern, head outputs, `mid`, MLP preactivations or output, `final`) and recomputes everything downstream. A *mechanism-dissection* intervention clamps one site (for example, a pattern) while another changes. It is labeled as such and is not an ordinary recomputed intervention.

---

## 1. Corrected prerequisites

### SS* — same-cut screening, no bypass (corrects the earlier Theorem SS)

**Hypotheses.** Writers `c_1, …, c_K` feed an aggregate `S = B + Σ_k c_k`. A downstream variable `Y` satisfies **no bypass**: every directed path from any `c_k` to `Y` passes through `S`. Equivalently, `Y = F(S, U)` where no variable in `U` is a descendant of any `c_k`, except through `S`.

**Claim.** Every edit `c_k ↦ c_k + δ_k` with `Σ_k δ_k = 0` leaves `Y` unchanged. Any relation among the writers that is not a function of `S` is causally inert for `Y` once `S` is held fixed.

**Proof.** `S` is unchanged and `U` is unchanged, so `F(S, U)` is unchanged. ∎

**Counterexample to the earlier wording.** `S = c_1 + c_2` and `Y = S + c_1`. Here `c_1` is "upstream of `S`", which the earlier wording allowed. The edit `(2, −1) → (5, −4)` keeps `S = 1` and changes `Y` from 3 to 6 (`test_SSstar_…`; independent audit §5).

**Transformer instance.** The heads of one attention layer are consumed only through the residual add. A standard pre-norm or parallel block therefore has no bypass, and SS* applies to zero-sum head-output edits (checked with two heads, 10 random zero-sum edits, to `1e-12`). SS* does **not** apply when an analysis reads a head output separately (a hook, a per-head gate), or to writers at different layers.

**Scope reminder (mandate §6).** SS* rules out effects at a fixed sum `S`. It does not rule out relational computation when `S` is a *vector* that keeps the sources apart. In the positive control, `S = a e_1 + b e_2` keeps both, and the MLP reads both.

### ID — three levels of identification (correction B)

Let `θ` be a parameterization, `x` an input, and `G` a group of reparameterizations with `f_{g·θ} = f_θ` for every `g`.

- **Fixed-parameterization observability.** `Q(θ, x)` can be computed from `θ` and the activations of `f_θ`. Example: a per-head direct effect in one checkpoint.
- **Causal intervention attribution.** `Q(θ, x) = f_θ(x; do(site ← v)) − f_θ(x)` for a site in the computational graph of `θ`. It is well defined given `θ`. Example: patching head `h`.
- **Identifiability modulo `G`.** `Q(g·θ, x) = Q(θ, x)` for all `g ∈ G`. **Function-level** quantities, which depend only on `f_θ` restricted to an intervention family on inputs, are identifiable modulo every `G`.

**Proposition ID1 (head redistribution).** If heads `h_1, h_2` of one layer have identical `W_Q`, `W_K`, `W_V`, then `W_O^{h_1} ↦ W_O^{h_1} + E` together with `W_O^{h_2} ↦ W_O^{h_2} − E` preserves `f_θ`. Per-head writes, per-head direct effects, per-head consensus scalars, and **per-head patching effects** all change in general.

*Proof.* The two heads have identical patterns and values `Y`, so their summed write `Y(W_O^1 + W_O^2)` is unchanged. By SS*, everything downstream is unchanged. Individually, `Y(W_O^1 + E) ≠ Y W_O^1` whenever `YE ≠ 0`. ∎

*Verified (PI1).* Logits change by at most `4.4e−16`. Per-head direct effects go from `[2.111, 2.111]` to `[1.703, 2.520]`. Per-head patch effects go from `[−2.106, −2.106]` to `[−1.699, −2.514]`. Their sum is unchanged (`−4.212`).

So per-head causal attribution is observable within one checkpoint, but not identifiable modulo `G`. The previous package's statement that per-head `v_j`, `M`, and centered writes are invariant under the architecture's symmetries **is withdrawn**.

**Proposition ID2 (other gauges).** Each of the following preserves `f_θ` and changes the listed quantities:

- `(W_Q, W_K) ↦ (W_Q M, W_K M^{-T})` changes `Q` and `K` vectors; scores are unchanged.
- `(W_V, W_O) ↦ (W_V N, N^{-1} W_O)` changes `V` vectors; head writes are unchanged.
- For ReLU only, positive per-unit rescaling `(W_in[:,k], b_k, W_out[k]) ↦ (s W_in[:,k], s b_k, W_out[k]/s)` changes per-unit weights.
- Splitting a unit into two copies with output weights `t W_out[k]` and `(1−t) W_out[k]` changes per-unit attribution.
- For an orthogonal `R` with `R1 = 1` and uniform LN gains, mapping residuals `h ↦ hR^T` with all weights transformed accordingly changes raw residual coordinates. Reader preactivations are unchanged.

*Proofs.* Direct substitution. For the residual map, `LN(hR^T) = LN(h)R^T` because `R` fixes `1` and preserves norms. *Verified (PI2–PI5).*

**Identifiable objects used below.** Function-level effects of input-level interventions. Reader preactivations as *compositions* "source write → normalization → reader direction", modulo unit permutation (and positive rescaling for ReLU). Attention scores and patterns. Summed same-layer writes. Per-head and per-unit quantities are used only as fixed-parameterization statements and are labeled as such.

---

## 2. Mechanism A: agreement-sensitive MLP

Let `h(t) = φ(t) + φ(−t)` be the even part of the activation, times 2.

### A1 — agreement law (PROVED)

**Class `𝓕_A`.** At the read position, the MLP input is `ζ = LN_2(h_read)`. Define the source coordinates `x = ⟨p, ζ⟩` and `y = ⟨r, ζ⟩` for two reader directions `p` and `r`. The conditions:

- **(A1)** Four units with preactivations `c(x+y)`, `−c(x+y)`, `c(x−y)`, `−c(x−y)`, `c > 0`, zero bias.
- **(A2)** Output vectors `λv`, `λv`, `−λv`, `−λv`, with `λ > 0`.
- **(A3)** `h` is strictly increasing on `[0, ∞)`.
- **(A4)** `m = s·Δu^T(h_read + MLP(ζ)) + κ`, with `s > 0` (the final-LN scale, or 1), `Δu^T v > 0`, `Δu^T h_read = 0` (no direct path), and `κ = 0`.

**Claim.**
1. `sign(m) = sign(xy)` for every real `x, y`.
2. Under ReLU, `MLP·Δu = 2λc·(Δu^T v)·sgn(xy)·min(|x|, |y|)` exactly.
3. Under GELU, `q(x,y) := h(c(x+y)) − h(c(x−y))` behaves like `4√(2/π)·c²xy` for small signals and like `2c·sgn(xy)·min(|x|,|y|)` for large ones.

**Proof.** `m = sλ(Δu^T v)·q`, and `q = h(c(x+y)) − h(c(x−y))`. Because `h` is even and strictly increasing in `|t|`, `sign q = sign(|x+y| − |x−y|)`. Squaring, `|x+y|² − |x−y|² = 4xy`. For ReLU, `h(t) = |t|`, and `|x+y| − |x−y| = 2 sgn(xy) min(|x|, |y|)`. For tanh-GELU, `h(t) = t·tanh(√(2/π)(t + 0.044715t³))`, which is `√(2/π)t² + O(t⁴)` near 0 and `|t| + o(1)` at infinity. ∎

**Corollaries.**
- **A1.1.** `m(x, 0) = m(0, y) = 0`. A single source has no effect; the effect is pure interaction.
- **A1.2.** `m(x, y) = m(−x, −y)`. Agreement on `c'` pushes toward `c`. The rule is **not candidate-relative**.
- **A1.3.** `sign m` is invariant under separate positive rescalings `x ↦ αx`, `y ↦ βy`. So any attention pattern that gives positive weight to both sources, and any LN scale, leaves `sign m` unchanged (*verified* PA3: 50 random clamped patterns).

**Verified.**
- PA1 to PA7 and PA12.
- The positive control gives margins `±3.2142216080` with final LN and `±3.8183907764` without.
- The pre-MLP readout margin is below `1e−16`.
- An independent pure-Python implementation agrees to `1.8e−15`.
- The law holds on 144-cell grids in the fully recomputed continuous family for GELU (both forms), ReLU, SiLU, and softplus. The ReLU formula error is below `1.1e−15`.

**Removing one condition at a time.** Witnesses are found by search on a 60×60 grid (`test_A1_each_removed_condition_has_a_witness`). Grid error fractions are from PA10.

| Removed | Construction | Result |
|---|---|---|
| A1 (symmetric read) | read `2e_1 + e_2` instead of `e_1 + e_2` | 8.3% of grid cells wrong |
| A1 (zero bias) | preactivation offsets `(0.3, 0.3, 0, 0)` | 4.2% wrong |
| A2 (equal output weights) | `(1.3, 1.3, −1, −1)` | 0% on the preregistered grid (**PA10 failed as tested**). Post hoc, a 60×60 grid finds 60 errors, all at `|b| = 0.05`: near the axis, `q → 0.3h(cx) > 0` |
| A3 (even part increasing) | tanh: `h ≡ 0`; sigmoid: `h ≡ 1` | `m ≡ 0` |
| A4 (no direct path) | readout `e_3 + 0.5e_1` | 4.2% wrong; evidence wins when the interaction is weak |
| A4 (`κ = 0`) | final-LN bias with `Δu^Tβ = 0.3` | 4.2% wrong |

**Perturbations (PA11, VERIFIED).** i.i.d. relative weight noise of size `η` on `W_V`, `W_O`, `W_in`, `W_out`, and `W_U`, 20 seeds each:

| `η` | binary cells all correct | grid error rate |
|---|---|---|
| 0.01 | 20/20 | 0 |
| 0.03 | 20/20 | 3.8% |
| 0.1 | 12/20 | 24% |
| 0.3 | 1/20 | 46% |

Errors concentrate where the weaker source is small: the median of `min(|a|,|b|)` at errors is 0.25–0.5. Exact pairing is a measure-zero condition. Approximate pairing is correct only where `|q|` exceeds the perturbation.

### A2 — the binary positive control cannot identify relation computation (PROVED)

Take the sum-only reader with units `±c(x+y)`, output weights `2λv`, and margin offset `−2λ(Δu^Tv)h(2c)`. On the four binary cells it reproduces the designed circuit's margins **exactly**.

*Proof.* The designed circuit gives `q = ±h(2c)`. The sum-only reader gives `2h(S) − h(2c)` with `S ∈ {±2c, 0}`. ∎

*Verified.* The maximum difference is `3.1e−15`. Two implementations with identical decisions on the binary family differ in whether they read `x − y` at all.

### A3 — interaction and agreement contrast do not separate them (PROVED)

For any even `h`, the sum-only function `s(x,y) = h(x+y)` has odd–odd component `E_oo[s] = [s(x,y) − s(−x,y) − s(x,−y) + s(−x,−y)]/4 = q(x,y)/2`, which is exactly half of the designed circuit's.

The finite interaction `I(a,b) = m(a,b) − m(a,0) − m(0,b) + m(0,0)` measures non-additivity of `m` on the chosen scale. For the designed circuit `I = q`. For `s` it is `h(a+b) − h(a) − h(b) + h(0)`, which has sign `sign(ab)` in the small-signal regime (`h ≈ kt²`).

*Proof.* Substitute and use evenness. ∎ *Verified* in tests.

So a nonzero interaction, and even the right sign of the agreement contrast, is produced by an even nonlinearity of **net evidence**, a function of a single scalar. Neither is evidence that a relation between the sources is computed.

### A4 — matched post-normalization sum: what does separate them (PROVED; post hoc after PA9)

Let `Ŝ = ⟨p + r, ζ⟩` and `D̂ = ⟨p − r, ζ⟩` be the *reader's* coordinates. A reader that depends on the sources only through `Ŝ` is constant on every family with fixed `Ŝ`. The designed reader is `q = h(cŜ) − h(cD̂)`. At fixed `Ŝ` it is strictly decreasing in `|D̂|` and changes sign at `|D̂| = |Ŝ|`.

*Proof.* Immediate from A1. ∎

*Verified post hoc.* Matched at `Ŝ = 1.4771`, the sum-only margins are `1.270372` three times, while the designed margins are `1.270, 0, −1.675`.

**The preregistered PA9 failed** because it matched the *raw* sum `x + y` at the residual. The reader's LN divides by `‖h_read‖`, which depends on `x² + y²`. The sum-only margins were then `1.270, 1.122, 0.830`. This failure is kept (`test_A4_raw_sum_matching_is_not_enough…`). It shows that pre-norm LN is itself a non-sum (norm) coupling of the sources. Matched-statistic controls must therefore be matched at the reader's actual input.

---

## 3. Mechanism B: query–key agreement gate

### B1 — QK gate (PROVED for the class; VERIFIED for the construction)

**Class `𝓕_B`** (two layers, positions A, B, read). The conditions:

- **(B1)** Layer 1 delivers to the read residual a feature whose LN-coordinate `x̂` along `f_Q` has `sign x̂ = sign a`. It may also add a direct vote `v_A(a)` toward `c` that depends on `a` only.
- **(B2)** Layer 2 has head scores `s_{rB} = λ x̂ ŷ /√d_h` with `λ > 0`, where `ŷ` is B's LN-coordinate along `f_B` and `sign ŷ = sign b`. The other `n` unmasked scores in the read row equal a constant `s_0`.
- **(B3)** The head's value at B projects to `k ŷ` on `Δu` with `k > 0`. Other positions project to 0.
- **(B4)** No component after this head reads its output, except the linear readout (no final LN, or final LN at a fixed positive scale with `κ = 0`).

**Claim.** `m = v_A(a) + T` with `T = k ŷ w(λ x̂ŷ)` and `w(z) = e^{z/√d_h}/(e^{z/√d_h} + n e^{s_0})`. Then:

1. `w` is strictly increasing in `x̂ŷ`. B's vote counts more the more A agrees with it.
2. `sign T = sign b`, and at fixed `|x̂|` and `|ŷ|`, `|T|` is larger when `ab > 0` than when `ab < 0`.
3. If `ab > 0` and `v_A` has the sign of `a`, then `sign m = sign a`.
4. If `ab < 0`, A's candidate wins iff `|v_A| > k|ŷ| w(−λ|x̂ŷ|)`. Relative to fixed routing at `w(0) = 1/(1+n)`, the decision reverses toward A exactly on the band `k|ŷ| w(−λ|x̂ŷ|) < |v_A| < k|ŷ|/(1+n)`.
5. Substituting the query alone and the key alone gives non-additive effects, because the score is bilinear.

**Proof.** Under (B2)–(B4), the head's output projected on `Δu` is `k ŷ w`. The rest follows from the strict monotonicity of the logistic function. ∎

**Verified.**
- PB1: score `= λx̂ŷ/√2` exactly, other scores 0, `w` monotone.
- PB2: unanimity holds in all agreeing cells.
- PB4: query-only `−4.212`, key-only `−4.212`, both together `0.000`; the non-additivity is `+8.424`.
- PB5: 242 conflict cells reverse relative to fixed routing.
- PB8: with two B sources, attention concentrates on the one that agrees with A (0.99 against 0.00).
- PB1 and the test check the exact margin formula `m = v_A + 2kŷw` to `1e−12`.

**Dissection (PB3; mechanism-dissection interventions).** With the layer-2 pattern clamped to its `a = 0` value, B's vote varies by 0.104 across `a`, against 4.238 when fully recomputed. The residual 0.104 is **not zero**. Layer 1 also copies A into B's residual, which changes B's LN scale and hence `ŷ`. That is an LN coupling the structural class idealizes away. With values clamped, the spread is 4.651: routing carries the effect.

**Negative controls.**
- `W_Q = 0` in layer 2 (PB6): relation-blind under RB. It is not exactly additive (residual 0.208), because of that same LN coupling.
- B placed after the read position (PB7): `b` has no effect, to `1e−16`.

**Counterexample to the decision-level extension (PB9).** A layer-2 MLP that re-reads `Δu` and subtracts above a threshold produces 26 cells in which the gated head's direct change and the total margin change have opposite signs. Example: `a: −1 → 0.5` at `b = 0.5`, head change `+1.905`, margin change `−7.315`. Final LN with bias produced **no** such cell on the tested grid. B1 is therefore a theorem about the class with no downstream re-reader (B4). Without B4, the head-level identity does not determine the decision.

---

## 4. Mechanism C: two-block reconciliation

### C1 — agreement-gated amplification of net evidence (PROVED for the class)

**Class `𝓕_C`.** Block 1 computes an agreement signal as in `𝓕_A`, written to a direction `e_3` that `Δu` does not read. Block 2's MLP reads `Ŝ = ⟨e_1 + e_2, ζ_2⟩` and `q̂ = ⟨e_3, ζ_2⟩`, with units

`ReLU(βŜ + γq̂ − θ)`, `ReLU(−βŜ + γq̂ − θ)` writing `±λ_2 Δu/‖Δu‖²`, and `ReLU(±εŜ)` writing `±Δu/‖Δu‖²`,

where `β, γ, θ, λ_2, ε > 0`. There is no other path to `m`, and the margin has a prior offset `−κ_0`.

**Claim.** Write `g(Ŝ, q̂) = ReLU(βŜ+γq̂−θ) − ReLU(−βŜ+γq̂−θ)`, so that `m = −κ_0 + εŜ + λ_2 g(Ŝ, q̂)`.

1. `g` is odd in `Ŝ` and `sign g ∈ {0, sign Ŝ}`. With `κ_0 = 0`, `sign m = sign Ŝ`: agreement never reverses the net-evidence direction.
2. At fixed `Ŝ`, `|m + κ_0|` is nondecreasing in `q̂`, and strictly increasing wherever the unit with the sign of `Ŝ` is active. Agreement amplifies the candidate favored by net evidence. **This is candidate-relative.**
3. Let `m^{ag}` and `m^{dis}` be the margins with `κ_0 = 0` of a matched-`Ŝ` pair with `q̂_ag > q̂_dis`. For any prior `κ_0` in `(m^{dis}, m^{ag})` (when `Ŝ > 0`), the decision is `c` under agreement and `c'` under disagreement, at fixed net evidence.
4. At fixed `Ŝ`, `q̂` is the only block-2 input that varies with agreement. Ablating block 1's MLP sets `q̂ = 0` and removes the amplification.

**Proof.** For `Ŝ > 0`, the first ReLU argument is at least the second, and ReLU is monotone, so `g ≥ 0`. Oddness follows from swapping the two units. `∂g/∂q̂ = γ(1[βŜ+γq̂>θ] − 1[−βŜ+γq̂>θ]) ≥ 0` for `Ŝ > 0`, and the case `Ŝ < 0` is symmetric. Claim 3 is the intermediate value property of a constant shift. ∎

**Verified.**
- PC1: `sign m = sign Ŝ` on 156 grid cells.
- PC2: matched-`Ŝ` pairs have `m^{ag} = 4.187` against `m^{dis} = 0.243`. With `κ_0 = 2.215` (the midpoint, chosen to *demonstrate* claim 3, not predicted), the decision is `c` under agreement and `c'` under disagreement.
- PC3: interchanging block 1's MLP output from the agreeing run into the disagreeing run moves `m` from `0.243` to `3.928` (the agreeing run gives `4.187`). The remainder is LN-scale coupling. Ablating block 1's MLP shrinks the agreement gap at fixed `Ŝ` from `3.944` to `0.182`.
- PC5: an unread residual component flips no sign.

### C2 — bottleneck screening (PROVED, by SS*)

If block 1 writes both sources onto `e_1 + e_2` (identity lost), the read residual depends on the sources only through `x + y`. Every matched-sum pair then gives identical margins. *Verified* (PC4): the maximum difference is `1.8e−15`, with the pairs matched exactly in input space by inverting the source LN.

### C3 — the I/O function does not identify the two-block mechanism (VERIFIED, with a sketch of exactness)

A single wide ReLU layer on `(x, y)`, with 300 random features and a least-squares fit, approximates the C margin function with held-out relative RMSE `0.032` and sign agreement `0.923` (PC6). It has no `q` variable.

*Exactness sketch.* A one-hidden-layer ReLU network of `(x, y)` is smooth off a finite union of lines. The C margin has a kink on `{βŜ + γq̂ = θ}`, the zero set of a non-affine analytic function, which contains no line segment generically. So exact representation by one ReLU layer fails, while approximation succeeds. Depth gives an exact *implementation*, but behavior on the tested family does not reveal it. *Status: sketch; not a full proof.*

---

## 5. Attempted causal-bridge theorem (mandate §8)

**Strongest candidate.** Classes `𝓕_B` and `𝓕_C`, "agreement-gated evidence". They are candidate-relative, unlike `𝓕_A`.

- **Model class:** `𝓕_B ∪ 𝓕_C` as above.
- **Margin:** `m_c = L_c − L_{c'}`.
- **Source representations:** the reader coordinates `x̂, ŷ` (B) or `(Ŝ, D̂)` (C). These are compositions "input token → LN → write → LN → reader direction", identifiable modulo the groups of ID2.
- **Relation:** `C = x̂ŷ` (B, the bilinear score), or `C = q̂ = h(cŜ) − h(cD̂)` (C).
- **Operator:** a softmax gate on a bilinear score (B); a ReLU gate on `(Ŝ, q̂)` (C).
- **Interventions:** input substitution at the source slots, recomputed.
- **Assumptions:** (B1)–(B4) or the `𝓕_C` conditions. All are conditions on weights and LN-coordinates, checkable without the response.

**Implication.**
- B1.1–B1.4: agreement increases the weight of the agreeing source's vote toward *that source's* candidate. In conflict there is a characterized reversal band relative to fixed routing.
- C1.1–C1.3: agreement amplifies the net-evidence candidate at fixed net evidence, and there is a prior interval in which agreement decides the candidate.

**The five checks on the assumptions.**

1. **Independently identified.** Yes, in principle: they are weight and composition conditions, modulo `G`.
2. **Computationally meaningful.** Yes.
3. **Plausibly present in a Transformer.** QK composition is common. The exact gating structure is a measure-zero design condition. Approximate versions hold only where the gated quantity exceeds the deviation (PA11). Not established in any trained model.
4. **Noncircular.** Yes. No response is used.
5. **Not the output sign in disguise.** **Partly fails.** The assumptions say, in weight terms, "an agreement gate exists". The conclusions then follow by monotonicity in one or two lines.

**Classification.** **Implemented-mechanism structural theorems.** They are not an independently discovered decision theory. The decision-level extension to models with downstream re-readers is refuted (PB9). In the classes as stated it holds, but only because (B4) excludes those re-readers.

---

## 6. Register

| ID | Status |
|---|---|
| SS* | PROVED |
| ID1, ID2 | PROVED (verified) |
| A1 and corollaries | PROVED |
| A1 condition witnesses | VERIFIED (construction) |
| A2, A3 | PROVED |
| A4 | PROVED (post hoc) |
| PA9 (raw-sum matching) | FAILED prediction, kept |
| PA10 (A2 witness on the preregistered grid) | FAILED as tested; post-hoc witness found |
| B1 | PROVED (class); verified construction; dissection residual 0.104 from LN coupling |
| PB9 | COUNTEREXAMPLE (downstream re-reader) |
| C1, C2 | PROVED |
| C3 | VERIFIED; exactness is a sketch |
| Bridge (§5) | implemented-mechanism structural theorem; fails check 5 in part; not identified in trained models |
