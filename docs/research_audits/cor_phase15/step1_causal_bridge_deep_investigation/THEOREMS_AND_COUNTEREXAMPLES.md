# Theorems and counterexamples

Implementation: `src/cbmath.py`. Experiments: `src/experiments.py`, with the log in `logs/experiments.txt`. Tests: `tests/test_deep.py` (31 tests). This version incorporates the separate-context reviews `reviews/R1–R4`. Every change made in response to them is listed in `ADJUDICATION.md`.

Status labels:
- **PROVED**: a full proof is given here.
- **VERIFIED**: checked on constructions only.
- **FAILED / COUNTEREXAMPLE**: kept on record.

A numerical test is not a proof of a universal claim.

**Common setting.** Pre-norm residual Transformer layers: LN, multi-head causal attention, residual add, LN, MLP, residual add. An optional final LN with `ε > 0` precedes a linear unembedding. The margin is `m = L_c − L_{c'}`, read at the read position.

- A **recomputed intervention** replaces one named site and recomputes everything downstream. Sites are the input token, `Q`, `K`, `V`, scores, pattern, head outputs, `mid`, MLP preactivations or output, and `final`.
- A **mechanism-dissection** intervention clamps one site while another changes. It is always labeled as such.
- Several residual-site interventions (PA8; PA9 post hoc) set states that no input reaches, for example `x = 3 > √6`. Where this matters, the result is repeated using input substitution only.

---

## 1. Corrected prerequisites

### SS*: same-cut screening, no bypass (PROVED)

**Hypotheses.**
- Writers `c_1, …, c_K` and a base `B` feed `S = B + Σ_k c_k`.
- `B` is not a descendant of any `c_k`. Review R2's example: with `B = c_1²`, the edit `(1,0) → (2,−1)` keeps `Σc` fixed and moves `Y` from 2 to 5.
- `Y = F(S, U)`, where no variable in `U` descends from any `c_k` except through `S`. That is, there is no bypass.

**Claim.** If `Σ_k δ_k = 0`, then `Y` is unchanged. Any relation among the writers that is not a function of their sum is inert for `Y` once `S` is fixed.

**Proof.** `S`, `B` and `U` are all unchanged. ∎

**The earlier wording was wrong.** It allowed `Y = S + c_1`. Under the edit `(2,−1) → (5,−4)`, that `Y` moves from 3 to 6 while `S` stays fixed.

**Transformer instance.** The heads of one layer are consumed only through the residual add, so there is no bypass. SS* therefore holds for zero-sum head-output edits (tested to `1e−12`). This does not depend on the downstream map being linear.

### ID: identification (corrected after R2, R3, R4)

**Definitions.** Let `θ` and `θ'` be parameterizations, possibly of different widths (after a split, for example), with `f_θ = f_{θ'}`. A site correspondence `π` maps sites of `θ` to sets of sites of `θ'`: a split unit maps to its two copies, a permuted unit to its image.

1. **Fixed-parameterization observability.** `Q(θ, x)` can be computed from `θ` and the activations of `f_θ`.
2. **Causal intervention attribution.** `Q` is an interventional contrast on a site of `θ`.
3. **Identifiability modulo `𝓖`.** `Q(θ, x) = Q^π(θ', x)` for every function-preserving pair `(θ, θ')` in a stated family `𝓖`, where `Q^π` is `Q` computed through `π`. `𝓖` is a set of maps, not necessarily a group, because splitting changes width.
4. **Function-level.** `Q` depends only on `f_θ` restricted to an input intervention family. Such a quantity is identifiable modulo every `𝓖`.

**ID1: head redistribution (PROVED).** If heads `h_1` and `h_2` share `W_Q`, `W_K` and `W_V`, then replacing `W_O^{h_1}` by `W_O^{h_1} + E` and `W_O^{h_2}` by `W_O^{h_2} − E` preserves `f_θ`. Per-head writes, direct effects, consensus scalars and patching effects all change. This needs identical heads, so it is a **non-generic** case (R4).

*Verified (PI1).* Logits change by at most `4.4e−16`. Per-head patch effects move from `[−2.106, −2.106]` to `[−1.699, −2.514]`.

**ID2: further maps that preserve `f_θ` (PROVED by substitution; verified).**

| Map | What it changes |
|---|---|
| QK gauge `(W_Q M, W_K M^{−T})` | `Q` and `K` vectors |
| VO gauge `(W_V N, N^{−1} W_O)` | `V` vectors |
| Positive rescaling of a ReLU unit | per-unit weights |
| Splitting a unit into copies | per-unit attribution |
| Orthogonal `R` with `R1 = 1` acting on the residual | raw residual coordinates; summed writes rotate with it (R2) |
| Folding a nonuniform LN gain, even one with mixed signs, into the next weights (R3) | LN coordinates of fixed directions. The gain is itself a gauge; the earlier claim that non-uniform gains "shrink the group" is withdrawn |
| Row-constant score shift (R2, R3): an LN bias along `1` plus a `W_K` shift, or a constant key component | every score in a row moves by the same input-dependent amount; patterns are unchanged |
| Cancelling linear units (R3): since `φ(t) − φ(−t) = t` for GELU, ReLU and SiLU, a pair of units on `±w` with opposite output weights is an exact linear map | a reader can be made to "read `x − y`" with any coefficient; an exact zero-output combination can be added |
| Emptying a duplicated head | the emptied head's pattern becomes arbitrary |
| Redistributing ReLU linear pairs (R2) | C's `ε`-units can read `x̂` and `ŷ` separately instead of `Ŝ` |

**Identifiable quantities used in this package (corrected).**
- Function-level input effects.
- **Within-row score differences** `s_{rj} − s_{rk}`, and **pattern mass over positions whose value writes differ.** Raw scores and raw patterns are not identifiable.
- **The even part of the reader, grouped by read direction.** For each direction pair `±w`, this is `φ(⟨w,ζ⟩) + φ(−⟨w,ζ⟩)` times its output weight, together with the matched-`Ŝ` contrast of A4. Raw preactivations, and which directions a given unit reads, are not identifiable.
- **Products and readout-normalized signs only.**
  - B: `x̂ŷ` is identifiable as a score difference. `x̂` and `ŷ` separately are not, because the QK sign gauge flips both.
  - A: `sign(xy)` is identifiable only after fixing each unit pair's scale by its output along `Δu`, and only up to a joint sign flip. Otherwise ReLU rescaling changes the recovered coordinates, and 44 of 144 cells come out wrong (R3).
  - C: `q̂` is identifiable up to a positive scale.
- **Source polarity**, meaning which sign of a source "supports `c`", is **not** identified internally (R3).
  - In A, a single source has exactly zero effect, so polarity is purely a convention of the input coding.
  - In B and C, polarity is fixed only by the source's direct vote path to `Δu`, which is the outcome pathway.
  - All claims are therefore stated relative to the designer's input semantics.

---

## 2. Mechanism A: agreement-sensitive MLP

Throughout, `h(t) = φ(t) + φ(−t)`.

### A1: agreement law (PROVED; hypotheses strengthened after R1 S1 and R2 S5)

**Class `𝓕_A`.**
- **(A0) Source separation.** The MLP input at the read position is `ζ = LN_2(h_read)`. There are reader directions `p, r` with `x = ⟨p,ζ⟩ = f_A(a)` and `y = ⟨r,ζ⟩ = f_B(b)`, where each `f` preserves sign and depends on one source only. Also `Δu^T h_read = 0`.
- **(A1)** Four units with preactivations `±c(x+y)` and `±c(x−y)`, with `c > 0` and zero bias.
- **(A2)** Output vectors `λv, λv, −λv, −λv`, with `λ > 0` and `Δu^T v > 0`.
- **(A3)** `h` is strictly increasing on `[0, ∞)`.
- **(A4)** `m = s·Δu^T(h_read + MLP(ζ)) + κ` with `s > 0` and `κ = 0`, and nothing else writes along `Δu`. With a final LN present, `Δu ⊥ 1` and the gain is uniform; otherwise `s = 1`.

**Claim.**
1. `sign m = sign(ab)` for all real `a, b`.
2. For ReLU, `MLP·Δu = 2λc(Δu^T v)·sgn(xy)·min(|x|,|y|)` exactly.
3. For GELU, `q ≈ 4√(2/π)c²xy` for small signals and `q ≈ 2c·sgn(xy)·min(|x|,|y|)` for large ones.

**Proof.** First, `m = sλ(Δu^T v)[h(c(x+y)) − h(c(x−y))]`. Because `h` is even and increasing in `|t|`, the sign of this is `sign(|x+y| − |x−y|)`. Since `|x+y|² − |x−y|² = 4xy`, that equals `sign(xy)`. By (A0), `sign(xy) = sign(ab)`. For ReLU, `h(t) = |t|`. For tanh-GELU, `h(t) = t·tanh(√(2/π)(t + 0.044715t³))`, which gives both regimes. ∎

**Corollaries.**
- **A1.1.** `m(a,0) = m(0,b) = 0`: either source alone has no effect.
- **A1.2.** `m(a,b) = m(−a,−b)` in sign, and in value when `κ = 0` and the final LN has no bias. The rule is not candidate-relative.
- **A1.3.** `sign m` is unchanged by separate positive rescalings of `x` and `y`. In particular, any single attention pattern that gives positive weight to each source leaves it unchanged, **provided (A0) holds.**

**Counterexample to A1.3 without (A0) (R1 S1).** With two heads and moving attention, the second head's routing can flip the sign of `x` as a function of B. The circuit then follows A alone: 50% error against `sign(ab)`, and 0% against `sign a`. A single head with moving attention keeps the law (R1: 0 errors in 7,200 cells), but the source weights can fall to about `10^{−76}`, so the margin all but vanishes.

**Verified.**
- Positive-control margins of `±3.2142216080`, reproduced by three independent implementations.
- 144-cell grids for GELU (both forms), ReLU, SiLU and softplus, including with a final LN.
- The ReLU formula, to `1.1e−15`.

**Removing one condition at a time.** Each witness was found by search.

| Condition removed | Result |
|---|---|
| symmetric read directions (skewed read) | 8.3% grid errors |
| zero bias (read bias added) | 4.2% |
| no direct path | 4.2% |
| `κ = 0` (LN bias added) | 4.2% |
| (A3), with an odd activation | `m ≡ 0` |
| equal output weights | 60 errors at `|b| = 0.05`, found post hoc. **PA10 failed as tested**, because the preregistered grid was too coarse to reach them |

**Perturbations (descriptive).** PA11's preregistered test could not fail, so it is reclassified as **non-discriminating** (R1 S5). With relative noise 0.03, the binary cells stay correct in 20 of 20 seeds. Grid error rates are 3.8% at noise 0.03, 24% at 0.1, and 46% at 0.3.

### A2: binary indistinguishability (PROVED; scope corrected after R2 S6)

Assume no final LN and `h(0) = 0`. Take a reader of net evidence only: units `±c(x+y)`, output weights `2λv`, and a margin offset of `−2λ(Δu^T v)h(2c)`. On the four binary cells it reproduces the designed circuit's margins exactly (to `3.1e−15`).

**Scope.**
- With a final LN the equivalence fails: in the agreeing cells the margins are 3.214 against −0.067 (R2).
- With softplus, `h(0) ≠ 0` and the offset must change.
- The weight-level statement that the two "differ in whether they read `x − y`" is not invariant. Cancelling linear units can make the sum-only model "read" `x − y`, with a direct effect of about 9.5, without changing its function (R3). What is invariant is the grouped even part, and the A4 contrast.

### A3: interaction and the odd–odd component do not separate them (PROVED)

For even `h`, `E_oo[h(x+y)] = q/2`. Equivalently, `q` is exactly twice the odd–odd component of the single-index function `h(x+y)` (R4).

So no contrast built from sign flips can separate the designed circuit from a function of net evidence. The interaction `I(a,b) = m(a,b) − m(a,0) − m(0,b) + m(0,0)` measures non-additivity on a fixed scale. For `h(a+b)` it has the sign of `ab` in the small-signal regime. Strictly, `h(x+y)` is a nonlinear function of one additive index, not a removable interaction in Loftus's sense (R4).

### A4: matched post-normalization sum (PROVED; conditions added after R1)

Let `Ŝ = ⟨p+r, ζ⟩` and `D̂ = ⟨p−r, ζ⟩`. If a final LN is present, assume `Δu ⊥ 1` and a uniform gain.

**Claim.** A reader that depends on the sources only through `Ŝ` is constant on any family with `Ŝ` held fixed. The designed reader `h(cŜ) − h(cD̂)` decreases strictly in `|D̂|` at fixed `Ŝ ≠ 0`, and changes sign at `|D̂| = |Ŝ|`. At `Ŝ = 0`, `q ≤ 0` and there is no sign change.

**Verified on reachable inputs**, by input substitution only, with no residual hook:

| Reader | Margins at matched `Ŝ` |
|---|---|
| designed | `0.307`, `0`, `−0.720` |
| sum-only | `0.307412`, three times |

**The preregistered PA9 failed, and is kept.** It matched the raw residual sum, but the reader's LN rescales by the residual norm, so the margins were `1.270, 1.122, 0.830`.

---

## 3. Mechanism B: query–key agreement gate

### B1 (PROVED for the class; corrected after R1 S3 and R2 S3)

**Class `𝓕_B`.**
- **(B1)** Layer 1 gives the read token a feature `x̂` with `sign x̂ = sign a` (up to the joint QK sign gauge), and a direct vote `v_A(a)` with `sign v_A = sign a`.
- **(B2)** In layer 2, the read-row score difference is `s_{rB} − s_0 = λx̂ŷ/√d_h`, with `λ > 0` and `sign ŷ = sign b`. The other `n` unmasked scores in the row all equal `s_0`.
- **(B3)** The head's value at B projects to `kŷ` on `Δu`, with `k > 0`. At every other position it projects to 0.
- **(B4)** `m = v_A + kŷw` exactly. **No other component writes source-dependent content along `Δu`**: R1 S3 shows one extra head with `W_Q = 0` breaks unanimity in 348 of 450 agreeing cells. No downstream component reads the head's output, and there is no final LN.

**Claim.** The gate weight depends only on score differences:

`w = e^{λx̂ŷ/√d_h} / (e^{λx̂ŷ/√d_h} + n)`.

1. `w` is strictly increasing in `x̂ŷ`.
2. `sign(kŷw) = sign b`. At fixed `|x̂|` and `|ŷ|`, B's vote is larger when `ab > 0`.
3. **Unanimity.** If `ab > 0`, then `sign m = sign a`.
4. **Conflict.** A wins iff `|v_A| > k|ŷ|w(−λ|x̂ŷ|)`. Compared with fixed routing at `w_0 = 1/(1+n)`, the decision reverses toward A exactly on the band `k|ŷ|w(−λ|x̂ŷ|) < |v_A| < k|ŷ|w_0`.
   - In raw-score form with `s_0 ≠ 0`, `w_0 = 1/(1+n e^{s_0 − s_{rB}(0)})`.
   - If the row shift pushes the gated weight above `w_0`, the reversal goes toward B instead. R2 S3: with `n = 2` and `s_0 = −3`, the gated weight is 0.787 against 1/3.
5. Substituting the query site alone, or the key site alone, gives non-additive effects. **At the input level, however, every sign-symmetric square `(±a, ±b)` has exactly zero four-term interaction** (R2): `v_A` cancels, and the four `kŷw` terms cancel in pairs. So 2×2 sign-flip designs cannot detect the gate. Designs with unequal levels can: PB6 finds the gated table relation-sensitive.

**Proof.** The logistic function is strictly monotone; the rest is direct substitution. ∎

**Verified.**
- The score is exactly bilinear, and `w` is monotone.
- The margin formula holds to `1e−12`.
- Unanimity holds; 242 grid cells reverse; under competition the weights go 0.99 against 0.00.
- Dissection: with routing fixed, B's vote spreads by 0.104 across `a`, against 4.238 when fully recomputed.
- Robust to perturbation: unanimity holds in 20 of 20 seeds at noise 0.03 and 18 of 20 at 0.1, and the reversal band is non-empty in 20 of 20.

**Failures and counterexamples.**
- **PB6 failed** on its additivity part after the registered rescoring: residual 0.208, caused by LN coupling at B. Its relation-blindness part holds.
- **PB9.** A downstream MLP that re-reads the head produces 26 cells where the head's change and the total change have opposite signs. A final LN with bias gives 18 of 450 unanimity failures (R1).
- **Three or more candidates (R1 S2).** The gate's relation is an inner product of the two sources' feature vectors, which is not the same as "both prefer the same candidate". Example: both sources prefer `c_1`, but `⟨x,y⟩ = −1.70`, so the gate cuts B's weight from 0.333 to 0.003. R1 gives a full-block case in which the decision flips to `c_2`. **Candidate-relativity is established only for a binary candidate pair with scalar evidence.**

---

## 4. Mechanisms C and C′: multi-block

### C1: gated amplification of net evidence (PROVED for the class; C1.2 corrected after R2 S1)

**Class `𝓕_C`.** Block 2's MLP reads `Ŝ` and `q̂` through units `ReLU(±βŜ + γq̂ − θ) → ±λ_2Δu/‖Δu‖²` and `ReLU(±εŜ) → ±Δu/‖Δu‖²`. There is no other path to `Δu`. A prior `−κ_0` enters through the bias.

Let `g = ReLU(βŜ+γq̂−θ) − ReLU(−βŜ+γq̂−θ)`.

**Claim.**
1. `g` is odd in `Ŝ`, and `sign g ∈ {0, sign Ŝ}`. With `κ_0 = 0`, `sign m = sign Ŝ`.
2. At fixed `Ŝ`, `|m + κ_0|` is nondecreasing in `q̂`. It increases strictly **iff exactly the unit with the sign of `Ŝ` is active.** When both units are active, `g = 2βŜ` does not change with `q̂` (R2: `g = 1.8` at `q̂ = 1.1, 1.2, 1.3`).
3. Take `Ŝ > 0`, a pair matched exactly at `Ŝ` with `q̂_ag > q̂_dis`, and any prior `κ_0` between `m^{dis}` and `m^{ag}`. Then agreement decides `c` over `c'`.

**Proof.** ReLU is monotone. Swapping the two units gives oddness. The derivative in `q̂` is `γ(1[·] − 1[·])`. ∎

**Verified, with pairs matched exactly at the block-2 reader.** This corrects R1 S6 and R2 S2: the earlier pairs were matched at block 1, repeating the PA9 mistake.
- At `Ŝ = 1.0795`, the agreeing pair gives `m = 4.588` and the disagreeing pair `m = 0.270`, a gap of 4.318.
- **With block 1's MLP ablated, the gap is exactly 0.0000.** The earlier figure of 0.182, attributed to LN coupling, was a matching error.
- `sign m = sign Ŝ` in all 156 cells.
- Interchanging block 1's MLP output moves `m` from 0.243 to 3.928.

**Robustness and boundaries.**
- **C's sign law is fragile.** It holds across the whole grid in 0 of 20 seeds at noise 0.03, because perturbed output weights add generic paths. Amplification is robust: 20 of 20 at 0.03, 17 of 20 at 0.1.
- **Outside the class**, a final LN with bias gives 34 sign errors. A third block that re-reads `Δu` can make agreement decide *against* net evidence (`m = −8.02`, R1).
- **The direction is not architectural.** The mirrored gate (`γ < 0`) amplifies under disagreement instead: 3.185 against 0.246.
- **At `κ_0 = 0`, C's decision is exactly `sign Ŝ`**, which a single sum-only unit reproduces (R2). Agreement affects C's decision only through a prior (C1.3). Otherwise it is a stability, or confidence, property.

### C2: bottleneck screening (PROVED, by SS*)

If both sources are written onto one direction, every matched-sum pair gives identical margins (to `1.8e−15`).

### C3: what the I/O function identifies (revised after R2 and R3)

- **Exact non-representability (PROVED).** A one-hidden-layer ReLU network of `(x, y)` is piecewise affine. The C margin is not: it has nonzero second differences outside a measure-zero set. So no one-layer ReLU network represents it exactly, and the exact I/O function *does* distinguish the two implementations. This replaces the earlier kink-curve sketch, which R2 found incomplete.
- **Approximation.** A 300-unit random-feature fit reaches relative RMSE 0.032, but its sign agreement with C is only 0.923. Wide networks can approximate almost anything. The earlier headline, "the I/O function does not identify the mechanism", is withdrawn. The correct statement is that finite-precision behavior on a finite intervention family does not identify it.
- **Within C's own architecture**, R3 found no exact reparameterization that removes `q̂`.
- That computation can move between attention and MLP "only approximately" in general is a **CONJECTURE**.

### C′: conditional rereading through attention (post-review; addendum predictions)

**Construction.** Block 1 computes `q` with the Mechanism-A pairing. In block 2, a head's **query reads `q̂`**. Its keys are positive at the source positions and negative at a prior token P. When the sources agree, the head attends to them and copies their votes. When they conflict, it attends to P, which carries a vote for `c'`.

**C′1 (PROVED for the construction's structural form; VERIFIED).** In agreeing cells, `sign m = sign a`. In conflicting cells, `m < 0` whichever source is stronger: the margin stays at −2.631 as the stronger source grows from 0.6 to 4.0 (PR1, PR5).

**C′2: depth is necessary for relation-dependent routing (PROVED).** In a single pre-norm attention layer, the read row's score for position `j` is `⟨Q(LN x_r), K(LN x_j)⟩`, which depends only on `x_r` and `x_j`. Therefore `log(α_A/α_P) = s_{rA} − s_{rP}` does not depend on `x_B`. No single attention layer can make the allocation between A and P depend on how A relates to B. Verified with 30 random heads: the spread is at most `3.6e−15`. The mechanism requires an earlier layer to write a relation-dependent feature into the read token's query (Q-composition).

**C′3: mediation (VERIFIED).** Interchanging block 1's MLP output from an agreeing run into a disagreeing run flips the decision (from −2.631 to +1.879) and moves the source attention mass from 0.000 to 1.000. Ablating block 1's MLP makes block 2's routing identical whether or not the sources agree (uniform 0.25).

**PR2 failed, and is kept.** Across cells, source mass is not monotone in `q̂`. The score is `λq̂·k̂_j`, and the key coordinates `k̂_j` vary with each source's LN scale, so `q̂` alone does not order the routing.

---

## 5. Attempted causal-bridge theorem (mandate §8): rejected as an explanatory bridge

**The strongest candidate is agreement-gated evidence:** classes `𝓕_B`, `𝓕_C` and C′.

| Element | B | C | C′ |
|---|---|---|---|
| model class | `𝓕_B` | `𝓕_C` | C′'s structural form |
| source representations | `x̂`, `ŷ` | `Ŝ`, `D̂` | `Ŝ`, `D̂` |
| relation | `x̂ŷ` | `q̂ = h(cŜ) − h(cD̂)` | `q̂` |
| operator | softmax gate on a bilinear score | ReLU gate on `(Ŝ, q̂)` | query-composed attention routing |
| intervention family | input substitution | input substitution | input substitution |
| proved implication | B1.2–B1.4 | C1.1–C1.3 | C′1 |

The assumptions are the class conditions, and each implication is proved inside its class.

**The mandate's five checks.**
1. **Independently identified: no.** Source polarity is fixed only through the outcome pathway, and the recovered coordinates depend on the parameterization (ID).
2. **Computationally meaningful: yes.**
3. **Plausibly present in a Transformer: not established.** The gating structures are measure-zero design conditions. C's sign law is fragile. B is robust within its own class but breaks with one additional parallel writer.
4. **Noncircular: yes**, in the narrow sense that no response is used to define anything.
5. **Not merely the desired output sign in disguise: fails.** Every directional conclusion comes from a sign hypothesis: `sign x̂ = sign a`, `k > 0`, the signs of the unit outputs, `Δu^T v > 0`, and the key signs at the sources and at P. What remains is a restated threshold (R2). The mirrored circuits are just as easy to implement.

**Verdict.** The theorem is **rejected as an explanatory causal bridge** under mandate §8. In the mandate's own terms, these are **implemented-mechanism existence theorems**: deliberately programmed gates produce relation-dependent decisions, and the boundaries of each class are proved.

---

## 6. Register

| ID | Status |
|---|---|
| SS* | PROVED |
| ID1, ID2, corrected identifiable list | PROVED |
| A1, with (A0) | PROVED |
| A1.3 without (A0) | COUNTEREXAMPLE (R1) |
| A2 (no final LN, `h(0) = 0`) | PROVED; fails with a final LN |
| A3, A4 | PROVED |
| PA9 | FAILED (kept) |
| PA10 | FAILED as tested (kept) |
| PA11 | NON-DISCRIMINATING |
| B1, with (B4) | PROVED |
| PB6 | FAILED (additivity part) |
| PB9 | COUNTEREXAMPLE |
| three-candidate gate | COUNTEREXAMPLE (R1) |
| C1 (C1.2 corrected), C2 | PROVED |
| C3 | revised: exact non-representability PROVED; non-identification from finite behavior VERIFIED |
| C′1 | VERIFIED |
| C′2 | PROVED |
| C′3 | VERIFIED |
| PR2 | FAILED (kept) |
| bridge (§5) | rejected as explanatory; implemented-mechanism existence theorems |
| original preregistration, rescored | 29 passed, 3 failed, 1 non-discriminating, out of 33 |
| addendum | 4 of 5 passed |
