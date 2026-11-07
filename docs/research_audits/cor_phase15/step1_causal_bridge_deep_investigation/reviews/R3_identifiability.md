# R3 — Identifiability and reparameterization (adversarial review)

Scope: THEOREMS §1 (ID, ID1, ID2, "identifiable objects"), §2 A2–A4, §4 C3, §5; IDENTIFIABILITY_AND_REPARAMETERIZATION.md (IAR); `src/cbmath.py` gauge_*; PI1–PI5. Attack scripts: `/tmp/r3/attack.py` and `/tmp/r3/gamma.py`. They import `src/cbmath.py` read-only, and no package file was modified. Identifiability tests pass (3/3).

Verdict: the three levels are correctly defined, and function-level quantities are correctly kept separate. The list of "identifiable objects" is **over-claimed**. It holds only for the hand-picked `G` of ID2, and five exact function-preserving maps outside that `G` break it. The package's answer to the mandate's §9 question does not address the actual consistency quantities. The polarity of each source, which is what makes the relation "agreement", is fixed by convention or by the readout. It is not identified.

---

## OK

- **ID definitions.** The three levels (fixed-parameterization observability, intervention attribution given θ, invariance modulo `G`) are correct. Function-level quantities are correctly shown to be invariant under every `G`. The per-head and per-unit withdrawals (ID1, PI1) are correct and reproduce.
- **ID1 and ID2 proofs.** Correct by substitution. PI1–PI5 reproduce at ≤1.1e−15.
- **Input-level sign law A1 and A4's matched-Ŝ contrast.** These are function-level, given an input coding and a fixed input family. They survive every attack below, including (3), where the sum-only model is rewritten to "read x−y" but stays constant on fixed-Ŝ families.
- **B's relation `x̂ŷ`, taken as the within-row score contrast `s_rB − s_0`.** Invariant under attacks (1) and (7) and the γ-fold (6b). This is the right invariant for B.

## SERIOUS

**S1. "Attention scores and patterns" are not identifiable, even at the weight level, beyond the QK gauge.** Location: THEOREMS §1 "Identifiable objects"; IAR table row 4.
- (1) *Row-shift gauge.* Set layer-2 `b1 += 2·1` in Mech B. This is inert because all `W_Q`, `W_K`, `W_V` columns are mean-zero. Then set `W_K[:,1] = 1/d` so that every key gets the constant 2, and `W_Q[:,1] = 5 f_Q`. Logits change by 1.8e−15. Read-row scores move from `[0, −2.964, 0]` to `[9.391, 6.427, 9.391]`, and the shift depends on the input. The pattern changes by 4e−16. The same trick works with `d_h = 1` (add `u/d` to `W_K`).
- (2) *Zero-OV head.* Apply ID1 with `E = W_O^2`, then randomize head 2's `W_Q` and `W_K`. Logits are identical (0.0). Head 2's read-row pattern goes from `[.005, .99, .005]` to `[.998, .002, 0]`.
- *Argument, not built exactly:* in Mech B, positions A and read both have `V = 0`. The split of `1 − w_B` between them therefore has no functional meaning.
- *Correction.* Replace "scores" with "within-row score contrasts". Replace "patterns" with "pattern mass on classes of positions whose OV writes differ in the subspace that downstream components read, for heads with nonzero OV". B's claims depend only on `s_rB − s_0` and `w_B`, so they survive once restated this way.

**S2. MLP preactivations are not identifiable "modulo permutation and ReLU rescaling".** Location: THEOREMS §1, IAR row 28, A2 ("differ in whether they read x−y at all").
- (3) For GELU, ReLU and SiLU, `φ(t) − φ(−t) = t` exactly. Six units on `±c(p−r), ±c r, ±c p`, with outputs `(v, −v, v, −v, −v, v)` along `Δu`, sum to 0 because `(p−r) + r − p = 0`.
- Adding them to the **sum-only** model leaves logits unchanged on 169 cells: GELU 1.7e−15, ReLU 1.5e−15, SiLU 1.9e−15. The model now has a unit reading `x−y` with direct logit attribution (DLA) ≈ 9.5. ID2's list (split, rescale, permute) does not contain this. These are irreducible groups of linearly dependent odd pairs.
- *Correction.* The weight-level invariant is the **even-ridge decomposition**. For each direction class `{±w}` (up to positive scale for ReLU), take the net output `Σ v_k`. Add the total linear map `Σ v_k w_kᵀ`. Under ridge-function uniqueness, which needs a non-polynomial even part `h`, this identifies "reads x−y" for Mech A. Two caveats: ζ lies on the LN sphere, and in the small-signal regime `h ≈ kt²`, so quadratic identities make the decomposition only approximately identifiable there. Restate A2 as a fact about the two given parameterizations. Its identifiable content is A4's function-level contrast.

**S3. Source coordinates and the weight-level class conditions are not gauge-invariant without an unstated, readout-dependent gauge-fixing.** Location: 𝓕_A (A1–A2), 𝓕_B (B1–B2), 𝓕_C, §5 "identifiable modulo the groups of ID2"; IAR §4.
- (6) Apply ReLU rescaling `(3, 3, .5, .5)` to the ReLU agreement family. Logits change by 2e−16. Reading `p` and `r` from `W_in` as the class definition implies gives `sign m ≠ sign(x'y')` on **44/144** cells, and condition A1 ("same c") fails. Modulo positive per-unit rescaling, the preactivations `s1(x+y)` and `s2(x−y)` do not determine `sign(xy)`. That needs `s1/s2`, which is fixed only by normalizing each unit's output projection on `Δu`. So the source coordinates are defined through the candidate readout. The same algebra applies to 𝓕_C under PI3's rescaling: `(s1 w₊ + s2 w₋)/2` mixes `s` into the recovered `e_3`, so the recovered `q̂` is contaminated by `Ŝ`.
- (7) The QK gauge `diag(−1, 1)` flips the reader coordinate `x̂` and the key coordinate `ŷ` together (`Q`: 5.31 → −5.31, `K`: 1.42 → −1.42), with logits identical. B1's "sign x̂ = sign a" and B2's "sign ŷ = sign b" both fail in the new gauge. Only the product is invariant.
- (6b) Fold a non-uniform, sign-mixed `γ` into `W_Q`, `W_K`, `W_V` (`/tmp/r3/gamma.py`). Logits change by 1.6e−15. The "LN-coordinate along `f_Q`" then has the wrong sign on 132/144 cells. B1 and B2 define `x̂` and `ŷ` as LN-coordinates along *residual directions*, not as reader compositions, so they are not invariant. IAR §5's claim that with non-uniform γ the gauge "shrinks to maps commuting with diag(γ)" is wrong. γ is itself a gauge (it folds into readers when β = 0), so the group is conjugated, not shrunk.
- *Correction.* State every class condition on invariant objects: products `w_in,k ⊗ w_out,k`, and `W_Q W_Kᵀ` restricted to the reader subspaces. Declare the gauge-fixing explicitly ("each unit normalized so that `w_out·Δu = ±λ`"), and acknowledge that it uses the readout direction (but not the response).

**S4. The §9 answer does not address the proposed consistency quantities.** Location: IAR §3.
- Items 1 and 2 are **not** mathematically equivalent implementations. A2 agrees only on four inputs, and C3 is approximate (RMSE 0.032). Item 3 is about per-component values, not about `sign(xy)`, `x̂ŷ` or `q̂`.
- The direct answer, from attacks (1)–(7):
  - `x̂ŷ`, as a score contrast: invariant.
  - `x̂` and `ŷ` individually: not invariant (scale and sign).
  - `sign(xy)` (A): invariant as a composition only under a readout-based gauge-fix, and only up to a joint polarity flip.
  - `q̂` (C): invariant only up to positive scale (`γq̂`), and only as a composition. It is not invariant under per-unit rescaling unless normalized.
  - Per-unit and per-head values: not invariant.
- *Correction.* Add this table to IAR §3.

**S5. Source *polarity* is not identified and depends on the outcome pathway.** Location: IAR §4 "Source identification"; MLP_AGREEMENT_MECHANISM.md row "Agreement under a specified transformation"; §5 check 1.
- "Agreement" versus "disagreement" is fixed only up to independent sign relabeling of each source. Under `b ↦ −b`, the identical function satisfies `sign m = −sign(ab')`.
  - In **A**, single-source effects are exactly zero (A1.1). So no intervention on the frozen model can fix polarity, and polarity is purely exogenous (input coding).
  - In **B**, A's polarity comes from its direct vote `v_A` and B's from its value projection on `Δu`. If `v_A ≡ 0`, which B1 allows, "the gate amplifies B when A agrees" and "when A disagrees" describe the same function.
  - In **C**, polarity comes from the `εŜ` path to `Δu`. Flipping `y` maps `Ŝ ↔ D̂` and `q̂ → −q̂`.
- So the package's "independent of the outcome" holds only for the designer's input labels. Internally, polarity is identified through the candidate readout, or not at all.
- *Correction.* Add polarity as an explicit, independently required identification step. In a trained model, it has to come from single-source interventional effects on the margin (available in B and C, impossible in A). State that this uses the outcome map and is therefore not noncircular in the sense of check 4 unless it is restricted to single-source effects.

## MINOR

- **M1. Summed same-layer writes are not identifiable** (IAR row 27). (4) Adding `7·1` to layer 1's `W_O` row changes the summed write by 5.66, with logits unchanged at 2e−15. Every LN removes the `1` direction, and `Δu ⊥ 1`. (5) Adding an unread direction to Mech C's block-2 output (no final LN) changes the write by 8.0, with logits unchanged at 4e−16. *Correction:* summed writes are identifiable only modulo `span(1)` and the kernel of all downstream reads.
- **M2. ID definition.** "`Q(g·θ, x) = Q(θ, x)`" is ill-typed for per-site quantities under permutation. It needs equivariance (site relabeling) or invariance of multisets. Neuron splitting and adding units (ID2, PI5, attack 3) change the width, so they are not elements of a group acting on a fixed parameter space Θ. Define `G` as a groupoid or as an equivalence relation on architectures, and define "modulo `G`" against the coarsest function-preserving equivalence. Otherwise "identifiable modulo `G`" depends on which `G` the authors chose to list.
- **M3. C3** concerns function classes across architectures, not reparameterization within one architecture. Within C's architecture I found no exact map that removes `q̂`, because LN norm coupling blocks moving the write. That is a positive point the package could state, labeled as unproven.
- **M4.** IAR §5 "move computation between attention and MLP: only approximately". This is plausible, and LN makes the linear odd-pair transfer inexact. It is unproven. Label it as a conjecture.

## FATAL

None. The mechanism-level conclusions (B via score contrast and `w_B`, C via `γq̂` and the function-level matched-Ŝ contrast) survive once restated on invariant objects. The over-claims are in the identification bookkeeping, and in the claim that source variables and polarity are independent of the outcome.
