# R2 — Mathematical correctness and counterexamples

Reviewer role: adversarial math. Scope: THEOREMS_AND_COUNTEREXAMPLES.md, src/cbmath.py, tests/test_deep.py, logs/experiments.txt, judged against MANDATE §2, §8.
Scratch script: /tmp/r2/chk.py (imports src/cbmath.py read-only). All numbers below come from it.

No FATAL findings. Most "PROVED" claims are correct algebra. There are 6 SERIOUS findings: false-as-stated sub-claims, and identifiability and matching claims that the code does not support. The §5 verdict understates the failure.

---

## SERIOUS

**S1. C1 claim 2: the "strictly increasing" part is false, and PC2's own agreement cell is a counterexample.**
Location: §4 C1.2 and its proof.
The proof gives ∂g/∂q̂ = γ(1[u1 active] − 1[u2 active]). This is 0 when both units are active, i.e. when γq̂ − θ > β|Ŝ|. In that region g = 2β|Ŝ|·sgn Ŝ, which does not depend on q̂.
Construction, PC2 agree cell: Ŝ = 0.98526, q̂ = 1.20310. The unit arguments are 2.190 and 0.219, so both units are active. g(Ŝ, q̂ ± δ) = 1.970524 for δ = +0.3 and for δ = −0.1. The reported m_ag = 4.187 = (ε + 2λ₂β)Ŝ is the saturated value, so ∂m/∂q̂ = 0 there.
Correction: g = sgn(Ŝ)·min(2β|Ŝ|, max(0, β|Ŝ| + γq̂ − θ)). The increase is strict only for θ − β|Ŝ| < γq̂ < θ + β|Ŝ|. State the saturation, and note that PC2 sits in it.

**S2. The PC2/PC3 pairs are not matched in Ŝ, yet the document calls them "matched-Ŝ" and "at fixed Ŝ".**
Location: §4 C1 "Verified" (PC2, PC3); log line 69; experiments.py lines 367–375; test_C_decision_….
The pairs are matched in the source-LN sum x + y, which is block 1's input. They are not matched at block 2's reader. The log itself shows Ŝ = 0.9853 vs 0.9736 and 0.7161 vs 0.6471. This is the same raw-vs-reader error that the document diagnoses in PA9.
Exact Ŝ-matching (brentq): the disagree partners at Ŝ = 0.985262 give m = 0.24632 (q̂ = −0.727 and q̂ = −0.282). The qualitative conclusion survives.
The ablation "gap 0.182 at fixed Ŝ" comes entirely from Ŝ mismatch: ablated Ŝ = 1.0887 vs 1.0079. Recomputing with g(Ŝ, 0) reproduces m = 1.2495 and 1.0678 exactly. At truly fixed Ŝ, C1.4 makes the ablated gap exactly 0.
Correction: relabel the pairs as "matched in source sum". Either add pairs matched at the block-2 reader, or report ΔŜ with the bound (ε + 2λ₂β)|ΔŜ|. Restate PC3's residual as Ŝ mismatch, which the theorem predicts.

**S3. B1 claim 4 (the reversal band) is false when s₀ ≠ 0, which the class allows.**
Location: §3 B1.4.
(i) w(0) = 1/(1 + n·e^{s₀}), not 1/(1+n).
(ii) Let w_fix denote the uniform (W_Q = 0) weight, 1/(1+n). Then w(−λ|x̂ŷ|) < w_fix iff s₀ + λ|x̂ŷ|/√d_h > 0.
Counterexample: n = 2, s₀ = −3, λ|x̂ŷ|/√d_h = 1. Then w(−z) = 0.787 > 1/3. For k|ŷ|/3 < |v_A| < 0.787·k|ŷ|, fixed routing picks A and the gate picks B. The reversal goes toward B, so "exactly on the band … toward A" fails.
Claim 4 also silently assumes that v_A has the sign of a. That hypothesis appears only in claim 3.
Correction: write w_fix explicitly. State the band as the interval between k|ŷ|·w_gate and k|ŷ|·w_fix. Its direction is set by sgn(s₀ + λ|x̂ŷ|/√d_h). Add sgn v_A = sgn a as a hypothesis.

**S4. The list of identifiable objects (§1, last paragraph) is overclaimed. This feeds §5 check 1.**
- *Attention scores are not identifiable; only patterns are.* Set the layer-2 LN bias to b₁ += 0.5·1/d. The mean-zero reader coordinates do not change. Then set W_K += 1⊗u with u = (0, 3). The read-row scores go from [0, −1.942, 0] to [1.072, −0.870, 1.072], a row shift. Patterns are identical and the logit difference is 5.6e−17. A key bias does the same in standard architectures. B2 ("s_rB = λx̂ŷ/√d_h, others = s₀") is therefore stated in a non-identified quantity. Restate it in terms of s_rB − s₀.
- *ReLU reader preactivations are not identifiable modulo permutation and rescaling* when units form linear pairs: ReLU(z) − ReLU(−z) = z. Counterexample: replace C's ε-units ReLU(±εŜ) with ReLU(±εx̂) and ReLU(±εŷ). The logit difference is ≤ 2.8e−17, but block 2 no longer "reads Ŝ" through that path. Identifiability holds only generically, for units with distinct, non-paired kink sets.
- *Summed same-layer writes* are residual vectors. They transform as h ↦ hR^T, which PI4 itself shows (raw coordinates change by 2.026). They are identifiable only through compositions with readers.
Correction: amend the list as above. ID2 is a list of gauges, not the full group G. "Identifiable modulo ID2" is weaker than "modulo G", so check 1 cannot be answered "Yes" on that basis.

**S5. A1 has no hypothesis linking the reader coordinates to the sources. As written, it is a theorem about an MLP with symmetric units.**
Location: §2 A1 class, A1.1, A1.3, and §5 "source representations".
p and r are recovered from any four units ±u, ±w: take p = (u+w)/2c and r = (u−w)/2c. So sgn m = sgn(xy) holds by construction for the reader's own coordinates.
The input-level statements need extra conditions. A1.1 (m(a,0) = 0) and A1.3 (any pattern or LN scale preserves the sign) require all of the following:
- A's write has zero r-component.
- B's write has zero p-component.
- The read token's own residual has zero (p, r)-components.
- sgn x = sgn a and sgn y = sgn b.
If any of these fails, for example if A writes onto r, then m(a,0) ≠ 0.
Correction: add a source-separation hypothesis (A5) and derive A1.1 and A1.3 from it. In §5, say plainly that the "sources" are defined by the reader, not independently of it.

**S6. A2 is false as stated when final LN is present, which the common setting allows.**
Location: §2 A2.
With final LN, the stated sum-only reader gives these margins (designed vs sum-only):
- agree cell: 3.2142 vs −0.0674 (the decision differs)
- disagree cell: −3.2142 vs −3.8184
Without final LN they match (3.818391 in both cells). The test uses final_ln=False.
A second gap: the stated offset assumes h(0) = 0. Softplus has h(0) = 1.386. The designed q is ±(h(2c) − h(0)), and the required offset is −λΔu^Tv·(h(2c) + h(0)).
Correction: add "no final LN (or s constant), h(0) = 0" or give the general offset. The existence of *some* matching sum-only reader under final LN would need its own proof.

---

## MINOR

- **SS*, hypotheses.** B must not be a descendant of any c_k. Counterexample: S = B + c₁ + c₂ with B = c₁² and Y = S. Every path to Y passes through S, yet the edit (1,0) → (2,−1) has Σδ = 0 and moves Y from 2 to 5. Add "B is held fixed (not a descendant of c_k)". The Transformer instance satisfies this.
- **ID1 "their sum is unchanged".** The sum of the individual per-head patch effects changes by (Y′−Y)E − (Y′−Y)E = 0 only to first order. It is exactly invariant only because the downstream map is linear in qk_gate. Under a nonlinear downstream map it is not invariant. Mark it construction-specific.
- **A1 (A4) hidden conditions.**
  - Final LN centers and scales by g. Δu must be the effective readout (diag(g)Δu, with Δu^T1 = 0), and κ must absorb Δu^Tβ + Δb.
  - No other MLP unit, and no b_out, may project onto Δu.
- **A1.2 equality needs s symmetric.** With final LN, m is not a function of (x, y). At reader coordinates ±(0.6231, 0.3894) (the read residual scaled by 3), the margins are 0.4986 and 0.1694. The sign version holds. The equality holds without final LN.
- **A1 claim 3.** "Large" is asymptotic in min(|x+y|, |x−y|) → ∞, or as a ratio. This is fine, but say "ratio → 1".
- **A4 at Ŝ = 0.** q = −h(cD̂) ≤ 0, with zero only at D̂ = 0. It does not change sign there. Add Ŝ ≠ 0.
- **PA11.** "Approximate pairing is correct only where |q| exceeds the perturbation" is a heuristic sufficient condition, not a shown "only if".
- **B1 claim 5 scope.** The claim is true only for *site-level* (dissection) Q-only and K-only substitutions, as in PB4. At the input level, every sign-symmetric square has zero interaction exactly in 𝓕_B, because T(−x,y) = −T(x,−y) and v_A is additive. Numerically: I(±1,±1) = 2.7e−15, I(±0.5,±1.3) = −8.9e−16, I(±1.2,±0.4) = 0. A non-symmetric square, a: 0.2→1 with b: 0.2→1, gives I = 0.933. So the binary sign family cannot detect the gate through interaction. State this; it parallels A2.
- **B4 "fixed positive scale".** Real final LN has a scale that depends on the input. Only the sign claims (1–4) survive. The identity m = v_A + T, and any head-change-to-margin-change inference, do not.
- **B1 notation.** The class formula is m = v_A + kŷw, while the test checks v_A + 2·1.5·ŷw. k absorbs the factor 2 from W_U. State this.
- **B1 hidden conditions.** Δu^T h before the layer-2 head must equal v_A(a) exactly (no b-dependent terms). No other layer-2 head may project onto Δu.
- **C2 "by SS*".** It also needs the routing at the read position to be independent of the sources (W_Q = 0 here). Otherwise keys are a bypass. Add this to the statement.
- **§5 notation.** C1's q̂ = ⟨e₃, ζ₂⟩ = λ₁q/σ₂ is a positively rescaled copy of h(cŜ₁) − h(cD̂₁), not equal to it. Ŝ in block 1 and Ŝ in block 2 are different normalizations.

---

## C3 assessment (the exactness sketch)

- The headline conclusion (no single ReLU layer represents the C margin exactly) is true, but the sketch is incomplete and unnecessary.
  - *Incomplete:* "Contains no line segment generically" is asserted, not shown. A non-affine analytic F can vanish on a line, e.g. (x−1)·G. Nor does the sketch show that the gradient jump λ₂γ∇F is nonzero on the curve, or that the u₂ kink does not cancel it.
  - *Unnecessary:* a one-hidden-layer ReLU net is piecewise affine. The C margin is non-affine on open sets where both gates are off: there m = εŜ, and Ŝ is divided by the LN norm. Second differences on the PC6 family at (0.1, −0.05), with h = 0.02, are −2.9e−6 (x direction) and −8.9e−7 (mixed). These are nonzero, so exact non-representability follows in one line.
- **The sketch contradicts the C3 headline.** Exact non-representability *distinguishes* the two-block mechanism from the one-layer competitor at the I/O level. "The I/O function does not identify the mechanism" is supported only approximately. Approximation by wide shallow nets is a generic universal-approximation fact, not something specific to C. Exact non-identification would need an exactly equivalent alternative mechanism, and none is given.
- **The approximation is weak at the decision level.** PC6 sign agreement is 0.923. At κ₀ = 0 the C decision is exactly sgn(x+y) (fraction 1.000 on 400 random points), which a single linear "sum-only" unit reproduces with no error. Report this. It means that at κ₀ = 0, C's *decision* is relation-blind; agreement enters only through magnitude and the tuned prior band.
- Correction: replace the sketch with the piecewise-affine argument. Downgrade the C3 headline to "approximate, finite-sample non-identification (generic)". Add the sum-only decision observation.

## §5 assessment ("partly fails check 5")

The judgment is understated. **Check 5 fails** for the directional content:
- Every candidate direction in the conclusions is supplied by sign hypotheses: sgn x̂ = sgn a, sgn ŷ = sgn b, k > 0, λ > 0, units writing ±Δu with their sign tied to Ŝ, and Δu^Tv > 0.
- What remains are threshold restatements of m = v_A + T (B1.4) and an intermediate-value shift (C1.3).
- B1.4 is also wrong as stated for s₀ < 0 (S3), and the C1.2 strictness is wrong (S1).

Check 1 "Yes" is also overstated (S4, S5): scores are not identified, ReLU readers are not identified under linear pairs, and the "sources" are reader-defined. Check 3 is not met.

Under mandate §8, failing these checks requires rejecting the theorem *as an explanatory bridge*. The mandate's own label is "implemented-mechanism **existence** theorem". "Structural theorems" over classes defined by the presence of the gate is a renaming, not a stronger result.

Correction: write "fails check 5 (directional content is assumed); check 1 is qualified; check 3 is unmet; rejected as an explanatory bridge; implemented-mechanism existence theorems for the classes 𝓕_B and 𝓕_C". Update the §6 register row.

## OK (verified line by line)

SS* proof (given the added B-fixed hypothesis) and its counterexample; the Transformer zero-sum head instance; ID1 proof; ID2 QK, VO, ReLU-rescale, split, and rotation gauges (LN(hR^T) = LN(h)R^T needs R^T1 = 1, which follows from orthogonality, and uniform g; the code transforms b₁, b₂, b_f); A1 claims 1–2 (including ties and zero coordinates) and the GELU coefficient 4√(2/π)c², which is the same for erf-GELU; the strict monotonicity of the even part for GELU, SiLU, softplus, and ReLU; the A1 removal witnesses (the A2-removal mechanism near the axis, 0.3h(cx) > 0, is correct); A3 identities (E_oo[s] = q/2, E_oo[q] = q, small-signal I_s = 2kab; for ReLU, I_s ≤ 0, correctly scoped); A4 apart from Ŝ = 0; B1 claims 1–3 (masked positions: n counts only unmasked positions, and PB7 is consistent); the PB9 counterexample; C1 claims 1, 3, and 4 (Ŝ = 0 gives m = 0, a tie; the C1.3 interval is empty when the pair lies in the same flat region); C2 within the construction.
