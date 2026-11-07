# R_math — adversarial mathematical review

Scope 1: `step1_learned_mediator_identification` (commit a899a7c). Scope 2: post-review additions in
`step1_mechanism_identification_continuation` (T-MS, T-RF′, N2, corrected T-CO, C2 three-family table).
Re-run scripts are in `reviews/r_math_scripts/`. `s3.py` and `c2.py` import a copy of the previous package's `src/` at
`/tmp/rv/w`, so its logs are never touched. The re-run of `mediator_sign_test.py` from that copy reproduced
`logs/mediator_sign_test.txt` byte-for-byte.

Severity: FATAL = a conclusion is false; MAJOR = a stated conclusion does not follow, or is overstated, in a way that
changes how a headline reads; MINOR = a wording, scope or provenance defect.

## Findings

**1. MAJOR — T-MED: hypothesis (H2) is identical to (H1), so it adds no evidence.** Location: `DECISION_THEOREM.md`
T-MED (H1), (H2), Conclusion 1; `src/mediator.py` `iia`, `complement_interchange`.
Evidence: both functions pool the directions (A→C) and (C→A).
- In `iia`, base x with source x′ gives the state `(I−P)h_x + P h_x′` and expects δ(x′).
- In `complement_interchange`, base x′ with partner x gives the same state `(I−P)h_x + P h_x′` and expects δ(x′).
- So the two tests evaluate the same set of hybrid states against the same targets.
- Measured: `iia.S_rel == sufficiency` exactly at all 150 sites in `logs/protocol_E*.json`. This includes the
  non-trivial values 0.901, 0.847, 0.735.

Correction:
- State a single hypothesis, (H1) in both directions. It establishes that, within the 2×2 hybrid set of each pair,
  δ depends only on which member the U-coordinates come from.
- Remove "and keeping it keeps the decision" as separate support. Set ε₂ ≡ ε₁.
- If real sufficiency is wanted, patch the complement from an *unrelated* input, not from the matched partner.

**2. MAJOR — T-MED: Conclusions 2–3 ("the relation causally determines the decision") do not follow from (H1)–(H4).**
Location: `DECISION_THEOREM.md` T-MED Conclusions 2–3; `FAILURE_MATRIX_UPDATE.md` F1.
Evidence:
- (H4) bounds r only relative to the *non-additive* part. It says nothing about how much of the *decision* difference
  within a pair is carried by `f(a,p)+g(b,p)`. Inside M, a ≠ a′ and b ≠ b′, so the per-source terms also differ within
  each pair.
- The package's own additive-surrogate test measures exactly this. With Uᵀh replaced by its additive part, decisions
  still differ on 0.338 of pairs (TF2 XNOR s4) and 0.372 (s8); on gauss: 0.290 and 0.348.
- So in two of the five headline models, the per-source part alone produces about a third of the within-pair decision
  differences. Only s2, s7 and s9 (0.010–0.020) support the conclusion.
- (H4) is also an L²-average over the 61×61 grid on [−3,3]², not a per-pair statement on M. The preamble's
  "measured on held-out pairs from M" is false for H4: U is fitted on the `pairs_fit` seed and H4 is evaluated on the
  grid.

Correction:
- Add (H5) for content on the decision: replacing Uᵀh by its additive surrogate removes the within-pair decision
  difference on a fraction ≥ 1 − ε₅ of pairs.
- Report ε₅ per model: 0.020, 0.338, 0.010, 0.372, 0.018 on train.
- Restrict Conclusion 3 to the models with small ε₅.

**3. MAJOR — T-MED Conclusion 1 overclaims "whatever the rest of the state is".** Location: same.
Evidence:
- The complement was only ever that of the matched partner, which has the same z and p.
- The hybrid states lie far off the manifold (NN ratio 75–189 at the s2, s4, s7 and s9 sites).
- Nothing constrains δ when the complement comes from another input.
- "Determined by the k coordinates" would require, for all reachable x, y, that `Uᵀh_x = Uᵀh_y` ⇒ δ(x) = δ(y). This
  was not tested.

Correction: say "within each matched pair's 2×2 hybrid set".

**4. MAJOR — The F1 "chain" joins evidence from different models.** Location: T-MED Conclusion 3 ("through the
identified computation (T-ATT for TF1)"); `FAILURE_MATRIX_UPDATE.md` F1; `MAIN_ANSWER.md` items 17–20.
Evidence:
- T-ATT is a one-layer theorem. Its Consequence 3 is false for L = 2, because in TF2 the B position's layer-0 output
  already depends on (a, b).
- The T-MED instantiation, k = 1 with sign(ab) R² 0.997–1.000, is entirely TF2.
- The only TF1 XNOR relation-computing site is s4 `resid_mid0@R`. There:
  - the additive R² is 0.789, so 79% of the U-variance is per-source;
  - the additive surrogate still gives pair-flip 0.580.
- In TF2, the normalizer attribution is empirical, not exact: freezing pattern 0 leaves 0.015–0.24 of the
  interaction, and s7 is redundant.

Correction: present "generator relation → identified U → decision" for TF2, and "interaction created exactly by the
normalizer" for TF1, as separate results on separate models.

**5. MAJOR (scope 2) — T-RF′ checks: the test was run on rows only, and "additive-gate twin 0/435" is an artifact of
that.** Location: `CONDITIONAL_CAUSAL_THEOREMS.md` T-RF′ table; `MATCHED_RELATION_INTERVENTIONS.md` S3;
`ADJUDICATION.md` #9; `src/mediator_sign_test.py` S3; `src/learned_review_checks.py` `k4`.
Evidence:
- For `h(f(a)+g(b))` with h strictly monotone, *columns* (b, b′) can never reverse either. The code only compares rows.
- Re-run (`s3.py`), with the same 30-point grid:

  | circuit | rows reversing | columns reversing | columns reversing by more than 0.05 SD |
  |---|---|---|---|
  | TWIN | 0/435 | 79/435 at tolerance 1e−3 (110 at tolerance 0) | 9, max 0.079 SD |
  | RF | 0 | 0 | 0 |
  | C′ | 357 | 417 | 246, max 1.14 SD |

- The twin's margin also has interaction share 0.342.
- So the twin is **not** monotone-additive, and T-RF′ rejects it once both orientations are checked.

Correction:
- Count reversals over both rows and columns.
- Restate S3 as "C′ 357 + 417, twin 0 + 79 (small); RF 0 + 0".
- Withdraw "both twins have 0".

**6. MINOR — T-ATT displayed formula omits `bO`.** Location: `DECISION_THEOREM.md` Statement.
- `model.py` adds `lay["bO"]`, and it is nonzero (|bO|∞ = 0.053–0.065 in the checked checkpoints).
- Without it, the formula is off by exactly that. With it, the formula matches `forward` to 3e−7 (float32 rounding;
  1.4e−5 for T4_TF1_s2).
- Also write `v^h(LN x_j)` rather than `v^h(x_j)`. The additivity argument is unaffected, because bO is constant.

**7. MINOR — errata item 3 overstates the new test.** The additive surrogate is called "encoding-invariant".
- Two-way ANOVA commutes with *affine* maps of the U-coordinates, so the test is invariant to affine recodings only.
- A nonlinear invertible code φ(f(a)+g(b)) inside U is non-additive and would be scored as interaction. By T-ATT,
  the normalizer itself produces codes of this kind.

Correction: "invariant to affine reparameterization of U".

**8. MINOR — the identifiability note is weaker than the truth, and "proved in MAIN doc" is false.** The identity
`ab > 0 ⇔ |a−b| < |a+b|` holds on all of ℝ², not only "at fixed z" or "on this input family".
- So no input family can separate "agreement" from "thresholded spread". Only representational content can, and X2
  partly does: the R² is about 0.56 for ab versus about 1 for sign(ab), so the representation is thresholded, not
  graded.
- `MAIN_ANSWER.md` asserts the identity but proves nothing. A one-line proof: `ab = (z²−d²)/4`.

**9. MINOR — the gauss rows do not shift the pair distribution.** `generators.matched_pairs` ignores `dist`. The gauss
IIA, necessity and sufficiency figures come from the same pair distribution with seed +100. Only `ev` accuracy is
evaluated under gauss. The gauss NN ratio goes down to 13.9 (s8), below the quoted "18–189".

**10. MINOR — TF2 locus wording.** "Remainder (≤ 0.08) comes from MLP0" is inferred, not tested: MLP0 was never
frozen. Pattern freezing is not an additive attribution either. For REL-MIS s3:
- the interaction share is 0.034 with nothing frozen;
- it is 0.048 with pattern 0 frozen;
- it is 0.087 with both patterns frozen.

**11. MINOR — provenance.** No committed script produces `logs/aggnl_reversals.json`. The values are reproduced
exactly (`agg.py`) by the previous package's K4 rule: 31×31 grid, reversal if greater than 0.05·SD, p = 0.

**12. MINOR (scope 2) — K4 details.**
- The docstring says "61x61 grid", but the code uses 31×31 (465 = C(31,2)).
- T3 is evaluated at p = +1 only.
- T1's constant 255/465 is structural: 15·15 opposite-sign row pairs plus 30 pairs involving the zero row. It reflects
  sign(ab) symmetry, not the magnitude of the interaction. State this.

**13. MINOR (scope 2) — T-MS is a definition, not a theorem, and depends on orientation.**
- Errata 3 is correct as far as it goes.
- Add the positive part: the TWIN fails *every* monotone recoding as well, because its conflict value (−1.84) lies
  strictly between its two agreement values (+1.84 and −5.52). So the S1 separation of C′ from the twin is robust to
  monotone recoding, though not to arbitrary recoding.
- S2 (routing after interchange) is the encoding-free part of the evidence.

**14. MINOR (scope 2) — N2 has unstated conditions.** `X = P/σ` holds when P and Q sit on mean-zero orthogonal
directions plus a constant ρ-coordinate (d = 6, ρ = 1), so that LN reduces to RMS. "XNOR exactly" holds on the grid
|a|, |b| ≥ 0.5 (re-run gives 1.0). It fails near the axes, where tanh(5·) is not saturated.

**15. MINOR (scope 2) — the C2 table reports medians that understate the spread.**
- "q_B carries at most 1%" is a median. Per point, the asymmetric family ranges from −0.080 to +0.057.
- The b-flip value share has median 0.018 but a maximum of 0.237.
- All 200 bases are in Q++, so only the agreement→conflict direction is tested.
- "q_B = 0 forced by symmetry" checks out numerically: |share| ≤ 5e−16 even though |Δq_B| is up to 14.6.

## Verified claims

- **T-ATT (TF1 class).** Correct for `model.py`. Checked:
  - per-position LN with g1/b1;
  - query from LN(E_R) alone;
  - √DH = √8 scaling;
  - no QKV biases;
  - causal mask (R is last, so it attends to all tokens);
  - prior token present for T3;
  - H = 2.

  Numerically:
  - freezing only the normalizers Z_h (with ω_j still varying) gives interaction share ≤ 1e−28 (T1 s4, T4 s2, T3 s2
    at both p);
  - unmodified, the shares are 0.21, 0.32 and 0.001.

  "Every cross-source interaction at h_R is created by the normalizers" is exactly true, including E_R and bO, which
  are constants. Log check: all 7 TF1 relation-computing sites give pattern-frozen shares ≤ 4.1e−13.
  Consequence 3 is valid for L = 1 only (see finding 4).
- **T-MED instantiation numbers.** These match the logs:
  - ε₁ = ε₂ = 0, with S_rel IIA and sufficiency both 1.000 at every TF2 XNOR `resid_mid1@R` site, train and gauss;
  - ε₃ ≤ 0.152 (train maximum s4; gauss 0.134);
  - ε₄ ≤ 0.003 (X2 sign(ab) R² is 0.999, 0.999, 1.000, 0.997, 1.000);
  - the train NN ratio is 18.2–188.8.
- **Aggregate-only ceilings.** Recomputed: XNOR = MIS = 0.7539 (held-out rule 0.7533), GATE 0.8770, AGG 1.0.
  - XNOR and MIS are equal by necessity, because at fixed z the MIS label is sign(z)(1−2s), a bijection of s.
  - Caveat: the ceilings hold for the training mixture only. They are an in-sample bin estimate, with bias under
    1e−3.
- **Errata 1, 2, 4.**
  - Errata 1 (A1* narrowed; M7 8.9e−16) is correct.
  - In errata 2, the T-RF′ proof is correct. The AGG-NL counts 423–450/465 are reproduced. The withdrawal is
    justified, since a non-monotone h(z) reverses.
  - Errata 4 is correct.
- **T-RF′ proof.** Valid. The statement holds for strictly decreasing h too.
- **T-MS logs.** S1 (500/500 vs 0/500), S2 (1.000/1.000) and X4 are reproduced exactly.
- **Corrected T-CO (C′-CO).** The proof is correct:
  - T jumps only across the positive semi-axes and at the origin;
  - T is constant (−1) across the negative axes.

  Further checks:
  - m = min(a, b) implements T at every point with ab ≠ 0;
  - C′ has m(0, 0) = −0.6578, m(0.5, 0) = −0.139 and m(1, 0) = +0.265, all nonzero as claimed;
  - the withdrawal of "every relation gate needs a dead band" is correct.

  Implied but unstated: on the negative axes, m ≤ 0 is also necessary.
- **C2 table and C2-OFF.** All 15 cells match `logs/program_c.txt`, as do the TE median −4.918 and the manifold
  distances 5.49/3.74 against controls 0.35/0.29 on a 61×61 grid. C2-OFF is valid as stated: any q_R patch that moves
  the state leaves the reachable set.
