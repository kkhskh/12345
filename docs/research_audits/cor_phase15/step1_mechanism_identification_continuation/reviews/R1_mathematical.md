# R1 — Mathematical review (adversarial)

Package: `step1_mechanism_identification_continuation` @ d1aa97d. Reviewer scripts: `reviews/r1_scripts/`
(absolute paths; run with `python3 <script>`). Nothing outside `reviews/` was modified.

Severity scale: FATAL = a headline result is wrong; MAJOR = a stated theorem/claim is false or unproved as
written, but can be repaired; MINOR = gap, imprecision or unsound-in-principle step with no effect on the numbers.

No FATAL findings. The C′ closed form, S+/S−/S−0, the sign-of-G lemma, A1*, and the XNOR non-additivity proof
are correct. Several *restatements and extensions* are false.

---

## Findings

### F1 (MAJOR) — T-C′1* in CONDITIONAL_CAUSAL_THEOREMS.md drops the hypothesis G ≥ 0
*Location:* CONDITIONAL_CAUSAL_THEOREMS.md, T-C′1*, bullet 2 ("follows the sources if `G > (σ_P/μ)·ln(Π/(v_A+v_B+v_R))` with all values ≥ 0").
CPRIME §3 (S+) requires `G ≥ 0`, and the proof needs it (`e^{s_A}, e^{s_B} ≥ 1`). When the value mass exceeds Π, the
threshold is negative, so the restated condition admits conflict cells with G < 0.
*Evidence:* on a 300×300 grid of Q+− (a ∈ [0.05,3], b ∈ [−0.5,−1e−4]), 2,167 points satisfy the restated condition
and **657 of them have m < 0**. Example: (a,b) = (0.7406, −0.00177), G = −0.0365, mass 2.917, m = −0.0427.
*Correction:* add "G ≥ 0" to the bullet, or point to S+ verbatim.

### F2 (MAJOR) — B-TM / T-B-TM is false with a final LN (zero bias)
*Location:* DYNAMIC_ATTENTION_AND_DEEP_MECHANISMS.md B3 proof step 3; CONDITIONAL T-B-TM class ("a final LN, if
present, with zero bias"); T-C4 conclusion 2 inherits this.
Step 3 says the LN "multiplies the logit difference by 1/σ > 0, so the sign is unchanged". That holds for the sign of
*each* margin. It does not hold for the sign of the *difference* between two runs, because σ changes with the routing:
`m′/σ′ − m/σ` need not have the sign of `m′ − m`.
*Evidence:* `r1_scripts/btm.py`. In `qk3(final_ln=True)` against its uniform-routing ablation, on 4,000 N(0,2²) draws
(seed 5), the B-TM sign prediction fails on **33/4000**. Example: zA = (0.206, −1.73, 1.792), zB = (−2.597, −2.402, −2.565),
ΔΔ = +0.0053, predicted −1. (B3's 3000/3000 check used no final LN.)
*Correction:* drop the final LN from the class, or state the corrected identity
`ΔΔ_LN = m′/σ′ − m/σ` with the extra term `m(1/σ′ − 1/σ)`, and give a sufficient condition such as
`|ΔΔ|/σ′ > |m|·|1/σ′ − 1/σ|`.

### F3 (MAJOR) — C′-CO "consequence" overreaches: dead bands are not necessary in general
*Location:* CPRIME §5 ("no continuous circuit … can implement T on all of ℝ²", "its form was impossible");
§7 status ("every relation-gated sign rule needs a dead band"); CONDITIONAL T-CO "Consequence".
The proposition (parts 1–2) is correct. It needs `m ≠ 0` at a point of a positive axis. The margin
`m(a,b) = min(a,b)` is continuous and gives `sign m = T` at **every** point with ab ≠ 0, so its failure set is empty.
It is also implementable with ReLUs on LN coordinates (`min(u,v) = (u+v−|u−v|)/2`, and LN scaling is positive).
It vanishes on the positive axes, which the text concedes as the "only way out", but that makes the dead band a
**robustness** limit, not an existence limit. "Its form was impossible" is false for the historical claim in the
abstract. It is true only for C′, where m(1,0) = +0.265 ≠ 0.
*Correction:* "No continuous margin can satisfy T together with `|m| ≥ δ > 0` near the positive axes. Any margin
nonzero at a point of a positive axis fails T on a positive-measure set near it. C′ is such a margin (m(a,0) = 0 only at
a ≈ 0.653, and m(0,b) = 0 only at b ≈ 1.091)."

### F4 (MAJOR) — Unequal-weight cone (N4 violated) is missing its upper boundary
*Location:* MLP_NORMALIZATION_THEOREMS.md §4 "(N4) violated": "For agreement with t = v/u > 0, this is t > (r−1)/(r+1)".
For ReLU, `1+t > r|1−t|` gives `(r−1)/(r+1) < t < (r+1)/(r−1)` when r > 1. The cone is symmetric under t ↔ 1/t.
*Evidence:* r = 2 with ReLU gives q = +0.10 at t = 2.9, **q = −0.10 at t = 3.1**, and q = −7 at t = 10. GELU at scale 5
gives q < 0 at t = 3.5. M5 only scans t ∈ (0, 1], so it cannot see this.
*Correction:* state the two-sided cone, with the GELU small-scale limits `((√r−1)/(√r+1), (√r+1)/(√r−1))`. Say what
happens for r < 1 (conflict cells then become positive near |t| = 1). Extend M5 to t ∈ (0, ∞).

### F5 (MAJOR, interpretive) — "Only candidate-symmetric rules force a cross-source interaction" ignores LN coupling
*Location:* CONDITIONAL T-RF, last line, and the framing that XNOR's non-additivity means a unit must read two sources.
The 4-point proof that `f(a)+g(b)` cannot sign-represent XNOR is **correct**. But a pre-norm network in which **no unit
reads two sources** can still implement XNOR, because the shared LN couples them. This is the package's own N1 effect.
*Construction* (`r1_scripts/xnor_ln.py`):
- per-source units write P(a) = 1.5 − 0.5 tanh 5a and Q(b) = 2 − tanh 5b;
- the read token's LN gives X = P/√((P²+Q²+ρ²)/d + ε), with d = 6 and ρ = 1;
- the four sign cells then map to X ∈ [1.408, 1.420] (++), [0.739, 0.746] (+−), [1.993, 2.000] (−+) and [1.306, 1.313] (−−);
- a ReLU bump on the single coordinate X gives **XNOR with 100% accuracy** on the 120×120 grid with |a|, |b| ≥ 0.5.

*Correction:* say that T1 forces a non-additive cross-source dependence, which may be supplied by normalization
alone. Do not use T1 behaviour as evidence of a dedicated relation unit.

### F6 (MINOR) — Interval arithmetic is not outward-rounded everywhere (results survive)
*Location:* src/cprime.py `interval_m`, lines 170–176, 189–193.
- (a) `x = I(_dn(fx(a.lo)), _up(fx(a.hi)))` pads by 1 ulp, but `fx` has about 6 roundings. The measured error is up to
  **2.07 ulp** (`ia1.py`), and **11,842/200,000 = 5.9%** of degenerate endpoint intervals do not contain the true
  value (`ia2.py`; e.g. t = 2.188489682951995).
- (b) `sP`, `p.mu/sP`, `vP`, `p.lam/math.sqrt(2)` and `4.0/3.0` are plain floats, not enclosed.
- (c) `exp` relies on libm being < 1 ulp (true for glibc ≥ 2.28; platform-dependent).
- (d) The monotonicity shortcut for x is **correct**: `dx/da = (κ²+μ²)/d + ε` over `σ_A³`, which is > 0.
- (e) The `iheven` pad is adequate: the worst error is 0.17× the pad on t ∈ [−8, 8]. Even/monotone use is justified by L1.
- (f) qB enters σ_B2 correctly.

*Evidence that it does not matter:* `cprime_fixed.py` and `rerun.py` apply a 16-ulp relative pad to x and y and enclose
all constants. E3 then reproduces 99.640/0.309/0.051, 100/0/0, 99.708/0.222/0.070 and 99.845/0.100/0.056 exactly,
and E6 still certifies τ = 0.09/0/0.02/0.02. The minimum certified |F| bound is ≥ 1.3e−4 in Q++, Q+− and Q−+.
In Q−− the whole quadrant is certified as one box with F.hi = −2.5e−49, which is sound but razor-thin. S−0 proves that
quadrant analytically anyway.
*Correction:* apply the fix and drop "outward rounding on every operation" until it is true.

### F7 (MINOR) — E6 conflict-ratio "certification" does not cover the claimed region
*Location:* cprime_regions.py lines 185–199; CPRIME §4 last line.
The boxes `(a, a+.05, −a, −ra)` and `(ra, a, −a−.05, −a)` miss the strip near |weak| ≈ |strong|. **10,062/400,000**
uniformly drawn points of the claimed region lie in no box (e.g. (1.5828, −1.5538)). Q−+ is never run, although the
doc says "the conflict region" without a quadrant.
*Evidence:* `ratio.py` uses covering boxes `(a, a+.05, −(a+.05), −ra)` and the Q−+ mirror. **Both certify** at r = 0.05,
so the claim is true but was not certified by the shipped code.
*Correction:* use the covering boxes, add Q−+, and say "Q+− and Q−+".

### F8 (MINOR) — "Decision depends on relation and aggregate: PROVED from C′1*" is a non sequitur
Sufficient conditions cannot establish a dependence. A witness can: (0.08, 0.08) and (1, 0.012103) have **identical
G = 0.260736** but m = −0.061 vs +0.906. Further equal-G partners include (0.5, 0.01534), (2, 0.01554) and
(3, 0.01932), all with m > 0. Cite the witness instead.

### F9 (MINOR) — B-IMP proof uses a measure-zero slice; the corollary is narrower than stated
*Location:* DYNAMIC…md B1, step 1 ("Fix z_B = e_0").
- Disagreement on a positive-measure set of z_A at a single z_B is a null set in (z_A, z_B). The fix is easy: the
  argument holds for every z_B in the open set {argmax z_B = 0}, since C_0 depends only on argmax z_B. Then apply Fubini.
- The corollary covers only the **sign of one score against 0**. An exact switch relative to another position's
  score with a different, z_B-dependent σ (`f_B/σ_B − f_A/σ_A`) is not bilinear, and the corollary does not cover it.
  It applies to qk3 because s_RA = s_RR = 0.

State both.

### F10 (MINOR) — Cross-talk and gain/bias cones: wording and tests
- The mixed-sign case is not "one cone on one side of the diagonal". With t12 > 0 > t21 there are **two** failure
  cones: agreement fails for |v/u| ≤ |t21|, and conflict fails for |v/u| ≥ 1/t12. Example with t12 = 0.5 and
  t21 = −0.3: (1, 0.1) fails, and (1, −3) also fails.
- M4 and M6 only test the identity `sign q = sign(u′v′)`, which is true by construction. They do not test the
  interval descriptions `(t21, 1/t12)`.
- For gain/bias no cone is actually derived; "the offsets move the apex" is qualitative. Do not label it DERIVED.

The same-sign statements, including the requirement t12·t21 < 1, are correct; I re-derived them.

### F11 (MINOR) — Boundary-shape claims are numerical, not proved; E8 values are grid-quantized
- "One root per ray" and "no root for a ≤ 0.5 / b ≤ 1" come from 7 rays and 6 values. My scan of 400 rays,
  θ ∈ [0.01°, 89.99°], r ∈ [1e−4, 1000], finds exactly one sign change per ray. So the claims hold, but label them
  VERIFIED. The true conflict thresholds are a ≈ 0.653 and b ≈ 1.091, which are the axis roots of m.
- E8 reports first grid points, not roots: at λ = 6, π = 1 it gives 0.1253 against the exact 0.1223. The π = 2,
  λ ≥ 12 roots "−0.0005" are at grid resolution.
- "No finite parameter setting removes both bands" is unproved. It is provable: `m(0,0) = v_P/4 < 0` always (wrong
  in Q++ near 0), and m(a,0) is analytic and non-constant.

### F12 (MINOR) — Reachability theorem R: domain and ε
*Location:* MATCHED_RELATION_INTERVENTIONS.md §2.
- The proof is correct (∂/∂x and ∂/∂y of (x+y)/√(x²+y²+ρ²) are both > 0 on the conflict quadrant), but the
  supremum needs **a → ∞** and ignores ε.
- On the experimental domain |a|, |b| ≤ 3 (κ = 2, d = 6), the reachable conflict supremum is **1.376**, not 1.549.
  So "a partner exists iff |Ŝ| < 1.549" overstates the in-domain set.
- X4's claim that "(2,2) has no partner" uses a different statistic (m_blind), so it is not "the same limit".
  It is true numerically: the supremum of m_blind over conflict is ≈ 1.3072 as a → ∞, against m_blind(2,2) = 1.4728.

### F13 (MINOR) — Small hypothesis slips
- P3 needs γ > 0. A1* needs only γ ≠ 0.
- P4 `∂u/∂y = −γxy/(dσ³)` is zero when x = 0 or y = 0, so "≠ 0" holds only generically.
- X3 "impossible for any reader that sees both sources" should read "for any reader whose state determines
  (u, v)".
- T-C4 assumption (i) is exact in the proof but only ≈ satisfied (values carry 1.8%). State the error term.
- T-RF "never behavioural evidence" should say decision-level. Margins are behaviour too.

---

## Claims verified as correct (with how)
1. **C′ closed form** (CPRIME §1). I re-derived it from `cbmath.rereading`:
   - uniform causal head, giving (a+2x, 0), (4x/3, b+4y/3) and (x, y);
   - MLP output `λ1[h(c1(u+v)) − h(c1(u−v))]`, with q = 0 at P and A;
   - block-2 key `(mk_S − mk_P)`, the √2 scale for dh = 2, and the value and unembedding factors.

   It matches the forward pass to **3.3e−13 relative** at 60 random parameter settings, with |a|, |b| ≤ 50 and near 0
   (`cf_check.py`).
2. **F = Z·m**, Z > 0, so sign F = sign m.
3. **Lemma sign G = sign q_R = sign(ab).** h(t) = t·tanh(K(t+Ct³)) is even and strictly increasing in |t|, and
   u_R, v_R are positive multiples of x, y. This needs c1, λ, λ1 > 0.
4. **S+, S− and S−0** as stated in CPRIME §3, including the need for v_A, v_B, v_R ≥ 0 and G ≥ 0 in S+, and Π > 0.
   I found 0 violations in 3,600 extra checks at extreme parameters.
5. The "Reading" inequalities are algebraically equivalent to S+ and S−. Q−− has m < 0 everywhere, analytically by S−0.
6. E2 counterexamples and the E3/E6 numbers are reproduced, also under sound rounding (F6).
7. **C′-CO** parts 1 and 2 as stated, with hypothesis m ≠ 0 on the axis.
8. E7: `q_R ≈ 4Kc1²uv` near 0, α_P ≈ 1/4, and Π = 2.631.
9. **A1*** under (N1)–(N4). The proof is correct and needs no mean-zero w, since e1, e2 ⟂ 1. N1 holds:
   `σ² = (S² − 2xy + ‖w‖²)/d + ε`, and the M3 value 0.7947 checks.
10. **P5** ratio invariance. The same-sign cross-talk cone `(t21, 1/t12)` holds under t12·t21 < 1. The GELU limits are
    0.1716 and 1/3 for r = 2.
11. **T-RF** four-case table. The binary AND/OR reading holds. The **XNOR non-additivity proof** is a valid 4-point
    contradiction, and the LP gives T1 infeasible and T2, T3(±1), T4 feasible (I re-ran it at 24×24). I also found an
    explicit additive certificate for T4.
12. **B-TM identity** without final LN: `α′_i = α_i(1−α′_B)/(1−α_B)` gives ΔΔ = (α′_B−α_B)(w_B − w̄_{−B}).
13. **T-C4** conclusions 1 and 3: |Σα v| ≤ max|v_j|, and φ strictly increasing with φ(0) = 0 preserves sign.
14. **T-SS** (trivial), **T-R** (the supremum computation; see F12 for the domain), and **X3** inversion
    `σ² = (ρ²/d + ε)/(1 − (u²+v²)/d)`.
