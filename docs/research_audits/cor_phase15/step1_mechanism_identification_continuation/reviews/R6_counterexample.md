# R6: Counterexample review

Reviewer role: search outside the tested grids and parameter neighbourhoods for counterexamples to every stated theorem or claim.
Package: `step1_mechanism_identification_continuation` (commit d1aa97d), with `../step1_causal_bridge_deep_investigation/src/cbmath.py`.
All scripts are in `reviews/r6_scripts/` (c1–c9). Each takes the package `src/` path as its argument, e.g. `python3 c4_twin.py $PKG/src`.
Nothing outside `reviews/` was modified.

## Summary

| # | claim | counterexample | severity |
|---|---|---|---|
| CE1 | T-B-TM / Theorem B-TM "holds with a final LN with zero bias" (proof step 3) | final LN, zero bias: the sign of the routing effect is mispredicted on 35/4000 (default) and up to 471/4000 (other params) random draws | **MAJOR** |
| CE2 | T-RF check: "C′ has a decision-equivalent relation-free twin" (`relation_free_equivalent.py`) | the twin outputs c′ on strong agreement, e.g. (8, 8), where C′ and the target give c; 735/19,888 mismatches outside the dead bands on [−10,10]² | **MAJOR** (scope) |
| CE3 | T-CO "Consequence: every relation-gated sign rule needs a dead band" (also MAIN_RESEARCH_RESULT item 4) | a continuous ReLU reader `min(u,v)` on the shared-LN read coordinates implements T on all of {ab ≠ 0}: 0 violations in 200,000 points, including points within 1e−8 of the axes | **MAJOR** (wording) |
| CE4 | C′ certified rectangle τ = 0.02 (conflict) and "conflict band ≈ 0.6% of the strong source" | the rectangle does not extend past 3: m(10, −0.03) = +0.146 and m(3.5, −0.02) = +0.030 (target −). The absolute band saturates at ≈ 0.040 and the relative band falls to 0.004% | MINOR (scope) |
| CE5 | T2 learned models: "margin is a function of the aggregate alone, D_sum 1e−6–3e−5" | outside [−3,3] the equal-sum relative MSE is 1e−3 to 3e−2 at |a|,|b| ≤ 6 and 0.08 to 0.65 at ≤ 30, with up to 17% decision flips on equal-sum pairs | MINOR (scope) |
| CE6 | `cprime.closed_form` as the "exact decision function" | `OverflowError` at λ ≥ 300 for (1,1): `F = Σ exp(s_k) v_k` is unshifted. 113/40,000 random extreme-parameter evaluations crash | MINOR (implementation) |
| CE7 | Lemma sign G = sign(ab); A1* sign q = sign(ab) "for every input" | floating point: q_R = G = 0 at (1e−17, 1). With GELU, c = 1e−8 and ‖w‖ = 1e9, q = 0 on 2224/3000 draws. Zeros only, never a wrong sign | MINOR (numerical) |

No FATAL counterexample was found. Every PROVED statement survived inside its stated hypotheses except CE1 (where the hypothesis class is stated too broadly and the proof has a wrong step) and the consequence sentence of T-CO (CE3).

---

## CE1 (MAJOR): T-B-TM fails under a zero-bias final LN

**Claim.** `DYNAMIC_ATTENTION_AND_DEEP_MECHANISMS.md` §B3 and `CONDITIONAL_CAUSAL_THEOREMS.md` T-B-TM allow "a final LN with zero bias and a mean-zero readout difference". Proof step 3: "The final LN with zero bias multiplies the logit difference by 1/σ > 0, so the sign is unchanged."

**Why this is wrong.** σ = σ(h_R) is different in the two routing states being compared. So ΔΔ_LN = Δ′/σ′ − Δ/σ, and its sign is not the sign of Δ′ − Δ. Routing to B adds B's whole value vector to h_R. When that vector has a large component on a third candidate, it raises σ more than it raises the (k, j) margin.

**Exact counterexample** (`c3_btm.py`; `program_b.qk3` defaults with `final_ln=True`, versus `qk_off=True`):
- Inputs: zA = (1.8839, −2.7601, −1.4939), zB = (7.3496, −11.1922, 6.7408), k = 0, j = 2.
- α_B = 0.99999945 under full routing and 1/3 under uniform routing. w_B = +0.2378 and w̄_{−B} = 0, so the theorem predicts a positive routing effect.
- Without the final LN, the margin is 1.0543 (full) vs 0.8957 (uniform), a routing effect of +0.159. The prediction is correct here.
- **With the zero-bias final LN, the margin is 0.4563 (full) vs 0.5038 (uniform), a routing effect of −0.0476.** The sign is wrong.

**Rates of wrong signs** on random zA, zB ~ N(0, 2²)·U(0.1, 5), with random (k, j):

| parameters | without final LN | with zero-bias final LN |
|---|---|---|
| defaults | 0/4000 | 35/4000 |
| λ = 10, kB = 4, wU = 1 | 0/4000 | 258/4000 |
| λ = 1, kB = 0.5, wU = 3, ρ = 0.5, κ = 0.2 | 0/4000 | 471/4000 |

Without the final LN, the B-TM formula is exact to 9e−15 in every setting. That part survives.

The package's single final-LN test (B5, zA = zB = (2,0,0)) happens to keep the sign.

**Correction.** Restrict the class to "no final LN". Alternatively, state the final-LN version with each run's own scale: `ΔΔ = Δ′/σ′ − Δ/σ`. The sign is then guaranteed only when `sign(Δ′ − Δ)` is consistent with the change of σ, e.g. if `σ′ ≤ σ` and `Δ′ > Δ ≥ 0`. Update B5's row "final LN, zero bias" and the T-B-TM assumption table accordingly.

## CE2 (MAJOR, scope): the relation-free twin is not decision-equivalent to C′ beyond [−3,3]

**Claim.** The twin "reproduces C′'s decisions on 100% of the 60 × 60 grid and 100% of 19,577 random points outside C′'s dead bands". MAIN_RESEARCH_RESULT states without scope: "C′ has a decision-equivalent relation-free twin".

**Search** (`c4_twin.py`, `c5_twin_diag.py`). Random points outside the same dead bands (agreement min < 0.09, conflict min < 0.02):

| range | mismatches with C′ |
|---|---|
| \|a\|,\|b\| ≤ 3 | 0/19,594 |
| \|a\|,\|b\| ≤ 5 | 1/19,724 (4.933, −0.0201): C′ is wrong here, see CE4 |
| \|a\|,\|b\| ≤ 10 | **735/19,888** |

Dense sampling at the dead-band edges inside [−3,3] gave 0/40,000 mismatches. That part survives.

**Strong-agreement failure of the twin.**
- On the diagonal, twin(r, r) = +0.0019 at r = 7.0, −0.0087 at 7.5 and −0.0493 at 10.
- Example: (7.4204, 8.5424): C′ m = +3.308, twin = −0.0174.
- At these points C′ agrees with the target, so the twin is the one that is wrong.

**Mechanism.** The twin's block-1 MLP writes φ = clamp onto fU, but leaves the raw a·e1 (b·e2) in the residual. Block-2's per-position LN then divides A's and B's fU support by a scale that grows with |a| (|b|), while P's prior term −τ is not shrunk. So the prior wins for large agreeing evidence.

**Conflict beyond 3.** At (10, −0.03), (50, −0.035) and (3.5, −0.02), C′ gives m > 0 (wrong; CE4) and the twin gives m < 0. There the two disagree in the other direction.

**Correction.**
- Scope the claim to "on [−3,3]² outside the dead bands".
- Or repair the twin. For example, have block-1 also write −a·e1 and −b·e2 through relu(t) − relu(−t), so block-2's LN scale is input-independent. Then re-verify on an unbounded range.

T-RF itself, the discrete source-additive score argument and the XNOR impossibility, survives. It is a statement in sign space and was re-derived case by case.

## CE3 (MAJOR, wording): a dead band is not necessary

**Claims.**
- T-CO consequence: "every relation-gated sign rule needs a dead band".
- C′-CO: "no continuous circuit … can implement T on all of ℝ²"; the correct form of any claim is "outside a dead band … of certified width w".
- MAIN_RESEARCH_RESULT item 4 repeats this.

**Counterexample** (`c9_tco.py`).
- Construction: the read-token shared-LN coordinates (u, v) of `agreement_family` (d = 6, κ = 2, ρ = 3), followed by three ReLU units: `m = relu(u) − relu(−u) − relu(u − v) = min(u, v)`.
- Why it works: since sign u = sign a and sign v = sign b (P3), sign m = T(a, b) at every point with ab ≠ 0.
- Check: 0 violations in 200,000 points. These include 50,000 points in [−1e−6, 1e−6]² and 50,000 within 1e−8 of the positive a-axis.
- The rule is implemented by a continuous pre-norm Transformer component on the full open set {ab ≠ 0}, with a dead set of measure zero (the axes) and no band of positive width.

**What survives.**
- C′-CO parts 1 and 2 are correct as stated: they are conditional on m ≠ 0 on the axis.
- The CPRIME doc's own sentence "the only way out is a margin that vanishes identically on the positive axes" is correct.

The "Consequence" lines contradict that sentence.

**Correction.** Replace "needs a dead band" with: "needs either a dead band of positive width, or a margin that vanishes on the positive a- and b-axes, i.e. confidence → 0 near them; C′ is of the first kind since m(a, 0) ≠ 0."

## CE4 (MINOR, scope): C′ conflict rectangle and relative-band statement beyond 3

**Conflict roots** (`c2_rect.py`):

| | 3 | 5 | 10 | 20 | 50 | 1000 |
|---|---|---|---|---|---|---|
| b*(a) | −0.0180 | −0.0262 | −0.0336 | −0.0371 | −0.0390 | −0.0401 |
| relative width \|b*\|/a | 0.60% | 0.52% | 0.34% | 0.19% | 0.08% | 0.004% |

The mirror case gives a*(b) = −0.0111, −0.0178, −0.0242, −0.0295 at b = 3, 5, 10, 50.

**Consequences.**
- The certified τ = 0.02 conflict rectangle cannot be extended past |strong| ≈ 3.2. Examples: m(10, −0.03) = +0.1457, m(3.5, −0.02) = +0.0303, m(−0.025, 10) = −0.0255 (that one is correct).
- "Conflict band ≈ 0.4–0.6% of the strong source" is false asymptotically. The band is ≈ 0.04 in absolute terms for large |strong|.
- The ratio-region certificate (≥ 0.05·|strong|) is not contradicted, since the relative band shrinks.

The certificates themselves were not broken inside their boxes:
- dense sampling of 55,000 points per box (corners and edges oversampled): 0 wrong signs; the smallest correct-signed margin was 0.037 at (0.09, 0.091);
- a float32 forward pass at the corners: no sign change.

Agreement beyond 3 (a up to 1000, b ∈ (0, 1]) has no root. Q−− is m < 0 on all of ℝ² by S−0.

**Correction.** Add "|a|, |b| ≤ 3" wherever the τ values or the 0.6% figure are quoted, e.g. in MAIN_RESEARCH_RESULT row 6. Optionally add: "the conflict band saturates at ≈ 0.040 (Q+−) and ≈ 0.03 (Q−+)".

## CE5 (MINOR, scope): the T2 "function of a + b" claim holds only on the training support

**Method** (`c7_learned.py`, all 15 T2 TF1/TF2/TF2AO checkpoints): random equal-sum pairs (a1, S − a1) vs (a2, S − a2), with both points inside the box. The statistic is mean (m1 − m2)² / Var(m).

| box | relative MSE | decision flips |
|---|---|---|
| \|a\|,\|b\| ≤ 3 | 5e−6 to 9e−5 | consistent with the claim (≤ 0.25%, all near S = 0) |
| ≤ 6 | 9e−4 to 3.3e−2 | up to 3.5% |
| ≤ 10 | 6e−3 to 0.2 | up to 9.7% |
| ≤ 30 | 0.08 to 0.65 | **up to 17%** |

Example (TF2 seed 1, S = −8.337): m(−1.882, −6.455) = −5.657 and m(21.601, −29.938) = +3.124.

**Correction.** Qualify the claim as "on the training support [−3,3]²". Likewise for "decodability is not relation computation" (the R² values) and the T1 relation-unit statistics, which were measured on the same grid.

## CE6 (MINOR, implementation): `closed_form` overflows

At `closed_form(1, 1, P(lam=300))` the line `F = sum(math.exp(s[k]) * v[k] …)` raises `OverflowError`. m itself is computed with a shifted softmax and is fine at λ = 100 and 200 (1.89112).

`suff` also calls `math.exp(-G*mu/sP)` unshifted.

**Correction.** Compute F as `exp(s_max)·Σ exp(s_k − s_max) v_k`, or return only sign(F) = sign(m). Use a log-domain comparison in `suff`.

## CE7 (MINOR, numerical): exact-arithmetic sign laws collapse to ties in float64

- **C′.** `closed_form(1e−17, 1.0)` gives q_R = G = 0, although sign(ab) = +. At 1e−15, q_R = 3.33e−15 is visibly quantized.
- **A1\*** (`c8_a1.py`; GELU computed as gelu(t) + gelu(−t), as in the network). At c = 1e−8 the share of q == 0 grows with the distractor norm:

| ‖w‖ | draws with q == 0 |
|---|---|
| 1e6 | 10/3000 |
| 1e9 | 2224/3000 |
| 1e12 | 3000/3000 |

  The cause is cancellation in t/2 − t/2 when |t| ≲ 1e−16. Also, q(x = 1, y = 1e−17) = 0.
- ReLU never gave q == 0 in this search. **No wrong sign occurred in any of the 108,000 draws** (|x|,|y| up to 50, ε ∈ [1e−12, 0.1], c ∈ {1e−8, 1.5, 1e8}, ‖w‖ up to 1e12).

**Correction.** Add one sentence: the PROVED sign laws are exact-arithmetic statements; in float64 they degrade to ties (never flips, in this search) when the weaker coordinate falls below ≈ 1e−16 of the stronger, or when c·|u|, c·|v| falls below ≈ 1e−16.

---

## Claims that survived, with search ranges

**C′1\* (S+, S−, S−0)** (`c1_cprime.py`).
- Search: 400 random parameter settings, log-uniform over λ ∈ [1e−2, 3e2], μ ∈ [1e−2, 30], κ, ρ ∈ [1e−3, 30], λ1, c1 ∈ [1e−2, 30], wv ∈ [1e−3, 100], π ∈ [1e−4, 1e3], ε ∈ [1e−15, 0.1]. Each setting had 100 points: uniform on [−50, 50]², on [−1e−3, 1e−3]², and log-magnitude over [1e−8, 50].
- Result: **0 violations in 39,887 evaluations** (29,782 decided). The 113 overflows are CE6.
- Outside the hypotheses (expected):
  - π < 0: 183 violations; μ < 0: 780 violations.
  - wv < 0, λ < 0, λ1 < 0, c1 < 0, π = 0 and wv = 0: 0 violations, because `suff` reads G and v directly. The hypotheses are sufficient, not necessary.

**Closed form = forward pass.** Max abs difference 8e−13 over 300 extreme parameter settings with |a|, |b| ≤ 50. No sign mismatch.

**Certified rectangles inside [τ, 3].** 4 × 55,000 dense samples gave 0 wrong signs. The float32 forward pass at the corners and the closest-to-boundary points gave no sign change.

**A1\*.** No wrong sign over 108,000 draws; ranges as in CE7.

**A1\* cones** (cross-talk, unequal weights). Re-derived algebraically, they are correct: conflict is correct iff |ρ| ∈ (t21, 1/t12) for t12, t21 > 0 with t12·t21 < 1, and the GELU small/large limits are (√r − 1)/(√r + 1) and (r − 1)/(r + 1). No numeric counterexample was sought beyond the package's M4–M6.

**T-B-TM without a final LN.** Exact to 9e−15 in three parameter settings × 4000 draws, with zB norms up to ≈ 50 (`c3_btm.py`).

**T-C4 (1) bypass dominance.** Elementary (|Σ α v| ≤ max |v|). Correct.

**T-R (reachability).**
- Re-derived: ∂Ŝ/∂y ∝ x² − xy + ρ² > 0 and ∂Ŝ/∂x ∝ y² − xy + ρ² > 0 in conflict, so the supremum d/√(d + ρ²) is approached at x → √d, y → 0⁻ and is not attained. Correct (with ε ignored, as stated).
- The C′ analogue (X4: "(2, 2) has no partner with equal m_blind") survives: m_blind(2, 2) = 1.4728, against a conflict supremum of 1.3072 over a ∈ (0, 1e4] and b ∈ [−10, −1e−8] (attained near (1e4, −1e−8)), and 0.706 for a < 0 < b (`c6_blind.py`).

**C′-CO parts 1 and 2, as conditional statements.** Correct. Only the "consequence" wording fails (CE3).

**T-RF (discrete) and the XNOR non-representability.** Proofs re-checked. Correct.

**T2 sum-only within [−3, 3]².** Equal-sum relative MSE ≤ 9e−5 for all 15 checkpoints. This is consistent with D_sum on the package's lattice; my 121-point grid gives 1.2e−4 because each bin holds two sum values.

## Not examined in depth

- T1 relation-unit counts and ablation numbers outside [−3, 3]. These are expected to drift, as in CE5.
- T5 (K = 3) checkpoints.
- Program C variants other than C′ (redundant copies).
