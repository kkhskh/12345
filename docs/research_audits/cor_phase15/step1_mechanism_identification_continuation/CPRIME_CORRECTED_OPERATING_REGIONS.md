# C′ corrected operating regions

Correction 2.1 of the continuation mandate. Code: `src/cprime.py` (closed form and interval arithmetic) and `src/cprime_regions.py` (experiments E1–E8). Log: `logs/cprime_regions.txt`. The historical package is untouched; its claim C′1 is superseded here, not edited there.

## 0. What was wrong

The historical claim C′1 said that in agreeing cells `sign m = sign a`, and in conflicting cells `m < 0` whichever source is stronger. **As an unrestricted statement it is false.** The independent audit's counterexamples are reproduced exactly (E2):

| (a, b) | m | target |
|---|---|---|
| (0.01, 0.01) | −0.6308419542 | + |
| (0.05, 0.05) | −0.3576101163 | + |
| (0.08, 0.08) | −0.0605622099 | + |
| (1, −0.001) | +0.1946247153 | − |
| (0, 0) | −0.6577909845 | undefined |
| (1, 0) | +0.2651922423 | undefined |

The historical grid never sampled `|b| < 0.6` in conflict or `|a|, |b| < 0.5` in agreement. The claim was an over-extrapolation from that grid.

## 1. Closed form (exact; derived from the circuit, not fitted)

All token embeddings are mean-zero in the 10-dimensional residual stream, and every LN has unit gain and zero bias. Hence `LN(z) = z/σ(z)` with `σ(z) = sqrt(‖z‖²/d + ε)`. Parameters: `d=10, μ=3, κ=2, ρ=3, λ1=1.5, c1=1.5, λ=6, wv=1, π=prior_v=1, ε=1e−5`. Let `h(t) = GELU(t) + GELU(−t) = t·tanh(K(t + 0.044715 t³))`, where `K = sqrt(2/π)`.

1. **Block-1 attention.** The head is uniform and causal, with `WO = 4(e1e1ᵀ + e2e2ᵀ)`. Let `x = a/σ_A` and `y = b/σ_B` be the source LN coordinates, where `σ_A = sqrt((a²+κ²+μ²)/d + ε)` and `σ_B` is defined likewise. The e1/e2 coordinates after the head are:
   - A: `(a + 2x, 0)`
   - B: `(4x/3, b + 4y/3)`
   - R: `(x, y)`
2. **Block-1 MLP (agreement detector).** At B and at R it writes `λ1·q·e3`, where `q = h(c1(u+v)) − h(c1(u−v))` and `(u, v)` are the LN2 coordinates of e1/e2 at that position.
3. **Block-2 head.** Let `q̂ = λ1 q_R / σ_R2` and `G = λ q̂/√2`. The scores and values are:

   | position j | score s_j | value v_j |
   |---|---|---|
   | P | `−Gμ/σ_P` | `−πμ/σ_P` |
   | A | `+Gμ/σ_A2` | `wv(a+2x)/σ_A2` |
   | B | `+Gμ/σ_B2` | `wv(4x/3 + b + 4y/3)/σ_B2` |
   | R | `0` | `wv(x+y)/σ_R2` |

4. **Margin.** `m = Σ_j α_j v_j`, with `α = softmax(s)`. **Decision function:** `F = Σ_j e^{s_j} v_j = Z·m`, so `sign F = sign m`.

**E1 check.** On 400 fresh points (seed 101) the closed form matches the recomputed forward pass of `cbmath.rereading` to `1.4e−14`.

## 2. Exact decision boundary

**The exact boundary is `{(a,b) : F(a,b) = 0}`** with F as above. It is an explicit, elementary, analytic function, so the boundary is that level set. E4 locates it by bracketing and bisection:

- **Agreement (Q++).** There is one root per ray. The floor is `r*` = 0.235 at 5°, 0.159 at 15°, 0.128 at 30°, **0.122 at 45°**, 0.133 at 60°, 0.172 at 75° and 0.270 at 85°. m < 0 inside the floor and m > 0 outside it.
- **Conflict, a > 0 > b.** For `a ≤ 0.5` there is no root in `b ∈ [−3, 0)`: the prior wins throughout. For `a = 1, 1.5, 2, 3` the root is `b* = −0.0036, −0.0076, −0.0114, −0.0180`, so the error band has relative width `|b*|/a ≈ 0.4–0.6%` on `[0, 3]`.
  - **Beyond 3 (post-review, E6b):** the band grows in absolute terms and saturates. `b*` is −0.026, −0.034, −0.037 and −0.039 at `a` = 5, 10, 20 and 50. So the "≈ 0.6% of the strong source" description holds only up to `a ≈ 3`.
  - Uniqueness of the root along each ray and line is established numerically, not proved.
- **Conflict, a < 0 < b.** The analogue has roots `a* = −0.0028, −0.0059, −0.0111` for `b = 1.5, 2, 3`, and none for `b ≤ 1`.
- **Both negative (Q−−).** There is no boundary: m < 0 everywhere.

## 3. Corrected theorem C′1* (analytic sufficient conditions)

These conditions are derived from the sign structure of the scores and values, not from thresholds chosen to exclude the counterexamples.

**Lemma (sign of the routing gain).** `sign G = sign q_R = sign(xy) = sign(ab)`.

Proof:
1. `h` is even and strictly increasing in `|t|`; see `MLP_NORMALIZATION_THEOREMS.md` L1.
2. Hence `q_R > 0` iff `|u+v| > |u−v|` iff `uv > 0`.
3. `u, v` are positive multiples of `x, y`, which have the signs of `a, b`. ∎

**Theorem C′1*.** For every parameter setting of the structural form with `μ, λ, λ1, c1, wv, π > 0`, let `Π = πμ/σ_P`. Then:

- **(S+)** If `G ≥ 0`, `v_A, v_B, v_R ≥ 0`, and `v_A + v_B + v_R > Π·exp(−Gμ/σ_P)`, then `m > 0`.
- **(S−)** If `G ≤ 0` and `Π·exp(|G|μ/σ_P) > v_A⁺ + v_B⁺ + v_R⁺`, then `m < 0`.
- **(S−0)** If `v_A, v_B, v_R ≤ 0`, then `m < 0`.

Proof:
- (S+): `G ≥ 0` gives `s_A, s_B ≥ 0`, so `e^{s_A}v_A + e^{s_B}v_B + v_R ≥ v_A + v_B + v_R > Π e^{s_P} = −e^{s_P}v_P`. Hence `F > 0`.
- (S−): `G ≤ 0` gives `e^{s_A}, e^{s_B} ≤ 1`, so the positive part of `F − e^{s_P}v_P` is at most `v_A⁺ + v_B⁺ + v_R⁺ < Π e^{s_P}`.
- (S−0) is immediate. ∎

**Reading.** The circuit follows the sources when the routing gain `G` overcomes the log-ratio of prior mass to evidence mass, `G > (σ_P/μ)·ln(Π/(v_A+v_B+v_R))`. It defers to the prior when `|G| > (σ_P/μ)·ln((v⁺ mass)/Π)`. Both conditions involve two separate computed quantities:
- the relation feature, through `G`;
- the aggregate evidence magnitude, through the value mass.

**The decision is not a function of `G` alone.** This needs a witness, because sufficient conditions alone do not imply it (post-review, R1 F8). The witness: `(0.07, 0.07)` and `(0.2202, 0.02296)` have the same routing gain, `G = 0.200124`, but margins −0.159 and +0.0008.

**Validation on new points (E5, E8).**
- On 20,000 fresh points (seed 202), the sufficient conditions have **0 soundness violations**. They decide 99.93% of the points where the target rule is correct, so they are nearly tight.
- At 40 fresh random parameter settings × 300 points (seed 303), there are **0 violations in 12,000**. The median decided fraction is 0.997 and the minimum is 0.953.

## 4. Certified inner and outer approximations

The certification is an interval-arithmetic implementation of F with outward rounding on every operation, combined with branch-and-bound down to box width `2e−3` (E3).

**Post-review hardening (R1 F6).** The original single-ulp outward rounding did not enclose every endpoint. The fix:
- relative padding of about 4 ulps on every operation;
- enclosed constants;
- doubled padding on the endpoint evaluation of `x(a)`.

Every number in this section was reproduced unchanged with the hardened arithmetic.

| region | certified correct (inner approx.) | certified WRONG | undecided |
|---|---|---|---|
| Q++ agree, target + | 99.640% | 0.309% | 0.051% |
| Q−− agree, target − | 100% | 0% | 0% |
| Q+− conflict, target − | 99.708% | 0.222% | 0.070% |
| Q−+ conflict, target − | 99.845% | 0.100% | 0.056% |

For each quadrant, the certified-correct boxes form an **inner approximation** of the correct region, and the correct boxes together with the undecided boxes form an **outer approximation**. The certified-WRONG boxes are a certified error set, i.e. positive-measure counterexample regions rather than isolated points.

**Certified rectangular operating regions (E6).** The rule is certified on `{τ ≤ |a|, |b| ≤ 3}` with:
- τ = 0.09 (Q++)
- τ = 0 (Q−−)
- τ = 0.02 (Q+− and Q−+)

It is also certified on the conflict region `{0.05·|strong| ≤ |weak| ≤ |strong|, |strong| ∈ [0.2, 3.05]}`, in **both** conflict quadrants.

The original code's boxes left about 2.5% of this region uncovered, and the Q−+ quadrant was never run (R1 F7). The region is now covered by superset boxes and certified for both Q+− and Q−+ at ratios 0.05, 0.1, 0.2, 0.3 and 0.5.

The rectangles are claimed only on `[τ, 3]`: the conflict band grows beyond 3 (E6b).

## 5. What continuity does and does not force (corrected after review)

**Proposition C′-CO.** Let m be continuous on ℝ². Every Transformer margin with ε > 0 is. Let `T(a,b) = sign a` if `ab > 0`, and `T = −1` if `ab < 0`.

1. If `m(a0, 0) ≠ 0` for some `a0 > 0`, then T fails on a set of positive measure in every neighbourhood of `(a0, 0)`. The same holds on the positive b-axis.
2. If `m(0, 0) ≠ 0`, T fails on a positive-measure set in every neighbourhood of the origin.

*Proof.*
1. T is +1 just above the positive a-axis and −1 just below it.
2. m has a constant sign on a small ball around `(a0, 0)`, so it disagrees with T on one of the two half-balls.
3. Part 2 follows the same way. ∎

**Correct consequence.** A continuous margin implements T everywhere off the axes **only if it vanishes on the positive axes and at the origin**. Example: `m = min(a, b)`, or `min(u, v)` on shared-LN coordinates. Such a margin gives zero confidence at the boundary but has **no dead band**: it implements T at every point with `ab ≠ 0` (R1, R5, R6; 0 errors in 10⁶ points).

**What this implies for C′.**
- C′'s dead bands come from its design: `m(a, 0) ≠ 0` and `m(0, 0) = −0.658` are nonzero because of the prior term.
- The historical unrestricted C′1 was **false for C′**. Its form was not impossible.
- An earlier version of this section claimed "every relation-gated sign rule needs a dead band". **That claim was wrong and is withdrawn.**

C′'s bands are:
- an agreement floor of about 0.12 at the diagonal;
- a conflict band of about 0.6% of the strong source on [0, 3].

The trade-off below is a property of the prior-plus-gate design, not of all continuous circuits.

## 6. Failure mechanism and the parameter trade-off (E7, E8)

**Mechanism of the failures (E7).**
- **Near-zero agreement.** `q_R ≈ 4Kc1²·uv` is quadratic in the evidence, so `G → 0` and routing becomes uniform (`α_P ≈ 0.25`). The evidence value mass is linear and small (0.07 at (0.01, 0.01)), while the prior mass is a constant 2.63. **The prior wins.**
- **Weak-conflict.** At (1, −0.001), `G = −0.02` is too small to concentrate attention on P (`α_P = 0.27`), while the strong source's value mass is 3.69. **The sources win.**
- In both cases the failure arises because the relation feature is second order in the weaker source, while the aggregate is first order.

**Parameter trade-off (E8).**
- Raising the routing gain λ shrinks both bands. Going from λ = 3 to λ = 24 at π = 1:
  - the agreement floor goes from 0.160 to 0.075;
  - the conflict root at a = 2 goes from −0.023 to −0.003.
- Raising the prior π shrinks the conflict band but raises the agreement floor. At λ = 6:
  - π = 0.25 gives floor 0.080 and root −0.034;
  - π = 2 gives floor 0.145 and root −0.001.
- No tested parameter setting removes both bands. This is a property of C′'s prior-plus-gate form. It is not forced by continuity alone; see §5.

## 7. Status

| claim | status |
|---|---|
| historical C′1 (unrestricted) | **FALSE** (positive-measure certified error sets, E3) |
| C′1* sufficient conditions S+/S−/S−0 | **PROVED**; 0 violations on fresh points and fresh parameters |
| exact boundary = `{F = 0}` with explicit F | **DERIVED**, verified against the forward pass to 1e−14 |
| certified operating rectangles (τ = 0.09 / 0 / 0.02 / 0.02, on [τ, 3]); ratio regions in both conflict quadrants | **CERTIFIED** (hardened interval arithmetic) |
| continuity obstruction C′-CO | **PROVED** in its conditional form. The corollary "every gated rule needs a dead band" is **withdrawn** (counterexample: `min(a, b)`) |
| decision depends on relation **and** aggregate magnitude | **SHOWN** by witness (equal G, opposite margins) |
