# Conditional causal and decision theorems (§13, Stage C4)

This version follows the review. `ADJUDICATION.md` records every change. Each theorem below gives its model class, conditions and conclusion.

**Status labels**

| label | meaning |
|---|---|
| PROVED | proof given here or in the referenced document |
| VERIFIED | checked numerically on fresh points or parameters |
| CERTIFIED | checked by interval arithmetic |

All of these results are known mathematics, such as threshold logic, convexity, continuity and softmax algebra, applied to this question. None is claimed as new mathematics (R5).

---

## T-A1*. MLP agreement law under shared LN2

**Elements**

- **Class:** one pre-norm block. The read position's pre-LN2 residual is `x e1 + y e2 + w`, with `w ⟂ e1, e2`.
- **Conditions:**
  - LN2 gain is a scalar, `γI` with `γ > 0`;
  - LN2 bias is orthogonal to e1 and e2;
  - the read directions are `±c(e1 ± e2)`;
  - the output weights are `(w, w, −w, −w)`.
- **Consistency:** D1 (`ab > 0`).

**Conclusion:** `sign m = sign(ab)`.

- Status: PROVED; 0 violations in 120,000 checks.
- Outside the conditions, the derived cones apply instead. These include the ReLU unequal-weight cone with **two** bounds, `(r−1)/(r+1) < t < (r+1)/(r−1)`, and two failure cones in the mixed-sign cross-talk case.
- Details: `MLP_NORMALIZATION_THEOREMS.md`.

---

## T-C′1*. Conditional rereading

**Elements**

- **Class:** the structural form of C′.
- **Candidates:** {c, c′}, with the prior voting c′.
- **Frozen parameters:** `λ, μ, λ1, c1, wv, π > 0`.

Let `Π = πμ/σ_P`. Each condition below implies the stated decision:

| case | condition | decision |
|---|---|---|
| (S+) agreement for c | `G ≥ 0`, all of `v_A, v_B, v_R ≥ 0`, and `v_A + v_B + v_R > Π·e^{−Gμ/σ_P}` | **m > 0**: follow the sources |
| (S−) conflict | `G ≤ 0` and `Π·e^{\|G\|μ/σ_P} > v_A⁺ + v_B⁺ + v_R⁺` | **m < 0**: defer to the prior |
| (S−0) | all values ≤ 0 | **m < 0** |

The `G ≥ 0` hypothesis in (S+) was missing from the reviewed version of this document. Without it, 657 of 2,167 conflict points satisfying the rest are violations (R1 F1). It is restored here. The statement in `CPRIME_…` §3 was correct.

**Status:**

- PROVED, with 0 violations at fresh points and at 40 fresh parameter settings.
- 0 violations in 39,887 extreme-parameter evaluations (R6).
- Operating rectangles CERTIFIED on [τ, 3].

---

## T-CO. Continuity obstruction (corrected)

**Statement.** Let `m` be a continuous margin that implements "sign a if the sources agree, c′ if they conflict" at every point off the axes. Then `m` vanishes on the positive axes and at the origin.

**Contrapositive.** If `m` is nonzero anywhere on the positive axes, it fails on a set of positive measure near that point. That is a dead band.

**Example.** `min(a, b)` meets the statement with no dead band.

**Withdrawn.** The earlier corollary "every relation-gated sign rule needs a dead band" is false.

**Status:** PROVED.

---

## T-B-IMP. Bilinear impossibility for K ≥ 3 (scoped)

**Statement.** Let K ≥ 3, and let `f(z_A, z_B) = z_Aᵀ M z_B + αᵀz_A + βᵀz_B + γ`, with `z_A, z_B` **linear** encodings of the evidence. Then `sign f ≠ 1[argmax z_A = argmax z_B]` on a set of positive measure.

*Proof.*

1. For each fixed `z_B` with argmax k, `{z_A : f > 0}` is a half-space. The agreement region `C_k` is a polyhedral cone whose complement is not convex: there is a midpoint witness with balls around it.
2. So the two sets differ on a positive-measure set of `z_A`.
3. This holds for every `z_B` in an open set. By Fubini, they differ on a positive-measure set in `(z_A, z_B)`. ∎

**Corollary.** A single attention score that is bilinear in linearly encoded evidence cannot implement argmax agreement exactly. This assumes zero LN bias, so that LN is a positive rescaling.

**Scope (corrected, R3).** The corollary does **not** extend to queries and keys that pass through a per-source nonlinearity before the score. With one-hot(argmax) features, a single bilinear score detects agreement exactly (B1b: 1.000).

**Empirical:** the best bilinear + linear classifier on raw evidence reaches 0.821, against 0.987 for an MLP.

**Status:** PROVED.

---

## T-B-TM. Total-margin consequence of a gate (corrected)

**Class.**

- The score change affects only `s_B`.
- The value margins `w_i` are frozen.
- No competing writer or rereader depends on the gate.
- **There is no final LN.** The reviewed version claimed the result also held under a zero-bias final LN; that clause is **withdrawn**. The LN scale depends on the routing state, so it does not preserve the sign of a margin *change*. Measured agreement was 2,980/3,000 and 2,960/3,000 (R1 F2, R3, R6).

**Conclusion:** `ΔΔ = (α′_B − α_B)(w_B − w̄_{−B})`.

**Status:** PROVED. Checks:

- 3,000/3,000, including a nontrivial version with nonzero value at A;
- exact to 9e−15 (R6).

**Attacks:**

| attack | effect |
|---|---|
| competing writer | cancels the effect at kC = 3 and reverses it beyond |
| rereader | reverses it |
| final-LN bias | changes the decision |

---

## T-C4. Multi-layer gate with bypass and downstream map (corrected)

**Class.**

- `m = Σ_j α_j(G) v_j + m_bp`, where `m_bp` is the block-2 input projected on the readout. This is the corrected site: `resid_in[1]`.
- Assumption (i): the relation feature enters `m` only through the gate's scores.
  - This was measured as **q_B's** share of TE: 0, 0 and −0.010 across three counterfactual families.
  - The value path does carry large shares of a *source* change (0.91 for an a-flip). That is the source's own value effect, not the relation feature's.

**Conclusions.**

1. **Bypass dominance.** If `|m_bp| > max_j |v_j|`, then `sign m = sign m_bp`.
   - VERIFIED: 107 points, 0 violations (recomputed at the corrected site).
2. **Gate-decided region.** If `|m_bp|` is smaller than every reachable `|Σ α v|`, the decision follows the gate. The direction follows T-B-TM, without a final LN.
3. **Downstream maps.**
   - A downstream component that is a strictly increasing function of the margin with `φ(0) = 0` preserves decisions.
   - The tested downstream MLP is *not* such a scalar map: it reads an LN-normed residual. It preserved decisions at γ ≤ 1 and reversed them at γ = 3 (0.680 / 0.440).
   - A rereading head adds to `m_bp`, so conclusion 1 applies to it.

**Status:**

- Conclusion 1: PROVED and VERIFIED.
- Conclusion 3, scalar case: PROVED.
- Conclusion 3, tested MLP: empirical.

---

## T-RF. Fixed-prior deferral rules are threshold logic (scoped)

**Statement.** For K ≥ 2 and a prior candidate p, define the rule `F(k_A, k_B, p) = k` if `k_A = k_B = k`, and `p` otherwise. F is exactly the argmax of the additive scores `1[k_A = k] + 1[k_B = k] + 1.5·1[p = k]`.

**Binary case.** `y = sign(sign a + sign b + p)`, which is a linear threshold function (Minsky–Papert).

**Check:** B-signs, a linear model on per-source sign one-hots plus the prior, scores 1.000 on T3 and T5b.

### Scope corrections

1. **Not every candidate-directed rule.** The K = 3 rule "agree → that candidate, else the candidate *neither* picked" is not additively representable. LP infeasibility shows it (R2). So the reviewed claim "only candidate-symmetric rules force a cross-source interaction" is **withdrawn**.
2. **XNOR.** The candidate-symmetric agreement rule (XNOR) is not additively sign-representable. *Proof:* the sign constraints give `f(a⁺) < −g(b⁻) < f(a⁻) < −g(b⁺) < f(a⁺)`, a contradiction. ∎
3. **Grid LP results.** T2, T3 (p = ±1) and T4 are feasible on grids up to 48 × 48. T1 is not.
4. **Decision level only.** T-RF is a statement about decisions. "No unit reads two sources" is not equivalent to it in a pre-norm Transformer, because shared LN couples the sources (N2).

### The relation-free twin

The twin is `src/relation_free_equivalent.py`. Its margin is exactly additive, with interaction ≤ 4e−16 (R3).

- On [−3, 3]², outside C′'s dead bands, it reproduces C′'s decisions.
- On [−10, 10]², it **disagrees with C′** on 735 of 19,888 points (R6). The block-2 LN shrinks its support at large |a|.
- The equivalence claim is therefore scoped to [−3, 3]².

---

## T-RF′. Graded-output single-cancellation test (added after review; requires a fresh review)

**Statement.** Suppose `m(a, b) = h(f(a) + g(b))` with h strictly increasing. Then for every pair `a, a′`, the sign of `m(a, b) − m(a′, b)` is the same for every b. Equivalently, there are no order reversals between rows.

*Proof.* `m(a, b) > m(a′, b)` iff `f(a) > f(a′)`, which does not depend on b. ∎

**Consequence.** A graded margin, i.e. a confidence and not just a decision, *can* behaviourally certify a cross-source interaction. It cannot certify that the interaction is **agreement** specifically.

**Checks** (`logs/mediator_sign_test.txt`, S3; `logs/learned_review_checks.txt`, K4):

| model | row pairs with order reversal |
|---|---|
| C′ | 357 / 435 |
| additive-gate twin | 0 / 435 |
| relation-free twin | 0 / 435 |
| learned T1 | 255 / 465 (up to 2.1 SD) |
| learned T2 | 0 |
| learned T3 | ≤ 0.07 SD (TF1/TF2); ≤ 0.25 SD (TF2-AO) |
| learned T4 | 28–337 pairs, up to 1.1 SD |

---

## T-MS. Mediator sign test for a candidate-symmetric relation gate (added after review; requires a fresh review)

**Statement.** Suppose a gate feature `q(a, b)` mediates the decision and is to be called a *candidate-symmetric relation*, in the D1 sense. Then it must satisfy `sign q(±a, ±b) = sign(ab)` in **all four** sign quadrants, including agreement for c′.

**Separating test.** An additive AND gate can match a relation gate's decisions and relation-blind pathway. It fails the (−,−) quadrant.

**Checks.** S1 and S2 in `src/mediator_sign_test.py`:

| circuit | sign pattern (500 draws) | (−,−) → (+,−) interchange routes to |
|---|---|---|
| C′ | holds 500/500 | sources: 1.000 |
| additive twin | holds 0/500 | prior: 1.000 |

The S2 interchange is off-manifold.

---

## T-SS and T-R. Sufficient statistic and reachability

- **T-SS:** with no bypass, equal statistic implies equal output.
- **T-R:** the supremum of `|Ŝ|` for conflicting partners is `d/sqrt(d+ρ²)` = 1.549 for A, approached as `|a| → ∞`. Within `|a| ≤ 3`, the maximum is 1.376 (R1 F12).
- For C′, the corresponding reachability limits in X4 are numerical, not derived.
- **Status:** PROVED for A.
