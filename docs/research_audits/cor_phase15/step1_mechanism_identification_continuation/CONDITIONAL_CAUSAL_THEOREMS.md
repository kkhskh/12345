# Conditional causal and decision theorems (§13, Stage C4)

Each theorem below states the elements that §13 requires: the model class, candidate set, frozen parameters, source semantics, consistency definition, intervention family, signed conditions, competing pathways, readout and conclusion.

**Status labels**

| label | meaning |
|---|---|
| PROVED | proof given here or in the referenced document |
| VERIFIED | checked numerically on fresh points or parameters |
| CERTIFIED | checked by interval arithmetic |

---

## T-A1*. MLP agreement law under shared LN2

**Elements**

- **Class:** one pre-norm block. The read position's pre-LN2 residual is `x e1 + y e2 + w`, with `w ⟂ e1, e2`.
- **Candidates:** {c, c′}.
- **Frozen parameters:** scalar LN2 gain, LN2 bias orthogonal to e1 and e2, read directions `±c(e1±e2)`, and output weights `(w, w, −w, −w)` onto the margin direction.
- **Source semantics:** `sign a`, `sign b`.
- **Consistency:** D1 (`ab > 0`).
- **Interventions:** any recomputed source change.
- **Competing paths:** none. If other paths exist, add their margin.
- **Readout:** linear, or a final LN with zero bias and a mean-zero readout difference.

**Conclusion:** `sign m = sign(ab)`.

- Status: PROVED; 0 violations in 120,000 checks.
- Without the conditions, the derived cones hold (cross-talk, gain and bias, unequal weights), again with 0 violations.
- Details: `MLP_NORMALIZATION_THEOREMS.md`.

---

## T-C′1*. Conditional rereading

**Elements**

- **Class:** the structural form of C′.
- **Candidates:** {c, c′}, with the prior voting c′.
- **Frozen parameters:** `λ, μ, λ1, c1, wv, π > 0`.

**Conclusion:**

- The circuit defers to the prior if `|G| > (σ_P/μ)·ln(v⁺-mass/Π)` with `G ≤ 0` (conflict).
- It follows the sources if `G > (σ_P/μ)·ln(Π/(v_A + v_B + v_R))` with all values ≥ 0 (agreement for c).
- If all values are ≤ 0, the result is c′.

**What decides each outcome**

| question | determined by |
|---|---|
| source versus prior selection | sign of G (= sign(ab)) together with its magnitude relative to the log value-mass ratio |
| direction of the margin effect | the signs of the source values versus the prior value |

**Status:** PROVED. 0 violations at fresh points and at 40 fresh parameter settings. Operating rectangles CERTIFIED. Details: `CPRIME_CORRECTED_OPERATING_REGIONS.md`.

---

## T-CO. Continuity obstruction

- **Statement:** no continuous margin implements "sign a if the sources agree, c′ if they conflict" on any neighbourhood of a point on the positive axes where the margin is nonzero.
- **Consequence:** every relation-gated sign rule needs a dead band.
- **Status:** PROVED.

---

## T-B-IMP. Bilinear impossibility (K ≥ 3)

- **Statement:** no sign of a bilinear-plus-affine form equals argmax agreement on a set of full measure. Hence no single attention score with zero LN bias can switch exactly on candidate-relative agreement.
- **Status:** PROVED.
- **Empirical gap:** the best bilinear classifier reaches 0.821, against 0.987 for an MLP.

---

## T-B-TM. Total-margin consequence of a gate

**Class:** a gate head whose score change touches only `s_B`, with:

- frozen value margins `w_i`;
- no competing writer or rereader that depends on the gate;
- a final LN, if present, with zero bias.

**Conclusion:** `ΔΔ = (α′_B − α_B)(w_B − w̄_{−B})`.

**Status:** PROVED; 3,000 of 3,000 signs correct.

**Each assumption fails when removed:**

| attack | effect |
|---|---|
| competing writer | cancels the effect at kC = 3 and reverses it beyond |
| rereader | reverses the sign |
| final-LN bias | changes the decision but not the sign of the routing effect |

---

## T-C4. Multi-layer gate with bypass and downstream map

**Elements**

- **Class:**
  - an L-layer pre-norm network;
  - margin `m = Σ_j α_j(G) v_j + m_bp`, where `m_bp` is every contribution that does not pass through the gate head;
  - followed by a downstream scalar map `φ` acting on the margin.
- **Assumptions:**
  - (i) The relation feature affects `m` only through the gate's scores. This was checked by C2: values carry 1.8% of the effect.
  - (ii) `v_j` and `m_bp` are inspectable frozen-path quantities.
  - (iii) `φ` is strictly increasing with `φ(0) = 0`.

**Conclusions**

1. **Bypass dominance.** If `|m_bp| > max_j |v_j|`, then `sign m = sign m_bp` whatever the routing. The gate cannot change the decision.
   - VERIFIED: 0 violations at 107 points where the condition held.
2. **Gate-decided region.** If `|m_bp| < min over gated choices of |Σ α v|`, the decision is the gate's. The direction of the effect is given by T-B-TM, with `w̄` taken over the non-gate positions.
3. **Downstream preservation.**
   - A strictly increasing `φ` with `φ(0) = 0` preserves every decision, as with the compressive MLP at γ ≤ 1 (VERIFIED: rule 1.000 / 1.000).
   - A non-monotone `φ` can reverse it, as with γ = 3 (rule 0.680 / 0.440).
   - A downstream component that re-reads the sources adds to `m_bp`, so conclusion 1 applies to it. With the rereading head at w = 4, the conflict rule drops to 0.844.

**Status:** PROVED (each conclusion is elementary given the decomposition); VERIFIED in the C3 variants.

**Scope.** The decomposition in assumption (i) is interventionally meaningful only when the gate head's contribution can be isolated. That holds when the gate is one head and its output is not re-read. It is not a property of arbitrary networks.

---

## T-RF. Relation-free representability of candidate-directed agreement rules (new, exploratory)

**Statement.**

- Take any candidate set of size K ≥ 2 and any prior candidate p.
- Define the candidate-directed agreement rule `F(k_A, k_B, p) = k` if `k_A = k_B = k`, and `= p` otherwise.
- **Claim:** F is implemented exactly by the argmax of source-additive scores, `score_k = 1[k_A = k] + 1[k_B = k] + 1.5·1[p = k]`.

**Proof.** Check the four cases:

| case | winning score | runner-up | winner |
|---|---|---|---|
| both sources pick `k ≠ p` | k: 2 | p: 1.5 | k |
| both pick p | p: 3.5 | — | p |
| sources disagree, one picks p | p: 2.5 | others ≤ 1 | p |
| sources disagree, neither picks p | p: 1.5 | others ≤ 1 | p |

∎

**Binary case.** The rule is `y = sign(sign a + sign b + p)`, a linear threshold function in sign space:

- AND when `p = c′`;
- OR when `p = c`.

**Contrast with candidate-symmetric agreement.** The rule "output c iff the sources agree, in either direction" (XNOR, T1) is **not** source-additively sign-representable. Proof:

1. Take `f(a) + g(b)` and use the sign constraints from `a > 0` and from `a < 0`.
2. They give `f(a⁺) < −g(b⁻) < f(a⁻)` and `f(a⁻) < −g(b⁺) < f(a⁺)`.
3. These are contradictory. ∎

**Checks**

- The LP feasibility test on grids up to 48 × 48 (`src/additive_lp.py`) gives:
  - T2, T3 (`p = ±1`) and T4: feasible;
  - T1: infeasible.
- An explicit pre-norm Transformer in which **no unit reads two sources** (`src/relation_free_equivalent.py`) reproduces C′'s decisions on:
  - 100% of the 60 × 60 grid with `|a|, |b| ≥ 0.1`;
  - 100% of 19,577 random points outside C′'s dead bands.

**Consequence.** A candidate-directed agreement rule, such as "follow agreeing sources, else defer", is **never behavioural evidence** of relation computation. The same decisions are produced by threshold logic on per-source support.

Only internal interventions can tell C′ apart from its relation-free twin. Examples:

- the relation-blind matched pairs (X4);
- the `q_R` interchange (C2);
- ablation of the block-1 MLP.

Only candidate-symmetric rules (T1-type) force a cross-source interaction.

---

## T-SS and T-R. Sufficient statistic and reachability

- **T-SS:** with no bypass, equal statistic implies equal output.
- **T-R:** matched-aggregate conflict partners exist only for `|Ŝ| < d/sqrt(d+ρ²)`.
- **Status:** PROVED. Details: `MATCHED_RELATION_INTERVENTIONS.md`.
