# R2 — Causal review (identification, mediation, bypasses, off-manifold interventions)

Package: `step1_mechanism_identification_continuation` @ d1aa97d. Reviewer role: causal / identification.
All checks below were re-run by the reviewer. Scripts are in `reviews/r2_scripts/`:

| script | what it tests |
|---|---|
| `x4_twin.py` | X4, C2 and block-1-MLP ablation on a same-architecture twin with a per-source (relation-free) gate feature |
| `c2_check.py`, `c2_aflip.py` | C2 value-route share, its direction dependence, and routing asymmetry |
| `learned_ctrl.py`, `sites_ctrl.py` | specificity controls for the probe-direction interchange (readout direction, per-source probes, subspaces, all sites) |
| `ablate_ctrl.py` | mean ablation vs resample ablation vs complement-unit ablation |
| `trf_scope.py` | T-RF scope: graded-behaviour (margin) test, and a K = 3 candidate-directed counterexample |

Run them from inside `reviews/r2_scripts/`. Each one takes under 2 minutes.

---

## Findings

### 1. FATAL — X4, C2 and block-1-MLP ablation do not separate "uses the relation" from a relation-free gate

**Where.**
- `MAIN_RESEARCH_RESULT.md`, Positive 2: X4 is "an on-manifold identification criterion that separates 'uses the relation' from 'uses the aggregate'".
- `MATCHED_RELATION_INTERVENTIONS.md` §4: "The only computational difference between them is the relation feature".
- `CONDITIONAL_CAUSAL_THEOREMS.md`, T-RF Consequence: "Only internal interventions can tell C′ apart from its relation-free twin. Examples: X4; the q_R interchange (C2); ablation of the block-1 MLP."

**Evidence (`x4_twin.py`).** I built a twin of C′ with exactly C′'s weights, except for the block-1 MLP. Its four ReLU units each read **one** source direction (`k·e1` or `k·e2`), and together they write `q = (λ/2)(φ(u) + φ(v) − 1)` onto e3, where `φ = clamp(k·)`. This is an additive per-source feature, thresholded: T-RF's AND. No unit reads two sources. The only cross-source coupling is the shared LN2 scale at R, which does not change any sign. Results:

| test | C′ | additive twin |
|---|---|---|
| decisions on the 60 × 60 grid, \|a\|,\|b\| ≥ 0.1 | — | identical on 100% |
| relation-blind model (block-1 MLP removed) | — | **bit-identical to C′'s** (max diff 0.0), so the matched pairs are the same pairs |
| X4 (1,1) vs (3,−0.238) | +1.890 / −2.630, flip | +1.994 / −2.344, **flip** |
| X4 (1.5,0.5) vs (3,−0.200) | flip | **flip** |
| X4 (0.6,0.6) vs (2,−0.634) | flip | **flip** |
| C2 q_R interchange: median share of TE, decision → cf | 1.000, 1.000 | **1.000, 1.000** |
| block-1 MLP ablation flips matched pairs | yes | **yes** |

The package's own relation-free twin (`relation_free_equivalent.py`) gives the same verdict. With its block-1 MLP removed, its margin is the constant −0.197, so every pair is "m_blind-matched", yet its full decisions differ (+0.211 vs −0.243 on the X4 pair). By the X4 criterion, it too "uses the relation".

**Why this happens.** Matching `m_blind` fixes one scalar functional of the value path. A difference in decision then shows only that the edited-out component (block-1 MLP → e3 → query) carries *some* input dependence beyond `m_blind`. It does not show that this dependence is a function of the cross-source relation. X4 is a D5 test relative to the declared aggregate `m_blind`. It is not a relation test. X3 in the same document concedes this relativity, but §4 and MAIN then drop it.

**What distinguishes them (`c2_check.py`).** A factorial on the mediator across all four sign quadrants does:
- C′ has `q_R` = +3.68 at (1,1) and +3.68 at (−1,−1), and the gate is open at (−1,−1) (`α_P` = 0).
- The twin has `q_R` = +1.84 at (1,1) and −5.52 at (−1,−1), and the gate is closed at (−1,−1) (`α_P` = 1).

The interaction contrast on the mediator is what identifies the relation, and so does testing the (−,−) quadrant. Every X4 base is agree-for-c, so the decisive quadrant is never probed.

**Correction.**
- Restate X4 as "uses information beyond `m_blind`".
- Remove X4, C2 and block-1 ablation from the list of tests that separate C′ from relation-free twins.
- Add a mediator-level 2 × 2 sign factorial, including (−,−), at `q_R` or `α`. Report its interaction contrast as the relation criterion.
- In MAIN, Positive 2 and arrow 5 must be downgraded accordingly.

### 2. FATAL (for §4 of LEARNED as evidence) — The probe-direction interchange measures alignment with the readout, not relation use

**Where.**
- `LEARNED_SYNTHETIC_MECHANISMS.md` §1.1 ("interchange along the agreement probe transfers the decision (1.0, against 0.0 for the random direction)").
- §1.3 and §4 ("agreement is partly decodable … but not used").
- `MAIN_RESEARCH_RESULT.md`, Negative 5.

**Evidence (`learned_ctrl.py`, T1/T3/T4 × TF1/TF2/TF2AO × seeds 0 and 3, final residual site, decision-differing matched pairs).**

| direction patched | T1 transfer | T3 transfer | T4 transfer | \|cos\| with the readout `gf⊙ΔW_U` |
|---|---|---|---|---|
| sign(ab) probe | 1.000 (all) | 0.000 | 0.000 | **T1: 0.90–0.999**; T3/T4: 0.002–0.08 |
| readout direction itself | **1.000 (all 18)** | **1.000** | **1.000** | 1 |
| sign(a) probe, sign(b) probe | 0.000 | 0.000 | 0.000 | ≤ 0.19 |
| ridge(a+b) | 0.000 | 0.000 | 0.000 | ≤ 0.06 |
| random | 0.000 | ≤ 0.007 | 0.000 | up to 0.58 |

- In T1 the label *is* sign(ab), so a logistic probe for sign(ab) at the final residual recovers the decision direction. Its 100% transfer is the trivial result "patching the logit direction moves the logit".
- In T3/T4 the sign(ab) probe is nearly orthogonal to the readout, so 0% at the final site is guaranteed whatever happened upstream.

**Power check (`sites_ctrl.py`).**
- In T3 (TF1, TF2), patching the sign(a) probe, the sign(b) probe, span{sign a, sign b}, or span{sign a, sign b, sign ab} transfers **0.000 at every residual site**. Yet T-RF says T3's decision *is* a function of sign a and sign b. By the package's own logic, per-source support would be "decodable but not used", which is absurd.
- In T1/TF2, the sign(ab) probe transfers 0.0 at resid_post0 and 0.5 at resid_mid1. It reaches 1.0 only at the final site.

The test has power only along the readout direction at the last site. It has no power to detect use anywhere upstream.

**Two further problems.**
- The random control is not magnitude-matched. Random directions with cos up to 0.58 to the readout still transfer 0, because base and source barely differ along them.
- A single-direction linear patch at one position (R only) also ignores information read from the A and B positions by later attention.

**Correction.**
- Drop the "1.0 vs 0.0" claim as evidence of relation use, and drop "decodable but not used" as evidence of non-use.
- Replace both with:
  - a readout-orthogonalised probe direction;
  - patches at intermediate sites at all positions;
  - a positive control showing the method detects a known-used per-source feature (sign a) in T3.
- The T1 use claim must rest on the ablation evidence (finding 6 qualifies that too).

### 3. MAJOR — The C2 claim "values carry 1.8%; the value route carries nothing measurable" is an artifact of the intervention choice

**Where.**
- `DYNAMIC_ATTENTION…` C2 table and "Path-specific conclusion".
- `MAIN_RESEARCH_RESULT.md`, arrow 4.
- `CONDITIONAL_CAUSAL_THEOREMS.md`, T-C4 (i).

**Evidence (`c2_check.py`, `c2_aflip.py`).**
- At (1,1) the block-2 routing is `α` = (P 0, A 0.995, B 0.005, R 0). An LN-scale asymmetry gives A the larger key. So under the package's source change `(a,b) → (a,−b)`, the patched value v_B gets almost no weight.
- With the symmetric change `(a,b) → (−a,b)` on the same 200 points, the medians are:

| site | median share of TE |
|---|---|
| q_R | 1.000 |
| scores | 1.000 |
| values | **0.909** (10th–90th percentile 0.62–0.98) |

  The values-only patch moves the decision to the counterfactual in **100%** of cases.
- The value share in the reverse direction (agree values into the conflict run) is exactly 0, because `α_P` = 1 there.

**Additivity.**
- Score share plus value share is about 1.9 under the a-flip, so the shares are not a decomposition.
- q_R → scores are nested sites on one chain, not competing mediators.
- With a saturated softmax, 1/0 shares are close to tautological.

**Correction.**
- Report shares for both source flips and both directions.
- State that they are non-additive and conditional on saturated routing.
- Withdraw "value route carries nothing measurable". What holds is narrower: *the relation feature* q affects m only through the scores. That is checked correctly by q_B = 0, and q_A ≡ 0 structurally (verified: the q_A share is 0.000 under the a-flip).
- T-C4 (i) should cite q_B and q_A, not "values carry 1.8%". The value share is the source change's route, not q's.

### 4. MAJOR — T-RF's implication "behaviour cannot identify relation use" holds for decisions only, not for graded behaviour

**Where.**
- `MAIN_RESEARCH_RESULT.md`, opening ("Behaviour cannot identify it") and Negative 1 ("Behaviour alone never shows relation use").
- T-RF Consequence ("never behavioural evidence").

**Evidence (`trf_scope.py`).**
- C′'s margins violate conjoint-measurement independence (single cancellation). Of 435 row pairs (a, a′) on a 30 × 30 grid, **357** reverse their order across b. Example: m(1,1) > m(−1,1), but m(1,−1) = −2.63 < m(−1,−1) = −1.89.
- So *no* monotone transform of any `f(a) + g(b)` reproduces C′'s margins or confidences.
- The package's relation-free twin has 0/435 reversals, and its margins differ visibly: (−1,−1) gives −0.606, the lowest cell, whereas C′ gives −1.89, above conflict.

Graded output (logit, probability, confidence) therefore *does* behaviourally refute every additive-plus-monotone account of C′. The package itself uses a behavioural margin statistic (`R²_add`) as evidence of additivity in T3/T4, which is inconsistent with "behaviour never shows".

A residual caveat: a cross-source interaction in the output is not yet "the relation" as a mediator (cf. finding 1).

**Correction.** Scope every instance to "binary decision behaviour". Add that margin-level independence tests are behavioural, and are discriminating for C′.

### 5. MAJOR — "Only candidate-symmetric rules force a cross-source interaction" is false in general

**Where.**
- T-RF, last line.
- MAIN, "The contribution …": "It has computational content only for candidate-symmetric relations".

**Evidence (`trf_scope.py`).** Take K = 3 and the rule "if k_A = k_B = k output k; otherwise output the candidate neither source picked". This rule is candidate-directed. An LP over fully general per-candidate additive tables `score_k = f_k(k_A) + g_k(k_B)` is **infeasible**.

The LP uses margin 1 and bounds ±50; because the constraints are homogeneous, this is equivalent to the strict-inequality problem. A hand proof is left to the authors. Whatever the proof, the solver result alone is enough to refute the "only" claim.

T-RF is correct for its family F ("agree → k, else prior p"). The generalisation to all non-symmetric rules is unproved and false.

**Correction.** Restrict the claim to the family F, and to rules that are threshold functions of per-source support. Characterise the boundary, for example as "additively sign-representable by LP".

### 6. MAJOR — The ablation evidence for T1 depends on the ablation operator, and unit selection has low specificity

**Where.** `LEARNED_SYNTHETIC_MECHANISMS.md` §1.1 and §3 ("three models where ablation leaves accuracy intact … relation computed redundantly").

**Evidence (`ablate_ctrl.py`, same preregistered unit sets, last position, 20k validation).**

| model | flagged units | mean-ablation | resample-ablation | complement units ablated (mean) |
|---|---|---|---|---|
| T1/TF1 s0 | 22/32 | 0.746 | 0.685 | 1.000 |
| T1/TF1 s1 | 26/32 | **0.998** | **0.504** | 1.000 |
| T1/TF2 s0 | 39/64 | 0.9996 | 0.995 | **0.749** |
| T3/TF1 s0 | 20/32 | 0.999 | 0.999 | 1.000 |

- In TF1 s1 the "intact after ablation → redundant" reading is an artifact of mean-ablation. A resample (on-distribution) ablation drops accuracy to chance.
- In TF2 s0 the necessary units are in the *unflagged* complement, so "redundancy via attention" is not the explanation there.
- The criterion flags 62–81% of units, so set-level effects are only weakly attributable to "relation units".
- Mean-ablation sets every flagged unit jointly to its marginal mean, which is an unreachable joint state. That is the same off-manifold objection the package levels at C2, and it is not addressed.

**Correction.**
- Report resample (or matched-input interchange) ablation alongside mean ablation.
- Add a size-matched random-unit control and a complement control.
- Revise the "redundant" explanation per model.

### 7. MINOR — X4 reachability for C′ is asserted by analogy, not proved

**Where.** MATCHED §4: "This is the same reachability limit as in §2". MAIN, arrow 5 cites T-R.

**Evidence.**
- T-R is proved for A's reader `Ŝ`, not for C′'s `m_blind`. X4 searches only a′ ∈ {2, 3, 4} with b′ ∈ [−6, 0).
- My wider search (|a|, |b| ≤ 30) gives a maximum conflicting `m_blind` of 1.299, below `m_blind(2,2)` = 1.473. So the claim looks true, but it is unproved.
- "(1,1) → a′ = 2 unreachable" holds only for that a′.
- All reachable partners are "one strong source plus one near-zero opposing source" (|b′| ≈ 0.2–0.6). The contrast lives in a thin, atypical region, adjacent to the gate's sign discontinuity.

**Correction.** Prove a C′ analogue of T-R or label it empirical. Describe the partner region.

### 8. MINOR — "The only computational difference … is the relation feature" is literally false

**Where.** MATCHED §4.

The paired inputs differ in a, b, LN scales, and every per-position value v_A, v_B, v_R. Only the uniform average of the values is matched. The gated margin `Σ α_j v_j` reads the values individually.

**Correction.** "They agree on `m_blind`; they differ in the block-1-MLP output, among other things."

### 9. MINOR — B4: one pair is outside D1's domain, and one null is structural

**Where.** `DYNAMIC_ATTENTION…` B4.

- `z_B = (0, −2, 0)` has a tied argmax (0 and 2), so semantic agreement (D1, unique-argmax set) is undefined for it. Use a perturbed non-tied example.
- The Q-only null is structural: the query reads `z_A` only, and the source change is to `z_B`. Present it as a design fact, not an experimental finding.

### 10. MINOR — Mis-citation in the MAIN table

**Where.** MAIN, arrow 7 cites T-SS ("sum-only Δm = 0 exactly") as a control for C′.

T-SS and X5 concern A's sum-only reader at matched `Ŝ`. No sum-only control for C′ at matched `m_blind` is reported.

### 11. MINOR — The sign-prediction claim is selectively summarised

**Where.** MAIN, Positive 3.

- MAIN cites "86–94% for TF1". TF2 is 100/141 and 73/97 (71–75%), and it is omitted from MAIN.
- The predictor uses the uncentred `gf⊙ΔW_U`, while the final LN centres the residual. It also ignores layer-1 mediation for layer-0 units in TF2.

---

## Verified-correct claims (reproduced by the reviewer)

1. **X4 numbers.** `m_blind` = 0.822 for (1,1) and (3,−0.2381); full margins +1.890 / −2.630; the other two pairs also match. The *computations* are right; the interpretation is challenged in finding 1.
2. **C2 table, as defined (b-flip, agree base).** Median shares q_R 1.000, q_B 0.000, scores 1.000, values 0.018 (`c2_check.py`).
3. **Proposition C2-OFF is correct.** R's e3 coordinate is a deterministic function of its e1/e2 coordinates, so the `q_R`-patched state is unreachable.
4. **T-RF is correct for its family F.** The case table checks out. The XNOR non-additivity proof is correct. `relation_free_equivalent.py` matches C′ on 100% of the grid.
5. **B-TM algebra is correct.** It is the softmax proportional-rescaling identity. The sign statement holds within its class.
6. **T-C4 conclusions 1–3 are correct and elementary.** For example, `|Σα v| ≤ max|v_j| < |m_bp|`. Assumption (i) holds for C′ via q_B = 0 and q_A ≡ 0. Note that the "0 violations at 107 points" check is tautological given the inequality.
7. **The learned-model interchange numbers reproduce exactly.** T1 transfers 1.000 everywhere, and T3/T4 transfer 0.000. T1/TF1 s0 mean-ablation accuracy is 0.746.
8. **X3 injectivity** (inversion error at the 1e−15 level) and **X5/T-SS** for A's sum-only reader hold.
9. **The redundancy result in C3** (either copy ablated → 200/200 preserved) is correctly read as defeating single-site necessity.

## Bottom line

The engineered circuits' operating theorems survive. The *identification* layer does not survive as stated:

- **X4, C2 and block-1 ablation** cannot tell C′ from a same-architecture relation-free gate (finding 1).
- **The learned-model interchange** is a readout-alignment test with no upstream power (finding 2).
- **The value-route null** depends on which source is flipped (finding 3).
- **"Behaviour never identifies"** holds only for decisions (finding 4).
- **"Only symmetric rules need interaction"** is false for K = 3 (finding 5).

The main result needs a mediator-level factorial criterion and a qualified scope before the identification claims in MAIN (Positive 2, Positive 3, Negative 1, Negative 5, arrows 4–5) can stand.
