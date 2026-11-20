# Pre-run review B: mathematics and functional equivalence (Investigation XX, design draft v1)

**Reviewer:** an independent AI agent reviewer (Claude). I am not a human peer reviewer. I did not build this package. The review was done before any confirmation compute.
**Scope:** Track C (all of it) and Track B (B1–B5: candidate rules, the functional tests, the internal test, and the classification). Track A is outside my specialty. I mention it only where Track B depends on it.
**Mandate sections applied:** 2 (design gate), 4, 5, 6 and 10 of `../MANDATE_XX.txt`.
**Files read:** `protocols/DESIGN_XX.md`, `cor_research/protocol.py`, and `cor_research/mathematics/{od_original,od_repaired,t4_order,run_track_c}.py`. Also `dev/track_c/track_c_math.json`, `cor_research/generators/t4.py`, `cor_research/scoring/{functional,track_b,trackb_run}.py`, `cor_research/analysis/hier.py`, `dev/DEV_NOTES_XX.md` and `dev/trackb_dev_rules*.json`. From the vendored code: `genlib.py`, `relpairs.py` and `xvitask.py`. From Phase 1: `02_formal_definitions.md` and `03_obstruction_equivalence_and_spectral_analysis.md`. I also read the historical F_G definition (`step1_relational_acquisition_limits/LEARNING_SIGNAL_AND_STAGE_VALIDATION.md`) and the power-mean ledger entry N20.
**Not available:** the project note `claude/od_repair_proposal.md` is not on disk, so I cannot read it from here. I take its definitions as restated in `od_repaired.py` and in DESIGN §3, and I flag every place where my conclusions depend on that restatement. I did not reconstruct `od_repair_check.py`.
**My own checks:** scratch scripts in `/tmp/revB/` (not in the package), CPU only, each under about 2 minutes. No package file was modified.

---

## 1. Summary verdict

**Overall: APPROVE WITH REQUIRED CHANGES.** The two tracks get different decisions:

- **Track C (mathematics): APPROVE WITH REQUIRED CHANGES. No blockers.**
  - I independently confirmed every numerical claim. This includes the order-18 interaction bound (40–60-digit independent enumeration) and the LP non-representability at orders 1 and a (exact rational Farkas certificates).
  - I also confirmed that the factor-deletion (FD) destination equals Bayes with the source marginalised out of the joint. This is a truly independent reference, unlike the one in the code.
  - The repairs needed are about how claims are scoped. Some checks are guaranteed by construction and are presented as tests. Some are tautological in code. Some theorems are stated too broadly (log-odds versus decisions; GATE versus general). Some labels are unsupported.
  - **Decision on neural O/D:** I agree that no O/D neural experiment should be run. I found one non-Bayes candidate prediction that the design missed (§3.6). It is identical to mean-field / log-linear pooling, a standard alternative. It is not a COR-specific prediction, so the C5 decision stands, but it must be documented.
- **Track B (functional identification): REJECT as frozen. Two blockers and several majors.**
  - **Blocker 1:** the predicted outcome (B2, "same rule, different implementation") is assigned when the functional tests fail to reject. There is no equivalence test, so an underpowered null gives B2 by default.
  - **Blocker 2:** HB_a cannot separate "different rule" from "same rule with an imprecise internal parameter". In my check, exact Bayes computed with α = 0.7 instead of 0.8 has KL 0.029, inside the matched band. It is read as w_nocount = 0.095, three times delta_w. So a B3 world can be classified B1.
  - Track B is scored offline on saved checkpoints. Track A training can therefore go ahead while Track B is fixed, but the corrected Track B code and thresholds must be **re-reviewed and refrozen before any `trackb_run` job executes on confirmation checkpoints**.

---

## 2. Independent verification results (reviewer-run, CPU)

| Claim in the package | My check | Result |
|---|---|---|
| T4 interaction coefficients nonzero on 18 coordinates (float64 plus rounding bound) | I re-enumerated the T4 posterior over roles in mpmath (60 digits for sizes 2/6/12; 40 digits for size 18), with my own implementation of the role semantics | All 15 small-size values reproduced to ~1e-13. Size 18: **−20.8558394526408** (base 0) and **138.635712661052** (base 1), equal to float64 to 1e-12. The exact zeros are exact in high precision too, so they are structural, not rounding. |
| LP: no order-1 or order-a additive complex reproduces Bayes decisions | Farkas LP on the same constraint matrix; certificate rationalised and **checked exactly** (y > 0 and yᵀA = 0 in rationals) | Exact certificates exist for order 1 (support 19) and order 2 (support 60) under (2, .9, .9, 1/3) and (2, .9, .55, .05). The claim is proven, not merely floating-point. |
| M2: FD destination = Bayes without the factor | Reference computed independently by marginalising the evidence source out of `xvitask.joint` | max error 2.2e-16 for h1, h2 and h0 in all 5 conditions. The claim holds. |
| T5: source-level D never follows a gate flip | Same code, with weights over all six sources (as the `gate_source_level` docstring says) instead of only the three evidence sources | With gate sources weighted, the argmin changes on up to 100% of decisive flips. T5's "holds" depends entirely on excluding x_{a−1}. It is guaranteed by construction. |
| Decisive gate flips: 216 (package) vs 162 (note) | Counted unordered pairs | The 216 are **ordered** pairs, i.e. 108 unordered. Neither matches 162. The package's definition matches the historical F_G definition (LEARNING_SIGNAL §1). |
| Soft-gate pooling (my candidate, §3.6) | Expected-energy rule qĝ(e₁+e₂)+(1−qĝ)e₀ vs the arithmetic mixture at the same ĝ | On-support decision disagreement mass of 0.29–0.38 for ĝ ∈ {0.1, 0.3}. Zero for ĝ ≥ 0.5 and zero for the exact q(g \| evidence). |
| Mean-field role pooling on T4 (geometric pooling across reliable-group hypotheses) | 12k natural episodes | Disagrees with Bayes on 0.39% of decisions. In practice it differs only in calibration. |
| Rule-mixture identifiability (planted w = 0.10, τ = 1, ε = 0.03) | Fit each single-rule family to the planted outputs | Own rule recovered exactly. **Cross-loading:** planted noelig gives w_local = 0.177; planted local gives w_nocount = 0.061 and w_noelig = 0.032. |
| Precision-only null | (i) iid logit noise at KL 0.027; (ii) exact Bayes with α = 0.70 (KL 0.029) and α = 0.75 (KL 0.007) | (i) w ≤ 0.0015. (ii) **w_nocount = 0.095** and 0.036. On the discriminating set, α = 0.70/0.75/0.85 Bayes have **0.0%** decision disagreement with exact Bayes. |
| Hybrid support (`t4.relational_pairs`) | Checks whether some group agrees at all 6 questions in each hybrid | 100% on-support. 288/3000 pairs change the decision. |
| OQI donor matching | Compares group-agreement patterns at question q between base and donor | 70.3% of pairs differ at q (the donor's q-question evidence differs from the base's). |
| maprole | Tie frequency of argmax_g W_g m_g; agreement with the full role MAP | Ties in **22.6%** of natural episodes, broken toward the lowest index (group 0 picked in 62% of ties). 6.7% of argmaxes differ from the full role MAP, which includes question q's evidence. |

---

## 3. Findings: Track C

### 3.1 Original O/D (T1–T5): are these correct statements of Phase 1?

- **T1** is Phase-1 03 §2, stated correctly. The proof is already in Phase-1 03 and is valid for M ≻ 0. The numerical check confirms it.
- **T2** is Phase-1 03 §3, but the code checks a tautology. `O_of(d, W*, s)` and `D_formula(d, M, s)` evaluate the **same expression** rᵀ(dM⁻¹dᵀ)⁺r, which is why the error is exactly 0.0. T2 only gains content when combined with T1. See C-MIN-1.
- **T3** (evenness) is trivially true for any quadratic form. The cited "Theorem Sign" is not in Phase-1 02/03; it is Step 1B `20_directionality_and_response.md`. Evenness in s implies "cannot encode the direction of a decision change" only if the margin is an odd function of s. That premise is not stated. See C-MIN-2.
- **T4** is Attack F (Phase-1 02 §2, 10). It is implemented as s = 0, which is equivalent to P = 0. Correct.
- **T5** is guaranteed by construction. A gate flip changes only x_{a−1}, and the weights are restricted to the evidence sources (indices 3–5), so the argmin cannot change. The function docstring says "ALL six sources", which contradicts the code. With all-six weights the argmin does change (§2).
  - The historical "Proposition S" is a **T4** sufficient-statistic result (JOINT_RELATIONAL_FEATURE_AND_REPAIR.md). It is not this GATE statement, so the citation is wrong.
  - The non-trivial, correct form of the claim is the order-1 LP certificate. I confirmed it exactly.
  - See C-MAJ-1.

### 3.2 Repaired O/D on GATE (R1–R5, M1, M2)

- **R1, R3, R5 and M2 are identities by construction.**
  - Each energy E_h is −log of a likelihood factor of the declared generator. The gate is observed and the prior is uniform. So D_c = −log p(x | c) + const(x) exactly.
  - R1 (Δ = log posterior ratio), R3 (argmin = Bayes argmax) and R5 (deletion effect = −ΔE_h) follow in one line each.
  - M2 as coded is circular. Its "Bayes without factor h" is the product of the remaining factors, which is algebraically exp(−D_{−h}).
  - The equivalence is **true**: my independent marginalised-joint reference gives 2.2e-16. But the code does not establish it independently.
  - Mandate C2 asks for a proof and its scope, not numerical identities. See C-MAJ-2.
- **R2:**
  - The LP certificate is correct and now verified exactly.
  - The docstring claims that least squares shows the "gate-blindness of every order ≤ a additive model". The package's own output contradicts this: the LS projection has gate-flip following F_G = 0.23 (a = 2) and 0.16 (a = 3).
  - Least squares on log-odds, with uniform weighting over inputs, is not the decision claim. See C-MIN-3.
- **R4** (O = 0 implies deletion invariance) follows in one line: a common minimiser of every factor minimises every sub-sum. Correct.
- **M1** (O is a function of the Bayes posterior) is correct on GATE but nearly trivial: there are only 3 sorted-posterior classes (g = 0; g = 1 with agreement; g = 1 with disagreement). **It is false in general.**
  - Counterexample with K = 3: E₁ = (0, 2a, 2a), E₂ = (a, 0, a) gives ΣE = (a, 2a, 3a) and O = a.
  - E₁′ = (0, a, 2a), E₂′ = 0 gives ΣE′ = (0, a, 2a), the same posterior, with O = 0.
  - The C3 entry already says "on GATE", which is correct. But C5 is justified with M1, and M1 does not transfer to T4, where the neural models live. See C-MAJ-3.
- **Discrepancies with the note:**
  - The latent-marginalisation masses reproduce the note when ties are excluded. Good.
  - "216 vs 162, definition not recoverable" is partly inaccurate. The historical F_G definition is recoverable and is the one implemented. 216 are ordered pairs (108 unordered). See C-MIN-4.

### 3.3 The T4 interaction-order lower bound (18)

- **The argument is valid.**
  - Suppose f = Σ_h φ_h(x_h) on a product cube {x0, x1}^T in which every point is on-support.
  - Then any φ_h whose hyperedge h misses some j ∈ T cancels in pairs across j in the alternating sum.
  - So I_T ≠ 0 implies some hyperedge contains T, which implies order ≥ |T|.
- **The inputs are on-support.**
  - On-support means some group agrees at all questions under some role.
  - The construction never touches the colluding pair, so the true colluder still agrees everywhere at every cube vertex.
  - All posteriors are strictly positive (α < 1, and m_g ≥ 1 for the non-colluders), so f is finite.
- **Rounding:**
  - The stated bound Σ|f|·4nε is **not a rigorous floating-point error bound**. The absolute error of each f is set by the magnitude of the log-likelihood terms, not by |f|.
  - The 9–10 orders of magnitude of margin make this moot, and my high-precision independent enumeration confirms the values.
  - The bound should be relabelled "heuristic", with the high-precision recomputation as the certificate. See C-MIN-5.
- **Scope (major):**
  - The bound is about representing the **log-odds** exactly (R1-type).
  - It says nothing about **decision** reproduction (R3-type). R3 is the level at which the repaired O/D was claimed on GATE.
  - DESIGN §3 says the complex cannot "reproduce T4 Bayes", which is ambiguous. A decision-level order bound for T4 is open; it may well be much lower.
  - The "therefore" also needs one more clause: with order unrestricted, a single all-coordinate hyperedge (a lookup table) trivially represents f.
  - So the correct statement is: either the complex degenerates to a lookup table of order ≥ 18, or latent roles must be added, and then it becomes standard factor-graph marginal inference.
  - See C-MAJ-4.

### 3.4 `run_track_c.classify`: sound and non-circular?

- **`original_OD = C4`** rests on T5 (guaranteed by construction) and T3 (trivial). So the label is predetermined, and it is also scoped wrongly:
  - C4 ("fails the proposed test") holds for the **source-level, order-1 instantiation**. The right justification is the exact order-1 LP certificate.
  - For the original **germ-level** definition (analyst-chosen P, ρ, M, W), Phase-1 Attack F and metric arbitrariness show the theory is **underdetermined**: it neither fails nor passes. That is C5 ("no valid neural prediction follows"), not C4.
  - See C-MAJ-1.
- **`repaired_OD_gate = C2`**:
  - It requires r1 ∧ m1 ∧ m2, all identities by construction. That is acceptable for a mathematical classification, because proofs are guaranteed by nature, provided it is presented as a proof with scope.
  - Including m1 as a necessary condition for C2 is not justified. A reformulation can lack distinctive content even if O is not a function of the posterior.
  - See C-MAJ-2.
- **`neural_prediction = C5 if m1 and m2`**:
  - This is the designer's list of two candidate predictions, hard-coded. It is not computed.
  - This is the one real circularity: the code cannot discover a candidate it was not given. The C5 label is a **reviewer-adjudicated** judgement and must be recorded as such, with every considered candidate and the reason it was rejected (my §3.6 included).
  - See C-MAJ-3.
- **Code inconsistency:** the docstring references `protocols/od_design/od_protocol.json`, but the file is `track_c_protocol.json` (C-MIN-6).

### 3.5 Track C design gate (the 8 questions)

| Test | 1 Guaranteed? | 2 Literature? | 3 Competing predictions differ? | 4 Metric discriminates? | 5 Controls? | 6 Positive establishes? | 7 Negative weakens? | 8 Inconclusive possible? |
|---|---|---|---|---|---|---|---|---|
| T1/T2 (O ≡ D up to weighting) | yes (theorem; T2 tautological in code) | Phase-1 03; standard least-squares projection | n/a (theorem) | n/a | n/a | yes, as a corroborated theorem | n/a | no (deterministic) |
| T3 (evenness) | yes (trivial) | Step 1B Theorem Sign | n/a | only with the odd-margin premise | n/a | only the scoped statement | n/a | no |
| T4 (Attack F) | yes | Phase-1 02/10 | n/a | n/a | n/a | yes | n/a | no |
| T5 (source-level gate blindness) | **yes, by construction** | XVI/XVII order results | no | **no** (cannot fail) | — | nothing beyond its construction | — | no → replace with the LP order-1 certificate |
| LP order-k certificate | no (it could have been representable) | XVI review A F6 (order ≤ 2 gate slope 0) | yes | yes | exact Farkas (now verified) | yes | yes | no |
| R1/R3/R5/M2 | yes (identities) | standard: a factor graph's log-likelihood ratio | no (by construction) | — | the M2 reference must be independent | establishes C2 **with proof and scope** | — | no |
| M1 | yes on GATE (3 classes) | — | no | weak (3 groups) | — | restricted C3 (GATE only) | — | no |
| T4 order bound | no | standard Möbius / Walsh interaction argument (Grabisch; O'Donnell) | yes | yes (log-odds level) | high-precision recomputation | yes, for **log-odds only** | yes | yes (a size could have been zero; some are) |
| Neural O/D test (none planned) | — | — | §3.6: the only divergence equals mean-field / log-linear pooling | — | — | — | — | C5 is correct; must be recorded as adjudicated |

### 3.6 Is there any distinctive neural prediction the design missed?

**Candidate found: soft-gate (expected-energy) pooling.**

- Suppose an O/D proponent replaces the hard gate indicator g in the repaired energies g·(e₁+e₂) + (1−g)·e₀ with an internal graded estimate ĝ ∈ (0, 1). This could happen during gate acquisition or after a partial lesion of the gate route.
- The result is the expected-energy rule D_c = ĝ(e₁+e₂) + (1−ĝ)e₀, i.e. geometric (log-linear) pooling.
- Exact Bayesian inference with the same internal gate belief ĝ instead predicts the arithmetic mixture ĝ·L₁ + (1−ĝ)·L₀.
- These **differ on-support at decision level**: 29–38% of input mass for ĝ ∈ {0.1, 0.3} (§2).

**Why this does not reverse C5:**

1. **It is not COR-specific.** It is exactly the mean-field (variational) versus exact-marginal distinction, equivalently product-of-experts versus mixture-of-experts (Hinton 2002; Genest & Zidek 1986 on log-linear versus linear opinion pools). Mandate C4 requires comparison against such alternatives, and here the "O/D prediction" is identical to one of them.
2. **It is not stated in the note** (as restated in the package). The note's own gate-removal analysis uses the exact marginal, i.e. Bayes.
3. **It needs an unstated bridge assumption** (mandate C3): that the network represents a scalar ĝ which multiplies the energies linearly.
4. **It concerns GATE models at intermediate gate acquisition.** None are in XX's Track A/B cohort, which is T4.

On T4, the analogous candidate is geometric pooling across role hypotheses weighted by the role posterior (latent-role O/D with expected energies). It disagrees with Bayes on only 0.39% of natural decisions, so it differs in calibration only and cannot be confirmed at decision level.

**Decision:** no neural O/D experiment. Both candidates must be listed in the Track C record as considered and rejected, with these reasons.

---

## 4. Findings: Track B

### 4.1 Candidate rules (B1) and their correctness

- **bayes**: the closed form p ∝ Σ_g W_g m_g L_g is correct for C = 1, β = 1. It matches brute-force enumeration to 2e-15, and my independent derivation gives the same: P(x | g rel, h col) ∝ W_g L_g(y) K^Q 1[h agrees everywhere], so m_g = Σ_{h≠g} e_h.
- **local**: `posterior_closed(Env(Q=1), x_q)` equals the exact marginal p(y | x_q) of the Q = 6 model, because the questions are conditionally independent given the roles. Correct.
- **noelig**: equals the exact Bayes posterior of a **misspecified model in which the colluding group is noise**. That is a meaningful "ignores copying" rule. Correct.
- **nocount**: Σ_g m_g L_g. Correct as defined, and well defined on-support, since some m_g ≥ 1 always.
- **maprole: not what the docstring says.**
  - (i) It uses argmax_g W_g m_g, where W_g **excludes** question q. The full role MAP is argmax_g W_g m_g Σ_y L_g(y). The two differ on 6.7% of episodes.
  - (ii) Ties occur in 22.6% of episodes and are broken toward the lowest group index. That is a position-dependent artefact which no permutation-equivariant learner would share.
  - This plausibly explains why w_maprole ≈ 0 for every dev model, i.e. why the rule is dead in HB_a. See B-MAJ-5.
- **On-support:** the natural and discriminating sets are generator samples, and the hybrids are 100% on-support (§2). Correct.
- **Mandate B1 gaps** (B-MAJ-4):
  - The mandate asks to "reconstruct the behavioral models previously used to describe early and mixture organizations". These are the nested power mean ρ (0.54–0.62 vs 0.87–0.98, ledger N20) and the threshold-4 rule.
  - It also asks for an explicit classification of every candidate: equivalent / calibration-only / different posterior combination / off-support-only / genuinely different on valid inputs, plus which parameters are non-identifiable.
  - None of the historical models is a candidate, and the classification table is missing. My check shows the ρ → 0 / mean-field-role limit is calibration-only (0.39% decision disagreement). That is exactly the kind of entry the table must record.

### 4.2 HB_a: is the rule-mixture family identifiable?

- Each rule is fitted **separately**: (τ, ε, w_k) per rule.
- **Planted recovery is exact for the planted rule, but it cross-loads onto other rules:** planted noelig gives w_local = 0.177, more than the planted 0.10 itself. So "rule k" attribution is not identified, and the per-rule profile D_k is not a profile.
- **w tracks precision.** In the dev data, w_local rises from about 0.02 at KL 0.012 to about 0.07 at KL 0.030 within one class. The matched band (±0.006 KL) therefore moves w by about ±0.017, more than half of delta_w.
- **Blocker 2.** The precision-only family F0 (temperature + lapse) does not contain "same rule, imprecise parameter". Exact Bayes with α = 0.70 lies in the matched band (KL 0.029) and gets w_nocount = 0.095. If the E and M organisations differ in their effective reliability estimate (a B3 difference), HB_a reports FUNCTIONAL_DIFF. Combined with HB_d, which is likely positive, the result is **B1**.
- Also, HB_a uses the fitted w even when the held-out dKL ≤ 0. The held-out criterion is computed but never used.

### 4.3 HB_b: is the noise null a fair precision-only baseline?

- **No.** iid Gaussian logit noise is orthogonal to every structured rule direction, so it gives w ≤ 0.0015. Any smooth approximator gives w of about 0.02–0.09, including exact Bayes with a mis-set α.
- HB_b ("fraction of models with w − w_null > 0.02") is therefore **guaranteed** to be close to 1 for every trained model, whatever its rule. It is labelled descriptive, but as built it cannot discriminate, and it must not be reported as "systematic non-Bayes structure". See B-MAJ-2.

### 4.4 HB_c: calibration-invariance and the between-vs-within contrast

- **The calibration-invariance argument is correct, with one caveat.**
  - argmax is invariant to τ > 0 and to the ε/K lapse. My check also shows it is invariant to α-misestimation in [0.70, 0.85] on the discriminating set (0.0% disagreement).
  - So HB_c is **the** functional test that is robust to the B3 confound. It should carry the functional inference, not HB_a.
  - The caveat: "after calibration (F0 fitted on half A)" is a no-op for decisions. The wording should be removed.
- **The contrast is unbalanced (B-MAJ-1).**
  - Between-class pairs are required to share D, and they include same-unit pairs (fullB vs fullI of one unit, differing only in the 100-step block).
  - Within-class pairs are required to have **different** D.
  - So the nuisance (data order, seed, unit) differs systematically between the two terms. With iid noise alone, two copies of the same rule disagree on 8.8% of the discriminating set at the matched KL, more than 4× delta_dis. The sign of the resulting bias is unknown before compute.
  - Same-D, same-class, cross-unit pairs exist: the E-base fullB and the M-base fullI are both EARLY, as are the E-base fullI and the M-base fullB (both MIXTURE). The contrast can therefore be balanced.
- **HB_c is not directional.**
  - Two implementations of the **same** rule with different structured error locations at equal KL will disagree more across classes than within. That makes HB_c positive in a B2 world.
  - B1 needs the disagreement to be **explained by a candidate rule**. The design needs a rule-directed decision test: the difference in agreement with rule k on the rule-k-discriminating subset, between classes, beyond the within-class spread.

### 4.5 HB_d: interchange and its ideal reference

- **The ideal reference is correct** for the hybrid input x_h: exact Bayes of an on-support, role-matched hybrid, verified at 100% support.
- **The donor is not source-matched to that reference (B-MAJ-3).**
  - OQI_l takes the other-question positions from **run(x_d)**. After layer 0, those positions can contain the donor's question-q evidence, through cross-question attention (for example group-agreement aggregation that includes q).
  - The donor's q-question agreement pattern differs from the base's in 70.3% of pairs.
  - The source-matched donor is **run(x_h)**: its other-question positions saw the base's question q, so Bayes(x_h) is exactly the right target.
  - The contamination can depend on the class, because EARLY mixes at layer 0. That could produce part of the HB_d class difference artefactually.
- **HB_d is near-implied by the class definition.**
  - The causal class is defined by the necessity of the layer-0 versus layer-2 own route. Where the cross-question evidence sits at the layer-2 input is close to a restatement of that.
  - The design correctly lets HB_d alone yield only B2. But this means B2 versus B3 is in practice decided by HB_d, which is close to guaranteed. A positive test that both organisations compute **the same sufficient statistic** (mandate B4) is missing.
  - Example of such a test: a per-model abstract-variable interchange on the Proposition-S statistic m, with similar high IIA in both classes at different sites. That would positively establish B2.
  - See B-MAJ-6.

### 4.6 HB_e and the classification logic

- **HB_e:** the dev anchors differ by about 0.015 KL at step 1211 (EARLY is worse in both bases). A PRECISION_DIFF at delta_kl = 0.005 is very likely, though not guaranteed. That is fine, but it should be stated as expected.
- **Blocker 1 (B-BLK-1).** "No functional difference" is computed as `not (HB_a or HB_c)`, i.e. failure to reject. The frozen prediction ("not B1; HB_d positive") is therefore obtained whenever the functional tests are underpowered or noisy. Combined with a near-guaranteed HB_d, **B2 is the default outcome of a weak experiment.**
  - The mandate requires that B1/B2/B3 be distinguished and that non-identifiability be available.
  - A "same rule" verdict needs an equivalence test: the CI of every functional statistic must lie inside the equivalence margin. Otherwise the result is NON_IDENTIFIABLE.
- **Can each outcome occur?**
  - **B1 can occur for the wrong reason:** through HB_a in a B3 world (Blocker 2), or through a non-directional HB_c in a B2 world (B-MAJ-1).
  - **B2 can occur for the wrong reason:** by absence of evidence (Blocker 1).
  - **B3** needs HB_d negative, which is unlikely, so B3 is nearly unreachable whatever the truth.
  - **NON_IDENTIFIABLE and INSUFFICIENT_UNITS** are reachable and are reported. Good.
- **Statistics (B-MIN-3).**
  - Each parent contributes one difference, so the "two-stage" bootstrap reduces to a parent-level percentile bootstrap.
  - With 4 parents this is anti-conservative: if all 4 parent means are positive, the CI excludes 0 automatically.

### 4.7 Thresholds

None of the four thresholds is derived in the package.

| Threshold | Value | Evidence available | Assessment |
|---|---|---|---|
| delta_w | 0.03 | w varies ±0.017 across the KL band; α-imprecision gives 0.095; cross-loading 0.03–0.18 | Not justified. It only means something after the precision family is extended and KL is adjusted for. |
| delta_dis | 0.02 | iid nulls disagree with each other 8.8%; no dev estimate of within-class or between-class disagreement | Not justified. It must come from the within-class spread (dev, same-D cross-unit pairs) and a stated minimum effect. |
| delta_oqi | 0.10 | dev difference 0.22, n = 2+2, with the contaminated donor | Plausible, but it must be re-estimated with the source-matched donor. |
| delta_kl | 0.005 | anchors differ by ~0.015 | Fine as a minimum effect, but stated nowhere. |

All four need a written derivation (noise SD from dev, minimum effect of interest) and a planted power simulation at the expected parent count (B-MAJ-7).

### 4.8 Track B design gate (the 8 questions)

| Test | 1 Guaranteed? | 2 Literature? | 3 Competing predictions differ? | 4 Metric discriminates? | 5 Controls? | 6 Positive establishes? | 7 Negative weakens? | 8 Inconclusive possible? |
|---|---|---|---|---|---|---|---|---|
| Matched precision | no | standard | — | KL band ±20%; class-correlated step | censoring reported; **no KL covariate** | — | — | yes (censoring); report by class |
| HB_a | no, but **confounded** | mixture-of-rules fitting; XIII-B non-uniqueness | B1 vs B3 **do not** differ (α-imprecision gives w > δ) | **no** (cross-loading; precision leak) | F0 lacks parametric precision | **no** (B3 can produce it) | weakly | yes |
| HB_b | **yes** (≈1 for any smooth model) | — | no | no | the null is unfair | no | no | no |
| HB_c | no | decision-level comparison (standard) | B1 vs B3: yes (decisions α-invariant); B1 vs B2: **no** (non-directional) | partially | within-class term **unbalanced** | no, not alone | yes, if balanced and powered | yes |
| HB_d | **near-implied by the class definition** | interchange intervention (Geiger et al. 2021) | B2 vs B3 nominally | yes (dev 0.22) | ideal reference correct; **donor not source-matched** | "different location"; not "same statistic" | yes (B3) | yes |
| HB_e | no; likely positive | — | yes | yes | — | precision difference | yes | yes |
| Classification | B2 is the **default under low power** | — | — | — | needs equivalence tests | B1 not secure (Blockers 1–2) | — | yes (NON_IDENTIFIABLE / INSUFFICIENT_UNITS) |

---

## 5. Numbered objections and required changes

### Blockers

**B-BLK-1 (BLOCKER, Track B classification): "no functional difference" is failure-to-reject.**
Required change:
- Add equivalence criteria in `protocol.TRACK_B.thresholds` and `track_b.score`.
- **"Functionally equivalent"** means both of the following:
  - the hierarchical 95% CI of HB_c's balanced statistic (B-MAJ-1) lies entirely below delta_dis;
  - for every rule k, the CI of the KL-adjusted D_k (B-BLK-2) lies within ±delta_w.
- B2 and B3 may be assigned only when functional equivalence holds.
- If there is neither a functional difference nor equivalence, assign `NON_IDENTIFIABLE (underpowered)`.
- Add a planted test in which a wide-CI null yields NON_IDENTIFIABLE, not B2.

**B-BLK-2 (BLOCKER, HB_a): parametric imprecision is read as a rule mixture.**
Required change:
- Extend the precision-only family to F0′ = (1−ε)·softmax(τ·log P_bayes(α′)) + ε/K, with α′ free. Optionally add a free colluder-copy belief β′.
- Fit the rule-mixture families on top of F0′.
- HB_a uses only rules whose held-out dKL against F0′ is > 0 in the model.
- D_k must be adjusted for each model's KL: regress w_k on KL within class, or interpolate to KL* = 0.030.
- Make HB_a **descriptive**, outside the B1 disjunction, unless a planted simulation shows that F0′-relative w separates a rule mixture from α′-imprecision at matched KL with error rate < 5%.
- B1 must then require the rule-directed decision test (B-MAJ-1).

### Majors

**B-MAJ-1 (MAJOR, HB_c): the contrast is unbalanced and non-directional.**
Required change:
- (i) Define between-class pairs as same parent, **same D**, different class, **different unit** (exclude same-unit fullB/fullI pairs, or analyse them separately). Define within-class pairs as same parent, same D, same class, different unit (E-base fullB with M-base fullI; E-base fullI with M-base fullB).
- (ii) Add HB_c-dir: for each rule k, on the subset where rule k disagrees with Bayes, the between-class difference in agreement with rule k minus the within-class difference, with a sign-consistency requirement.
- (iii) A functional difference for B1 requires HB_c-dir. The non-directional HB_c alone gives at most `NON_IDENTIFIABLE (functional difference, rule unidentified)`.

**B-MAJ-2 (MAJOR, HB_b): the noise null is not a fair precision-only baseline.**
Required change:
- Replace or augment the iid-logit null with the parametric-precision nulls Bayes(α′) at the model's KL, and with the F0′ fit.
- Report HB_b against both nulls.
- Remove the phrase "systematic non-Bayes structure" unless the excess holds against F0′.

**B-MAJ-3 (MAJOR, HB_d): the OQI donor is not source-matched.**
Required change:
- In `functional.interchange`, compute OQI_l with donor activations from **run(x_h)** as the primary measure. Keep the run(x_d) donor as a secondary measure, labelled "unmatched donor".
- Report both measures.
- Recompute the dev OQI_2 estimate and delta_oqi from the source-matched version before freezing.

**B-MAJ-4 (MAJOR, mandate B1): the historical models and the equivalence classification are missing.**
Required change:
- Add to `generators/t4.py` the nested power-mean family (ρ, with a lapse-aware fit as in XVIII B1) and the threshold-4 rule.
- Add the role-posterior geometric pooling ("softrole") as the ρ → 0 limit.
- Add a frozen table (protocol plus DESIGN §2) that classifies every candidate as one of: equivalent / calibration-only / posterior-combination / off-support-only / genuinely different on-support. Give its on-support decision disagreement with Bayes and with each other rule, and list which fitted parameters are non-identifiable.
- My check shows softrole is about 0.4% different at decision level, so it goes under "calibration-only". It must be listed, not tested.

**B-MAJ-5 (MAJOR, maprole): the implementation contradicts its definition and breaks ties by position.**
Required change: either
- implement the full role MAP, argmax_g W_g m_g Σ_y L_g(y), and average L_g over tied maximisers; or
- rename the rule ("other-question MAP, tie-averaged"), still with tie-averaging.

Re-run the dev disagreement numbers afterwards.

**B-MAJ-6 (MAJOR, HB_d / mandate B4): there is no positive test of "the same sufficient statistic".**
Required change:
- Add, at least as a frozen secondary test, a per-model abstract-variable interchange on the Proposition-S statistic (m_H counts, or the eligibility/weight pair). Learn the alignment on a training split and evaluate IIA on held-out data.
- **B2 is positively supported** only if both classes achieve IIA above a frozen floor for the same abstract variable.
- If they do not, report "different location of unidentified content".
- Rationale: HB_d differences are close to implied by the route-based class definition.

**B-MAJ-7 (MAJOR): the thresholds are not justified.**
Required change:
- For each of delta_w, delta_dis, delta_oqi and delta_kl, write in the protocol: (a) the dev-estimated noise SD it is compared against; (b) the minimum effect of scientific interest; (c) a planted power and false-positive simulation at 4 and at 8 parents, run through `track_b.score`.
- Freeze the results in `tests/planted_results.json`.

**C-MAJ-1 (MAJOR, original O/D): T5 is guaranteed by construction, and C4 is mis-scoped.**
Required change:
- Replace T5 as the basis for C4 with the exact order-1 LP / Farkas certificate. Fix the docstring ("ALL six" versus the evidence-only code).
- Drop the "Proposition S" citation: that name is a T4 result.
- Change `classify` so that:
  - `original_OD (source-level instantiation) = C4`, justified by the order-1 certificate;
  - `original_OD (germ-level definition) = C5 / underdetermined`, justified by Phase-1 Attack F and metric arbitrariness (03 §5).

**C-MAJ-2 (MAJOR, repaired O/D): C2 needs a proof and an independent M2 reference.**
Required change:
- In `od_repaired.py` and DESIGN §3, add the analytic proof: E_h = −log likelihood factor, observed gate and uniform prior give D_c = −log p(x|c) + const, from which R1/R3/R5/M2 follow.
- State the scope: an observed selector variable, each evidence source in exactly one factor, and a uniform class prior.
- Replace the M2 reference with marginalisation of the source out of `xvitask.joint`. It holds at 2.2e-16 (§2).
- Remove m1 from the C2 condition, or justify keeping it.

**C-MAJ-3 (MAJOR, neural C5): the label is designer-enumerated, and M1 is GATE-only.**
Required change:
- `classify` must record C5 as `"C5 (reviewer-adjudicated)"`, together with the list of candidate predictions considered and why each was rejected:
  - FD destination: equals Bayes without the factor;
  - O-tie sensitivity: a function of the posterior on GATE only;
  - soft-gate expected-energy pooling: equals mean-field / log-linear pooling; not in the note; needs a bridge assumption; GATE-only;
  - T4 latent-role expected energy: calibration-only;
  - T4 additive complex: not representable below order 18 at the log-odds level.
- State in the C3 entry that M1 is false in general (give the counterexample in §3.2).

**C-MAJ-4 (MAJOR, the T4 order theorem): the scope is overstated.**
Required change:
- Restate the result as: "No additive complex over observed inputs with all hyperedges of order < 18 **reproduces T4's exact Bayes log-odds** (R1-type). Decision-level (R3-type) representability is not addressed."
- Add: "with order unrestricted, a single all-coordinate hyperedge represents f trivially; the only non-trivial completion adds latent roles, which is standard factor-graph marginal inference".

### Minors

**C-MIN-1 (MINOR):** T2 is a tautology in code. Compare `O_of(d, W*, s)` against `D_direct` instead.

**C-MIN-2 (MINOR):** T3 cites Phase 1 for "Theorem Sign", which is actually Step 1B. Add the premise that the decision margin is an odd function of s.

**C-MIN-3 (MINOR):** the R2 docstring claims least squares shows gate-blindness, but the LS projection's F_G is 0.16–0.23. Delete the claim, or report the LS fit as a descriptive approximation only.

**C-MIN-4 (MINOR):** report decisive gate flips as both 216 ordered and 108 unordered. State that the definition matches the historical F_G definition (LEARNING_SIGNAL_AND_STAGE_VALIDATION §1), so the 162 discrepancy lies in the note, not in an unrecoverable definition.

**C-MIN-5 (MINOR):** relabel the rounding bound as heuristic. Add a high-precision recomputation (mpmath, ≥ 30 digits) of at least one nonzero size-18 coefficient as the certificate. I did this: −20.8558394526408 and 138.635712661052.

**C-MIN-6 (MINOR):** fix the protocol filename reference in the `run_track_c` docstring (`od_protocol.json` → `track_c_protocol.json`). Add the exact Farkas verification to `lp_margin` so that "exact certificate" is literally true.

**B-MIN-1 (MINOR):** remove "after calibration (F0 fitted on half A)" from the HB_c text; decisions do not depend on it.

**B-MIN-2 (MINOR):** report matched-step censoring and the matched-step distribution **by class**. The dev data show MIXTURE matched at around step 600 and EARLY at around 1211. Add a sensitivity analysis using KL-interpolated statistics.

**B-MIN-3 (MINOR):** with one difference per parent, the "two-stage" bootstrap is a parent-level percentile bootstrap, which is anti-conservative at 4 parents. Use an exact sign-flip / permutation test, or a t-interval on parent means, alongside it.

**B-MIN-4 (MINOR):** state in the frozen prediction that HB_e is expected to be positive (dev ΔKL ≈ 0.015 > delta_kl), so that a "+B3" flag is not later read as a discovery.

---

## 6. Per-test decision table

| Track | Test | Decision | Condition |
|---|---|---|---|
| C | T1 (D formula) | ACCEPT | — |
| C | T2 (W* ⇒ O ≡ D) | ACCEPT-WITH-CHANGE | C-MIN-1 |
| C | T3 (evenness) | ACCEPT-WITH-CHANGE | C-MIN-2 |
| C | T4 (Attack F) | ACCEPT | — |
| C | T5 (source-level gate-flip following) | **REJECT as a test** | Guaranteed by construction. Replace with the order-1 LP certificate (C-MAJ-1). |
| C | LP order certificates | ACCEPT-WITH-CHANGE | Add the exact Farkas verification (C-MIN-6) |
| C | R1, R3, R4, R5 | ACCEPT-WITH-CHANGE | Present as proved identities with scope (C-MAJ-2) |
| C | R2 (least-squares order) | ACCEPT-WITH-CHANGE | C-MIN-3 |
| C | M1 | ACCEPT-WITH-CHANGE | GATE-only; give the general counterexample (C-MAJ-3) |
| C | M2 | ACCEPT-WITH-CHANGE | Independent marginalised-joint reference (C-MAJ-2) |
| C | Latent-marginalisation / note discrepancies | ACCEPT-WITH-CHANGE | C-MIN-4 |
| C | T4 interaction-order bound (18) | ACCEPT-WITH-CHANGE | Log-odds scope; high-precision certificate (C-MAJ-4, C-MIN-5) |
| C | `classify` (C2 / C3 / C4 / C5) | ACCEPT-WITH-CHANGE | C-MAJ-1, C-MAJ-2, C-MAJ-3 |
| C | Neural O/D experiment | **None planned: ACCEPT (correctly omitted)** | Document the candidates considered (§3.6) |
| B | Candidate rules: bayes, local, noelig, nocount | ACCEPT | — |
| B | Candidate rule: maprole | ACCEPT-WITH-CHANGE | B-MAJ-5 |
| B | Candidate set as a whole (mandate B1) | ACCEPT-WITH-CHANGE | B-MAJ-4 |
| B | Discriminating set (on-support) | ACCEPT | — |
| B | Relational hybrids / ideal reference | ACCEPT | Verified on-support |
| B | Matched precision | ACCEPT-WITH-CHANGE | B-BLK-2 (KL adjustment), B-MIN-2 |
| B | HB_a | **REJECT as a B1 input** (descriptive at most) | B-BLK-2 |
| B | HB_b | **REJECT as built** (guaranteed) | B-MAJ-2 |
| B | HB_c | ACCEPT-WITH-CHANGE | B-MAJ-1, B-MIN-1 |
| B | HB_d | ACCEPT-WITH-CHANGE | B-MAJ-3, B-MAJ-6 |
| B | HB_e | ACCEPT | B-MIN-4 |
| B | B1 / B2 / B3 classification | **REJECT as frozen** | B-BLK-1, B-BLK-2, B-MAJ-1 |
| B | Thresholds | ACCEPT-WITH-CHANGE | B-MAJ-7 |

**Re-review required** (mandate §2: "any corrected design must be reviewed again") for all Track B changes and for the revised `classify`. This must happen before `protocols/FREEZE.json` records the Track B code hash and before any Track B job runs on confirmation checkpoints.

*Reviewer's scratch checks (not part of the package): `/tmp/revB/{t4exact,t4exact18,gatechk,mf,farkas,recov,hbc,softrole,ties}.py`.*
