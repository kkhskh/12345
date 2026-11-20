# Pre-run review B, round 2: mathematics and functional equivalence (Investigation XX, design v2)

**Reviewer:** an independent AI agent reviewer (Claude), specialty mathematics and functional equivalence. I am not a human peer reviewer. I did not build this package, and I did not write the round-1 review B. I read it, and the adjudication, as inputs.
**Scope:** I verified in the code every round-1 item from `reviews/PRE_RUN_REVIEW_B_mathematics_functional_v1.md` against `reviews/ADJUDICATION_v1.md` and `protocols/DESIGN_XX.md` (v2). I also looked for new problems that the v2 changes introduced.
**Files read:**
- Track C: `cor_research/mathematics/{od_original,od_repaired,t4_order,run_track_c}.py`, `dev/track_c/track_c_math_v2.json`, `tests/test_track_c.py`, `tests/independent/t4_highprecision_certificate.py`, and the vendored `xvitask.py`.
- Track B: `cor_research/generators/t4.py`, `cor_research/scoring/{functional,track_b,trackb_run}.py`, `cor_research/protocol.py` (TRACK_B, TRACK_C), `tests/test_trackb_oc.py`, `tests/test_scorers_planted.py`, `tests/trackb_oc_results.json`, and `dev/DEV_NOTES_XX.md`, `dev/trackb_dev_rules.json`, `dev/trackb_ic/*.json`.

**Checks I ran:**
- `python3 tests/test_trackb_oc.py`: 67 s, PASS. It reproduces `tests/trackb_oc_results.json` exactly, and the file was byte-identical after the run.
- My own CPU scratch scripts, in `/tmp/revB2/{leak,tost,alpha,ties,leak2}.py`.

No package file was modified except this review.

---

## 1. Verdict

**Overall: APPROVE WITH REQUIRED CHANGES. Track C is approved. The Track B classification is still REJECTED as frozen, because of one new blocker (N1).**

**Track C: APPROVE.** Every round-1 major is fixed in the code.
- The Farkas verification is genuinely exact (rational arithmetic on the ±1 integer constraint matrix).
- The M2 reference is genuinely independent of the energies.
- The C5 list is complete and honestly labelled as reviewer-adjudicated.
- Only documentation residues remain (N8).

**Track B: much improved, but not freezable.**
- The round-1 blockers are fixed as designed:
  - TOST equivalence replaces failure-to-reject.
  - HB_a is removed from B1.
  - The α′ profile absorbs the round-1 counterexample exactly.
- I also verified that **every** decision-level test is immune to α-imprecision: Bayes(α′) for α′ from 0.60 to 0.95 has **0.0000** decision disagreement with Bayes on the frozen discriminating set.
- **The new blocker (N1):** the rule-directed test HB_c_dir, and through it the equivalence criterion, is computed against **argmax tie-break artefacts**.
  - On the frozen discriminating set, **100%** of `maprole`'s and `nocount`'s disagreements with Bayes are exact ties in the rule's output that argmax breaks toward the lowest class index. So is 37% of `local`'s.
  - Tie-aware, `maprole` (and `softrole`) never disagree with Bayes on any natural episode.
  - The consequence is that HB_c_dir for `maprole` measures the model's error rate. A pure precision difference at the dev error rates (EARLY 2.6%, MIXTURE 1.4% Bayes-disagreement on the discriminating set) shifts it by **+0.085**, almost 3× delta_dir.
  - So equivalence is unreachable whenever HB_e is positive, which the frozen prediction itself expects. The predicted B2-LOCUS(+B3) and B3 outcomes are therefore unreachable at the final level, and the headline reads "NON_IDENTIFIABLE (underpowered)" for a reason that is not power.
  - Combined with any non-rule HB_c difference, the same leak would yield a spurious "B1-FUNCTIONAL (maprole)".
- **Why the OC simulation missed this:** by construction, its noise can never move a decision toward a candidate rule (N3).

**Consequence for the run:** as in round 1, Track A training may proceed. The Track B code and thresholds must be corrected (N1–N4) and re-checked before `protocols/FREEZE.json` records the Track B hash, and before any `trackb_run battery` runs on confirmation checkpoints. The changes are mechanical. A re-check limited to N1–N4 is sufficient.

---

## 2. Status of every round-1 item

| id | round-1 issue | adjudication | verified in code | status |
|---|---|---|---|---|
| B-BLK-1 | "no difference" = failure to reject | TOST | `track_b.score_level` l.81–99, 116–124: HB_c equivalence = one-sided upper 95% bound (two-sided 90% t-interval) < delta_dis; HB_c_dir equivalence = 90% CI within ±delta_dir for every rule (correct TOST at α = .05; intersection–union, so no multiplicity correction is needed); NON_IDENTIFIABLE otherwise; planted test `B_underpowered_is_NON_IDENTIFIABLE_not_B2` | **Resolved in construction**, but its dir component is invalidated by N1 |
| B-BLK-2 | parametric imprecision read as rule mixture | F0′ with α′ grid; HB_a descriptive; B1 needs decision tests | `functional.rule_fits` profiles α′ on a 10-point grid; HB_a is outside the classification. Reviewer checks: planted Bayes(0.70) gives α* = 0.70 and w = 0 for every rule; Bayes(0.67) and Bayes(0.72, τ = 0.8) give w_maprole ≈ 0.05 but dKL ≤ 8e-5; Bayes(α′) has 0.0000 decision disagreement on the disc set for α′ ∈ [0.60, 0.95] | **Resolved** (minor residue N7) |
| B-MAJ-1 | HB_c unbalanced, non-directional | balanced same-unit D1 ≠ D2 contrast + same-D 2×2 secondary + HB_c_dir | l.68–80: balanced and correct. Reviewer check: a pure precision difference gives HB_c = 0.0018 (KL 0.03 vs 0.012) with per-parent sampling SD 0.0002, so HB_c is precision-robust. HB_c_dir exists but is broken (N1, N2) and does not match its text (N5) | **Partially resolved** |
| B-MAJ-2 | iid null unfair | parametric null added | `functional.nulls` (bisection over α ∈ [0.4, 0.8]); HB_b descriptive vs the parametric null | **Resolved** (minor N7) |
| B-MAJ-3 | OQI donor not source-matched | donor = run(x_h); unmatched secondary; base-run null; IIA_norm | `functional.interchange` l.103–117 does exactly this | **Resolved** |
| B-MAJ-4 | historical models, equivalence table | power-mean descriptive; softrole; `equivalence_table`; threshold-4 not located | Present. The table reproduces the DESIGN numbers exactly (local 0.2391, noelig 0.6158, nocount 0.0899, maprole 0.1353, softrole 0.0088, Bayes(0.7) 0.0234). But they are raw-argmax numbers, mostly tie artefacts, and there is no category column or list of non-identifiable parameters (N4). Not locating threshold-4 is acceptable under mandate C1 | **Partially resolved** |
| B-MAJ-5 | maprole definition and position ties | full role MAP, role-level tie-averaging | `t4.role_rules`: role posterior includes question q; ties averaged (tie fraction 0.204, reproduced). Reviewer check: role-sum = `posterior_brute` to 0.0. **But the position artefact has moved to the decision level:** the tie-averaged rule output has exact top-2 ties on 14% of the disc set, broken to index 0 in 66% of cases (N1) | **Partially resolved** (the definition is fixed; the artefact remains) |
| B-MAJ-6 | no positive same-statistic test | documented and deferred; B2 renamed B2-LOCUS | DESIGN §2, protocol classification text | **Acceptably deferred** |
| B-MAJ-7 | thresholds not justified | OC simulation | The simulation runs and reproduces. Its generative model makes precision effects on rule agreement impossible by construction; the "noise floor 1%" justification is not the operative variance; 20 reps (N3) | **Partially resolved** |
| C-MAJ-1 | T5 guaranteed; C4 mis-scoped | C4 on exact order-1 LP; germ-level C5 | `classify` l.70–73; v2 JSON: order-1 infeasible with `farkas_exact.verified_exact = True` in all 5 conditions (support 19/13); germ-level = "C5 / underdetermined"; T5 descriptive. `gate_source_level` function docstring still says "ALL six sources" (N8) | **Resolved** |
| C-MAJ-2 | proof; independent M2 | proof and scope in `od_repaired`; M2 vs marginalised joint | Proof docstring is correct and scoped. M2 reference (l.221–228) marginalises the source out of `xvitask.joint`'s p(x, y), which is built directly from the generative description and shares no code with `energies`; max error 2.2e-16. m1 removed from the C2 condition | **Resolved** |
| C-MAJ-3 | C5 designer-enumerated | "reviewer-adjudicated" plus 5 candidates | `classify` l.78–85: all five round-1 candidates, with honest reasons. I found no further candidate. The softrole entry ("about 1%") should read "0% tie-aware; differs only in tie-breaking" (N8) | **Resolved** |
| C-MAJ-4 | T4 order-theorem scope | log-odds scope, trivial completion, latent roles | `t4_order.run` statement is exactly as required | **Resolved** |
| C-MIN-1 | T2 tautology | vs `D_direct` | l.50 | Resolved |
| C-MIN-2 | T3 premise and citation | | module docstring | Resolved |
| C-MIN-3 | LS gate-blindness claim | descriptive | `od_repaired` docstring | Resolved |
| C-MIN-4 | 216/108; F_G definition | | JSON reports both. DESIGN does not say that the definition matches the historical F_G definition (LEARNING_SIGNAL §1) | Mostly resolved (N8) |
| C-MIN-5 | rounding bound heuristic; high-precision certificate | | statement says "heuristic"; independent script included. The `t4_order` module docstring still says "bounded and reported" | Mostly resolved (N8) |
| C-MIN-6 | protocol filename; exact Farkas | | `classify` docstring fixed; **module** docstring of `run_track_c` still says `od_protocol.json`. Farkas exact: yes | Mostly resolved (N8) |
| B-MIN-1 | "after calibration" wording | | removed | Resolved |
| B-MIN-2 | censoring by class | | `censoring_by_class`, `matched_step_by_class`, gap rule | Resolved |
| B-MIN-3 | anti-conservative bootstrap | t-interval plus sign-flip | `tci` | Resolved |
| B-MIN-4 | HB_e expected positive | | protocol HB_e text and frozen prediction | Resolved (but see N1: it now breaks equivalence) |

### 2.1 Specific questions in the brief

**Is the Farkas verification genuinely exact? Yes.**
- `farkas_exact` takes the LP's Gordan multiplier y, rationalises it with `limit_denominator(10**6)`, and recomputes Mᵀy in `Fraction` arithmetic over the integer matrix entries.
- Mx has only ±1 entries (the c* and c keys are distinct). The function requires Mᵀy = 0 exactly, y ≥ 0, and Σy > 0, which is Gordan's alternative to "∃F: MxF < 0", i.e. to strict reproduction of every unique-argmax Bayes decision.
- If rationalisation had broken exactness, the flag would read False, so the check is honest.
- Two optional hardenings (N8):
  - assert that every matrix entry is integral, rather than relying on `int(round(v))`;
  - note that c* is identified in float. The GATE posterior gaps are far above the 1e-9 tolerance, so this is not a practical risk.

**Is the M2 reference independent? Yes.** See C-MAJ-2 above.

**Is the C5 list complete and honest? Yes.** It is honestly labelled, and it is complete as far as I can determine.

**Is the TOST construction correct? Yes in construction. Achievability:**
- For HB_c alone, equivalence is easy: per-parent sampling SD is about 0.0002. It holds unless the true between-parent SD of HB_c exceeds about 0.017 at 4 parents or about 0.03 at 8.
- For the dir component (after N1 is fixed), my simulation at matched precision gives P(equivalence):

  | parents | between-parent SD 0.01 | between-parent SD 0.02 |
  |---|---|---|
  | 4 | 0.95 | 0.10 |
  | 8 | 1.00 | 0.85 |

- The between-parent SD is unknown, because there is one dev parent. The design should say plainly that at 4 parents equivalence needs a between-parent SD of the directed effect ≲ 0.01 (N3).

**Is HB_c_dir well-defined and not guaranteed?**
- It is well-defined for any rule whose disagreement subset S_k is non-empty, and it is not guaranteed in sign.
- **But for `maprole` and `nocount` it is not a rule test** (N1). For the remaining rules it is precision-confounded at the 0.002–0.01 level (N2).

**Does the α′ profile absorb parametric imprecision? Yes, and it is moot for decisions.**
- Bayes(0.70) → α* = 0.70, w ≡ 0.
- Off-grid α leaves w_maprole ≈ 0.05, but with dKL ≈ 1e-4. That is harmless now that HB_a is descriptive; N7 asks for a dKL floor before any w is reported.
- Decisions are α-invariant on the disc set, so the round-1 counterexample cannot reach B1 any more.

**Are delta_dis = 0.02 and delta_dir = 0.03 justified by the OC, and is its generative model realistic? No.** See N3.
- The model is unrealistic in five ways:
  1. synthetic rules with no ties;
  2. noise that by construction never lands on a rule's decision;
  3. no class-dependent decision precision;
  4. E-base units only, so the native-vs-induced requirement is vacuous;
  5. 20 reps.
- On the "1% per-model noise": the real dev models disagree with Bayes on 1.4–2.6% of the disc set (`dev/trackb_ic`, `model_agree.bayes` 0.974–0.986). So 1% is too low for EARLY, but the right order of magnitude.
- HB_c's between − within cancels per-model noise to first order. So "delta_dis ≈ 2× the noise floor" is not the relevant justification. The relevant quantities are the minimum effect of interest and the between-parent SD.

---

## 3. New objections

### N1 (BLOCKER, Track B: HB_c_dir and equivalence): rule "decisions" are argmax tie-break artefacts

**Evidence** (`/tmp/revB2/ties.py`, frozen disc set: stream 91_410_000, pool 40000, 20141 kept):

| rule | \|S_k\| (raw argmax ≠ Bayes) | share where the Bayes decision is inside the rule's exactly tied top set | genuine S_k |
|---|---|---|---|
| local | 8289 | 0.37 | 5262 |
| noelig | 18671 | 0.00 | 18671 |
| nocount | 2268 | **1.00** | **0** |
| maprole | 1678 | **1.00** | **0** |

On natural episodes, the tie-aware decision disagreement with Bayes is:

| rule | tie-aware | raw argmax |
|---|---|---|
| maprole | **0.0000** | 0.1353 |
| softrole | 0.0000 | 0.0088 |
| nocount | 0.0302 | 0.0899 |
| local | 0.1360 | 0.2391 |
| noelig | 0.5719 | 0.6158 |
| Bayes(0.7) | 0.0123 | 0.0234 |

Bayes itself is exactly tied on 14% of natural episodes. Ties are broken to class 0 in 66–74% of tied cases.

- So the full role MAP is **decision-equivalent to Bayes up to tie-breaking**, and nocount differs from Bayes only where Bayes is near a tie, which the disc set excludes.
- On S_maprole the "maprole answer" is the lowest-index member of {Bayes answer, runner-up}. A model's errors land there.
- HB_c_dir(maprole) therefore measures the class difference in error rate:

  | scenario | HB_c_dir(maprole) |
  |---|---|
  | iid logit noise at KL 0.03 vs 0.012 | +0.073 |
  | calibrated to the dev disc-set error rates 2.6% vs 1.4% | **+0.085** |
  | error rates 2.0% vs 1.4% | +0.041 |
  | equal error rates | +0.0007 |

- **Consequences:**
  - EQUIVALENT (all four rules within ±0.03) fails with probability ≈ 1 at the final level whenever HB_e is positive (P(EQ) = 0.00 at 4 and 8 parents in `/tmp/revB2/tost.py`). It fails with probability 0.02–0.13 at the matched level, even when the classes are one snapshot apart (KL 0.03 vs 0.02).
  - So B2-LOCUS, B3 and EQUIVALENT_NO_DIFFERENCE_DETECTED are effectively unreachable in the world the frozen prediction describes, and the headline becomes "NON_IDENTIFIABLE (underpowered)", which is mislabelled.
  - DIRECTED_DIFF fires on maprole: P = 1.0 at the final level, 0.12–0.52 at the matched band. Together with any non-rule HB_c difference, that yields **B1-FUNCTIONAL attributed to a rule that never disagrees with Bayes**.

**Required change:**
1. In `t4.rules` (or in `functional.run_checkpoint` when saving `rule_dec`), save each rule's **top set** (relative tolerance 1e-9) rather than its argmax.
2. Define S_k = {x : Bayes top set is a singleton, and it is disjoint from rule k's top set}. Define "agreement with k" as: the model's decision lies in k's top set.
3. Apply the same tie-aware definition in `t4.discriminating` (7.3% of the current disc set has no genuine rule disagreement) and in `equivalence_table`.
4. Rules with an empty or tiny genuine S_k (frozen floor, e.g. < 200 inputs; currently `maprole` and `nocount`) are **removed from HB_c_dir, DIRECTED_DIFF and the equivalence conjunction**. They are recorded in the equivalence table as "decision-equivalent to Bayes up to tie-breaking" (maprole, softrole) or "differs only near Bayes ties; no on-support decisive disagreement" (nocount).
5. Re-run the planted tests and the OC (N3) after the change.

### N2 (MAJOR, Track B: HB_c_dir): no precision adjustment for the directed statistic

- Even after N1, a pure precision difference moves rule agreement. On genuine S_k, at iid-KL calibration (0.03 vs 0.012), local shifts by +0.008 and noelig by +0.010. At the dev error-rate calibration the shifts are +0.002 and +0.005.
- That is up to a third of delta_dir, in the direction "the less precise class is closer to every rule".
- HB_c is robust to this (+0.002); HB_c_dir is not.

**Required change** (either option):
- (a) Per model, subtract a surrogate expectation: agreement_k of Bayes + logit noise, with σ calibrated so that the surrogate's disc-set Bayes-disagreement equals the model's. That is, test the *excess* rule-k agreement. Or
- (b) Use the error-conditional statistic: among the model's non-Bayes decisions on genuine S_k, the share in k's top set, minus the surrogate's share.

Show in the OC (N3) that the adjusted statistic has |bias| < delta_dir/5 under a precision-only difference at error rates of 2.6% vs 1.4%.

### N3 (MAJOR, Track B: thresholds and OC): the OC generative model cannot produce the failure modes that matter

`tests/test_trackb_oc.py` and the Track B planted tests have the following problems:
- (i) The rules are independent random +1 shifts of a random "Bayes" vector, with no ties.
- (ii) Noise flips go to (base + 2) % 3, which **never equals any rule's decision**, so noise cannot touch rule agreement by construction.
- (iii) The classes have identical decision noise. Even the planted `B_B3_precision` case changes only the KL field, never the decisions.
- (iv) Only E-base units exist, so `native_vs_induced.same_sign` is None and the requirement passes vacuously.
- (v) Matched and final are identical copies.
- (vi) 20 reps, so "P(B1 | null) ≤ 0.05" is checked at a resolution of 0.05.
- (vii) The between-parent SD (effective about 0.006 on HB_c and 0.01 on dir) is asserted, not derived.

**Required change:** rebuild the OC on the **real frozen discriminating set and real (tie-aware) rule outputs**, with the following specification:
- Model decisions = argmax(log Bayes + σ_c z), with σ_c calibrated to disc-set Bayes-disagreement of about 2.6% (EARLY) and 1.4% (MIXTURE), the dev values. Add an equal-precision arm.
- Add planted class-specific systematic shifts toward each rule's genuine S_k.
- Use both bases with fullB/fullI and the real arm-to-class mapping.
- Give the matched and final levels different precision gaps.
- Run ≥ 200 reps per cell, at 4, 6 and 8 parents.
- Bracket the between-parent SD over 0.005, 0.01, 0.02 and 0.03.
- Report P(B1), P(EQ), P(B2-LOCUS), P(B3) and P(NON_IDENTIFIABLE) under each of: null; precision-only; locus-only; rule-directed effect sizes of 0.02, 0.05 and 0.10; and a non-rule functional difference plus precision.
- Freeze these gates:
  - P(B1 | precision-only) ≤ 0.05;
  - P(B1 | non-rule difference + precision) ≤ 0.05;
  - P(EQ | precision-only, 8 parents, SD 0.01) ≥ 0.8.
- In the protocol, replace "delta_dis ≈ 2× the per-model noise floor" with the actual derivation: the minimum effect of interest plus the between-parent SD needed for the stated power.

### N4 (MAJOR, mandate B1: equivalence table)

- `t4.equivalence_table` and DESIGN §2 report raw-argmax disagreement. That overstates maprole (0.135 vs 0.000), nocount (0.090 vs 0.030), local (0.239 vs 0.136) and softrole (0.009 vs 0.000).
- The mandate-required category per candidate (equivalent / calibration-only / posterior-combination / off-support-only / genuinely different on-support) and the list of non-identifiable fitted parameters are still only in prose.

**Required change:**
- Report tie-aware disagreement, both on natural episodes and on the disc set.
- Add the category column. With tie-aware numbers:
  - maprole and softrole: equivalent at decision level (differ only in tie-breaking; calibration-level);
  - nocount: differs only near Bayes ties;
  - local and noelig: genuinely different on-support;
  - Bayes(α′): calibration-only on the disc set.
- List the non-identifiable parameters: (τ, ε, α′) are partially confounded, and w_k has cross-loading (round-1 §2).
- Update DESIGN §2 and the protocol `candidates` string.

### N5 (MINOR, HB_c_dir definition mismatch)

- The protocol and DESIGN define HB_c_dir as "the between-class minus within-class difference in agreement with k".
- `track_b.score_level` l.78–80 computes the plain class-mean difference EARLY − MIXTURE per parent. Within-class spread enters only through the parent-level t-test.
- Either text can be statistically acceptable, but they must agree. **Required change:** align the text with the code (preferred), or implement the stated contrast.

### N6 (MINOR, native-vs-induced requirement can pass vacuously)

- In `DIRECTED_DIFF`, `nat_ind[k]["same_sign"] is not False` passes when the contrast is None, i.e. when no parent has both bases.
- **Required change:** for B1, require `same_sign is True`. Otherwise label the outcome "B1-FUNCTIONAL (induced/native consistency not testable)".

### N7 (MINOR, descriptive HB_a / HB_b residues)

- (i) The parametric null bisects α over [0.40, 0.80], while the F0′ grid is [0.60, 0.95]. For model KL ≳ 0.08 the null's α falls off the grid. The null is also one-sided (α < 0.8 only).
- (ii) α* is selected by held-out KL on half B, and dKL is then evaluated on the same half B. That gives an optimistic F0′ and a conservative dKL.
- (iii) Off-grid α leaves w ≈ 0.05 with dKL ≈ 1e-4.

**Required change:**
- Widen the grid to the null's range (or restrict the null to the grid range).
- Select α* on half A.
- Report w_k only where dKL_heldout exceeds a frozen floor (e.g. 5e-4). Otherwise report "not identified".

### N8 (MINOR, Track C documentation residues; no effect on results)

- The `run_track_c` module docstring still cites `protocols/od_design/od_protocol.json`. The file is `track_c_protocol.json`.
- The `t4_order` module docstring says the float64 rounding is "bounded and reported". It must say "heuristic", as the statement does.
- The `od_original.gate_source_level` docstring says "ALL six sources", but the code uses the evidence sources only.
- DESIGN §3 should state that the 216-ordered / 108-unordered definition matches the historical F_G definition, so the 162 discrepancy lies in the note.
- In `classify.candidates_considered`, replace "about 1% of decisions" for the T4 latent-role candidate with the tie-aware value: 0% (it differs only in tie-breaking), and 0.9% raw.
- Optional: assert integrality of the matrix entries in `farkas_exact`.

### N9 (MINOR, test claims and outputs)

- The `t4.py` docstring says "verified: their role-sum equals posterior_brute to 1e-12 in tests", but no test does this. I verified it: max abs 0.0 on 3000 episodes. **Required change:** add that check to `tests/test_track_c.py` or to the planted tests.
- `tests/test_trackb_oc.py` and `tests/test_scorers_planted.py` write their result files into `tests/` whenever they are run. That is fine if the freeze hashes these files intentionally. Otherwise they should write to a temporary directory and compare.

---

## 4. Design gate (mandate §2): new or changed tests only

| test (v2) | 1 Guaranteed? | 2 Literature | 3 Competing predictions differ? | 4 Metric discriminates? | 5 Controls | 6 Positive establishes | 7 Negative weakens | 8 Inconclusive possible |
|---|---|---|---|---|---|---|---|---|
| Track C: order-1 LP + exact Farkas (C4, source level) | no (it could have been feasible) | XVI review A F6; Gordan's theorem | yes | yes | exact rational verification; order a+1 feasible as positive control | no order-1 additive rule reproduces GATE Bayes decisions (proof) | n/a (proof) | no (deterministic) |
| Track C: M2 vs marginalised joint | yes, as a theorem (now with an independent reference) | factor-graph marginalisation | no | n/a | independent reference from `xvitask.joint` | C2 with scope | — | no |
| Track C: `classify` v2 / C5 adjudicated | C5 is a judgement (labelled) | — | the five candidates are listed with reasons | — | planted M2 / LP flips change the label (`test_scorers_planted`) | — | — | yes (UNRESOLVED / C1-candidate paths exist) |
| HB_c (balanced, same unit, D1 ≠ D2) | no | decision-level model comparison | B1 vs B2/B3: yes. Robust to α-imprecision (0 disagreement) and to precision (+0.002) | yes | shared-D planted null; same-D 2×2 secondary | a decision-function difference (not its rule) | yes, with TOST | yes |
| HB_c_dir | no, **but for maprole/nocount it is a precision measure (N1)**; for local/noelig it is precision-confounded at ≤ 0.01 (N2) | — | only after N1/N2 | **no** for maprole/nocount (empty genuine S_k) | needs a precision surrogate (N2) | after the fixes: which rule a difference points to | yes | yes |
| TOST equivalence | no | Schuirmann TOST; intersection–union | yes | HB_c part yes; dir part broken by N1 | — | functional equivalence at ± delta | — | yes (NON_IDENTIFIABLE) |
| F0′ α′ profile / HB_a (descriptive) | no | parametric-calibration families | descriptive only | w not identified (cross-loading); needs a dKL floor (N7) | parametric null | — | — | — |
| HB_b vs parametric null (descriptive) | not guaranteed any more (the null is a fair smooth approximant) | — | — | weak | both nulls | — | — | — |
| HB_d (hybrid donor, base null, IIA_norm) | OQI_0 and RS_3 equal the hybrid run by construction (sanity checks, not tests); OQI_2 is not | interchange interventions (Geiger et al. 2021) | locus only (declared) | yes (dev 0.44/0.49 vs 0.24/0.20, source-matched) | base-run null; unmatched-donor secondary | a locus difference only | — | yes |
| maprole (full role MAP, tie-averaged) / softrole / equivalence table | — | — | maprole and softrole are decision-equivalent to Bayes (N1, N4) | maprole: no | — | — | — | — |
| Matching (κ_p, earliest snapshot ≤ κ_p) | no | — | — | It leaves residual precision gaps between classes at the matched level (snapshot granularity of about 26 per run). This matters only through N1/N2: report the matched-level KL and disc-set error rate by class | censoring rule | — | — | yes |

---

## 5. Decision table (changes since round 1)

| track | item | decision |
|---|---|---|
| C | all tests and `classify` v2 | **ACCEPT** (N8 documentation fixes before freeze) |
| C | no neural O/D experiment | **ACCEPT** |
| B | HB_c (primary, balanced), TOST for HB_c | **ACCEPT** |
| B | HB_c_dir and the equivalence conjunction | **REJECT as built** (N1). Accept after N1 + N2 |
| B | maprole and nocount as decision-level rule tests | **REJECT** (decision-equivalent to Bayes on the disc set). Keep them in the equivalence table only (N4) |
| B | local and noelig as decision-level rule tests | ACCEPT-WITH-CHANGE (tie-aware; N1, N2) |
| B | F0′ / HB_a / HB_b (descriptive) | ACCEPT-WITH-CHANGE (N7) |
| B | HB_d v2 | **ACCEPT** |
| B | classification v2 | **REJECT as frozen** until N1–N3 are done. The logic itself is acceptable; N6 is a small fix |
| B | OC simulation / thresholds | ACCEPT-WITH-CHANGE (N3; re-derive thresholds from the rebuilt OC) |

*Reviewer's scratch checks (not part of the package): `/tmp/revB2/leak.py` (precision leak, iid-KL calibration), `/tmp/revB2/tost.py` (TOST / DIRECTED_DIFF operating characteristics on the real disc set), `/tmp/revB2/alpha.py` (role_terms vs posterior_brute, α-invariance of decisions, α′ profile), `/tmp/revB2/ties.py` (tie-driven rule disagreements), `/tmp/revB2/leak2.py` (leak calibrated to the dev disc-set error rates).*
