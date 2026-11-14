# Independent mathematical review: Workstream C (MATHEMATICAL_SUFFICIENCY_AND_GENERALIZATION.md)

Package: `docs/research_audits/cor_phase15/step1_relational_decision_computation` (below, "the doc" means MATHEMATICAL_SUFFICIENCY_AND_GENERALIZATION.md).
Reviewer scratch: `/tmp/review_math/`. I made no edits to the package. I copied both package scripts with their output paths redirected and ran them with `python -B`, so that no `__pycache__` was written into `src/`. Both reproduce the logged JSON exactly: `sufficiency_boundary.json` is byte-for-byte equal after key sorting, and `verify_shared_posterior.json` gives identical numbers. Each script takes about 10–15 s, not minutes. My own checks are in `/tmp/review_math/checks.py`, which runs in about 20 s.

## Verdict summary

The core mathematics is correct:
- `posterior_shared` is the exact posterior of `genlib.sample`.
- The brute-force enumerator is independent of it in the respects that matter.
- R1–R4 are correct as formulas. I derived each one and re-implemented each independently. All match `posterior_shared` to about 1e-14 at Q=6.
- The counterexample is valid: the per-group statistics are identical and the exact posteriors differ.

There are two must-fix errors, both of description rather than mathematics:
- **The counterexample is described wrongly.** x₂ does not differ from x₁ "only in whether A and B report identical values". It also makes B and C identical at question 2, and that coincidence is what produces most of the reported posterior shift.
- **The reported accuracies of the hard rule count undefined (NaN) posteriors as predictions of class 0.**

The main over-claims are in the sufficiency boundary (C3, R4):
- "Any per-group rule" and "a joint statistic is required" are false unless the class of rules is restricted. With an unrestricted per-group embedding and pooled readout (deep sets), every symmetric posterior is representable. The claims must be about fixed low-dimensional per-group statistics, or about separable (sum-of-products) structure.
- "Needed exactly on S = 3" is too broad. On a third of the S = 3 slice the product rule is exact.

## Confirmed

1. **`posterior_shared` is exact for `genlib.sample`** (xii_bayes.py:10–25; genlib.py:29–44).
   - The sampler draws `Y`, `col`, `cop`, `off`, `u` and the baseline `X` independently per (episode, question), with the role `ridx` fixed per episode and uniform over `env.roles`.
   - Given the role, questions are therefore i.i.d., and the per-question likelihood is: reliable pair `p1·p2`, using `(Y+off)%K` with `off∈{1..K−1}`, i.e. uniform over wrong answers; colluder set `β·1[all 2|H| equal]/K + (1−β)/K^{2|H|}`; noise `1/K²`.
   - `posterior_shared` uses exactly this, with a uniform role prior, a marginal over Y at q'≠q, and y kept at q.
   - The colluder overwrite (line 43) and the reliable overwrite (line 42) never touch the same group, so line order does not matter.
2. **The enumerator is independent** (verify_shared_posterior.py:11–26).
   - It sums explicitly over y, the shared cop, and the shared c, with per-source probabilities.
   - It shares nothing with `_group_lik` or `posterior_shared` except `env.roles` (which also drives the sampler, so sharing it is harmless) and the assumption that questions factorise given the role.
   - The support sizes check out analytically. C1β1: 729·217 = 158,193. C2β1: 3·27² − 3·9 + 9 = 2,169.
3. **The factorisation was checked against the sampler itself** (my check 1).
   - Factorisation is the one assumption shared by the enumerator and `posterior_shared`, so the exhaustive Q=2 test cannot detect a violation of it. I therefore compared 10⁶ samples of `genlib.sample` (Q=2) with the exact joint p(X₀, X₁) built from an independently written table.
   - Results: C1β1 χ² = 18,334 on 18,225 cells; C2β1 χ² = 2,159 on 2,169 cells; C2β0.6 χ² = 6,126 on 6,309 cells. There were zero off-support samples. The p(X₀, Y₀) deviations are within sampling noise.
   - Because `posterior_shared` is Q-generic (no hard-coded Q) and the per-question structure is the same for every Q, the Q=2 exhaustive test plus this check validates the general-Q formula.
   - I also ran the exhaustive test at q=1 (the package only tests q=0). Maximum error was 1e-16 (C2β1) and 3e-16 (C2β0.6).
4. **R1** (C=1, β=1): χ_h/noi = K·1[agree], the product over questions is K⁶e_h, and summing over h≠g gives Σ_g W_g L_g (S − e_g).
   - e_g in code (genlib.py:81–91) is `n_g(q) == Q−1` (agreement at every *other* question) AND agreement at q. So e_g = 1[agrees at all six questions], as the doc states.
   - My implementation, `posterior_closed`, and `deepsets_factor` all match `posterior_shared` (max error 2e-15, 2e-15 and 9e-14).
5. **R2** (C=1, β<1): χ_h/noi = 1 − β + βK·1[agree]. c_h = Π_q(·) and p ∝ Σ_g W_g L_g Σ_{h≠g} c_h. My implementation matches (3e-14). The formula holds for any G with C=1. c_h depends only on the number of agreeing questions.
6. **R3** (C=2, β=1): χ_H/noi² = K³·1[all four equal], so J_H is as defined and the product rule replaces J_H by e_{h₁}e_{h₂}. My J-form matches (1e-16).
   - Support argument: the true colluders have e = 1, so S ≥ 2. **S = 1 cannot occur at C=2, β=1.** The sampled S distribution is [0, 0, 0.916, 0.084], so the "S ≤ 2" row is exactly S = 2.
   - At S = 2 the reliable group has e = 0. Every wrong role's colluder set contains it, so both J and the e-product vanish there, and the product rule is exact. Confirmed.
   - The claim that on the S = 3 support the posterior is determined by cross-group equality at q is **true**. Equality is transitive, so at q either exactly one pair agrees (the true colluders, and p ∝ L_odd), or all three agree (then every L_g is equal and p ∝ L regardless of roles). My odd-one-out rule matches the exact posterior on S = 3 with max error 2e-18.
7. **R4** (C=2, β<1): χ_H/noi² = 1 − β + βK³·1[all four equal]. My J^β implementation matches (3e-15).
8. **Counterexample statistics** (sufficiency_boundary.py:39–47).
   - Every group self-agrees at all six questions, so e_g = 1 and c_g = (1+2β)⁶ for all g. Each W_g equals (relm/noi)^5 = (0.22·9)^5 = 1.98^5.
   - The values at q = 0 are (A, B, C) = (0, 0, 1) in both inputs, so L_g is identical.
   - Recomputed by hand, the exact posteriors are x₁ = (0.0152, 0.9697, 0.0152), x₂ = (0.498, 0.487, 0.015), and the product rule gives (0.652, 0.333, 0.015). x₂ is off-support at β = 1. All confirmed.
9. **Rank.** χ = (β/K)I + ((1−β)/K⁴)11ᵀ has rank K for β > 0. The script only evaluates β = 1 (sufficiency_boundary.py:33); I checked β = 0.6 and β = 0.1 (rank 3). The six-question kernel J^β on self-agreeing sequences is a Kronecker power with rank 3⁶ = 729 (full).
10. All table numbers in C1–C3 match the logs.

## Errors (must-fix)

**E1. The counterexample description is factually wrong** (doc line 110: "The two inputs differ only in whether groups A and B report identical values at the other five questions").
- In x₂, B2 = [0,2,0,1,2,0] and C = [1,0,0,2,0,1] are **equal at question 2**. That makes the all-four-equal event true for H = {B, C} at q' = 2. This coincidence drives the shift of x₂ toward (0.50, 0.49, 0.015): roles A and C get equal weight 16.6·0.4⁵.
- If B is chosen to differ from both A and C at every q' ≥ 1 (for example B3 = [0,2,1,1,2,0]), the exact posterior of x₂ is (0.059, 0.926, 0.015). It still differs from x₁, so the counterexample still stands, but the difference is far smaller.
- Correction: either reword to "differ only in cross-group identities at q' ≠ q: in x₂, A and B are no longer identical at q' = 1..5, and B and C coincide at q' = 2". Or replace B2 by B3 in sufficiency_boundary.py:39 and report (0.059, 0.926, 0.015). The script comment (line 37, "differ ONLY in cross-group identity at q' ≠ q") is accurate; the doc line is not.

**E2. Hard-rule accuracies include NaN rows counted as predictions of class 0** (doc lines 60, 100; sufficiency_boundary.py:18).
- Where S = 0 the hard rule returns NaN, and `np.argmax` of a NaN row returns 0. Those 77% (C=1, β=0.6) or 70% (C=2, β=0.6) of episodes are therefore scored as "predict 0" (accuracy about 1/3).
- The reported 0.397 (R2) and 0.361 (R4 table) are artefacts of this. On the rows where it is defined, the C=1, β=0.6 hard rule scores 0.612 against a Bayes accuracy of 0.643 (log, S12 slice). My independent sample gives 0.624 on the defined rows.
- Correction: report accuracy on the defined rows together with the undefined fraction, for example "undefined on 77% of episodes; on the remaining 23%, accuracy 0.612 vs Bayes 0.643, KL 0.313". Alternatively, state explicitly that undefined outputs are scored as a fixed guess. Apply the same fix to the R4 table row, "hard C=1 rule 0.361".

## Over-claims / wording

**O1. "Any per-group rule" and "a joint, cross-group statistic is required"** (doc line 93; C3 item 2, lines 119–121).
- With an unrestricted per-group embedding (for example, a one-hot of each group's full 12-value record) and an arbitrary pooled readout, every permutation-invariant function of the three groups, including the exact C=2 posterior, is representable (deep-sets universality on a finite domain). So "per-group statistics are insufficient" is false without a restriction. The counterexample only defeats rules of the form F({(agreement pattern_g, W_g, L_g(·))}_g); group B's full data differs between x₁ and x₂.
- Correction for line 93: "No rule that sees, per group, only (e_g or c_g, W_g, L_g(·)) — whatever the pooled readout — is sufficient here (counterexample below)."
- Correction for C3.2: "The exact posterior is not a function of the fixed low-dimensional per-group statistics (c_g, W_g, L_g); it depends on the cross-group statistic J_H^β. Equivalently, each role's weight is not separable across groups: the pairwise kernel J^β on self-agreeing sequences has separation rank K^Q = 729, so any sum-of-products (factorised) representation needs per-group features of dimension ≥ 729." The rank statement is a bound for sum-of-products representations only, not for arbitrary nonlinear readouts.

**O2. Wording of C3 item 1, "per-group statistics with a pooled readout suffice for C = 1, for every β"** (line 117).
- The statement is true, but it is non-trivial only because the statistic is fixed and low-dimensional: φ_g = (c_g, W_g L_g(·), c_g W_g L_g(·)), i.e. 1 + 2K numbers per group, with readout p ∝ (Σc)·A − B. It also holds for any G at C=1.
- Correction: name the statistic and its dimension, and add "(as opposed to C=2, where no such fixed low-dimensional statistic suffices)".
- "Shown by exact enumeration and closed form" is also slightly over-stated. The exhaustive enumeration was of `posterior_shared` at Q=2. The R1/R2 closed forms at Q=6 were checked on 10⁵ Monte-Carlo episodes, and only through `posterior_brute(C=1)`, which is algebraically equal to the closed form but is not the closed form. I verified the closed forms directly (Confirmed 4–5). Suggest: "closed form, checked numerically against the exact posterior".

**O3. "The joint statistic is therefore needed exactly on S = 3"** (lines 79–84; C3.2 first bullet).
- On the third of S = 3 episodes in which all three groups agree at q (measured 0.333), all L_g are equal and the product rule is exact. The product rule's KL is > 0 on 5.6% of episodes, not 8.4%.
- Correction: "needed only within S = 3, and there only when exactly one pair agrees at q (≈2/3 of the slice, 5.6% of episodes)".
- Also line 77, "J_H selects the identical pair": the selection is not unique in 0.17% of S = 3 episodes, where the reliable group is identical to the colluders everywhere. Say "the identical pair(s)".

**O4. Line 86, "the pair that agrees at q must be the identical pair"**: this is false when all three groups agree at q. The conclusion still holds, because the posterior is then L(y) whatever the roles. Correction: "if exactly one pair agrees at q it is the colluder pair; if all three agree, every L_g is equal and the roles do not matter."

**O5. Scope of "C ≥ 2"** (C3.2, line 119). Only G=3, C=2 was analysed (the only C ≥ 2 option when G=3). Correction: "for C = 2 (G = 3)".

**O6. The rank argument is offered as a proof** (lines 102–104). That χ is not a product shows only that the role factor does not factorise. It does not show that the posterior differs on the support: at β=1, χ has rank 3, yet the product rule is exact on 94% of episodes. The proof of insufficiency is the counterexample (and the S = 3 numerics). Correction: present the rank as the structural reason, and the counterexample as the proof. Also note that the script computes the rank only at β=1 (sufficiency_boundary.py:33), while the text claims it for β > 0. Either loop over β, or cite the analytic eigenvalues β/K (multiplicity K−1) and β/K + (1−β)/K³.

**O7. "At β < 1, on most of the input space"** (C3.2, line 121). This is supported for the posterior: the product rule's KL > 0.01 on 85% of episodes. It is not supported for decisions, which differ from Bayes on 37%. Correction: state both numbers.

## Minor

- **M1. Line 72, "e_H = 1 for the true H"**: e_H is undefined. Write "e_{h₁} = e_{h₂} = 1 and J_H = 1 for the true H".
- **M2. Line 83**: label the row "S = 2" (S = 1 is impossible at C=2, β=1), or note that the S ≤ 2 bucket contains only S = 2.
- **M3. Lines 111–112**: say explicitly that c_g (the β < 1 statistic) is also identical between x₁ and x₂. The doc lists only (e_g, W_g, L_g), but the relevant per-group statistic at β = 0.6 is c_g.
- **M4. Line 122**: "because colluders' high W_g dominates" is incomplete. At C=2 every colluder has an eligible partner, so the C=1 rule always gives positive weight to "colluder as reliable", and W_colluder ≈ 1.98⁵ ≫ W_reliable does the rest.
- **M5. Line 20, Verification 1**: add a sentence that cross-question factorisation given the role is assumed by both implementations. It is justified by the sampler's code (independent per-question draws), and a direct sampler-vs-likelihood test would make the check self-contained; my check 1 is one such test. Optionally also test q ≠ 0 exhaustively (I did q = 1: it passes).
- **M6. Line 11**: say "reliability α, wrong answers uniform over the K−1 others" for completeness. The W_g/L_g values (agree ratio 1.98, L = 5.76 or 0.09) depend on it.
- **M7. Hygiene**: both scripts write into `logs/` and, if run without `-B`, into `src/__pycache__`. A reviewer reproducing them modifies the package. Consider accepting an output-path argument.
