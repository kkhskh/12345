# R1 — Formal mathematics review (pre-review commit 9389496)

Scope: PROPOSITION_3_CORRECTION.md (P3C), EXACT_BAYES_BOUNDARY_THEORY.md (EBT), THEOREMS_AND_COUNTEREXAMPLES.md (TC), and
the code and logs behind them. Reviewer checks ran from a scratch directory; no package file was edited.

**Verified correct:** the six P3′ log-weight changes; the counterexample (0.0 null, 0.63 F45); the Prop. 1 tilt algebra;
Lemmas B1, B2; B3's E = (γ−1)Δm_B and −0.18; all EBT/TC numbers against the log except item 4.

## Findings

1. **Major — P3C §2 A3 and the hard-gate step of the proof.**
   - *Problem:* A3 ("each pair (k, k+1) is moved by some edit") does not give affinity. The hard-gate branch only gives
     d_a = d_b for counts a and b that move together. Affinity needs the graph on {0..3}, with an edge a–b for each such
     pair, to be connected. For example, edits with a = b only would cover every k but constrain nothing.
   - *Problem:* "[N] All four pairs occur" is not computed anywhere in the script or the log.
   - *Correction:* Restate A3 as connectivity of that graph. It holds trivially, because all 96 abstract pairs are
     enumerated and natural(v) does not depend on (a, b). Log the check.

2. **Major — P3C §2 A2 and "Excluded degenerate cases" (mandate §3 C/D).**
   - *Problem:* The claim that mixed gates "can be invariant under weaker conditions" is false.
     - Argument: if φ_col is finite on a proper non-empty subset S, some edit takes a count from k ∈ S to k ± 1 ∉ S. That
       sends the weight of (h*, g₁) from finite to 0 while (g₁, h*) stays finite. This is not a common factor, so by A1
       the response is nonzero.
     - Reviewer run, five choices of S × {constant, Bayes} φ_rel: maximum null response 2.3–5.0 in every case.
   - *Problem:* Limits of the family are not treated. With affine φ_rel and a constant finite gate c, the maximum null
     response is 0.86 at c = −5, 0.51 at c = −10 and 6e-5 at c = −20. Branch (a) is therefore a non-uniform limit of
     non-invariant members.
   - *Correction:* Drop A2 for φ_col; the theorem then states that mixed gates are never invariant. Add the remark on the
     limit, which is the form that matters for fitted surrogates (mandate F).

3. **Minor — P3C §2 A1.**
   - *Problem:* "For each X_q … determine the weights" is false per X_q: each X_q gives 3 equations for 6 unknowns.
     "Equivalently" is wrong.
   - *Problem:* A kernel of dimension 1 has projective dimension 0, not 1.
   - *Problem:* The SVD checks one dense random w over all patterns. The proof needs the edit-compatible X_q (h* agrees at
     q) and arbitrary w, including the sparse hard-gate w.
   - *Reviewer check:* restricted to the 243 compatible patterns for each h*, with both dense and hard-gate w, null_dim = 1
     and the six J-functions have rank 6. The conclusion stands.
   - *Correction:* State A1 jointly over the compatible set and for every w ≥ 0, w ≠ 0. Replace the script check with the
     restricted one.

4. **Minor — P3C §2, the "518 own-question patterns" figure.**
   - *Problem:* The log has n_unknowns = 519 = 6 + 513. There are **513** positive-probability patterns (reviewer
     confirmed); 518 is the rank.
   - *Correction:* Change to "513 patterns, rank 518".

5. **Minor — P3C §2 definitions.**
   - *Problem:* The minimal null edit changes **two** reports, one per group (eqv.pi_preserving_edit), not "one report".
   - *Problem:* 𝔽_mix omits the output temperature τ that cri.mix_logits includes. It is harmless here.
   - *Correction:* Fix both statements.

6. **Minor — P3C §4, final paragraph.**
   - The corrected Prop. 1 is valid.
   - *Problem:* "No single-episode tilt reaches B below δ_min·KL_{x₀}(3B/2)" is false. A downward tilt by t of a
     non-argmax y keeps the argmax, gives the same (2/3)t response, and has KL ≤ t·p_y < t(1−p_ŷ).
   - *Correction:* Delete the sentence, or restrict it to the ŷ-up tilt.

7. **Major — TC T1(b) proof.**
   - *Problem:* "c_g is the mode of J(·|g, h*) ⇒ the oriented margin rises along the segment" is invalid. Counterexample:
     p0 = (.3, .69, .01), p1 = (.4, .2, .39), with c first; the margin's derivative at t = 0 is ≈ −18.6.
   - *Correction:* Use likelihood-ratio dominance: if r_y = p1(y)/p0(y) is maximised at c_g, then m(t) is non-decreasing.
     In gen_family this holds structurally. Noise is y-independent, and r_y ∝ P(g|rel, y)/P(f|rel, y), which is maximal at
     c_g for every report pattern of f, with equality exactly in R2. That makes T1(b) genuinely [P] and yields Lemma B1 as
     a corollary.

8. **Major — TC T1(c) and EBT §2 "Reading".**
   - *Problem:* "Exactly where the Bayes update changes sign whenever c_g does not agree with the colluder candidate" is a
     universal claim drawn from a sample. It is also too broad on two points:
     - It omits the |H| = 1 restriction. The |H| = 2 R3 row is mixed (59 % negative, 41 % positive).
     - In R2 the sub-boundary response is exactly 0, so the change there is 0 → negative, not a sign change.
   - *Correction:* Restrict to |H| = 1 and say "zero or positive → negative". Then either enumerate the finite domain
     exactly or label the claim [N-sample].

9. **Major — EBT B3 "Consequence" and TC T4(b).**
   - *Problem:* Under pure gain, D-R1(P) − D-R1(T1) = −0.18(γ_P − γ_T). Its sign depends on which arm is more attenuated,
     not on γ < 1. If γ_P > γ_T, pure gain *predicts* the frozen P2 direction (T1 > P). "A confirmed P2 is evidence against
     gain" holds only if γ_P < γ_T.
   - *Correction:* State this condition, and report per-arm gains, e.g. γ fitted on F34/F23.

10. **Minor — TC T4(c) and EBT B3.**
    - *Problem:* "Any finite set of mean excesses" fails on F34/F23 R2, where Δm_B ≡ 0 pointwise and so every gain gives
      E = 0.
    - *Problem:* B3's "negative excess on F34/F23" should read "non-positive; zero in R2".
    - *Correction:* State the exception. It is useful: a nonzero model response on those pairs is a gain-free test.

11. **Minor — TC T2.**
    - *Problem:* The witness (the counterexample versus exact Bayes) does not satisfy "differ only in φ_rel(5)".
    - *Problem:* "(zero)" needs branch (a) or (b). "Identical" holds always.
    - *Correction:* Use φ_rel = φ_col ≡ 0 with φ_rel(5) ∈ {0, 1}, giving F45 responses of 0 and 0.63.
    - *Note:* T2 sits close to the trivial restatement that mandate §10 warns against.

12. **Minor — TC T3(a).**
    - *Problem:* The gauges are the additive constants of φ_rel and φ_col as wholes. They are *fixed by* setting
      φ_rel(0) = φ_col(5) = 0; those two values are not themselves the gauges.
    - *Problem:* "Below −30" is a numerical statement, not [P].
    - *Problem:* Under (b), φ_col(0..4) − φ_col(5) *is* identified, because it enters F45.

13. **Minor — TC T5.**
    - *Problem:* In (b), "half the change" requires a linear readout.
    - *Problem:* (c) is a criterion list, not a theorem; "IIA near 1" is not formal.
    - *Problem:* (a) is elementary.
    - *Correction:* Relabel (c) as [D] (definition) and (a) as elementary.

## Overall verdict

**Accept with required revisions.**

- The core repair is sound: the counterexample holds, Prop. 3′ is correct for this generator, A1 holds in restricted
  form, and the Prop. 1 fix is valid.
- Required: items 1, 2, 7, 8, 9 (item 2 strengthens the theorem).
- No logged number is wrong; the only numerical misstatement is item 4.
