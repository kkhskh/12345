# Mathematical review: XIII-B, step1_relational_decision_computation_followup

Reviewer checks are in /tmp/review13b_math/. Package files were not modified.
- check1_lik.py: per-question likelihood factors, enumerated over all 3^6 joint reports under the genlib sampler semantics.
- check2_post.py: an independent implementation of FULL and LOCALQ; PERGROUP compared with posterior_brute; the proposition's sets A and A'.
- check3_jpairs.py: the J-pair construction, regenerated with adapt2's seeds.
- check4_fit.py, check5_fit4.py: refits of the algorithm families, including alternative POE/MIX parametrisations, on Bayes and on uP160/161/163/165 (conf13 seed-20 models, fit_family's stream).
- check6_ties.py, check7_dd.py: argmax-tie robustness.

## Verdict summary

- **The Proposition (beta = 1, G = 3, C = 2) is true.** Numerically, max |p_FULL − p_LOCALQ| = 1.2e-16 on 40k episodes. The proof has a gap, but it can be repaired: the two displayed sums do not carry the same weights W_g, and the case A' = A with |A| ≥ 2 is not handled. Both are fixable in two sentences.
- **censored_bayes.py is correct for FULL and LOCALQ.** Each factor matches exhaustive enumeration to 2e-16. Two independent implementations agree to 3e-16.
- **PERGROUP equals the independent-copying product rule.** It agrees with genlib.posterior_brute to 4e-15, so the 0.749 / 0.500 accuracies are those of the product rule. But it is *not* the exact posterior of any censored observation under the true generator, and it is *not* "the best rule that never compares groups". A rule that keeps the shared copy event through joint self-agreement bits uses no cross-group value comparison and scores 0.671 at beta = 0.6 (0.747 at beta = 1, the same as PERGROUP there). Most of the 0.500 is a misspecification penalty, not the price of having no cross-group comparison.
- **The J-pair construction is valid at beta = 0.6 and the reported numbers reproduce exactly.** Mean |Δ| is 0.733, 21.5% of decisions change, the product rule's change is exactly 0, and so is LOCALQ's. The claim "at beta = 1 they are off-support" is false for 31.5% of pairs. Those pairs (99% of them had the reliable group relabelled) stay on support, and their exact change is exactly 0, which is consistent with the Proposition.
- **The fit_family conclusion "neither class implements log-linear pooling / same computation" is not warranted as stated.**
  - The POE family is handicapped by unnormalised exponents. A 3-parameter POE whose exponents are normalised across groups reaches R² 0.971 on Bayes and 0.969–0.973 on models, against 0.83 for the script's POE.
  - A nested power-mean family (rho = 1 is the mixture, rho → 0 is log-linear pooling) fits **both early-readout models with rho ≈ 0.59–0.60**, with held-out R² improving on MIX. Late models fit rho ≈ 0.90–0.94 with no gain.
  - The κ = 1.62 "floor" explanation is right: with an indicator exclusion, Bayes gives κ = 1.000. But the inference "lower κ means softer exclusion" does not hold up once the parameters are separated.

## Confirmed

1. **Likelihood factors** (scripts/censored_bayes.py L16–47).
   - At q' ≠ q: the reliable pair agrees with probability pr = α² + (1−α)²/(K−1), and the bit is sufficient within the group because values are uniform given the bit after marginalising Y_q'.
   - Colluder bits: β·1[both agree] + (1−β)·Π_h (1/K)^{a_h}(1−1/K)^{1−a_h}.
   - At q (joint): p1·p1·(β·1[all four equal]/K + (1−β)/K⁴).
   - All match enumeration of the genlib sampler (shared `cop` and `col` per question) to 2.2e-16 (check1).
2. **posterior_shared** agrees with an independent implementation (check2) to 1.8e-15.
3. **The Proposition's statement holds** on every sampled on-support input at beta = 1: FULL = LOCALQ, KL 1e-17, 0 decision changes. A ⊆ A' always holds. A' \ A ≠ ∅ occurs in 2.8% of episodes, and in 100% of those all six reports at q are equal, exactly as the proof's case argument says.
4. **The colluder constant is right.** At beta = 1 the FULL colluder factor is Π_q 1[all four equal at q]/K = K^{-6}·1[…]. The LOCALQ colluder factor is K^{-1}·1[both self-agree at all q' ≠ q, all four equal at q]. Each is constant over admissible roles (the constants differ between FULL and LOCALQ, which is harmless).
5. **Admissibility at q and "excluded pair not self-agreeing at q".** Neither can occur: the A' condition is all four sources equal at q (the script's `allEq`), which implies both colluder groups self-agree at q. At q' the A' condition explicitly requires self-agreement. The case analysis over A' is therefore complete, apart from the gap in Error E1.
6. **PERGROUP ≡ posterior_brute** (check2: 2.2e-16 at beta = 1, 3.7e-15 at beta = 0.6). The reason: under every role, a group's values given its bit are uniform over consistent pairs. The bit-to-value conversion factor is therefore the same for every role and cancels.
7. **J-pairs** (adapt2.py L33–45; check3 reproduces them with the same seeds):
   - self-agreement bits and values at q are unchanged in 1500/1500 pairs, so e_g, W_g, counts and L_g are unchanged;
   - the product-rule change is exactly 0.0, and so is LOCALQ's;
   - the exact mean |Δ cen log p| is 0.733 and decision changes are 21.5%;
   - 87.7% of pairs change some pair's n4, and 24.4% have zero exact change;
   - at beta = 0.6 every input has positive likelihood (β < 1 and α < 1), so "on support" holds trivially.
8. **Extra information at beta < 1.** Given LOCALQ, the FULL likelihood depends additionally only on n4_H restricted to q' ≠ q (for each colluder pair H, the number of other questions where all four sources agree). J_H^β = Π_q (1−β+βK³·1[…]) is K⁴·colH: correct.
9. **fit_family κ = 1.62 on Bayes is a floor artefact.** With f = 0 mapped to weight 0, Bayes gives (α, τ, κ) = (1.000, 1.000, 1.000) and R² 1.0. MIX and POE have 3 parameters each. The 3000/3000 iid held-out split is sound.
10. **Ties.** Exact argmax ties are frequent: 34% at beta = 1 for both FULL and PERGROUP. They do not bias the accuracies: random tie-breaking changes them by ≤ 0.0005. The 6.1% decision-difference figure at beta = 0.6 is tie-robust (6.05% tie-aware).

## Errors (must-fix)

**E1. Proof of the Proposition** (JOINT_RELATIONAL_COMPUTATION_AND_ADAPTATION.md L21–33).
- The same symbol W_g is used in both displays, but the weights differ. In FULL, W_g = Π_{q'} relm_g uses value probabilities pr/K or (1−pr)/(K(K−1)). In LOCALQ, W'_g = Π pr^{a}(1−pr)^{1−a}. Their ratio Π (1/K)^{a_g}(1/(K(K−1)))^{1−a_g} depends on g's own agreement pattern, so "the factor is constant, so p ∝ Σ W_g L_g" is not literally correct for the LOCALQ display.
- The proof only treats g' ∈ A' \ A. It never states why the posteriors agree when A' = A, which needs equal *relative* weights.
- Correction: add that if |A(X)| = 1 both posteriors equal the normalised L_{g*}. If two roles g1, g2 are in A' (hence also if |A| ≥ 2), then the two A' conditions together force all six reports at q to be equal, and every group to self-agree at every q' ≠ q (and, for A, to be identical everywhere). All L_g are then equal, so the weights are irrelevant.
- Also fix L32. "h equals the other colluder g' at q" follows from g* ∈ A(X), not from the definition of A'. Say so.
- Also fix L27 and L36. Write "all four sources of the other two groups are equal at q", not "equal at q".

**E2. PERGROUP is mis-described** (censored_bayes.py docstring L3–7; JOINT L11, L13 column header).
- PERGROUP is not an "exact Bayes posterior of a censored observation under the true shared-copy generator". It is the posterior of the misspecified independent-copying model (identical to posterior_brute).
- It is also not "the best rule that never compares groups". Keep the shared copy event in the joint bit likelihood (β·1[both agree] + (1−β)·product at q', and β·1[both pairs agree]/K² + (1−β)/K⁴ at q). That rule uses only per-group self-agreement and the reliable candidate's values, with no cross-group value comparison, and scores 0.671 against 0.500 at beta = 0.6 (identical at beta = 1).
- Correction: relabel the column "independent-copying product rule (= posterior_brute)". State that at beta = 0.6 its 0.500 mostly reflects ignoring the shared copy event, and report the coupled-bit rule (≈ 0.67) as the "no cross-group value comparison" benchmark, or drop the "best rule" wording.

**E3. "At β = 1 they are off-support, as the proposition requires"** (JOINT L49; adapt2.py docstring L8).
- This is false for 31.5% of beta = 1 J-pairs. When the relabelled group is the reliable one, x' stays on support, and its exact change is exactly 0.
- Correction: "at beta = 1 a J-pair is either off-support or has zero exact change (Proposition)".

**E4. "Neither class implements log-linear pooling"; "both classes implement the same computation"** (ALTERNATIVE_NATIVE_DECISION_ALGORITHMS.md L83–85; the POE row of the L64 table).
- The POE family c_g = softplus(a log W_g + b log f_g + c0) uses unnormalised, per-group-only exponents, which cannot express responsibility splitting. Its R² ≈ 0.83 is identical for Bayes and all models, so it discriminates nothing.
- A 3-parameter normalised log-linear pool, log p = λ Σ_g softmax_g(α log W + κ log f) log L_g, reaches R²:

  | Bayes | uP161 | uP160 |
  |---|---|---|
  | 0.971 | 0.969 | 0.973 |

- A nested generalised (power) mean, p ∝ (Σ_g r_g L_g^ρ)^{τ/ρ} with r = softmax(α log W + κ log f, excluded-cell offset), gives (held-out R²):

  | | Bayes | uP160 | uP163 | uP161 | uP165 |
  |---|---|---|---|---|---|
  | rho | 1.00 | 0.59 | 0.60 | 0.94 | 0.90 |
  | R², power mean | 1.000 | 0.9876 | 0.9847 | 0.9953 | 0.9948 |
  | R², MIX4 | 1.000 | 0.9852 | 0.9826 | 0.9953 | 0.9947 |

- The early-readout class is therefore best described as *between* mixture and log-linear pooling. This is a class signature, plausibly the "class-shared approximation signature" of L90.
- Correction: replace the dichotomy with the nested fit. Say "late class ≈ mixture (rho ≈ 0.9); early class is a tempered mixture (rho ≈ 0.6); unnormalised POE is rejected". Do not claim that log-linear pooling in general is excluded.

## Over-claims

- **O1. "Lower κ means softer eligibility exclusion"** (ALT L70; also "softer exclusion" at L90 and in NOTES_DEV L14). With the floor, κ mixes two things: the excluded-cell weight (f = 0) and the f = 2 : f = 1 ratio. Fitting them separately (MIX4: κ on log f for f ≥ 1, plus a free offset for f = 0) gives:

  | | uP160 | uP163 | uP161 | uP165 |
  |---|---|---|---|---|
  | κ | 1.65 | 1.14 | 1.26 | 0.55 |
  | excluded offset | −9.0 | −9.0 | −10.3 | −10.4 |

  κ has no class pattern. The exclusion offsets are slightly softer in the early class, but both classes exclude almost completely. Correction: report the excluded-cell offset (or the MIXF cells), and drop the reading of κ.
- **O2. "This is the exact Bayes form at α = κ = τ = 1"** (ALT L61). This is exact only if log f_g = −∞ at f_g = 0. With the ε = 1e-3 floor the family does not contain Bayes exactly (best fit R² 0.9999 at κ 1.62). Say so here, or switch to the indicator parametrisation.
- **O3. "J_H is never needed"; "exactly sufficient"** (JOINT L35–36). These are correct only within G = 3, C = 2, beta = 1, on support. Say so in the sentence. The sufficient statistic is also more than "equality of groups at q plus self-agreement": it includes the full joint values at q, in particular the reliable group's values, which L_g needs. Write "LOCALQ (joint values at q + self-agreement bits elsewhere) is sufficient".
- **O4. "cross-question identity"** (JOINT L35, L39, L44, C5). What LOCALQ lacks is cross-*group* equality at the *other* questions q' ≠ q (summarised by n4 over q' ≠ q). It is not a relation across questions. Rename it, or define it once.
- **O5. "the gap … is exactly the information that only the cross-question relation provides"** (JOINT L108). An accuracy gap is not an information quantity. Write "the accuracy gap attributable to the information lost by censoring cross-group equality at q' ≠ q". Also note that "exceeds the LOCALQ ceiling" for models (0.771–0.781 vs 0.758) compares a 3000-episode evaluation against a 200k-sample ceiling. Give a CI: the evaluation's own Bayes is 0.792 vs 0.794, and the episode-clustered SE is ≈ 0.005.
- **O6. "Restoration through an independent pathway"** (CAUSAL L64–68). The mixture rows lie downstream of route Q on the same causal path; "independent" means only "not a trained subspace". Write "through a downstream natural pathway". "E is identified" (L100) should keep its single-site qualifier inline (limit 4 at L121).

## Minor

- censored_bayes.py L8 and L23 mention noise groups, but G = 3, C = 2 has none. This is harmless.
- censored_bayes.py L11 says "LOCALQ and PERGROUP predict exactly zero change" on J-pairs. That is confirmed, but the script never computes it (only adapt2 checks the product rule). Add the check or cite check3.
- fit_family.py: there are two starts per family. Nelder-Mead was used for the 3-parameter families, but MIXF's offset of 63.7 on Bayes indicates an unidentified or saturated cell. Report which cells are identifiable.
- fit_family.py: R² is computed on centred log-probabilities, which weights confident episodes heavily. The KL_test reported alongside it is the better summary. Its ranking agrees here.
- adapt2.py: transposition permutations fix one value, so some q' are left unchanged. The pairs are still valid; just mention it.
- Accuracies are argmax-first. There are frequent exact ties (34% at beta = 1), but they do not change accuracy in expectation (checked).

## Must-fix list

1. E1: repair the proof. The weights W_g differ between FULL and LOCALQ, and the case where two or more roles are admissible needs the "all L_g equal" argument. State where "h = g' at q" comes from, and write "all four sources equal at q".
2. E2: PERGROUP is the misspecified independent-copying posterior (= posterior_brute), not exact censored Bayes and not "the best rule without cross-group comparison". A coupled-bit rule with no value comparison gets 0.671 at beta = 0.6.
3. E3: at beta = 1, 31.5% of J-pairs are on support, with zero exact change. Correct "off-support".
4. E4 (with O1 and O2): the claims that early and late are the "same computation" and that neither implements log-linear pooling are not supported. Normalised POE gives R² ≈ 0.97, the power-mean fit gives rho ≈ 0.6 for early models, and κ's "softer exclusion" reading is a parametrisation artefact.
