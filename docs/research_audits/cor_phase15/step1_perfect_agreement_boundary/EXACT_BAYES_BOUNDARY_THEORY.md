# Exact-Bayes reference for the 4→5 boundary

**Status labels**
- **[P]** proved.
- **[N]** exact computation over 20,000 natural T4 episodes (stream 14_000_000, `scripts/exact_boundary.py`,
  `logs/exact_boundary.json`). These are exact posteriors, not estimates. Only the episode sample is random.

## 1. Quantities

- **Posterior over configurations.** For a decision (X, q), π(g, h) ∝ RHO^{n_g} · [n_h = 5], with RHO = 66/17 and
  n = n^{(−q)}. The predictive is p*(y | X) = Σ_cfg π(cfg) J(x_q, y | cfg) / Σ_cfg π(cfg) J(x_q, · | cfg).
- **The edit.** A *4→5 edit* of group g changes one report so that n_g goes from 4 to 5. The world, X_q and every other
  count stay fixed.
  - [N] For F45, F34 and F23: same world 1.0, X_q fixed, one report changed, and exactly one count changed by exactly one.
- **Orientation.** The edit is oriented toward c_g, g's common vote at q. It is undefined when g's two q-sources disagree.
- **Margin and response.** The oriented margin is m = log p(c_g) − ½ Σ_{c ≠ c_g} log p(c). The model response is
  Δ45 m_M and the Bayes response is Δ45 m_B.
- **Excess.** The mandated excess is **E45(M) = Δ45 m_M − Δ45 m_B**. The probability-space analogue is
  Δp_M(c_g) − Δp*(c_g), reported separately.

Information change and decision change are distinguished as follows:
- *information change* is the change in the role posterior (who is reliable and who colludes);
- *decision change* is the change in the predictive over candidates, and in its argmax.

## 2. Regimes and the structure of the Bayes response

Before a 4→5 edit, g is a non-colluder (n_g = 4), so the colluder-candidate set is H = {h*}, or rarely two groups. The
edit adds g to H. That opens the configurations (h*, g) and (f, g), in which **g is the colluder**, where f is the third
group.

**Regimes**
- **R1**: c_g agrees with the colluder candidate h*.
- **R2**: c_g agrees only with the third group f.
- **R3**: c_g agrees with no group.

**F45 Bayes response by regime [N]** (oriented pairs with |H_before| = 1):

| regime | n | mean Δ45 m_B | sign pattern |
|---|---|---|---|
| R1 | 1439 | **+0.243** | 87 % positive, 13 % exactly 0, none negative |
| R2 | 363 | **−2.383** | 100 % negative |
| R3 | 2617 | **−2.059** | 100 % negative |
| \|H_before\| = 2, R3 | 37 | −0.098 | mixed |

**Overall F45.**
- Mean Δ45 m_B = −1.32: 67 % negative and 29 % positive.
- The argmax reverses in 22 % of oriented pairs.
- In 68 % the probabilities change without an argmax change.
- In 9.8 % the role posterior changes while the predictive does not: information change without decision change.

**Comparison families.** F34 and F23 only re-weight *who is reliable*. They never create a colluder candidate.
- Their Bayes responses are non-negative: F34 has mean +0.443 and 87 % positive; F23 has mean +0.645.
- Their argmax reversals are rare: 4.2 % (F34) and 5.3 % (F23).

**Reading.** At the perfect-agreement boundary the exact Bayes update is mostly a **discount**. Reaching 5/5 makes g a
possible colluder, so its vote at q stops counting as independent evidence. Below the boundary, an additional agreement is
always (weakly) a **credit**. The sign of the Bayes response therefore changes at 4→5 in R2 and R3 and does not change in R1.

## 3. Results used later

**Lemma B1 (exact zeros in R2 below the boundary) [P].** Take F34 and F23 in R2 with |H| = 1. Then Δ m_B = 0 exactly.

*Proof.* With |H| = {h*} unchanged, the edit multiplies π(g, h*) by RHO and leaves π(f, h*) unchanged. In R2 both g's and
f's q-sources vote c_g. The likelihoods J(x_q, · | g, h*) and J(x_q, · | f, h*) are therefore equal: the configurations
differ only by exchanging two groups whose q-reports are identical, and J is exchangeable in group labels. Re-weighting
two configurations with identical predictive distributions leaves the mixture unchanged. ∎

[N] For F34 and F23 in R2, 100 % of the responses are exactly 0 (|Δ| < 1e-15).

**Lemma B2 (F0 is Bayes-null) [P].** F0 changes the same report as F45, but to the third value. The two sources of g still
disagree in that question, so n is unchanged. π depends on X only through n and on x_q only through J, so p* is unchanged.

[N] Max |Δp*| = 0.

**Lemma B3 (temperature and gain confound the excess) [P].** Suppose ℓ_M = γ ℓ_B + const per decision, i.e. a pure gain
(γ = 1/T). Then E45 = (γ − 1) Δ45 m_B on every pair. The same identity holds for every family.

*Consequence.* An attenuated model (γ < 1) shows a **negative** excess in R1 and a **positive** excess in R2/R3 of F45,
and a negative excess on F34/F23. That is the qualitative pattern of the development P-count models.

A single-family excess therefore cannot separate a boundary-specific operation from global attenuation. Under pure gain:

  D-R1 := B45-R1 − B34-R1 = (γ − 1)(Δm_B,45,R1 − Δm_B,34,R1) ≈ (γ − 1)(0.24 − 0.42).

This is **positive** for γ < 1. Pure attenuation in the P-count arm would therefore make D-R1(P) > D-R1(T1), the
**opposite** of the frozen P2 prediction. A confirmed P2 is evidence against "global gain" as the whole explanation.
(The frozen protocol did not state this reason; it is noted here as theory, and the frozen rule is unchanged.)
Gain is not the only miscalibration, so a confirmed P2 does not exclude a regime-dependent gain. See
THEOREMS_AND_COUNTEREXAMPLES.md, T4.

## 4. Why the orientation makes signed averaging meaningful

Oriented toward c_g, a positive E45 means "M credits g's vote at q more than Bayes does after g reaches 5/5" in every
regime. Equivalently, M discounts less than Bayes in R2/R3 and credits more in R1. The pooled B45 is therefore a coherent
"credit-to-g" index.

Because Δ45 m_B has opposite signs across regimes, the pooled B45 mixes the gain confound (Lemma B3) in opposite
directions. The R1 restriction removes the sign mixing. That is why B45-R1 was frozen as co-primary. The weighting is
natural (each natural episode equally), as fixed before training.

## 5. Information without decision

[N] In 9.8 % of F45 pairs the role posterior changes but p* does not. These are cases where the configurations gaining or
losing weight have equal predictives (as in Lemma B1). A model can represent the information change ("g is now a
colluder candidate") and correctly make no decision change, so decodability of the role posterior does not imply use.
This is the basis of T2 in THEOREMS_AND_COUNTEREXAMPLES.md.
