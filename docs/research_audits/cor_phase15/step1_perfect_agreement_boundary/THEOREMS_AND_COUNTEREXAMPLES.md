# Theorems and counterexamples (T1–T5)

**Status labels**
- **[P]** proved.
- **[N]** exact numerical verification.
- **[E]** empirical / exploratory.

Notation follows EXACT_BAYES_BOUNDARY_THEORY.md.

## T1 — Boundary-update theorem

**(a) [P] Structure.** A 4→5 edit of g does two things to the posterior over configurations. It multiplies π(g, ·) by RHO
(reliability credit). It also creates the configurations (h*, g) and (f, g), in which g is the colluder, each with weight
∝ RHO^{n_reliable}. No edit below the boundary (k→k+1, k ≤ 3) creates a configuration. Such edits only re-weight
existing ones by RHO.

**(b) [P] Below the boundary.** For k→k+1 with k ≤ 3 and |H| = 1, the edit multiplies only π(g, h*) by RHO > 1. So
p*(·|X) moves along the segment toward J(·|g, h*) normalised. When c_g is J(·|g, h*)'s mode, that movement is weakly
positive on the oriented margin.
- It is exactly zero when the configurations being re-weighted have equal predictives (Lemma B1, regime R2).
- [N] Every F34 and F23 response with |H| = 1 is ≥ 0 (no negatives in 20,000 episodes).

**(c) [N] At the boundary.**
- In regimes R2 and R3, Δ45 m_B < 0 for 100 % of pairs (mean −2.38 and −2.06).
- In R1, Δ45 m_B ≥ 0 (87 % positive, 13 % exactly zero, mean +0.24).

The boundary is therefore exactly where the Bayes update changes **sign** whenever c_g does not agree with the existing
colluder candidate.

A full analytic proof of (c) needs a sign analysis of a four-component mixture across all X_q, and it is not claimed. The
statement is verified exactly on the natural distribution sample.

## T2 — Representation insufficiency [P, with N witness]

**Claim.** Decodable, even perfectly linear, information about "group g has reached 5/5" or about the exact role
posterior does not determine the 4→5 decision update.

**Construction.** Consider a network that computes the exact counts n (a representation shared by every member below), then
outputs f_θ ∈ 𝔽_mix. Take two members θ, θ′ that agree on φ_rel(0..4) and on the gate, and differ only in φ_rel(5).
- They have identical internal representations of n, and of any function of n, including the Bayes role posterior.
- By Proposition 3′ they have identical (zero) null responses.
- Their F45 responses differ. [N] The Inv. VI counterexample member (s = 1, finite constant gate) reaches 0.63, and
  exact Bayes has s = log RHO. The s-profile in FITTED_OPERATOR_IDENTIFIABILITY.md shows predicted B45-R1 varying
  monotonically with s at fixed representation.

**The other direction.** [N] In 9.8 % of F45 pairs the role posterior changes while p* is unchanged. So even the *correct*
transformation does not have to turn a representation change into a decision change.

**Consequence.** Probing studies, including Investigation IV/V's decodability of counts or agreement, cannot establish
use. Separate tests of representation, transformation, selection and decision are required. See CAUSAL_EVIDENCE_UPDATE_MECHANISM.md.

## T3 — Fitted-parameter nonidentifiability

**(a) [P] Gauges.** In 𝔽_mix:
- φ_rel(0) and φ_col(5) are additive gauges. They are fixed to 0 in the parameterisation.
- With a hard gate, every value of φ_col(0..4) below about −30 gives the same outputs to machine precision, so those
  values are unidentified.
- Under branch (b) of Proposition 3′, φ_col(0..4) is identified only up to the constant.

**(b) [P] The F45 response depends on several parameters jointly.**
- Before the edit, g contributes weight exp(φ_rel(4) + φ_col(5)) as reliable and exp(φ_col(4) + φ_rel(n_other)) as
  colluder.
- After the edit, these become exp(φ_rel(5) + φ_col(5)) and exp(φ_col(5) + φ_rel(n_other)).
- The observable Δ45 therefore depends on s = φ_rel(5) − φ_rel(4) **and** on the gate increment φ_col(5) − φ_col(4).

A lower s with a softer gate can produce a similar 4→5 response to a higher s with a harder gate. s is identified only to
the extent that other families (F34, F23, base episodes) pin down φ_col(4) and φ_rel(4).

**(c) [E] Development profiles.** See FITTED_OPERATOR_IDENTIFIABILITY.md: profiled s ranges, alternative operator families
with comparable fit, and s-values rejected by the observables.

## T4 — Calibration limits [P]

**Gain.** If ℓ_M = γ ℓ_B (up to a per-decision constant), then E = (γ − 1) Δm_B on every pair of every family (Lemma B3).
In the R1/R2/R3 pattern this is negative where Bayes credits and positive where Bayes discounts.

**(a)** No single-family excess, pooled or regime-restricted, separates "boundary-specific operation" from "global gain
γ ≠ 1".

**(b)** A contrast across two families with known, unequal Bayes responses, such as D-R1 = B45-R1 − B34-R1, has the
pure-gain prediction (γ − 1)(Δ̄45,R1 − Δ̄34,R1) ≈ −0.18 (γ − 1). For γ < 1 this is positive. A curriculum difference with
the opposite sign is therefore not produced by global gain alone.

**(c) Limit.** Every regime- or family-dependent gain γ(family, regime) can reproduce any finite set of mean excesses.
Calibration *within a family*, i.e. fitting γ on F34 and predicting F45, removes global gain but cannot exclude a
boundary-dependent gain, because that is the hypothesis. Matching on KL or on temperature T* controls quality on average.
It does not control the response pattern on the measure-small F45 pairs (Proposition 1, corrected).

**(d) Probability space.** Δp has the additional confound that, for fixed log-ratio excess, its size scales with
p(1 − p). The probability-space contrast is therefore reported only for sign agreement (P5).

## T5 — Causal identifiability of the update mechanism

**(a) [P] Behaviour does not identify mechanism.** Two networks that compute the same input–output map have identical
responses on every family, so no behavioural statistic distinguishes them. A mechanism claim needs interventions on
internal variables.

**(b) [P] Interchange success does not imply uniqueness, and failure does not imply non-use.** Consider a network with
two exact copies Z₁ = Z₂ of the variable "n_g = 5", whose output depends on Z₁ + Z₂.
- Interchanging Z₁ alone between the before and after episodes of an F45 pair produces half the counterfactual change, so
  IIA on Z₁ < 1 even though Z₁ is used.
- Ablating Z₁ leaves the response intact through Z₂.

Redundancy must therefore be tested by joint interventions.

**(c) [P] Sufficient conditions** for "the curriculum difference in the 4→5 response is mediated by variable Z":
1. *Representation.* Z is computed from the inputs, and its value differs between the episodes of an F45 pair.
2. *Interchange.* Patching Z from the after-episode into the before-episode reproduces Δ45 m_M (IIA near 1 on F45 pairs),
   and patching on F0 pairs produces no change (specificity).
3. *Transformation.* Replacing the downstream map from Z to the output with that of the other curriculum (cross-model
   patching across arms, on aligned subspaces) transfers the curriculum difference.
4. *Redundancy.* Joint ablation of Z and its candidate copies removes the response.
5. *Replication.* 1–4 hold in independent parents of each curriculum.

Without condition 3, a mechanism claim can say *where* the boundary information flows. It cannot say *which operation*
differs between curricula.

**(d) Status in this investigation.** P6 (mechanism) was not preregistered with the behavioural protocol. Whether it is
attempted, and with which subset of 1–5, is decided after the behavioural result and frozen separately.
