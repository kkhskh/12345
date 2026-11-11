# Formal theorems and limits

**Status labels.** [P] proved here; [N] exact numerical verification (script named); [E] empirical; [elementary] a standard
identity, labelled as such.

**Common notation** (R1_INFORMATION_REQUIREMENTS.md §1). Decision (X, q). n_g is the leave-q count of group g, a_g whether g's
q-pair agrees, L_g(y) g's reliable-pair likelihood, m_g = #{h ≠ g : n_h = 5, a_h = 1}. The **mixture class** 𝕄 is the set of rules
p(y) ∝ Σ_g W_g(X) L_g(y), with W ≥ 0 and the exact own-question likelihoods. The oriented margin toward candidate c is
m_c = log p(c) − ½ Σ_{y ≠ c} log p(y).

## Proposition 1 (closed form of exact T4 Bayes) [P; N]

p*(y | X) ∝ Σ_g RHO^{n_g} m_g L_g(y). The proof is in R1_INFORMATION_REQUIREMENTS.md §1. [N]: maximum difference from the exact
posterior is 1.3e-15 (`logs/r1_information.json`).

**Statistical sufficiency.** The triple (n_g, m_g, q-reports of g), g = 1, 2, 3, is sufficient for the exact candidate decision.
m_g is a deterministic function of (n, a), so the counts plus q-agreement are sufficient too. Sufficiency says nothing about
whether a learner can *compute* m_g; §5 separates the two.

## Lemma 1 (likelihood-ratio dominance) [P, elementary]

For p_t(y) ∝ e^t A(y) + R(y), with A, R > 0:

d m_c / dt = s(c) − ½ Σ_{y ≠ c} s(y), where s(y) = e^t A(y) / (e^t A(y) + R(y)).

So if A(c)/A(y) ≥ R(c)/R(y) for all y, then m_c is non-decreasing in t, with equality iff the ratios are equal for all y.

In T4, a pair unanimous on c has A(c)/A(y) = α²/ε² = 64. That is the maximum over all pair report patterns, so the dominance
condition holds against **any** non-negative combination R of pair likelihoods.

## Theorem 1 (separable weights cannot represent the R1/R3 Bayes signs) [P; N]

Let W_g = w(n_g) for some w : {0, …, 5} → (0, ∞), optionally multiplied by any common factor u(X), which cancels. On every
supported F45 pair with exactly one perfect group before the edit, in each of the regimes R1, R2 and R3:

sign(Δm_{c_g}) ∈ {sign(w(5) − w(4)), 0}.

*Proof.* Only W_g changes (w(4) → w(5)). g's q-pair is unanimous on c_g. Apply Lemma 1 with t = log W_g. ∎

[N] 300 random w × all |H| = 1 pairs (`logs/formal_checks.json`): no violation.

**Corollary 1.** Exact Bayes has R1 > 0 and R2, R3 < 0. No separable rule has both R1 > 0 and R3 < 0. Every rule in 𝕄 that
reproduces the Bayes R1/R3 signs therefore has weights that depend on *other* groups' counts.

## Theorem 2 (the R1 sign in the eligibility family) [P; N]

Let W_g = w(n_g) v(m_g) with v(0) = 0, so a lone perfect group is never reliable. On every |H_before| = 1 R1 pair:

sign(Δm_{c}) = sign( w(5)/w(4) − v(2)/(2 v(1)) ),

or 0 if the third group is unanimous on c.

*Proof.* The two components unanimous on c weigh w(4)v(1) before and 2w(5)v(1) after. The third group weighs w(n_f)v(1) before
and w(n_f)v(2) after. Apply Lemma 1 to the log-ratio of the c-weight to the third-group weight. ∎

[N] 300 random (w, v): no violation. The analogous R3 statement is false and is not claimed.

**Reading.** Given correct exclusion of a lone perfect group, the R1 sign is set by a single balance:

- the reliability gain w(5)/w(4) (Bayes: RHO = 3.88);
- against the inflation v(2)/(2v(1)) of the third group's eligibility when two groups are perfect (Bayes: 1).

The theorem predicts the R1/R2/R3 sign pattern under an identified computational condition:

- R1 fails when v(2)/(2v(1)) > w(5)/w(4);
- R2/R3 stay negative whenever v(0) ≈ 0.

**Empirical fit (exploratory, 4 development models; `logs/separability_fit_dev.json`).** The fitted values of log v(2) − log 2
are:

| model | log v(2) − log 2 |
|---|---|
| T1 parent 70 | 0.14 |
| T1 parent 73 | 0.73 |
| P-count parent 80 | 1.44 |
| P-count parent 81 | 1.29 |

The P-count fits are on the wrong side of the Theorem-2 boundary or close to it. These are fitted parameters of a family that
fits P-count models only moderately (R² 0.83–0.92). They are **not** used as evidence of the models' algorithm; the observables are.

## Proposition 2 (sufficient information, different signs) [P, elementary]

Two operators can receive identical sufficient inputs (n, a, q-reports) and still produce opposite R1 signs:

- exact Bayes;
- the separable rule with w = RHO^n · 0.1^{[n = 5]} (R1 −0.135, R3 −0.700; `logs/r1_information.json`).

This is the elementary counterexample the mandate asks for. Theorems 1 and 2 strengthen it to a characterisation within 𝕄:

- *separability* fixes R1 and R3 to the same sign;
- *within the eligibility family*, the sign is fixed by one inequality.

## Limits: what these results do and do not license

1. **Scope of the class.** 𝕄 contains the exact rule and the restricted rules considered, but trained Transformers are not
   members of 𝕄. Theorem 1 constrains *mixture-form* computations. It applies to a network only to the extent that the network's
   response behaves like such a mixture. No theorem here says a Transformer must be separable.
2. **Sufficiency versus accessibility.** Proposition 1 makes (n, a, q-reports) sufficient. A frozen representation can contain
   them, in principle (high probe accuracy, perfect information), while a given downstream learner cannot compute m_g from it
   within a budget. FROZEN_REPRESENTATION_SUFFICIENCY.md separates these distinctions empirically:
   - statistical sufficiency;
   - computational accessibility (linear or shallow-nonlinear decodability at the sites layer 2 reads);
   - architectural expressivity (a one-block downstream versus a two-block downstream or an MLP);
   - optimisation and budget (oracle-feature readers at 4000 versus 16,000 steps).
3. **No information-theoretic impossibility is claimed.** No exact representation collision was found or sought exhaustively.
   High-capacity readers improve P-count R1 substantially, so a lower-bound argument from collisions would be false or vacuous
   here.
4. **Causal use.** The injection repair (SELECTIVE_RESCUE.md) changes behaviour by *supplying* m_g. That shows m_g is
   behaviourally sufficient to remove the error under the original learner. It does not show the original P-count models
   compute an inaccurate m_g internally (T5 of Investigation VI). The internal reconstruction (FULL_LAYER2_CAUSAL_RECONSTRUCTION.md)
   is the evidence on what they do compute.
