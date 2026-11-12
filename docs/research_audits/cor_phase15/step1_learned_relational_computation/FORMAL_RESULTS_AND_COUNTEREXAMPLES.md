# Formal results and counterexamples

Tags: **[P]** proved (elementary unless stated); **[N]** numerical, on the stated sample; **[C]** counterexample. Nothing
here is presented as a new theorem about COR. Each result is an elementary fact about mixtures and aggregation rules, or a
standard Bayesian computation, stated because the experiments need it.

## Results

1. **[P] Closed form.** p*(y) ∝ Σ_g RHO^{n_g} m_g L_g(y), with m_g = #{h ≠ g : n_h = 5 and h's q-pair agrees}. This is
   Bayes' rule for the T4 generator, carried over from Investigation VII. **It is not a new theory.**
2. **[C] The mandate's m′ is not the relation.** m′ is vote-matching with g. It agrees with m in 42.3 % of cells. Bayes with
   m′ in place of m gives an R1 response of 0 and R2 +1.94 (`logs/relational_math.json`). Investigation VIII uses m.
3. **[N] Sufficiency table.** Counts + q-reports are sufficient (spread 0). m + q-reports is not (0.946). The perfect set +
   q-reports is not (0.946).
4. **[P] Proposition A.** Log-additive cross-group aggregation is separable, so R1 and R3 share a sign. [N] 0/300 random
   rules violate this.
5. **[P] Proposition B.** A gating sum multiplied into RHO^{n_g} realises the pattern (it is the Bayes rule).
6. **[P] Proposition C.** Logit rules that are additive over groups give regime-independent responses to own-group
   edits. They therefore cannot produce the R1/R2/R3 sign pattern on R features. [N] Confirmed by the self-only
   ablation (R1/R2/R3 dm +0.17 / +0.16 / +0.17).
7. **[P] Representability.** Group attention with ≥ 1 block and a nonlinear readout can represent the closed form on R
   features: it can attend with weights that select eligible partners, then form the product in a feed-forward layer. This
   is a constructive existence statement only; nothing is proved about training dynamics.

## Counterexamples to tempting general claims

- **[C] "Linearly decodable ⇒ accessible."** The failed mlp R 16k readers decode m at 0.99 and have E_R1 −0.46. The frozen
  P-count states plus one block are a second counterexample (Investigation VII).
- **[C] "m available ⇒ any reader succeeds."** The linear O reader has KL 0.335.
- **[C] "Supervising the intermediate ⇒ the model uses it."** auxm gives E_R1 −1.10, worse than orig at the same step.
- **[C] "Hand-computed m injection is the right repair."** It overshoots (+0.47 on xP80). The learned adapter does not.
- **[C] "Aggregate quality reflects relational correctness."** At 1k steps, R learners reach KL 0.017 while E_R1 = −1.65.
  The R1 sign is wrong in 2.8 % of decisions, and KL barely registers it.

## Theorem targets left open

- A sample-complexity or gradient-flow result explaining the separable phase. The phenomenon is documented [N], with no
  theorem.
- A necessary and sufficient condition on (Z, 𝓡) for accessibility at budget T. Only the sufficient representability
  statement (7) is given.
- A characterisation of when an interchange share differs from 1 (overshoot or cancellation inside a learned reader).
