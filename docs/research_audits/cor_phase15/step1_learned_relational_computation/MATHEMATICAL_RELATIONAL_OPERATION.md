# The relational operation: definition, equivalence and requirements

**Status labels.** [P] proved; [N] exact numerical computation on generator-exact posteriors (scripts named); [elementary]
marks a standard consequence. Source: `scripts/relational_math.py` → `logs/relational_math.json`. Development evaluation pairs are
Investigation VII's `eval_dev.npz`; stream 18_000_000 holds 40,000 natural episodes × 6 decisions.

## 1. Which eligibility factor: a correction to the mandate's formula

The mandate writes the eligibility factor as

  m′_g = Σ_{h≠g} 1[n_h = 5] · 1[r_h(q) = r_g(q)]

that is, other perfect groups *whose target-question report equals g's*. Investigation VII's factor, the one in the exact closed
form, is

  m_g = Σ_{h≠g} 1[n_h = 5] · a_h(q),

where a_h(q) = 1 when h's two sources agree on q.

**Why the definition is a_h(q).** The colluder must agree on every question, so a perfect group qualifies as a possible colluder
when it agrees at q. *What* it votes there does not matter. Agreement with g's vote is not required.

**The two factors differ [N].**
- They agree on only 42.3 % of (decision, group) cells. In the other 57.7 %, m > m′.
- Substituting m′ into the closed form does **not** reproduce Bayes. It gives an R1 response of 0, an R2 response of +1.94 (wrong
  sign) and an R3 response of 0. The maximum log-probability error on natural decisions is 4.07.
- With m, the closed form is exact (error 1.3e-15) and gives R1 +0.233, R2 −2.384, R3 −2.067.

The mandate's m′ is the "vote-matching" rule family that Investigation VII's review identified as a counterexample, not the
Bayes factor. **This investigation uses m (Investigation VII's definition) throughout.** Its reference condition O uses m.

## 2. Notions of equivalence

Let f_θ map an episode to candidate logits. Each notion below is strictly weaker than the one before, and each weaker one is used
only where the stronger cannot be established.

1. **Exact operational equality.** An internal variable u(X) equals m_g(X) on every supported episode.
2. **Equality up to an invertible representation.** u = T(m) for some injective T. Example: a code distinguishing m ∈ {0, 1, 2}.
3. **Downstream-response equivalence on a family 𝓕 of supported edits.** Δm_θ(x, x′) = Δm_B(x, x′) for all (x, x′) ∈ 𝓕,
   measured both in candidate log-ratio margins and in probabilities.
4. **Support-restricted equivalence.** Notion 3 holds on the training support only (as opposed to held-out configurations,
   COMPOSITIONAL_GENERALIZATION.md).
5. **Approximate equivalence.** Mean |E| ≤ ε on 𝓕, with E = Δm_θ − Δm_B. Mean |E| is reported always, signed mean E
   alongside it, and distance-to-Bayes separately from sign correctness.

Matching candidate rankings, or matching on the training distribution, never establishes notions 1–2 (see the examples in
FORMAL_RESULTS_AND_COUNTEREXAMPLES.md).

## 3. What the R1/R2/R3 pattern requires [P, N]

**Sufficiency (functional dependence) [N].** Decisions are grouped by a candidate statistic, and the largest within-group spread of
the exact posterior is measured. A spread of 0 means the statistic is sufficient on the sample.

| statistic per group (+ target-question reports of all six sources) | classes | max posterior spread |
|---|---|---|
| counts n_g + q-reports | 22,001 | **0** (sufficient) |
| (n_g where m_g > 0, else ×) + m_g + q-reports | 22,001 | **0** (sufficient) |
| m_g + q-reports, no counts | 985 | 0.946 (not sufficient) |
| perfect set + q-reports | 985 | 0.946 (not sufficient) |
| counts only | 91 | 0.955 |
| q-reports only | 513 | 0.953 |

**Reading.**
- The counts of *non-perfect* groups are necessary for the exact posterior. They set the reliability weights RHO^{n_g}.
- The cross-group factor m is necessary for the *sign pattern*. Theorem 1 of Investigation VII and §4 below show this.
- Neither the perfect set alone nor the counts alone is sufficient.

**The sufficient statistics.** Every sufficient statistic must determine the products RHO^{n_g} m_g up to a common factor, together
with the q-reports. Within the mixture class this is exact. The equivalence class of sufficient representations is therefore
{T(W_1, W_2, W_3, q-reports) : T injective up to a common scale of W}. No claim is made that m_g itself is *minimal*: the exact
posterior is the minimal sufficient statistic.

**What is needed, item by item.**

| information | needed? | role |
|---|---|---|
| every group's count | yes | reliability weight |
| the set of perfect groups | yes | only through m_g |
| target-question votes | yes | each group's own likelihood L_g |
| equality relations between votes | no | they enter only through the mixture of the L_g |
| source-group identities | no (the generator is symmetric over groups) | needed only to bind sources to groups |
| the candidate being evaluated | yes | only through L_g(y) |
| an explicit latent-role posterior | no | not required |

## 4. Aggregation structure: which pairwise operations can and cannot realise the pattern

**Proposition A (log-additive aggregation is separable) [P, elementary].** Let W_g = f(n_g) · exp(Σ_{h≠g} ψ(n_h, a_h)), for any
positive f and any real ψ. Then W_g = [f(n_g) e^{−ψ(n_g, a_g)}] · e^{Σ_h ψ(n_h, a_h)}. The second factor is common to every group,
so W is separable, and by Theorem 1 R1, R2 and R3 share a sign.

[N] 300 random (f, ψ) on the development pairs: no rule gave R1 and R3 opposite signs.

**Proposition B (additive, gating aggregation realises it) [P, constructive].** W_g = f(n_g) · Σ_{h≠g} e(n_h, a_h) with
e = 1[n = 5] a and f = RHO^n is exact Bayes. That gives R1 > 0 and R2, R3 < 0 (Proposition 1 of Investigation VII).

**Reading.** The required cross-group operation is *a sum of other groups' eligibility, multiplied into g's weight, with zero
acting as a hard gate*. Two common neural primitives cannot express it on their own:
- an additive contribution to a log-weight (Proposition A);
- a pointwise function of g's own features (Theorem 1).

It needs a multiplicative interaction between g's term and an aggregate over the other groups, or an equivalent nonlinear
re-encoding. This is a restricted statement about mixture-form rules. It is **not** a statement about what a Transformer can
represent.

## 5. Candidate responses and ties [N]

Responses are measured as differences in the oriented log-ratio margin, E = Δm_M − Δm_B, and separately in probability.

**Tie rates of the exact posterior on the development F45 pairs** (exact tie: top-two gap < 1e-9; near tie: < 0.05 nats):

| regime | before: exact / near | after: exact / near |
|---|---|---|
| R1 (n = 469) | 0.9 % / 0.9 % | 0 % / 0 % |
| R2 (n = 98) | 0 % / 0 % | 0 % / **79.6 %** |
| R3 (n = 793) | 0.8 % / 0.8 % | **39.5 % / 95.2 %** |

The after-edit R2/R3 states are mostly (near) ties, because two perfect groups support different candidates equally. **Argmax
correctness is therefore not used as an outcome for R2/R3.** Margins (log-ratio) and proper scores (KL, NLL) are used instead.
R1 states are almost never tied, so the R1 *sign* is well defined.

## 6. Frozen-information conventions

These were fixed before any Investigation VIII model outcome was seen:

- **Primary distance.** Mean |E| per regime, plus signed mean E. Raw-sign correctness of Δm_M is reported separately.
- **Ties.** No tie-breaking enters any metric. R2/R3 are evaluated only by margin and KL.
- **Prevalence.** Natural prevalence, used for weighting, is taken from Investigation VII:

  | state | share of decisions |
  |---|---|
  | one perfect group | 91.6 % |
  | two perfect groups agreeing at q | 2.8 % |
  | two perfect groups disagreeing at q | 5.6 % |
