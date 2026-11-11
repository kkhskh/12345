# R1 information requirements

**Status labels.** [P] proved; [N] exact numerical computation (generator-exact posteriors on sampled episodes, scripts named);
[E] empirical on models.

**Sources.** `scripts/r1_information.py` → `logs/r1_information.json` (stream 17_000_000, 40,000 episodes);
`scripts/formal_checks.py` → `logs/formal_checks.json`; `scripts/build_eval.py` → `logs/eval_dev.npz`.

## 1. Closed form of the exact T4 decision [P, N]

**Generator.** The T4 generator has pc = 1 (exactly one colluding group) and ρ = 1 (a colluding pair always agrees, at every
question). For a decision (X, q), define:

- n_g: the leave-q agreement count of group g (0–5);
- a_g: whether g's two sources agree at q;
- L_g(y) = Π_{s ∈ g} (α if x_{s,q} = y else ε), with α = 0.8 and ε = 0.1: the likelihood of g's q-reports if g is reliable.

**Proposition 1 (closed form).**

  p*(y | X) ∝ Σ_g W_g L_g(y),  W_g = RHO^{n_g} · m_g,  m_g = #{h ≠ g : n_h = 5 and a_h = 1},  RHO = 66/17.

*Proof.*
1. From Investigation V, π(g, h | X_{−q}) ∝ RHO^{n_g} [n_h = 5].
2. Given configuration (g reliable, h colluding, the third group noise), the own-question likelihood is
   J(x_q, y | g, h) = (1/K) L_g(y) · (ρ/K)[a_h] · K^{−2}.
   - A colluding pair contributes 1/K if it agrees and 0 otherwise.
   - A noise pair contributes K^{−2}, independent of y.
3. All factors except L_g(y) and [a_h] are common to every configuration.
4. Summing over h ≠ g gives W_g. ∎

[N] The closed form matches the exact posterior with maximum |difference| 1.3e-15 over 40,000 decisions.

**Reading.** The exact decision is a mixture over *which group is reliable*. Each group contributes its own q-evidence L_g. Its
weight has two factors:
- an own-count reliability factor RHO^{n_g};
- a cross-group eligibility factor m_g, which counts the groups that could be the colluder if g is reliable.

The weights are **vote-blind**: the other groups' votes enter only through their own L_h. Nothing in the rule is a
"vote × count" interaction.

## 2. Explaining away at the 4→5 boundary [P, N]

**Notation.** The edited group is g (4→5). The existing perfect group is h* (n = 5, agrees at q). The third group is f.
Only |H_before| = 1 pairs are considered.

**Weights (unnormalised) before and after the edit.**

| | W_g | W_{h*} | W_f |
|---|---|---|---|
| before | RHO⁴ · 1 | RHO⁵ · **0** (h* is the only possible colluder, so it cannot be reliable) | RHO^{n_f} · 1 |
| after | RHO⁵ · 1 | RHO⁵ · **1** (g could now be the colluder) | RHO^{n_f} · **2** |

The edit does three things at once:
1. it raises g's reliability factor (×RHO);
2. it makes h* *eligible to be reliable* (0 → 1);
3. it doubles the eligibility of the third group (1 → 2), because either perfect group could now be the colluder.

**Posterior over "which group is reliable"** (mean over pairs; `explaining_away_weights_H1`):

| regime | before: g / h* / f | after: g / h* / f | posterior mass on hypotheses whose reliable group votes c_g (before → after) |
|---|---|---|---|
| R1 (c_g = h*'s vote) | 0.916 / 0.003 / 0.081 | 0.486 / 0.487 / 0.027 | **0.926 → 0.976** |
| R2 (c_g = f's vote only) | 0.914 / 0 / 0.086 | 0.486 / 0.486 / 0.027 | 1.000 → 0.514 |
| R3 (c_g agrees with no group) | 0.917 / 0.002 / 0.081 | 0.486 / 0.487 / 0.027 | 0.917 → 0.486 |

**Why R1 is a credit.** After the edit, g and h* are exchangeable. Exactly one of them colludes, and either way the *other* one
is reliable and votes c. "Explaining away" g's perfect agreement as collusion does not remove support for c, because the
alternative explanation (h* colludes, g is reliable) supports the same candidate. Meanwhile the support for c rises:
- the c-supporting weight goes from RHO⁴ (g alone) to 2·RHO⁵;
- the third group's weight only doubles.

Net, c's hypotheses gain a factor of RHO (2·RHO against 2) relative to f.

**Why R2/R3 are a discount.** After the edit, h* becomes an equally weighted reliable candidate that votes *against* c_g. In R3,
c_g goes from the dominant hypothesis to a tie with h*'s candidate. In R2 the third group still votes c_g, but its weight is
small.

**Exact Bayes responses** on F45 pairs (stream 17_000_000):

| regime | mean response | sign |
|---|---|---|
| R1 | +0.233 | 89 % positive, 11 % exactly 0 (third group also unanimous on c) |
| R2 | −2.379 | |
| R3 | −2.050 | |

The nested sequence 3→4→5 (§4) gives the same picture.

## 3. Restricted computational rules [P, N]

Every rule below has the form p(y) ∝ Σ_g W_g L_g(y) with the exact likelihoods L_g, so the rules differ **only** in how evidence
is weighted. Responses are means over F45 pairs (`rules`).

| rule | weights W_g | R1 | R2 | R3 | F34-R1 |
|---|---|---|---|---|---|
| exact Bayes | RHO^{n} m_g | **+0.233** | −2.379 | −2.050 | +0.419 |
| count-only (no collusion; T1's own Bayes rule) | RHO^{n} | +0.044 | **+1.014** | **+1.093** | +0.024 |
| categorical perfect-agreement exclusion | RHO^{n} [n < 5] | **−3.734** | 0.000 | −4.228 | +0.403 |
| soft discount, δ = 0.1 | RHO^{n} δ^{[n=5]} | **−0.135** | −0.543 | −0.700 | +0.199 |
| soft discount, δ = 0.5 | | +0.037 | +0.468 | +0.523 | +0.052 |
| perfect-agreement bonus, δ = 2 | | +0.038 | +1.586 | +1.670 | +0.010 |

**Theorem 1 (separable weights cannot produce the Bayes sign pattern) [P].** Let W_g = w(n_g) for any positive w, so a group's
weight depends only on its own count. Then on every |H_before| = 1 F45 pair, in R1, R2 and R3 alike,
sign(Δm) ∈ {sign(w(5) − w(4)), 0}.

*Proof.*
1. Only g's count changes, so only W_g changes: from w(4) to w(5).
2. Write t = log W_g. The oriented margin m = log p(c_g) − ½ Σ_{y ≠ c_g} log p(y) has derivative
   dm/dt = s_g(c_g) − ½ Σ_{y ≠ c_g} s_g(y), where s_g(y) = W_g L_g(y) / Σ_h W_h L_h(y) is g's share of the mixture at y.
3. g's q-pair is unanimous on c_g, so L_g(c_g)/L_g(y) = α²/ε² = 64, the largest likelihood ratio any pair can have. Every other
   component, and so the rest of the mixture, has ratio ≤ 64.
4. Hence s_g(c_g) ≥ s_g(y) for every y, so dm/dt ≥ 0.
5. Equality holds iff the rest of the mixture also has ratio 64 at every y, i.e. all remaining weight is on groups unanimous
   on c_g.
6. Integrate t from log w(4) to log w(5). ∎

Multiplying by a global factor u(|H|) leaves the rule separable, because u cancels in the normalisation.

[N] Checked on 300 random separable rules × every |H| = 1 pair: no violation.

**Corollary.** Any rule that gives the Bayes signs (R1 > 0 together with R3 < 0) has **non-separable** weights. A group's weight
must depend on whether *other* groups are perfect. This is the eligibility factor m_g, or something equivalent. In Bayes it is
exactly this cross-group factor that turns h* from "certain colluder" into "possibly reliable".

**Theorem 2 (sign of R1 in the interaction family) [P; N].** Let W_g = w(n_g) v(m_g) with v(0) = 0, so a lone perfect group is
always taken to be the colluder. On every |H_before| = 1 F45 pair in R1:
- sign(Δm) = sign(w(5)/w(4) − v(2)/(2 v(1)));
- Δm = 0 when the third group is also unanimous on c.

*Proof.* The c-supporting components (g and h*, both unanimous on c) have weight w(4)v(1) before and 2w(5)v(1) after. The third
group has weight w(n_f)v(1) before and w(n_f)v(2) after. This is a two-component mixture, so by the dominance argument of
Theorem 1 the margin rises iff the ratio of the c-weight to f's weight rises:
2w(5)v(1)/(w(n_f)v(2)) > w(4)v(1)/(w(n_f)v(1)). ∎

[N] Checked on 300 random (w, v): no violation. The analogous claim for R3 is **false** in general, with violations found, so
no R3 statement is made.

**Reading of Theorem 2.** With collusion exclusion in place, the R1 sign is decided by a single balance. On one side is the
reliability gain of perfect agreement, w(5)/w(4) (Bayes: RHO = 3.88). On the other is the inflation of the third group's
credibility when two groups are perfect, v(2)/(2v(1)) (Bayes: 1). R1 turns negative when the model inflates the third group,
relative to the two agreeing perfect groups, by more than the reliability gain.

## 4. Supported edits used here [N]

| family | edit | Bayes response | notes |
|---|---|---|---|
| **F45 / F34 / F23** | Investigation VI's one-report count edits | as above | |
| **F0** | same token as F45, changed to the third value | exactly 0 | edit distance 1 |
| **FLOC** (agreement location) | a count-4 group moves its single disagreement to another question (repair k₁, break k₂) | exactly 0 (π depends only on counts) | edit distance 2 |
| **FID** (report identity) | in a question k ≠ q where a non-perfect group agrees on v, both reports become v′ ≠ v | exactly 0 | edit distance 2 |
| **Nested S** (3→4→5) | a count-3 group whose q-pair agrees has its two disagreeing questions repaired one at a time | S34-R1 +0.455, S45-R1 +0.281 | same group, same c_g, same other evidence at every stage |
| **F54o** | in a natural state with two perfect groups agreeing at q, one of them loses perfect agreement | −0.239 | built on its own natural pool (stream + 900) |

**Nested S checks.** 10,473 sequences on stream 17_000_000. All count and X_q checks pass, and each step changes exactly one
report.

| | S34 | S45 |
|---|---|---|
| R1 | +0.441 | +0.268 |
| R2 | 0 | −2.351 |
| R3 | +0.527 | −1.934 |

Stage-5 states are same-world (colluder agrees on all questions).

**F54o is the time-reversed R1 edit.** For every rule in the families above, sign(F54o) = −sign(R1). It is therefore
**not** an independent prediction. Its value is that it starts from *naturally sampled* two-perfect states rather than
constructed ones.

## 5. Minimum information for the correct R1 update

Under the closed form, the correct R1 update needs:

- **Own count** n_g, through RHO^{n_g}. Needed.
- **Other groups' counts**, but only through *whether they are perfect and agree at q* (m_g). Needed. Without it no rule can
  get both R1 and R3 right (Theorem 1).
- **The vote of each group at q.** Needed, but only as each group's own evidence L_g. The relation "c_g equals h*'s vote" is
  not an input to the weights. It enters through the mixture.
- **Other groups' votes.** Needed only through their own L_h, as above.
- **The full role posterior.** Not needed. The per-group statistic (n_g, m_g, q-reports of g) is sufficient for the predictive
  (Proposition 1).

The smallest sufficient summary is therefore the three triples (RHO^{n_g} m_g, q-reports of g), g = 1, 2, 3. A smaller
sufficient summary of the R1 *sign* is the balance in Theorem 2.

## 6. Prevalence (natural decisions) [N]

| state | share of decisions |
|---|---|
| exactly one perfect group | 91.6 % |
| two perfect groups agreeing at q (the R1 "after" state) | 2.8 % |
| two perfect groups disagreeing at q (the R2/R3 "after" states) | 5.6 % |
| three perfect groups | < 0.01 % |
