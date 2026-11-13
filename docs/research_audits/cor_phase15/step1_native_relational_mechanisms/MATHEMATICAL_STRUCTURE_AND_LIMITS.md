# Mathematical structure and limits (Workstream E)

Tags: **[P]** proved here (all elementary unless stated otherwise); **[N]** numerically verified (scripts and logs given);
**[K]** known in the literature. Nothing here is claimed as a COR theorem, and Bayes' rule is not renamed.

## 0. Setting: the role model

The **role model** is the generalised T4 generator (`src/genlib.py`).

- There are G groups of two sources, K candidates and Q questions.
- A uniformly random role assignment R picks **one reliable group** and **C colluding groups**. The other groups are noise.
- On each question:
  - a reliable source reports Y_q with probability α, and otherwise a wrong answer chosen uniformly;
  - with probability β, a colluding group's two sources report a common c_q ~ U(K), independent of Y_q; otherwise they
    act as noise ("partial copying" when β < 1);
  - a noise source reports uniformly.
- G = 3, C = 1, β = 1, K = 3, Q = 6, α = 0.8 is exactly the T4 of Investigations VI–VIII.

**[N] V1–V2** (`scripts/verify_math.py` → `logs/verify_math.json`):
- The brute-force role-enumeration posterior of `genlib` matches the latent-reliability package's exact
  `episode_posterior` on bayes.sample_fast T4 episodes (max |Δp| = 3.4e-15).
- The `genlib` sampler reproduces T4 statistics: the distribution of the number of all-agreeing groups is
  0 / 0.917 / 0.083 / 0.00005, against 0 / 0.915 / 0.085 / 0.0001 for bayes.sample_fast.

Notation:
- n_g(q) is the number of questions other than q on which group g's two sources agree.
- e_g = 1[n_g = Q−1] · 1[g's two sources agree at q] is the "eligible" (perfect and agreeing) indicator, and S = Σ_g e_g.
- W_g = Π_{q′≠q} p(x_{g,q′} | reliable) / p(x_{g,q′} | noise) is the reliability weight. In T4 it equals RHO^{n_g} up to
  a constant.
- L_g(y) = p(x_{g,q} | reliable, Y_q = y) / p(x_{g,q} | noise).

## 1. The eligibility factor (mandate 2.4): verified, with no vote matching

**[N] V3.** For G = 3, 4, 5, 6 with C = 1 and β = 1, take the closed form p(y) ∝ Σ_g W_g m_g L_g(y), where
m_g = Σ_{h≠g} e_h. It matches brute force to ≤ 3.7e-15.

The previously used vote-matching variant m′_g counts h only if h's vote equals one of g's reports. It deviates by up to
0.97 in posterior probability, and m = m′ in only 59–63 % of decisions. The m′ correction of Investigation VIII stands.

## 2. Theorem 1 (pooled-sum form) [P, elementary; N]

For the role model with β = 1 and any G and C:

  **p(Y_q = y | X) ∝ binom(S, C) · A_y − binom(S − 1, C − 1) · B_y,**

where A_y = Σ_g W_g L_g(y) and B_y = Σ_g W_g e_g L_g(y).

*Proof.*
- A role assignment with reliable group g has nonzero likelihood only if all C colluders are eligible. A colluder agrees
  on every question by construction.
- Given that, its likelihood ratio against all-noise is W_g · L_g(y) · Π_{h colluder} c_h, where c_h is a constant
  common to all eligible groups.
- The number of admissible colluder sets is binom(S − e_g, C).
- Because e_g ∈ {0, 1}, binom(S − e_g, C) = binom(S, C) − e_g · binom(S − 1, C − 1).
- Summing over g gives the result. ∎

**[N]** Verified against brute force for (G, C) = (3,1), (4,1), (4,2), (5,2), (5,3), (6,2), (6,4), with max |Δp| ≤ 1.9e-13
(`scripts/verify_math2.py`, `logs/verify_math2.json`). The C = 1 case, p ∝ S·A − B, was also checked separately for
G = 3–6 (V4).

**Consequences.**

1. **Fixed-dimension sum decomposition.**
   - The exact posterior is ρ(Σ_g φ(x_g)), with φ(x_g) = (e_g, W_g L_g(·), W_g e_g L_g(·)) ∈ ℝ^{2K+1} and ρ an explicit
     polynomial-then-normalise map.
   - The dimension 2K + 1 = 7 does **not** depend on G or C. This is a constructive, task-specific instance of the known
     sum-decomposition results for permutation-invariant functions [K: Zaheer et al. 2017; Wagstaff et al. 2019]. What
     is new here is only the explicit, low-dimensional, interpretable form for this generator family.
2. **The cross-group operation is one scalar.** Every cross-group dependence of the decision passes through S, the
   *number* of eligible groups, and through which groups carry e_g. No pairwise relation between particular groups is
   needed. m_g = S − e_g is a derived quantity.
3. **Group-identity-free and size-free.**
   - The form contains no group identity. A reader that computes it is exactly permutation-invariant.
   - The same φ and ρ are exact for every G, for any fixed C.
   - This is the basis of the frozen generalization predictions (`PROSPECTIVE_PREDICTIONS.md`).

## 3. Theorem 2 (interaction order of the evidence function = 1 + C) [P, elementary; N]

Define the evidence function f_y(x_1, …, x_G) = p(X, Y_q = y) / Π_g p_noise(x_g). It is a sum over admissible role
assignments of products of 1 + C per-group factors; noise groups contribute exactly 1. Its ANOVA/Möbius decomposition over
any product grid of group states therefore has **no interaction terms of order > 1 + C**. Its order-(1 + C) term is
nonzero whenever the per-role factors are non-constant.

**[N] V5** (normalised maximal Möbius term by order, G = 4, Q = 3, product grid with 3 states per group):

| order | C = 1 | C = 2 |
|---|---|---|
| 1 | 0.96 | 0.83 |
| 2 | 0.32 | 0.66 |
| 3 | 2e-16 | 0.33 |
| 4 | 2e-16 | 3e-16 |

**Corollary (a restricted separation) [P].** Consider a reader whose output score must equal f_y up to a constant that
does not depend on the input ("evidence-form" reader), of the form P(Σ_g φ(x_g)) with P a polynomial of degree d. Such a
reader has interaction order ≤ d. It therefore cannot represent the evidence function when d < 1 + C. For T4 (C = 1):

- additive scores (d = 1) are excluded;
- bilinear ρ is the minimal polynomial degree, and Theorem 1 shows it suffices.

**Limits of the corollary.**
1. It concerns evidence-form outputs. A normalised posterior can absorb an arbitrary input-dependent factor, and then
   interaction order is not preserved.
2. It does not cover non-polynomial ρ.
3. **[K / counterexample]** On a finite input domain, a sum decomposition with a 1-dimensional latent and an arbitrary ρ
   represents *every* symmetric function: encode each group state as a distinct power of a large base. So **no general
   impossibility follows for sum decompositions**. Every lower bound needs a restriction on ρ (degree, smoothness) or on
   φ. Investigation VIII's Proposition C (logits additive over groups ⇒ regime-independent response) is one such
   restricted result, and it remains valid.

## 4. Theorem 3 (soft gates under partial copying) [P; N]

For β < 1, eligibility is no longer a hard gate. Replace e_g by the all-question colluder likelihood ratio
c_g = Π_q p(x_{g,q} | colluding) / p(x_{g,q} | noise). Then the C = 1 form is still exact:

  p(y) ∝ (Σ_h c_h) · A_y − Σ_g W_g c_g L_g(y).

**[N] V6.** With β = 0.7 and G = 4 it is exact (max |Δp| = 2.8e-14). The hard-gate closed form (m_g counting perfect
groups) is wrong there by up to 0.97, and 68 % of episodes contain no "perfect and agreeing" group at all.

**Reading.**
- *Discrete eligibility* is a property of the β = 1 generator.
- The *pooled-sum structure* is the invariant.
- A model that learned a hard "perfect-group" detector would fail under partial copying, while one that learned the
  pooled likelihood-ratio form would not. This is one of the generalization tests in `RELATIONAL_GENERALIZATION.md`.

## 5. Information sufficiency (E1)

| statistic (per group, plus the target-question reports) | sufficient? |
|---|---|
| exact counts n_g + q-reports | **yes** [P, Theorem 1] |
| counts capped at 4, which cannot distinguish 4 from 5 | **no**: KL floor 0.081 nats [N] |
| capped at 3 / 2 / 1 / 0 | no: 0.165 / 0.220 / 0.258 / 0.276 |
| q-reports only (no cross-question information) | no: 0.278 |
| pairwise relations between particular groups | not needed: only S and the per-group e_g enter |
| higher-order (> 2) interactions | not needed for C = 1; order 1 + C in general (Theorem 2) |

The capped-count floors use a 200,000-decision sample. The best posterior given the statistic is estimated by cell
averaging, and its expected KL to p* is reported. This gives a reader-independent **information floor**: no reader whose
input cannot distinguish n = 4 from n = 5 can reach KL < 0.081. The floor is compared with native readers in
`PCOUNT_FAILURE_DIAGNOSIS.md`.

## 6. What is and is not new

- **Known:** Bayes' rule; sum decompositions of symmetric functions; ANOVA interaction order; the failure of additive
  models to represent interactions.
- **Specific to this work:**
  - the explicit pooled form of Theorem 1 for every (G, C), with S as the single cross-group scalar;
  - the exact interaction order 1 + C;
  - the soft-gate generalization;
  - the numerical information floor at the perfect-agreement boundary.

  All of these are elementary. Their value is as tools that make *falsifiable predictions* about learned models (§2,
  consequence 3; §4) and as a reader-independent information floor.
