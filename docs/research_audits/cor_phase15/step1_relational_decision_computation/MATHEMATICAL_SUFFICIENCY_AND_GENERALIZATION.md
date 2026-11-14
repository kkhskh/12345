# Mathematical sufficiency and generalization (Workstream C)

Investigation XIII, Phase 1.5 Step 1. This document covers the mathematics only. Mechanistic claims are in
`NATIVE_DECISION_COMPUTATION.md`, and adaptation experiments are in the section *Two-colluder repair* below.

Scripts: `scripts/verify_shared_posterior.py` → `logs/verify_shared_posterior.json`;
`scripts/sufficiency_boundary.py` → `logs/sufficiency_boundary.json`.

## C1. The exact posterior for the sampler that is actually used

`genlib.sample` draws one copied value c_q ~ U(K) and one copy event cop_q ~ Bern(β) per question. Both are **shared by every colluding
source**. When cop_q is false, colluder sources report independently and uniformly. The exact posterior therefore sums over roles
(g reliable, H colluders, the rest noise). The colluder-set factor at each question is

  χ_H(q) = β · 1[all 2|H| colluder sources are equal at q] / K + (1 − β) / K^{2|H|}.

`xii_bayes.posterior_shared` implements this. `genlib.posterior_brute` instead multiplies per-group factors
β·1[pair equal]/K + (1−β)/K², which is the posterior of an *independent-copying* model.

**Verification 1 (exhaustive, independent implementation).** I wrote a separate brute-force marginalisation directly from the sampler's
definition: sum over y, the shared copy event, and the shared copied value, with per-source report probabilities. I then enumerated
**every** input in a reduced environment (G=3, Q=2, K=3, so 3¹² = 531,441 inputs) for C ∈ {1, 2} and β ∈ {1, 0.6}.

| regime | Σ p(X) | support size | max abs error, `posterior_shared` | max abs error, `posterior_brute` | p(X)-weighted L1 error, `posterior_brute` |
|---|---|---|---|---|---|
| C=1, β=1 | 1.000000 | 158,193 | 2.2e-16 | 2.2e-16 | 8e-17 |
| C=1, β=0.6 | 1.000000 | 531,441 | 2.2e-16 | 2.2e-16 | 1e-16 |
| C=2, β=1 | 1.000000 | 2,169 | 1.1e-16 | **0.636** | **0.370** |
| C=2, β=0.6 | 1.000000 | 531,441 | 3.3e-16 | **0.694** | **0.649** |

`posterior_shared` is exact to machine precision in all four regimes, including partial copying and the shared copy event. The
independent-copying posterior is exact only for C = 1.

The exhaustive check uses q = 0 only, and it shares one assumption with `posterior_shared`: questions are independent given the role.
Independent review 1 (`reviews/REVIEW_1_MATH.md`, `reviews/review1_math_checks.py`) closed both gaps. It ran the exhaustive test at q = 1,
and it compared 10⁶ draws of `genlib.sample` with the exact Q = 2 likelihood (chi-square ≈ number of cells, no off-support draws).

**Verification 2 (Monte-Carlo calibration at full size, Q=6, 200,000 episodes each, on `genlib.sample`).**

| regime | ECE | accuracy | mean confidence | NLL, shared | NLL, independent |
|---|---|---|---|---|---|
| C=1, β=1 | 0.0027 | 0.741 | 0.743 | 0.545 | 0.545 |
| C=2, β=1 | 0.0006 | 0.800 | 0.800 | **0.400** | 0.454 |
| C=2, β=0.6 | 0.0007 | 0.793 | 0.793 | **0.424** | 0.885 |

## C2. Closed forms and where J_H enters

Write W_g = Π_{q'≠q} p(x_{g,q'} | g reliable) / p(x_{g,q'} | noise) and L_g(y) = p(x_{g,q} | g reliable, y) / p(x_{g,q} | noise). Each
role's likelihood, taken relative to the all-noise likelihood, factorises over questions.

**(R1) C = 1, β = 1.** The colluder factor ratio is χ_h/noi = K·1[pair h self-agrees at q]. Its product over the six questions is
K⁶·e_h, where e_h = 1[h self-agrees at all six questions]. Summing over the two possible colluders for each reliable g gives

  p(y | X) ∝ Σ_g W_g L_g(y) (S − e_g),  S = Σ_h e_h.

This is exact. Monte-Carlo comparison with the exact posterior (100,000 episodes, via `posterior_brute`, which is exact for C = 1) gives
KL 0.000 and 100% decision agreement. Review 1 also re-derived and re-implemented R1–R4 and matched the exact posterior to about 1e-14 at
Q = 6. The **fixed, low-dimensional** per-group statistic φ_g = (e_g, W_g L_g, e_g W_g L_g) with a pooled readout is therefore
sufficient (Investigation VII, Theorem 1).

**(R2) C = 1, β < 1.** The colluder ratio becomes 1 − β + βK·1[agree], so e_h must be replaced by the graded per-group statistic
c_h = Π_q (1 − β + βK·1[h self-agrees at q]):

  p(y | X) ∝ Σ_g W_g L_g (Σ_{h≠g} c_h).

The fixed per-group statistic (c_g, W_g L_g, c_g W_g L_g) is **still sufficient** (KL 0.000). The *hard* (S − e_g) rule is undefined
whenever S = 0, which is 77.4% of episodes at β = 0.6. On the 22.6% where it is defined, it reaches accuracy 0.612 against a Bayes
accuracy of 0.643 on the same rows (KL 0.31; its decision differs from Bayes on 17.8%). An earlier draft scored the undefined rows as
answers (NaN → argmax 0), which produced an artefactual 0.397; review 1 caught this. The (S − e_g) form is a β = 1 special case; per-group
sufficiency is not.

**(R3) C = 2, β = 1.** For colluder set H = {h₁, h₂}, χ_H/noi² = K³·1[all four sources equal at q]. Over six questions this gives
K¹⁸·J_H, where

  J_H = 1[h₁ and h₂ both self-agree, and report identical values, at all six questions],

and so p(y | X) ∝ Σ_g W_g L_g(y) J_{−g}. J_H is a **joint** (cross-group) statistic. The independent-copying product rule replaces
J_H by e_{h₁} e_{h₂}.

*Where the difference matters (proved, then checked numerically).* At β = 1 the colluders always self-agree everywhere, so e_H = 1 for
the true H, and S ≥ 2 on the support. S = 1 cannot occur, so the S ≤ 2 row below is exactly S = 2.
- If **S = 2**, the reliable group has e = 0. Every other candidate colluder set then contains a group with e = 0, so J and the e-product
  agree on all three roles, and the product rule is exact.
- If **S = 3**, every group self-agrees everywhere. The product rule gives all three roles equal weight, while J_H selects the identical
  pair.

The joint statistic can matter **only on S = 3**. Within S = 3 it matters only when the groups do *not* all report the same value at q.
When all three agree at q, every role gives the same answer and the product rule is exact (review 1):

| C = 2, β = 1 | fraction | Bayes accuracy | product-rule accuracy | product-rule KL |
|---|---|---|---|---|
| S ≤ 2 | 0.917 | 0.781 | 0.781 | **0.000** |
| S = 3, all agree at q | 0.028 | 0.973 | 0.973 | 0 |
| S = 3, not all agree at q | **0.055** | 0.973 | **0.013** | large |

On the support, cross-group equality *at q* already identifies the colluder pair whenever it is informative: if exactly two groups agree
at q, they must be the identical pair. If all three agree, the answer does not depend on the pair. Any S = 3 input that would break this
is off-support. This is why a pair-contrast design that breaks the colluders'
identity at one question while keeping every e_g fixed produces inputs of probability 0 under β = 1. The exact posterior is undefined
there (NaN), and the design was abandoned before any freeze (`logs/COMPUTE_LOG.txt`). The J_H-dependence test used in XIII is the
S = 3 subset.

**(R4) C = 2, β < 1.** The ratio becomes 1 − β + βK³·1[all four equal at q], and the sufficient statistic is the pairwise, cross-question
quantity J_H^β = Π_q (1 − β + βK³·1[all four equal at q]). Neither the hard rule nor the fixed per-group statistics suffice here. At
β = 0.6 the product rule's posterior is off (KL > 0.01) on 85% of episodes and its decision differs from Bayes on 37%:

| C = 2, β = 0.6 | accuracy | KL |
|---|---|---|
| exact | 0.795 | 0 |
| independent-copying product rule | 0.502 | 0.462 |
| graded C = 1 rule | 0.446 | 0.723 |
| hard C = 1 rule | 0.428 on the 30% of rows where defined (Bayes on those rows: 0.832) | 1.17 (defined rows) |

**Rank argument.** On self-agreeing colluder pairs with values (a, a) and (b, b), the one-question shared colluder kernel is
χ(a, b) = β·1[a = b]/K + (1 − β)/K⁴. That is βI_K/K plus a constant, with rank K = 3 for β > 0 (computed at β = 1 and 0.6). A single
product ψ(a)ψ(b) has rank 1, so the colluder factor does not factorise across the two colluder groups. Over six questions the kernel on
full self-agreeing sequences is the Kronecker power χ^{⊗6}, with rank 3⁶ = 729 (computed). Any sum-of-products (factorised) representation
of it therefore needs at least 729 features per group (review 1). The rank argument shows non-factorisation of the factor. That the
*posterior* differs is shown by the counterexample.

**Explicit counterexample (β = 0.6).** Take q = 0 and two inputs in which **every group's own statistics are identical**:
- all three groups self-agree at all six questions (e_g = 1, identical W_g);
- the values at q are identical (identical L_g).

In x₁, groups A and B report identical values at all six questions. In x₂, B differs from **both** A and C at every other question.
(The first draft's x₂ also had B = C at one question, a coincidence that inflated the shift; review 1 caught it.) Review 1 recomputed
W_g = 1.98⁵ and L_g for every group.
- Exact posteriors: x₁ → (0.015, 0.970, 0.015); x₂ → (0.059, 0.926, 0.015).
- Any rule F({(e_g, W_g, L_g)}_g) must output the same distribution for both. The product rule gives (0.652, 0.333, 0.015) for both.
- At β = 1, x₂ is off-support, consistent with (R3).

## C3. Boundary statement (what is and is not claimed)

The "per-group statistics" here are **fixed, low-dimensional** ones. A rich enough per-group encoding with a pooled readout can
represent any group-symmetric posterior, so "per-group" alone is not a meaningful restriction.

1. **A fixed three-component per-group statistic suffices** when there is at most one colluder group (C = 1), for every β. The
   statistic is (c_g, W_g L_g, c_g W_g L_g), which reduces to the hard eligibility e_g at β = 1. Evidence: closed form; Monte-Carlo
   agreement with the exact posterior; review 1's independent re-implementation. The exhaustive enumeration verifies the exact posterior
   itself.
2. **The fixed statistics (c_g, W_g, L_g) do not suffice once two colluder groups share copy events (C = 2, G = 3).** A cross-group
   (pairwise) statistic is needed:
   - at β = 1, only on the S = 3, not-all-agree-at-q slice (5.5% of episodes), where a cross-group equality at q suffices on the support;
   - at β < 1, through the cross-question pairwise statistic J_H^β. A factorised representation of it needs ≥ 729 features per group.
     The product-rule posterior is off on 85% of episodes; decisions differ on 37%.
3. **The C = 1 rule fails** on C = 2 data (accuracy 0.334 at β = 1, which is chance) because colluders' high W_g dominates. The
   independent-copying product rule fails on the S = 3 not-all-agree slice at β = 1 (accuracy 0.013 there), and broadly at β < 1.
4. Not claimed: that these regimes cover "relational inference" in general, that the T4 family is representative of other tasks, or any
   link to O/D.

## Two-colluder repair: selective adaptation (development and confirmation)

`scripts/adapt_sel.py` freezes layers 0–1 of a C = 1-trained model, which fixes resid2. It then trains, on C = 2 data
(1500 steps, batch 256, AdamW), one of five configurations:
- the readout (final LN + unembedding) only;
- layer-2 attention only;
- layer-2 MLP only;
- layer 2 plus the readout ("full");
- a freshly initialised layer 2 plus the readout ("fresh").

Each is evaluated on 3,000 fresh C = 2 episodes × 6 questions, overall and on the J_H-dependent S = 3 slice. Results are in
`FROZEN_PROTOCOL_AND_CONFIRMATION.md` (P5) and `logs/adapt_sel_{dev,conf}/`.
