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

This is exact: KL to the exact posterior is 0.000 and decisions agree 100%. A per-group (deep-sets) statistic
φ_g = (e_g, W_g L_g, e_g W_g L_g) with a pooled readout is therefore sufficient (Investigation VII, Theorem 1).

**(R2) C = 1, β < 1.** The colluder ratio becomes 1 − β + βK·1[agree], so e_h must be replaced by the graded per-group statistic
c_h = Π_q (1 − β + βK·1[h self-agrees at q]):

  p(y | X) ∝ Σ_g W_g L_g (Σ_{h≠g} c_h).

Per-group statistics are **still sufficient** (KL 0.000). The *hard* (S − e_g) rule fails: at β = 0.6 its accuracy is 0.397 against a
Bayes accuracy of 0.616, and it is undefined whenever S = 0, which is 77% of episodes. So the (S − e_g) form is a β = 1 special case;
per-group sufficiency is not.

**(R3) C = 2, β = 1.** For colluder set H = {h₁, h₂}, χ_H/noi² = K³·1[all four sources equal at q]. Over six questions this gives
K¹⁸·J_H, where

  J_H = 1[h₁ and h₂ both self-agree, and report identical values, at all six questions],

and so p(y | X) ∝ Σ_g W_g L_g(y) J_{−g}. J_H is a **joint** (cross-group) statistic. The independent-copying product rule replaces
J_H by e_{h₁} e_{h₂}.

*Where the difference matters (proved, then checked numerically).* At β = 1 the colluders always self-agree everywhere, so e_H = 1 for
the true H, and S ≥ 2 on the support.
- If **S = 2**, the reliable group has e = 0. Every other candidate colluder set then contains a group with e = 0, so J and the e-product
  agree on all three roles, and the product rule is exact.
- If **S = 3**, every group self-agrees everywhere. The product rule gives all three roles equal weight, while J_H selects the identical
  pair.

The joint statistic is therefore needed **exactly on S = 3**:

| C = 2, β = 1 | fraction | Bayes accuracy | product-rule accuracy | product-rule KL |
|---|---|---|---|---|
| S ≤ 2 | 0.917 | 0.781 | 0.781 | **0.000** |
| S = 3 | 0.083 | 0.973 | **0.336** | 0.649 |

On the support, cross-group equality *at q* already identifies the colluder pair when S = 3: the pair that agrees at q must be the
identical pair. Any S = 3 input that would break this is off-support. This is why a pair-contrast design that breaks the colluders'
identity at one question while keeping every e_g fixed produces inputs of probability 0 under β = 1. The exact posterior is undefined
there (NaN), and the design was abandoned before any freeze (`logs/COMPUTE_LOG.txt`). The J_H-dependence test used in XIII is the
S = 3 subset.

**(R4) C = 2, β < 1.** The ratio becomes 1 − β + βK³·1[all four equal at q], and the sufficient statistic is the pairwise, cross-question
quantity J_H^β = Π_q (1 − β + βK³·1[all four equal at q]). Neither the hard rule nor any per-group rule is sufficient here:

| C = 2, β = 0.6 | accuracy | KL |
|---|---|---|
| exact | 0.795 | 0 |
| independent-copying product rule | 0.502 | 0.462 |
| graded C = 1 rule | 0.446 | 0.723 |
| hard C = 1 rule | 0.361 | undefined when S = 0 |

**Rank argument.** On self-agreeing colluder pairs with values (a, a) and (b, b), the shared colluder kernel is
χ(a, b) = β·1[a = b]/K + (1 − β)/K⁴. That is βI_K/K plus a constant, which has rank K = 3 for β > 0 (numerical rank 3). A product
ψ(a)ψ(b) has rank 1, so no per-group-product rule can equal the shared-copy posterior.

**Explicit counterexample (β = 0.6).** Take q = 0 and two inputs in which **every group's own statistics are identical**:
- all three groups self-agree at all six questions (e_g = 1, identical W_g);
- the values at q are identical (identical L_g).

The two inputs differ only in whether groups A and B report identical values at the other five questions.
- Exact posteriors: x₁ → (0.015, 0.970, 0.015); x₂ → (0.498, 0.487, 0.015).
- Any rule F({(e_g, W_g, L_g)}_g) must output the same distribution for both. The product rule gives (0.652, 0.333, 0.015) for both.
- At β = 1, x₂ is off-support, consistent with (R3).

## C3. Boundary statement (what is and is not claimed)

1. **Per-group statistics with a pooled readout suffice** whenever there is at most one colluder group (C = 1), for every β. The
   statistic is graded for β < 1 and reduces to the hard eligibility e_g only at β = 1. This is shown by exact enumeration and closed form.
2. **A joint, cross-group statistic is required** as soon as colluders share copy events (C ≥ 2):
   - at β = 1, only on the S = 3 slice, where a cross-group equality at q suffices on the support;
   - at β < 1, on most of the input space, through a pairwise statistic across questions.
3. **The C = 1 rule fails** on C = 2 data (accuracy 0.334 at β = 1, which is chance) because colluders' high W_g dominates. The
   independent-copying product rule fails on S = 3 at β = 1, and broadly at β < 1.
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
