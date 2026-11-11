# Main result — Investigation VI: the perfect-agreement boundary

> **Post-review [R1–R5].** This document was revised after five independent reviews (REVIEWS_AND_ADJUDICATION.md).
> The frozen verdicts are unchanged. P1(a) is downgraded, the causal summary is weakened, the approximation-error argument
> is re-scoped, and the frozen-layer scope is made explicit. Pre-review text: commit 9389496.

**Question.** Does T1 vs P-count pretraining change how models use the 4/5 → 5/5 agreement transition when they update a
candidate decision, beyond exact Bayes and approximation error? If so, what causes it?

**Answer.** Yes, in one regime, prospectively, in the observable decision response. The setting is transfer with
**frozen pretrained layers 0–1** and a freshly trained layer 2.

**The regime.** The newly perfect group's vote agrees with the existing perfect group (R1). This is 32 % of oriented 4→5
pairs and about 3 % of natural decisions. Here exact Bayes says the candidate should go **up** (+0.23).

**The result.** In 16 new, independently seeded parents (8 + 8), trained after a frozen protocol:
- every T1 parent moves the candidate up (mean +0.19);
- every P-count parent moves it **down** (mean −0.51).

The Bayes-corrected difference is +0.70 log-ratio units (p = 7.8e-5, CI [0.59, 0.81]), with complete separation of the arms.

**What the difference survives:**
- it is specific to 4→5 relative to 3→4 in that regime, at step 4000 and from about step 500–1000 onward;
- it is present at every transfer stage, and it grows while P-count average quality improves;
- it is not produced by global gain or temperature;
- it is not reproduced by any fitted pooling operator examined;
- F0 controls show nothing.

**What the difference is.** A structured, curriculum-dependent approximation error concentrated on an "explaining-away"
subgroup. It is not a difference in average quality, and no distinct neural algorithm is identified. Its correlate is an
opposite-signed effect of the edit at the readout token's layer-2 input, carried by non-additive layer-2 attention × MLP
computation.

**Outcome.** A, scoped to R1. B is not reached.

## 1. Mathematical evidence

**Exact Bayes at the boundary** (EXACT_BAYES_BOUNDARY_THEORY.md, THEOREMS_AND_COUNTEREXAMPLES.md). With one existing perfect
group, a 4→5 edit opens the hypothesis "g is the colluder". The Bayes update toward g's vote then behaves as follows:
- it goes from positive (R3) or exactly zero (R2) below the boundary to **negative** (−2.1 to −2.4) at it [N-sample];
- in R1 it stays positive (+0.23);
- below the boundary it is non-negative and exactly zero in R2 [P, via likelihood-ratio dominance; R1-7].

**Proposition 3 corrected** (PROPOSITION_3_CORRECTION.md).
- The hard gate is not necessary. The mandate's counterexample is verified: null response 0.0, F45 response 0.63.
- Under nondegeneracy (A1, verified jointly over compatible patterns) and graph connectivity of edits (A3), null invariance
  holds iff one of:
  - the gate is hard and φ_rel is affine on 0–4;
  - the gate is finite and φ_rel and φ_col are both constant on 0–4.
- Mixed gates are never invariant.
- The hard-gate branch is a non-uniform limit of non-invariant members.
- s is unrestricted in both branches.
- Proposition 1's quantifier is corrected to the floor δ_min.

**Calibration limit** (T4). Under pure gain the predicted between-arm D-R1 contrast is −0.188 (γ_T − γ_P). The fitted
gains are γ_T ≈ 0.93 and γ_P ≈ 0.78, so pure gain predicts ≈ −0.03 where +0.66 is observed. On F34/F23-R2 pairs the Bayes
response is identically zero, so those pairs are gain-free.

**Elementary results**, labelled as such: representation does not fix use (T2), and s is confounded with the gate
increment (T3).

## 2. Behavioural evidence (frozen protocol, independent confirmation)

See INDEPENDENT_PARENT_REPLICATION.md and OBSERVABLE_BOUNDARY_RESPONSES.md.

| frozen prediction | result |
|---|---|
| P1(b) B45-R1, T1 > P | **confirmed**: +0.70, p = 7.8e-5 (Holm 1.6e-4). Arms do not overlap. Robust to every ITT reading, successful-only analysis and Bonferroni over 25 tests |
| P1(a) pooled B45, T1 > P | confirmed under the protocol/code ITT rule (+0.36, p = .018) but **fragile**: p = .066 under the stricter docstring reading, p = .076 without two weak T1 parents. It rests on quality-driven R2/R3 under-discounting. **Not counted as support for the hypothesis** |
| P2 D-R1 = B45-R1 − B34-R1 | **confirmed**: +0.66, p = 7.8e-5. The pooled analogue is n.s. (p = .09) |
| P3 stages | B45-R1 difference positive at all five stages (p ≤ 1e-4). D-R1 is −0.04 at 250, +0.16 at 500 (p = .03) and ≥ +0.45 from 1000. Pooled B45 has the opposite sign at 250/500 |
| P4 quality | KL-matched +0.68 (p = 7.8e-5; matched mean KL 0.079 vs 0.088). The ANCOVA arm coefficient is consistent with this (support-restricted 0.62). KL is nearly blind to an effect on about 3 % of decisions, so this is a weak control |
| P5 contexts | formal rule met (probability-space R1 sign +0.058). The frozen R2/R3 expectation is partly mismatched (R3 +0.23, n.s.) and not testable as stated |
| F0 control | max \|B0\| = 0.013 |

**Training dynamics** (TRAINING_STAGE_AND_QUALITY.md; exploratory).
- Over transfer, P-count KL falls from 0.44 to 0.09. Its 3→4 R1 deficit closes (−0.46 → −0.07) and its R3/R2 excesses
  shrink, while its R1 4→5 deficit deepens (−0.34 → −0.74).
- This is not what a uniform-shrinkage approximation account predicts. It is what a feature-limited approximator does when
  it improves its average while failing a small subgroup.
- The discount is acquired during transfer, under the same objective and layer-2 learner as T1.

## 3. Causal evidence (frozen P6 plus exploratory reviewer analyses)

See CAUSAL_EVIDENCE_UPDATE_MECHANISM.md.

**Frozen.**
- P6b, P6c, P6d and P6e were confirmed as stated.
- P6a (localisation) was **not** confirmed.
- After review, P6b and P6c are read as responses to non-linear, hybrid inputs at **asymmetric sites**. The P-count arm
  responds to arbitrary norm-matched site perturbations. The "count subspace" is not distinct from the principal subspace of
  any edit's change. P6d is near-vacuous. P6e does not show non-redundancy for R1 pairs.

**Exploratory.**
- At a symmetric site, the readout token's layer-2 input, the patched R1 response is positive in 8/8 T1 and negative in
  8/8 P-count parents.
- In 4 models, the P-count discount needs layer-2 attention and the MLP jointly.

**Not identified.** Which operation differs:
- (i) a frozen encoding that is hard to read conditionally; or
- (ii) a learned over-general "5/5 → discount" rule, possibly helped by the inherited readout.

These are undistinguished.

**Not tested.** Collateral effects, joint ablation, per-head/LayerNorm decomposition, and transplant.

## 4. Fitted operators (exploratory)

See FITTED_OPERATOR_IDENTIFIABILITY.md.

- The s ordering T1 > P-count replicates (0.76 vs −0.44). s is weakly identified, with overlapping flat sets.
- The best Bayes mixture reproduces about half of the P-count R1 effect.
- A mixture with a uniform "5/5 discount" fits with the wrong sign.
- No examined family produces a deep R1 discount together with near-Bayes R3 behaviour. The observable, not s, is the result.

## 5. What is and is not claimed

**Claimed.** A prospective, curriculum-associated difference in the **decision response** to perfect agreement, in R1, under
frozen-feature transfer:
- P-count-pretrained models discount where the generator calls for a credit;
- T1-pretrained models do not;
- this holds across independent seeds of one pipeline.

**Not claimed.**
- A distinct neural algorithm, or that the difference lies in *interpretation* rather than representation. The causal
  analysis cannot separate the two.
- A causal effect of the pretraining objective as such. Pretraining data and step counts (6000 vs 3000) differ too.
- Behaviour with unfrozen layers.
- Generalisation beyond R1 at the 4→5 boundary of this generator (K = 3, Q = 6, three groups of two, ρ = 1).
- Readiness, a paper thesis, or the closure of Step 1.

**New relative to Investigation V** [R5-e]:
- a fitted-parameter-free, prospective, sign-level difference in independent parents;
- the exact-Bayes regime decomposition;
- the P-count long-transfer arm;
- the D-R1 contrast with its gain prediction;
- the Proposition 3 correction;
- partial localisation.

**Not new.** The s ordering (now replicated), and the general lesson that Bayes-closeness can hide structured errors.

## 6. Earlier verdicts

All earlier verdicts are preserved. Investigation V's Proposition 3 ("the hard gate is necessary") and Proposition 1
quantifier are corrected here. No other earlier conclusion changes.
