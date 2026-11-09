# PP1: calibration, support and the sign of intervention effects (mandate §2)

Status: Stage I correction of atlas PP1. The frozen PP1 prediction and its verdict are unchanged in the original atlas.
This document re-analyses the same frozen edit population (1500 base episodes, stream 6,100,000; every query q, every
other question k ≠ q, every group g; 135,000 surgical edits). The new analyses are **post hoc relative to the frozen
PP1 run**. They are labelled as such and do not count as confirmation.

Scripts: `scripts/pp1_calibration.py`, `scripts/pp1_cx_matched.py`, `scripts/pp1_cx_targeted.py`, `scripts/pp1_learned_ece.py`.
Logs: `logs/pp1_calibration.json`, `logs/pp1_cx_matched.json`, `logs/pp1_cx_targeted.json`, `logs/pp1_learned_ece.json`.

## 1. Support notions (formal)

Let P be the T4 generator (pc = 1, ρ = 1). Its latent is R = (reliable group g, colluder group h). The answer is Y. The
observation is x = the 6×6 matrix of reports. A PP1 edit maps x to x′ = E_{k,g}(x) by toggling within-group agreement of
group g in question k ≠ q.

| notion | definition | what it licenses |
|---|---|---|
| S1 marginal | P(x′) > 0 | the Bayes posterior p(y \| x′) is defined; Δm is a well-posed target |
| S2 conditional on the latent | P(x′ \| R_true(x)) > 0 | the edit is a possible observation *for the same world* |
| S3 naturally reachable | x′ is obtainable by resampling one source mechanism given (R, Y) | an actual "natural" counterfactual |
| S4 surgical, observed | any edit of the input cells | only an input–output experiment |
| S5 surgical, internal | an edit of an internal activation | not a statement about P at all |

In this generator S3 = S2 (proof): the reliable and noise sources give every value positive probability; the colluder's
second source copies deterministically. Resampling one mechanism therefore stays inside P(· | R) exactly when the
colluder group still agrees in every question.

Population counts (135,000 edits):

| set | count |
|---|---|
| S1 | 93,570 |
| S2 = S3 | 90,000 |
| S1 \ S2 (marginally possible, impossible for the true world) | 3,570 |
| S2 \ S1 | 0 |
| outside S1 (no Bayes target; excluded from scoring) | 41,430 |

Scored set: S1 and |Δm| > 0.5 (47,005 edits).

Correction to the atlas wording: "on-support" in the atlas meant S1. Only the S2 subset is a natural counterfactual of
the sampled world.

## 2. Theorems: calibration-to-sign

Notation:
- m(x) = log p(c1 | x) − log p(c2 | x) is the Bayes margin. c1 and c2 are the base episode's top-2 candidates for q, held fixed across the pair.
- m̂ is the model margin. e(x) = m̂(x) − m(x).
- Δm = m(x′) − m(x) and Δm̂ = m̂(x′) − m̂(x).

**C1 (pointwise margin certificate).** If |Δm| > |e(x)| + |e(x′)|, then sign Δm̂ = sign Δm.
*Proof.* Δm̂ = Δm + e(x′) − e(x). ∎

**C2 (log-probability certificate).** Let δ(x) = max_y |log p̂(y | x) − log p(y | x)|. Then |e(x)| ≤ 2δ(x), so
|Δm| > 2(δ(x) + δ(x′)) suffices.

**C3 (total-variation certificate).** Let t(x) = TV(p̂(·|x), p(·|x)) and π(x) = min(p(c1|x), p(c2|x)). If t < π, then
|e(x)| ≤ 2t/(π − t).
*Proof.* |log a − log b| ≤ |a − b| / min(a, b), and |p̂ − p| ≤ t for each candidate. ∎

**C4 (distribution-level guarantee; density ratio and tail).** Let N be the natural distribution and E the
edit-endpoint distribution. Assume:
- (DR) a density ratio dE/dN ≤ W on the endpoints;
- an average natural divergence ε = E_N[KL(p ‖ p̂)].

By Pinsker and Markov, P_E(t(x) > τ) ≤ W ε / (2τ²). Then:

P_E(sign fails) ≤ P_E(|Δm| ≤ b_τ(x) + b_τ(x′)) + W ε / τ²,  where b_τ = 2τ/(π − τ).

The first term is a margin-tail term. It is the fraction of edits whose true effect is smaller than the error budget.
The second is the off-calibration mass, amplified by the density ratio.

What this needs: margin (the tail of |Δm| near 0), density ratio (W) and divergence (ε). None of these is "accuracy".

**C5 (monotone-transform invariance).** If m̂ = φ(m) on both endpoints for one strictly increasing φ, then sign
agreement is exact, regardless of calibration.
*Corollary.* Global temperature scaling never changes PP1 signs. Calibration in the ECE sense is neither necessary nor
sufficient.

**C6 (no accuracy-only theorem).** For every generator with non-degenerate margins, some model has Bayes-optimal
accuracy on N and sign agreement far from 1.
*Construction.* Let s(x) = 1/(0.05 + m_top(x)²) and set the logits to s(x) times the centred Bayes log-probs. This
preserves the argmax everywhere, but the margin is non-monotone in m across x.

**C7 (no average-KL theorem without (DR)).** Let E place mass on a region A with N(A) = η. A model that is exact off A
and arbitrary on A has KL ≤ η·B on N, yet any sign agreement on E(A). So the theorem needs W.

## 3. Learned models

| model | class | natural acc | ECE | natural KL | sign agreement | C1-certified | agreement when C1 fails |
|---|---|---|---|---|---|---|---|
| tf s0 | A′ | 0.739 | 0.004 | 0.0012 | 0.9986 | 0.991 | 0.84 |
| tf s1 | A′ | 0.737 | 0.007 | 0.0016 | 0.9961 | 0.990 | 0.61 |
| tf s2 | B | 0.740 | 0.008 | 0.0016 | 0.9958 | 0.986 | 0.71 |
| tf s10 | A′ | 0.740 | 0.003 | 0.0008 | 0.9998 | 0.997 | 0.93 |
| tf s11 | A′ | 0.739 | 0.005 | 0.0012 | 0.9967 | 0.990 | 0.66 |
| tf s12 | C | 0.741 | 0.006 | 0.0019 | 0.9998 | 0.987 | 0.99 |
| tf s13 | A′ | 0.738 | 0.006 | 0.0012 | 0.9988 | 0.995 | 0.78 |
| ds s0 | A | 0.742 | 0.003 | 0.0015 | 0.9987 | 0.993 | 0.82 |
| ds s1 | A | 0.735 | 0.009 | 0.0016 | 0.9989 | 0.992 | 0.86 |
| ds s2 | A | 0.739 | 0.005 | 0.0014 | 0.9993 | 0.994 | 0.88 |

Findings:
- **C1 is never violated.** Agreement where C1 holds is 1.000 in all 10 models, as it must be.
- **Every sign failure is an uncertified edit.** The C2 and C3 certificates are much looser: they cover 66–92% of edits.
- **Failures concentrate in two strata**, for the transformers:
  - the count boundary (leave-q count 4 ↔ 5): agreement 0.971–0.999;
  - S1 \ S2, edits impossible for the true world: agreement 0.968–0.998 for transformers. The DeepSets models score 1.000 on this stratum.
  - *Hypothesis* (one stratum, 2,605 edits, untested): transformer key-tag routing errs where the colluder's agreement pattern is broken. The DeepSets pool has no such dependence.
- **Across the 10 learned models, neither ECE nor natural KL ranks sign agreement.** Spearman correlation with sign agreement is −0.30 for ECE and −0.22 for natural KL. The models are too similar for a ranking test, so this is not evidence against C4; it is evidence that the learned population does not separate these quantities.

## 4. Constructed models (same edit population)

| model | natural acc | ECE | natural KL | sign agreement |
|---|---|---|---|---|
| Bayes | 0.746 | 0.003 | 0 | 1.000 |
| TEMP ×1/3 (underconfident) | 0.746 | **0.220** | 0.188 | **1.000** |
| TEMP ×3 (overconfident) | 0.746 | 0.082 | 0.275 | **1.000** |
| CX1, argmax-preserving margin distortion | 0.746 | 0.324 | 1.009 | **0.223** |
| CX2, permuted below the natural 2nd percentile of log P(x) | 0.739 | — | 0.02 | 0.994 |
| CX3, density-ratio region (count pattern (0,1,6), natural mass 0.15%) | 0.746 | 0.003 | 0.0004 | 0.999 |

- **The accuracy-matched triple (Bayes, TEMP, CX1)** is the required "matched accuracy, different calibration" contrast:
  - TEMP is badly calibrated and still has perfect sign agreement (C5).
  - CX1 has Bayes accuracy and fails most signs (C6).
  - So **margin fidelity, not calibration or accuracy, is the operative quantity.**
- **The density-ratio counterexamples (CX2, CX3) do not bite on this population.** This is a finding about the PP1 edit
  family, not a failure of C7:
  - single-cell toggles of natural episodes land in the bulk;
  - the largest count-pattern edit/natural ratio found was about 1.2;
  - only 0.18% of endpoints lie in the most over-represented class;
  - a targeted family ("make the noise group perfect") put 0% of endpoints below the 2nd percentile.
- **Conclusion for PP1.** W ≈ 1 here, so C4 reduces to "natural KL ≈ 10⁻³ plus a margin tail". That explains the 0.996+
  agreement *mechanistically, without invoking anything about the internal algorithm*. C7 would bite only for an edit
  family concentrated off the bulk. None was found among natural-like single-cell edits. This is recorded as an open
  construction in OPEN_FOUNDATIONAL_MAP.md.

## 5. Corrected PP1 claim

The atlas statement ("calibrated models predict intervention signs") is replaced by:

> For a natural-like edit family (W ≈ 1), any model with natural KL ε gets every sign right except on
> (i) edits whose true margin change is within its error budget and (ii) at most O(Wε/τ²) of the edit mass.
> In the ten learned models all sign failures are of type (i), concentrated at the count boundary and, for the
> transformers, in S1 \ S2.
> Sign agreement therefore *does not discriminate between implementation classes*: it is a consequence of being close to Bayes on N.
> It is not evidence about the computation.
> Accuracy alone guarantees nothing (CX1). ECE is irrelevant (TEMP).

Scope: one generator, one edit family, ten models. The S1 \ S2 transformer gap is a one-stratum observation, not a law.
