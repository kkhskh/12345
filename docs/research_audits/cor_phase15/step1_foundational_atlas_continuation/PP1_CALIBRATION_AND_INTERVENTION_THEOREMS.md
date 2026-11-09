# PP1: calibration, support and the sign of intervention effects (mandate §2)

**Status.** This is a Stage I correction of atlas PP1. The frozen PP1 prediction and its verdict are unchanged in the atlas.

**What this document analyses.** It re-analyses the same frozen edit population: 1500 base episodes from stream 6,100,000, with every query q, every other question k ≠ q and every group g, giving 135,000 surgical edits. The new analyses are **post hoc relative to the frozen run** and are not confirmations.

**Revision.** Revised after review round R2 (`reviews/ADJUDICATION_R2.md`). Corrected claims are marked [R]. Pre-review text is in git history.

**Scripts:**
- `scripts/pp1_calibration.py`
- `scripts/pp1_cx_matched.py`
- `scripts/pp1_cx_targeted.py`
- `scripts/pp1_learned_ece.py`
- `scripts/pp1_highratio.py` (added after review)

Logs: the matching `logs/pp1_*.json`.

## 1. Support notions (formal)

Let P be the T4 generator (p_c = 1, ρ = 1). The latent is R = (reliable group g, colluder group h); the answer is Y; the observation x is the 6×6 matrix of reports. A PP1 edit maps x to x′ = E_{k,g}(x) by toggling the within-group agreement of group g in question k ≠ q.

| notion | definition | what it licenses |
|---|---|---|
| S1 marginal | P(x′) > 0 | the Bayes posterior p(y \| x′) is defined, so Δm is a well-posed target |
| S2 conditional on the latent | P(x′ \| R_true(x)) > 0 | the edit is a possible observation *for the same world* |
| S3 naturally reachable | x′ is obtainable by resampling one source mechanism given (R, Y) | an actual natural counterfactual |
| S4 surgical, observed | any edit of the input cells | an input–output experiment only |
| S5 surgical, internal | an edit of an internal activation | not a statement about P at all |

**S3 = S2 for this edit family.**
- Reliable and noise sources give every value positive probability. The colluder's second source copies deterministically.
- So resampling one mechanism stays inside P(· | R) exactly when the colluder group still agrees in every question.
- [R] This equality is a property of this single-cell toggle design (reviewer ST-F8). It is not a general fact.

**Population.** The 135,000 edits are 27,000 distinct inputs, each scored at 5 queries (reviewer ST-F7). Counts below are edit counts.

| set | edits |
|---|---|
| S1 | 93,570 |
| S2 = S3 | 90,000 |
| S1 \ S2 | 3,570 (714 distinct endpoints from 119 base episodes) |
| outside S1 | 41,430 |

**Scored set.** S1 and |Δm| > 0.5: 47,005 edits.
- [R] This filter drops about half of S1, and that half is exactly where sign failures concentrate (ST-F6).
- So the filtered rates below are conditional on a Bayes-side selection. Unfiltered rates are given for the new high-ratio family (§4).
- **Correction to the atlas wording.** In the atlas, "on-support" meant S1.

## 2. Theorems: calibration-to-sign

**Notation.**
- m(x) = log p(c1 | x) − log p(c2 | x), with (c1, c2) the base top-2 pair, held fixed at both endpoints.
- e(x) = m̂(x) − m(x).
- Δm = m(x′) − m(x).

**C1 (pointwise certificate).** If |Δm| > |e(x)| + |e(x′)|, then sign Δm̂ = sign Δm.
- *Proof.* Δm̂ = Δm + e(x′) − e(x). ∎
- [R] This is an identity. Its "never violated in 10 models" check is a code check. The *empirical content* is the **coverage**: the fraction of edits on which the condition holds.

**C2.** |e(x)| ≤ 2δ(x), with δ(x) = max_y |log p̂(y|x) − log p(y|x)|.

**C3.** With t(x) the TV distance between p̂ and p, and π(x) = min(p(c1|x), p(c2|x)): if t < π, then |e(x)| ≤ 2t/(π − t).

**C4 (distribution level).** Assume a density ratio dE/dN ≤ W on the endpoints (DR), and natural KL ε. Then
P_E(fail) ≤ P_E(|Δm| ≤ b_τ(x) + b_τ(x′)) + W·ε/τ².

- [R] **Valid but vacuous on PP1.** With ε ≈ 10⁻³, the right-hand side is ≥ 1 for every τ (reviewers ST-F2, FM-F3, NV-A2), because ε/τ² is O(1) at any τ small enough to make b_τ useful.
- C4 therefore **does not explain** the 0.996+ agreement. The pre-review claim that it did "mechanistically" is withdrawn.

**C5 (monotone-transform invariance).** If m̂ = φ(m) at both endpoints for a strictly increasing φ, then signs are exact. So global temperature is irrelevant to signs.
- [R] Per-probability recalibrations, such as uniform mixing, are outside C5 (FM-F10).

**C6 (no accuracy-only guarantee).**
- [R] Stated as an existence result for this generator and edit family. CX1 attains Bayes accuracy with sign agreement 0.22.
- It is not "for every generator". A ranking-preserving distortion such as CX1 automatically gets right any edit that swaps the order of c1 and c2: 1.000 on that stratum, against 0.094 elsewhere (reviewer FM-F5).

**C7 (no average-KL guarantee without (DR)).**
- [R] It needs a bounded-KL assumption.
- It is realised by a multi-cell edit family with a model corrupted where that family lands. Reviewer FM: 46% of endpoints in the corrupted region, agreement 0.875 (reviewer-reported).

## 3. Learned models (frozen PP1 population; filtered |Δm| > 0.5)

| model | class | natural acc | ECE | natural KL | sign agreement | C1 coverage | agreement where C1 fails |
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

**Findings.**
- **The C1 condition covers 98.6–99.7% of scored edits.** The remaining edits are where every failure occurs.
- **The explanation of high agreement is empirical:** these models' pointwise margin errors at the edit endpoints are smaller than the true effects. No theorem with these constants implies that (§2, C4).
- **Strata.**
  - At the count boundary (4 ↔ 5), agreement is 0.971–0.999.
  - On S1 \ S2, agreement is **0.968–1.000** across Transformers [R: corrected from 0.968–0.998; TF s12 is 1.000], and 1.000 for the DeepSets models.
  - [R] The pre-review hypothesis that "transformer key-tag routing errs on S1 \ S2" is **withdrawn**. It is one post hoc stratum among about 100 model × stratum cells, with no multiplicity control, drawn from 119 base episodes.
- **Across the 10 models, ECE and natural KL do not rank sign agreement** (Spearman −0.30 and −0.22). The models are too similar for a ranking test.

## 4. Constructed models and a high-density-ratio edit family

### 4a. Accuracy-matched constructed models (same population)

| model | natural acc | ECE | natural KL | sign agreement |
|---|---|---|---|---|
| Bayes | 0.746 | 0.003 | 0 | 1.000 |
| TEMP ×1/3 (underconfident) | 0.746 | **0.220** | 0.188 | **1.000** |
| TEMP ×3 (overconfident) | 0.746 | 0.082 | 0.275 | **1.000** |
| CX1, argmax-preserving margin distortion | 0.746 | 0.324 | 1.009 | **0.223** |
| CX2, permuted below the natural 2nd percentile of log P(x) | 0.739 | — | 0.02 | 0.994 |
| CX3, corrupted on the count pattern (0,1,6) | 0.746 | 0.003 | 0.0004 | 0.999 |

- **The accuracy-matched triple (Bayes, TEMP, CX1)** shows that calibration (ECE) and accuracy are not what matter for signs. Margin fidelity at the endpoints is.
- [R] CX2 and CX3 corrupted **low-probability** or **coarse-pattern** regions. A high *density ratio* is a different property, so these were not tests of C7 (reviewer ST-F5).
- [R] The pre-review "W ≈ 1.2" was a coarse lower bound. The pointwise ratio on S1 reaches about 1.96 (reviewer ST-F4).

### 4b. High-ratio family NP: make the noise group perfect (added after review)

**Construction** (`scripts/pp1_highratio.py`; new stream 6,180,000; 2000 base episodes × 6 queries).
- For query q, the noise group's second source is set equal to its first in every k ≠ q.
- Every endpoint lies in A_q = {≥ 2 groups agree in every k ≠ q}. The natural probability of A_q is **0.130**, so the density ratio is **≥ 7.7**.
- All 12,000 endpoints are in S1.

| model | sign agreement, all S1 (clustered 95% CI) | \|Δm\| > 0.5 | \|Δm\| < 0.1 (n = 372) | 0.1–0.5 (n = 814) | 0.5–1 (n = 1043) | ≥ 1 (n = 8425) |
|---|---|---|---|---|---|---|
| tf s0 | 0.979 [0.976, 0.982] | 0.997 | 0.61 | 0.94 | 0.97 | 1.000 |
| tf s1 | 0.979 | 0.997 | 0.66 | 0.91 | 0.98 | 1.000 |
| tf s2 | **0.963** [0.959, 0.968] | 0.985 | 0.61 | 0.87 | 0.86 | 1.000 |
| tf s10 | 0.969 | 0.999 | 0.54 | 0.81 | 0.99 | 1.000 |
| tf s11 | 0.980 | 0.998 | 0.61 | 0.93 | 0.98 | 1.000 |
| tf s12 | 0.977 | 0.999 | 0.52 | 0.94 | 0.99 | 1.000 |
| tf s13 | 0.978 | 0.997 | 0.62 | 0.91 | 0.98 | 1.000 |
| ds s0, s1, s2 | 0.977–0.979 | 0.998–0.999 | 0.49–0.51 | 0.95 | 0.99 | 1.000 |

**Findings.**
- At W ≥ 7.7 the learned models still get every large-effect sign right (|Δm| ≥ 1: 1.000).
- Failures sit entirely in the margin tail. For |Δm| < 0.1 agreement is 0.49–0.66, near chance [R3-F14].
- A high density ratio alone does not break the learned models here, because they are accurate *on* this region. The region A_q has natural mass 0.13; it is not rare.
- CX4 (Bayes with the top-2 candidates swapped on R = {≥ 2 groups with total count ≥ 5}) has sign agreement 0.60. But R has natural mass 0.35, giving natural KL 0.82 and accuracy 0.52, so CX4 is **not** a low-KL counterexample. It is reported only for completeness.
- The C7-type construction that does bite at low KL is reviewer FM's (§2).

## 5. Corrected PP1 claim

The atlas statement "calibrated models predict intervention signs" is replaced by:

> **(i)** On the filtered PP1 edits, the C1 condition (|Δm| exceeds the summed pointwise margin errors at both endpoints) holds for 98.6–99.7% of edits. By C1 (an identity), signs agree on all of them. The 0.996+ agreement therefore reflects these models' *pointwise* accuracy at the edit endpoints.
>
> **(ii)** On the high-ratio family NP (W ≥ 7.7), every sign with |Δm| ≥ 1 is correct, and agreement falls toward chance as |Δm| → 0. C1 coverage on NP was not computed [R3-F10].
>
> **(iii)** The distribution-level theorem C4 is valid but vacuous at the observed ε.
>
> Sign agreement does **not** discriminate implementation classes, and is not evidence about the computation. Accuracy alone guarantees nothing (CX1), and ECE is irrelevant (TEMP, C5).

**Scope.** One generator, two edit families, ten models. The clustered intervals cover the high-ratio family only.
