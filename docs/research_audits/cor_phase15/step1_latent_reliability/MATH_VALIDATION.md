# Mathematical validation before training (mandate §1–§3)

Status: written and run before any model in this package was trained on a frozen protocol.
Code: `src/bayes.py`, `src/math_checks.py`, `src/math_checks2.py`. Logs: `logs/math_checks*.txt|json`.
Everything below is a statement about the **data-generating process and its exact Bayes posterior** (Bayes-level).
Nothing here says anything about a learned model.

## 0. Notation

- An episode has Q questions (primary setting Q = 6) and K candidates (primary setting K = 3; K = 5 is an extension).
- Each question has six sources in three fixed groups: G0 = {0,1}, G1 = {2,3}, G2 = {4,5}.
- Answers Y_q ~ Uniform(K), independently across questions.
- Reports x_{q,s} ∈ {0..K−1}.
- A **reliable** source reports Y_q with probability α = 0.8. Otherwise it reports one of the other K−1 candidates uniformly.
- A **noise** source reports uniformly at random.
- A **colluding** group (T4) makes both members report the same c_q ~ Uniform(K), independent of Y_q.
- The latent R is the episode's reliability configuration. It is shared by all Q questions in the episode and has a uniform prior.

| task | latent R | roles |
|---|---|---|
| T0 | none (known) | G0 reliable, G1 and G2 noise |
| T1, T2, T3 | g ∈ {0,1,2} | G_g reliable, others noise |
| T4 | (g, h), g ≠ h (6 values) | G_g reliable, G_h colluding, third group noise |
| T5 | none | all six sources reliable with α5 = 0.5 |

## 1. E0 — independent sources: additive log odds

**Setting.** Let the sources be conditionally independent given Y with *known* reliabilities, and let the prior on Y be uniform.

**Derivation.**

P(Y=y | x) ∝ Π_s P(x_s | y)

so for any two candidates y, y′,

log P(y|x)/P(y′|x) = Σ_s [log P(x_s|y) − log P(x_s|y′)].

**Result.** The log odds are a sum of per-source terms, one evidence vote per source. No interaction among sources is needed.

**Check (M2).** With 6 sources, K = 2 and random reliabilities, max |exact − additive| = 1.8e−15.

**Consequence.** In T0, where reliability is known and fixed, the per-question Bayes posterior uses only the reliable group's votes. Cross-question information has zero value: full Bayes equals own-question Bayes at 0.8031 (M4).

## 2. E1 — hidden shared reliability breaks additivity (mandate example)

**Setting.** Prior (0.025, 0.025, 0.95) over three candidates. Sources A1, A2 come from one pair, B1, B2 from the other. Exactly one pair is reliable (accuracy 0.88, errors uniform over the other two candidates), the other pair is noise, and it is unknown which.

**Exact posterior.** We marginalise over which pair is reliable.

**Results.** Computed analytically (`bayes.e1_example`) and, independently, by brute-force enumeration over the joint (Y, reliable pair, reports):

| reports | posterior over candidates |
|---|---|
| A = (0,0), B = (1,1) | (0.42523, 0.42523, 0.14954) |
| A = (0,1), B = (0,1) | (0.21782, 0.21782, 0.56436) |

Both reproduce the mandate's numbers to 5 decimals (M1).

**Why this is non-additive.** Both report sets have the same global histogram: two votes for candidate 0 and two for candidate 1. Yet the posteriors differ. The evidence a vote carries depends on whether its partner agreed: a pair that agrees is evidence that *this pair* is the reliable one. The posterior is not a function of per-source terms or of the vote histogram.

## 3. Six-source version (M3)

**Construction.** Two Q = 6 episodes in which every question has the identical global histogram (2,2,2):

- **Episode A:** G0 agrees on every question and the other groups never agree.
  - P(R | X) = (0.9994, 3e−4, 3e−4).
  - Question-0 posterior = (0.969, 0.015, 0.015).
- **Episode B:** no group is coherent.
  - Every posterior is uniform (1/3).

**Random search.** Over 20,000 episodes, matched-histogram pairs reach an L1 difference of 1.611 between question-0 posteriors.

**Restricted-statistic Bayes accuracies** (M4, exact). These include the *best* classifier that sees only the six per-question histograms, computed by enumerating all 3^6 report vectors.

| task | full Bayes | own question only | all six histograms | oracle R |
|---|---|---|---|---|
| T0 | 0.803 | 0.803 | 0.638 | 0.803 |
| T1 | 0.743 | 0.652 | 0.636 | 0.798 |
| T4 | 0.745 | 0.580 | 0.556 | 0.801 |
| T5 | 0.646 | 0.646 | 0.646 | 0.646 |

### What histogram insufficiency does *not* show

That the global vote histogram is insufficient (T1: 0.636 vs 0.743) shows only that a solver must use **which source said what**, not just how many said it.

It does **not** show that a solver must form a **separate agreement or consistency variable**. Three alternatives are equally consistent with the data:

- **Exact computation without a distinct variable.** A flexible function of the full report matrix X can compute the exact posterior with no isolable intermediate. For example, it can sum per-question log-likelihoods log P(X_q | R) for each R and softmax them. That computation never names an "agreement" variable, although it is extensionally a function of agreement counts (§5).
- **Direct mapping.** A lookup table over X, or a sufficiently large MLP, needs no intermediate at all.
- **Distributed representation.** Any representation that is an invertible re-encoding of the counts is equally sufficient (§6).

So the existence of a relation-dependent Bayes posterior is a statement about **what information is needed**. Whether a learned model **represents** that information as a separable, reusable variable is an empirical question. This package tests it; it is not assumed.

## 4. Multi-question episodes with shared R (§2)

**Exact reference.** For each R:

- the joint per question is J_q(R, y) = P(y) P(X_q | y, R);
- the marginal is M_q(R) = Σ_y J_q(R, y);
- P(R | X) ∝ Π_q M_q(R);
- P(Y_q | X) ∝ Σ_R π_{−q}(R) J_q(R, y), where π_{−q}(R) = P(R | X_{−q}).

**Exact reuse identity (M7).** Y_q ⟂ X_{−q} given (R, X_q), so

P(Y_q | X) = Σ_R P(R | X) · P(Y_q | X_q, R)   for every q.

**One** episode-level vector P(R | X) therefore suffices to answer every question. It is 2 free numbers in T1 and 5 in T4. Numerical reconstruction error: 1.9e−15 (T1) and 1.0e−15 (T4).

There are two equivalent "reusable summary" choices:

- the episode-level P(R | X), identical for all q;
- the leave-one-out π_{−q}, different per q but free of X_q.

Both are exact. They differ in whether the question's own evidence is folded into the reliability estimate before or after.

### The quantities §2 asks to distinguish

| quantity | definition | type | dimension |
|---|---|---|---|
| reliability posterior | P(R \| X) or π_{−q} | episode / leave-one-out belief | 2 (T1), 5 (T4) |
| pairwise agreement | A_{q,g} = 1[x_{q,2g} = x_{q,2g+1}] | deterministic function of X_q | 3 bits per question |
| group coherence | n_g = Σ_q A_{q,g} (or the leave-one-out n_g^{−q}) | deterministic function of X | 3 counts |
| candidate evidence | P(Y_q = y \| X_q, R) for each R | per question, per R | K × \|R\| |
| final answer | argmax_y P(Y_q \| X) | — | — |
| confidence | max_y P(Y_q \| X) | — | — |

## 5. Sufficiency theorem (agreement counts)

**Theorem (S).** Assume uniform Y priors and the roles above (reliable / noise / colluding, with errors uniform over wrong candidates). Then

- P(R | X) is a function only of the group coherence counts (n_0, n_1, n_2);
- π_{−q} is a function only of the leave-one-out counts n^{−q}.

**Proof.**

1. Fix R. For a noise group, both reports are uniform and independent of Y, so the group contributes K^{−2} whatever the reports.
2. For a colluding group, it contributes K^{−1} · 1[A_{q,g} = 1].
3. For the reliable group G_g, summing over y with uniform prior gives Σ_y K^{−1} P(x_a | y) P(x_b | y). Write e = (1−α)/(K−1).
   - If x_a = x_b this equals K^{−1}[α² + (K−1)e²], which is independent of the reported value.
   - If x_a ≠ x_b it equals K^{−1}[2αe + (K−2)e²], which is independent of which two values were reported.
4. So M_q(R) depends on X_q only through A_q.
5. Hence Π_q M_q(R) depends on X only through the counts n_g. ∎

**Numerical check (M5).** Grouping 20,000 episodes by their count vector, the maximum deviation of π_{−q} within a count cell is:

- T1: 6.4e−15 over 210 cells;
- T4: 8.8e−15 over 90 cells.

**Closed forms (M9, α = 0.8, K = 3).**

- P(agree | reliable) = 0.66; P(agree | noise) = 1/3; P(agree | colluding) = 1.
- Likelihood ratio, reliable vs noise: LR_agree = 1.98, LR_disagree = 0.51.
- T1: P(R = g | X) = softmax_g(n_g · log(LR_agree / LR_disagree)) = softmax_g(1.357 · n_g). This is **linear in the counts followed by a softmax**.
- T4: P(R = (g,h) | X) ∝ 1[n_h = Q] · ρ^{n_g} (1−ρ)^{Q−n_g} · (1/3)^{n_k} (2/3)^{Q−n_k}. This contains a **hard constraint** (the colluder must agree on every question), so the map from counts to reliability is **non-monotone**.
  - Holding n_1 = 6 and n_2 = 2, P(G0 reliable) rises with n_0 from 0.06 (n_0 = 0) to 0.98 (n_0 = 5).
  - It then **drops to 0.50 at n_0 = 6**, because a group that agrees on every question is as likely to be the colluder.

So the Bayes-relevant summary is literally a **consistency (within-group agreement) statistic**. This is a fact about the task's sufficient statistic, not about any model.

## 6. Alternative sufficient statistics

The following are all exactly sufficient for every question's posterior:

- the counts (n_0, n_1, n_2);
- any invertible function of them, including P(R | X) itself (in T1 the map from counts to the softmax posterior is not invertible, because it depends only on count *differences*; it is still sufficient);
- the per-R summed log marginals Σ_q log M_q(R);
- the leave-one-out versions of each.

**What is not sufficient:**

- per-question histograms;
- per-source vote frequencies across the episode. Under symmetric Y these carry no information about R.

**Consequence for mechanism claims.** A learned model "uses consistency" in a well-defined sense if it carries *some* sufficient statistic of the counts. Which encoding it uses (counts, count differences, posterior, log-likelihood sums) is not identified by behaviour. A mechanism test must therefore target the **equivalence class**: variation in group coherence with the target question's own reports held fixed. It must not target one specific coordinate system.

## 7. Tasks T0–T5: sufficient statistics and simpler solvers (§3)

Simple solvers below are exact non-learned heuristics (M8, 20k episodes).

| task | what must be shared across questions | sufficient statistic | full Bayes | majority (6 votes) | most-coherent-group vote | coherent group, excluding perfect agreement | own-question Bayes |
|---|---|---|---|---|---|---|---|
| T0 | nothing (R known) | G0's votes | 0.800 | 0.636 | 0.739 | 0.701 | 0.800 |
| T1 | P(R \| X): which group is coherent | counts n_g, softmax-linear | 0.743 | 0.637 | **0.736** | 0.699 | 0.652 |
| T2 | as T1, plus confidence (max posterior) | as T1 | 0.743 | — | — | — | — |
| T3 | as T1; R also supervised (positive control) | as T1 | 0.743 | — | — | — | — |
| T4 | P(R \| X) with a hard colluder constraint | counts, non-monotone | 0.741 | 0.551 | **0.359** | **0.719** | 0.575 |
| T5 | nothing (all symmetric) | per-question histogram | 0.647 | **0.647** | 0.616 | 0.617 | 0.647 |

**Implications fixed before training:**

- **T1.** The heuristic "find the group whose members agree most across the episode, then use its votes" reaches 0.736, against 0.743 for exact Bayes. A model can almost solve T1 with a hard "most coherent group" selector. So a T1 mechanism need not be graded probabilistic reliability; near-ceiling accuracy does not distinguish the two.
- **T4.** The same heuristic is *worse than chance-adjusted majority* (0.359), because colluders are the most coherent group. Solving T4 requires a non-monotone use of coherence: exclude perfectly coherent groups. **T4 is therefore the decisive test that a model's consistency computation is task-appropriate, not a generic "agreement = good" bias.**
- **T0 and T5** are controls in which cross-question coherence has no Bayes value. In T0 it is useless because R is known. In T5 all sources are exchangeable and the histogram is sufficient. A model that still routes decisions through cross-question coherence in T0/T5 shows that the computation is not driven by usefulness. T0's coherence heuristic still helps (0.739 vs 0.636 for majority) only because reliability is *fixed*, so learning "always use G0" is better still.
- **T2** adds a calibration requirement. Confidence must equal max_y P(Y_q | X), which depends on the strength of the coherence evidence, not only on the selected group. This separates graded reliability from hard selection.
- **T3** directly supervises R. It is a positive control for identification methods. Its results are reported separately and are never used as evidence that unsupervised learning yields the mechanism.
