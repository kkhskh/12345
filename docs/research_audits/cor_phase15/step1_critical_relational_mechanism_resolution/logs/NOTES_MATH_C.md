# XIII-C C1 mathematics: two-colluder shared-copying posterior; sufficiency of m; per-group form (proofs, development notes)

> **Correction (math review 1, accepted).** The V4 figure of 4.4 points came from an unconverged estimator: 48 importance-weighted
> relabellings, median effective sample size 4.7. The exact LOCALQ posterior factorises over questions. Computed exactly
> (`reviews/review1_scripts/r1_localq_exact.py`, `r1_localq_large.py`), the gap is:
> - 2.8 points on the same 3000 episodes;
> - **3.6 ± 0.07 points on 100,000 episodes**, identical to XIII-B's number. XIII-B's censored LOCALQ is the same exact quantity, so the
>   statement that "the two estimators differ" was wrong.
>
> Corollary P is a statement about the posterior. That a gating mixture *needs* per-group format is an implementation hypothesis, which
> the supG/supL2 arms test; the mathematics does not imply it. Those two arms also differ in which tokens receive the features, so they are
> not a pure format contrast.
>
> At β = 0.6 every input has positive probability, so the J-pair "on-support" check is vacuous.


Generator (genlib.sample, G = 3, C = 2, K = 3, Q = 6, α = 0.8): a uniformly random group g is reliable and the other two groups H(−g)
collude. At each question independently:
- the reliable group's two sources each report Y_q with probability α, otherwise a uniformly chosen wrong value;
- with probability β all four colluder sources report one shared value c_q ~ U(K); otherwise each of the four reports uniformly and
  independently.

**Likelihood.** Given role g, the episode likelihood factorises over questions:

p(X | g) = Π_q [ relm_g(q)^{[q ≠ q*]} · rel_g(q, y)^{[q = q*]} · colH_{−g}(q) ],

where:
- relm_g(q) = (1/K) Σ_y p(x_{g1} | y) p(x_{g2} | y);
- colH_{−g}(q) = β · 1[all four sources of H(−g) equal at q] / K + (1 − β) / K⁴.

This is exactly `xii_bayes.posterior_shared`. **V1:** an independent direct enumeration written from the sampler semantics agrees to
2.8e-15 (β = 0.6) and 1.3e-16 (β = 1).

**Proposition S (sufficiency of m).** Fix the per-group statistics:
- each group's self-agreement pattern over questions;
- the values at q*.

Then the exact posterior p(Y_{q*} | X) depends on the remaining information in X only through m = (m_H) for the three pairs H, where
m_H = #{q ≠ q* : all four sources of H equal}.

*Proof.*
1. relm_g(q) depends only on whether g's two sources agree at q. For two distinct values the sum is the same for every pair of values;
   for equal values it is again the same for every value, by candidate symmetry.
2. rel_g(q*, ·) depends on g's values at q*.
3. colH_{−g}(q) takes one of two values according to alleq_{H(−g)}(q). So Π_{q ≠ q*} colH_{−g}(q) = A^{m_{H(−g)}} B^{5 − m_{H(−g)}},
   with A = β/K + (1 − β)/K⁴ and B = (1 − β)/K⁴.
4. The factor at q* depends on the values at q*.

Hence p(g | X) ∝ [Π_{q ≠ q*} relm_g(q)] · rel_g(q*) · colH(q*) · (A/B)^{m_{H(−g)}} · B⁵, and p(y | X) = Σ_g p(g | X) p(y | g, X_{q*}). ∎

**V2:** 2495 relabellings that keep m change the posterior by at most 0.0 (exact). The 1505 that change m change it by 0.048 on average.

**Corollary P (per-group form).** The log-weight of the hypothesis "g is reliable" is linear in m_{H(−g)}, the statistic of the pair NOT
containing g, with coefficient log(A/B) = log(1 + βK³/(1 − β)). At β = 0.6, K = 3 this is log(41.5) = 3.73.

A downstream mixture that gates by group therefore needs, at group g's tokens, the scalar m_{H(−g)}. The vector m = (m_01, m_02, m_12) is
the same at every token, and selecting the complementary component requires a group-dependent selection. The supG / supL2 contrast tests
exactly this.

**β = 1 (XIII-B proposition, re-verified, V3).**
- On-support, the colluding pair must have all four sources equal at every question.
- The posterior is constant within cells defined by the self-agreement bits and the values at q*: 3985 multi-episode cells, maximum
  within-cell spread 0.0.
- Cross-group comparisons at other questions are not needed.

**β = 0.6 (V4).**
- The orbit posterior given only the self-agreement bits and the values at q* (importance-weighted Monte Carlo over 48 relabellings)
  reaches accuracy 0.747.
- The full posterior reaches 0.792, a gap of 4.4 points.
- XIII-B's censored LOCALQ gap was 3.6 points. The two estimators differ (Monte Carlo orbit average vs XIII-B's censored computation); both
  show a positive gap.

**J-pairs (V5).**
- They keep every self-agreement bit and the values at q*.
- 100% of them have a finite posterior, i.e. they are on-support.
- 88.9% change m. The 11.1% that keep m have exactly zero exact change (by S), and form the J_D0 control.
