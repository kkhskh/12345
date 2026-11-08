# Theorems, boundaries and impossibilities

**Status labels.**

| label | meaning |
|---|---|
| **[P]** | proved here, with a written proof |
| **[P+N]** | proved and checked numerically (the log is cited) |
| **[C]** | constructive: an explicit example establishes the claim |
| **[E]** | empirical statement about trained models |
| **[K]** | known in the literature; restated so that the dependencies are visible |

**Kinds of claim.** Bayes-level statements are about the generative model. Network-level statements are about trained checkpoints. The two are never merged.

**Code and logs.**
- `src/gen_family.py`: the extended generator family and exact Bayes.
- `scripts/bayes_boundaries.py` → `logs/bayes_boundaries.json`.
- `scripts/pp1_sign.py`, `scripts/pp2_shift.py`, `scripts/pp5*.py`.

## 0. Setting

**Episodes.**
- Q questions, K candidates, and six sources in three pairs (groups).
- A configuration R assigns each group a **role** r ∈ 𝓡 (reliable / noise / colluding, and variants).
- Roles are shared across all questions of an episode.
- Y_q is uniform on [K], and the questions are independent given R.

**Role-pair model.** Given role r and answer y, a pair of reports (x_a, x_b) has likelihood f_r(x_a, x_b | y). The extended family (`gen_family.Fam`) contains:

- **reliable:** f = (1−κ) p(x_a|y) p(x_b|y) + κ·1[x_a = x_b]·p(x_a|y), where p(x|y) = α if x = y, and e = (1−α)/(K−1) otherwise. With κ > 0 the reliable pair shares one report (correlated honest errors).
- **noise:** f = K⁻².
- **colluding (ρ):** f = ρ·K⁻¹·1[x_a = x_b] + (1−ρ)·K⁻². Colluders copy a common value, independent of y, with probability ρ.

**Configuration prior.** An episode contains a colluding group with probability p_c; otherwise it is a T1 episode.

**Original tasks.** T1 is (p_c = 0). T4 is (p_c = 1, ρ = 1).

**Agreement statistics.**
- A_{q,g} = 1[x_{q,2g} = x_{q,2g+1}] is the within-pair agreement of group g in question q.
- n_g = Σ_q A_{q,g} is the group's count.
- n_g^{−q} is the leave-one-out count.
- a_r = P(A = 1 | role r) is the role's agreement probability:
  - a_rel = κ + (1−κ)(α² + (K−1)e²);
  - a_noise = 1/K;
  - a_coll = ρ + (1−ρ)/K.

## 1. Sufficiency and reuse

### Theorem S (count sufficiency) [P+N]

**Assumptions.**
- **(U)** For every role r, Σ_y K⁻¹ f_r(x_a, x_b | y) depends on (x_a, x_b) only through 1[x_a = x_b].
- **(O)** At most one group per configuration has a y-dependent likelihood. This was added after review R1-a. With two reliable pairs, the likelihood depends on *whether they agree with each other*, so counts are not sufficient. Example: 0.1366 vs 0.0043.

**Statement.** Under (U) and the uniform Y prior:
- P(R | X) depends on X only through the counts (n_0, n_1, n_2);
- P(R | X_{−q}) depends on X only through the leave-one-out counts.

**Proof.**
1. The per-question marginal is M_q(R) = Π_g Σ_y K⁻¹ f_{r_g(R)}(·|y)... In fact M_q(R) = K⁻¹ Σ_y Π_g f_{r_g}(x_{q,g}|y). For the noise and colluding roles f does not depend on y, so it factors out.
2. Only the reliable group's term is summed over y. That sum is a function of 1[x_a = x_b] alone. When the pair agrees, every term is symmetric in the common value. When it disagrees, the terms are symmetric in the two distinct values.
3. Hence M_q(R) = Π_g h_{r_g}(A_{q,g}).
4. Therefore P(R | X) ∝ prior(R) Π_g Π_q h_{r_g}(A_{q,g}) = prior(R) Π_g a_{r_g}^{n_g}(1 − a_{r_g})^{Q − n_g}. ∎

**Check (B0).** Comparing the count-only posterior with the full posterior gives a maximum |Δ| of 2.1e−15, 2.2e−15 and 3.4e−15 in three extended settings: (ρ 0.8, p_c 0.5, κ 0.3), T4, and (ρ 0.6, p_c 0.2).

**Role of (U).** (U) is *used* by the proof. A failure of (U) can break count sufficiency, as in the counterexample below. It is not shown to be necessary in general (review R1-l).
- **Counterexample.** Suppose a noise source is biased (reports 0 with probability 0.6). Then the probability that a pair agrees on "0" differs from the probability that it agrees on "1". The value of the agreement then carries information about the role, so counts alone are not sufficient.
- **Status.** This is a counterexample by construction, not run.

### Theorem M (reuse identity) [P+N] (from the latent-reliability package)

**Statement.** P(Y_q | X) = Σ_R P(R | X) · P(Y_q | X_q, R).

**Consequence.** A single episode-level vector answers every question.

**Corollary J1 (what a relation summary does and does not determine).**
- (X_q, n^{−q}) is sufficient for the Bayes decision at q.
- The relation summary n^{−q} alone is not sufficient.
- **Witness.** Shift X_q's labels cyclically. n^{−q} is unchanged and the Bayes argmax shifts with them: the same relation gives the opposite preference.
- **What is missing.** The candidate-specific evidence X_q.
- **Is it accessible to the model?** Yes: the model receives X_q directly.

**Count-sufficiency in the models [E].**
- Exact Bayes changes its argmax under count-preserving resampling of X_{−q} in 0.5% of questions (only through ties). All T4 Transformers change theirs in 3.6–5.8% (`logs/mech/s*.json`, `input_invariances`).
- So the models are approximately but measurably not count-sufficient.

## 2. How agreement maps to reliability: shape and phase boundary

### Theorem U (quasi-concavity of role posteriors) [P+N]

**Assumptions.**
- (U) holds.
- The roles have per-question agreement probabilities a_1 < a_2 < … < a_J.

**Statement.** Fix a group g and the other groups' counts. Then:
- the posterior probability that g has role j, as a function of g's count n ∈ {0, …, M}, is **quasi-concave** (unimodal);
- it is non-decreasing for the role with the highest agreement probability;
- it is non-increasing for the role with the lowest agreement probability.

**Proof.**
1. By Theorem S, P(role_g = j | n, rest) ∝ W_j · a_j^n (1 − a_j)^{M−n}. Here W_j collects the prior and the other groups' likelihoods summed over the configurations that give g role j. W_j does not depend on n.
2. Write 1/P_j − 1 = Σ_{i≠j} (W_i / W_j) exp(n·s_{ij} + const), with s_{ij} = log(a_i/a_j) + log((1−a_j)/(1−a_i)).
3. s_{ij} has the sign of (a_i − a_j). A sum of exponentials of affine functions is convex in n, so 1/P_j − 1 is convex. Hence P_j is quasi-concave.
4. For j = J every s_{iJ} < 0, so 1/P_J − 1 is decreasing and P_J is increasing. For j = 1 the same argument runs in reverse.
5. **Degenerate case.** a_J = 1 (ρ = 1) means L_J = 0 for n < M. That is the limit, with a single jump at n = M.

**Check (B1).** 76,320 (setting × others-counts) rows over the grid ρ ∈ {1, …, 0.5}, p_c ∈ {0, …, 1}, α ∈ {0.6, 0.8, 0.95}, K ∈ {2, 3, 5}, Q ∈ {3, 6, 10}, κ ∈ {0, 0.3}: **0 violations**.

**Interpretation.** This is a statement about the *reliability posterior*. Its effect on candidate margins goes through Theorem D, not directly (review R1-l). For the *reliable* role, which sits between noise (a = 1/K) and copying/collusion (a_coll), the response to more agreement is
- positive below a peak n*;
- negative above it;
- never with more than one turn.

The peak exists inside [0, M] only if some role with a *higher* agreement probability than honest sources has positive prior mass. That role can be colluders with a_coll > a_rel, or copiers.

**Peak and boundary condition [P].** For the roles {noise, reliable, colluding} with b < a < c, write ρ_X(n) = L_X(n)/L_rel(n).

P(rel) drops from n = M−1 to n = M exactly when

W_C [ρ_C(M) − ρ_C(M−1)] > W_N [ρ_N(M−1) − ρ_N(M)],

where ρ_N is decreasing and ρ_C is increasing.
- When ρ = 1 (so c = 1), ρ_C(M−1) = 0 and ρ_C(M) = a^{−M}. The condition becomes W_C > a^M · W_N (ρ_N(M−1) − ρ_N(M)).
- W_C is proportional to p_c times the other groups' likelihood under configurations that make g the colluder.

### Corollary U2 (no reversal without a higher-agreement role) [P]

**Statement.** If no role with positive prior mass has a higher per-question agreement probability than the reliable role (in particular, if a_coll ≤ a_rel), then P(g reliable | counts) is non-decreasing in g's count in **every** context, whatever p_c.

**Proof.** The reliable role is then the highest-agreement role, so the monotone-extreme case of Theorem U applies.

**Credit.** Suggested by review R1-h.

**Consequence.** Reversal requires a role that agrees *more* than honest sources. Independent colluders with ρ ≤ ρ* do not suffice; ρ* = 0.49 for α = 0.8, K = 3, κ = 0.

### Phase diagram (B2) [N]

Settings: K = 3, Q = 6, α = 0.8, κ = 0. "Drop" is the fraction of naturally occurring contexts in which P(rel) falls at full leave-one-out agreement.

| ρ | a_coll vs a_rel = 0.66 | p_c = 0 | p_c = 0.05 | p_c = 0.2 | p_c = 0.5 | p_c = 1 |
|---|---|---|---|---|---|---|
| 1.0 | 1.0 > 0.66 | 0 | 0.99 | 0.99 | 0.98 | 0.66* |
| 0.9 | 0.93 | 0 | 0.99 | 0.99 | 0.98 | 0.97 |
| 0.8 | 0.87 | 0 | **0** | 0.97 | 0.95 | 0.91 |
| 0.7 | 0.80 | 0 | 0 | 0.96 | 0.96 | 0.93 |
| 0.5 | 0.667 ≈ 0.66 | 0 | 0 | 0 | 0 | 0.36 |

\* At ρ = 1 and p_c = 1, 29% of contexts are undefined: the other groups' counts make every configuration impossible (review R1-i).

The table reports the **incidence** of a drop, not its magnitude. At ρ = 0.5 and p_c = 1 the mean drop is 0.003 (review R1-k).

**Boundary statements, each with its domain:**
1. **No colluder prior (p_c = 0):** the agreement → reliability response is monotone increasing in every context. This is the T1 regime.
2. **High ρ:** a tiny collusion prevalence, p_c = 0.05, already reverses the response at full agreement in ≈ 99% of contexts. Perfect agreement is rare for honest pairs (0.66⁵ ≈ 0.125) but certain for colluders.
3. **Lower ρ:** reversal needs more prevalence. At ρ = 0.8 there is no drop at p_c = 0.05, but there is one at p_c = 0.2.
4. **When colluders agree about as often as honest pairs (ρ = 0.5):** the drop largely disappears. Agreement no longer separates the two roles; see §4, impossibility I2.

**Answer to mandate C1** (is the T4 non-monotonicity specific to the original generator?):
- It is **not** specific to ρ = 1. It is **stable across a region**.
- It **disappears** as a_coll → a_rel, or as p_c → 0.
- It is **conditional on the other groups' counts**. With ρ = 1 and counts (6, 2, 2), group 0 must be the colluder.
- **Partial collusion (ρ < 1)** smooths the step into an interior-peak response.

## 3. Decision consequences

### Theorem D (exact margin-change identity) [P+N]

**Setup.** Let an edit change only X_{−q}, so X_q is fixed. Let π and π′ be the leave-one-out configuration posteriors before and after the edit, and r(R) = π′(R)/π(R).

**Assumption.** Absolute continuity: π(R) = 0 implies π′(R) = 0.

**Statement.** Write m_{c,c′} = log P(c | X) − log P(c′ | X), and π^c(R) ∝ π(R) · P(X_q, Y_q = c | R). Then

Δm_{c,c′} = log E_{π^c}[r] − log E_{π^{c′}}[r].

**Proof.**
1. P(c | X) ∝ Σ_R π(R) J_q(R, c).
2. After the edit it is ∝ Σ_R π(R) r(R) J_q(R, c) = (Σ_R π J(c)) · E_{π^c}[r].
3. The normaliser cancels in the margin. ∎

**Check (B3).** On natural single-report edits, the maximum error is 1.4e−15 to 1.6e−15 over about 800 cases. About 2.5% of edits violate absolute continuity: they create support for a configuration that was impossible before. For those, the identity is replaced by a direct computation.

**Corollary D1 (sign rule, general form only).**
- Agreement elsewhere raises c over c′ if and only if E_{π^c}[r] > E_{π^{c′}}[r]. That is: the configurations the edit makes more probable must be, on average, more compatible with Y_q = c than with Y_q = c′.
- **Inputs.** The likelihood ratios r (generative model) and the candidate-conditioned configuration posteriors (generative model plus X_q). Neither uses the network's logits.
- **Withdrawn after review R1-b.** The earlier "specialisation" said that g's agreement gain helps g's candidate if and only if P(g reliable) rises more than P(g colluding). It is false as worded: the sign was wrong in 27,098 of 27,100 cases at (ρ 0.7, p_c 0.2, κ 0.3), and in 3.8% of T4 cases.

**Corollary D2 (no universal sign) [P].**
- The same relational change can have opposite Bayes-optimal margin effects under two generative models that share every other parameter: raise a group's count from 4 to 5 under T1 versus under T4.
- **Check.** On PP1's boundary edits, the T1 plug-in and T4 Bayes agree in sign only **16.6%** of the time (`logs/pp1_sign.json`).
- So any sign theorem stated in terms of agreement alone, without the source semantics, is false. That includes the original COR claim that lower discrepancy means higher support.

### Empirical counterpart: the decision rule transfers across implementations (PP1, prospective, passed) [E]

**Test.** 45,000 natural edits; 15,855 scored, with |Δm_Bayes| > 0.5.

| measure | result |
|---|---|
| sign agreement between model Δm and the T4-Bayes Δm | 99.4–100% in all 10 T4-capable models (frozen 1500-episode re-run: 99.6–100%; boundary 97.1–99.9%, cluster CIs ≥ 0.96; unfiltered boundary 0.79–0.93 vs 0.20–0.39 for the monotone plug-in) |
| at the perfect-agreement boundary | 96.4–100% |
| at the boundary, agreement with the monotone plug-in | 13–17% |

The 10 models are 7 Transformer curriculum seeds and 3 DeepSets seeds. They span every implementation observed:
- key-tag-and-route;
- the split key/query form;
- three-head redundancy;
- direct last-layer rereading;
- explicit pooling.

The decision-level prediction is therefore implementation-independent **on the support of the training distribution**. That support restriction is necessary: see PP2 and §5.

**Off-support edits.** 13,800 of the 45,000 edits break the colluder's agreement, which gives zero generator probability (Bayes undefined). Models still respond to them, with |Δm| ≈ 0.5–1.3. No Bayes prediction exists there. They are reported as stress tests, not as counterfactuals.

## 4. Identifiability and impossibility results

**I1 (noise re-pairing is unidentifiable and irrelevant) [P+N].**
- **Statement.** Under any configuration, the joint law of X is invariant under every permutation of the noise sources' positions. In particular it is invariant under any re-pairing of the noise sources.
- **Proof.** Noise reports are i.i.d. uniform and independent of everything.
- **Check.** Exact |Δ log-lik| = 0.0 between two such pairings.
- **Consequences.** With unknown groups, the pairing of noise sources cannot be recovered by any estimator. It is also decision-irrelevant, because noise sources carry no information about Y. Only the informative groups' pairing matters.

**I2 (honest copying vs collusion is unidentifiable when agreement rates match) [P+N].**
- **Statement.** Suppose a_rel(κ) = a_coll(ρ) and the honest pair's agreed value is marginally uniform, as it is under uniform Y. Then the episode distribution is invariant under exchanging the reliable and colluding roles.
- **Proof.**
  1. Both groups produce "agree with a uniform common value, with probability a; otherwise an ordered pair of distinct values", i.i.d. over questions.
  2. Y is never observed.
  3. The reliable group's correlation with Y is therefore invisible in X. ∎
- **Check.** |Δ episode log-lik| ≤ 1.4e−14 in two settings: (κ = 1, ρ = 1) and (κ = 0.5, ρ = 0.745).
- **Bayes accuracy collapses** to 0.566 and 0.586, against own-question 0.557 and 0.577 and the identifiable T4 value 0.743.
- **Consequence (restated after review R1-j).** No estimator, and so no network, can identify *which* of the two groups to trust in that regime. Bayes then averages both groups' evidence, and its residual gain over own-question is small: 0.566 vs 0.557. The same unidentifiability holds with independent honest errors (κ = 0) at ρ* = 0.49 (review R1-h). A failure there is statistical, not neural. This also bounds the phase diagram: as ρ → ρ*(κ), the non-monotone response must vanish.

**I3 (histogram bottleneck) [P+N]** (latent-reliability package).
- **Statement.** No function of the per-question vote histograms exceeds 0.636 on T1. Bayes reaches 0.743.

**I4 (unknown pairing costs information, but not always) [N].**

| task | known pairing | unknown pairing | own-question |
|---|---|---|---|
| T1 | 0.741 | 0.714 | 0.651 |
| T4 | 0.743 | 0.743 | 0.576 |

- In T4 the colluder's perfect agreement reveals its pair, and the reliable pair is then mostly determined. Unknown grouping costs almost nothing there.
- In T1 the reliable pair has to be found from excess agreement, which is a noisier signal.
- **Learnability is separate:** see `GENERALIZATION_AND_EMERGENCE.md` (PP4).

**I5 (head count is not a functional invariant) [C, exact].**
- **Statement.** For any attention layer, replace a head h by two heads with h's W_Q, W_K, W_V and W_O split as (W_O/2, W_O/2). The resulting network computes the same function.
- **Proof.** The output is linear in W_O given the pattern, and both copies have the same pattern.
- **Instance.** Applied to replication seed 11 (`scripts/pp5b_exact_split_posthoc.py`):
  - outputs are identical (max |Δp| 2.4e−7, argmax agreement 100%);
  - the preregistered R3 criterion (freeze the top-2 heads by group-mass R²) flips from **pass to fail**, with route-frozen IIA 0.21 → 0.58.
  - **Withdrawn after review R2-c:** the earlier claim that Shapley shares aggregated by pattern class are invariant. They are not conserved under the split, because the metrics are nonlinear.
- **Consequence.** Any criterion that counts or selects heads is not a property of the computed function. Seed 10's R3 failure is the natural version of this.
- **Related failure.** The *near*-preserving variant (overwriting an existing "inert" head) failed its frozen prediction PP5: 7% of argmaxes changed.

**I6 (query/key localisation not identifiable from input–output behaviour): CONJECTURE, downgraded after review R2-e.** The construction below ignores the pre-attention LayerNorm. LN makes scores nonlinear in the encoded vector, and a reliability vector left in the readout token also changes the final LN. A norm-compensating construction on the finite input set is plausible but has not been built. The original text follows.
- **Construction.**
  - Take a last-layer head whose score on source s is β·t_{g(s)}, where t is a group reliability tag.
  - Network A writes t into the source tokens (keys) and uses a constant query.
  - Network B writes a reliability vector π into the readout token (query) and group-identity features into the keys, with W_Q W_Kᵀ chosen so that q·k_s = β·t_{g(s)}.
- **Claim.** The scores, and therefore all outputs, are identical for every input.
- **Consequence.** Input-level interventions, PP1-type decision tests included, cannot distinguish A from B. Only internal interventions can, and their verdict depends on which representation is patched.
- **Status.** Constructed on paper, not implemented. The learned seeds 0–1 versus seed 2 are a natural, approximate instance (`latent_reliability/logs/posthoc/joint_*`).

**I7 (vacuity of unconstrained alignment) [K].**
- **Statement.** With arbitrarily complex alignment maps, causal abstraction can align any network to any algorithm, even untrained networks (arXiv:2507.08802, the "non-linear representation dilemma").
- **Consequence for this work.** Every localisation claim here is relative to a fixed class: linear subspaces of dimension ≤ 5–10 at named sites. A null under that class does not exclude a nonlinear alignment. A positive result is meaningful only because the class is restricted.

## 5. What the theorems do not cover (stated limits)

**Group size.**
- With pairs replaced by groups of m ≥ 3, the per-question sufficient statistic is the *partition type* of the group's reports, not a single agreement bit. Theorem S generalises: under (U) the likelihood depends only on the partition type.
- Theorem U does not generalise directly: the statistic is multi-dimensional. **Open.**

**Non-uniform errors.** These break (U), and with it count sufficiency (§1). The sufficient statistic must then include values. **Not analysed.**

**Adaptive collusion** (colluders who sometimes agree with the reliable group to blend in) and **reliability drift**: **not analysed.**

**From Bayes to the models.** Every network-level transfer rests on the measured on-distribution match (PP1). Off the support, the networks implement **different** extrapolations from the Bayes plug-in (PP2), so no Bayes-level theorem transfers there.
