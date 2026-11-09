# Theorems and counterexamples — Investigation III

Labels: **[P]** proof given here; **[N]** exact numerical validation; **[E]** learned-model evidence; **[X]** counterexample;
**[O]** open proof obligation.  Earlier registries (atlas, continuation) are not restated; dependencies are cited.

## 1. TRANS: what identifies the maps and their composition (Workstream C1)

Setting (`src/tasks3.py`, continuation `tasks_ext`): σ, τ independent uniform on S₃ (K = 3); per question Y uniform; A
reports Y, B reports σ(Y), C reports τσ(Y), each source correct w.p. β (else uniform on the K−1 others); a question is AB
(A, B shown), BC (B, C shown) or C-only, uniformly.

**Lemma T1 (factorisation) [P].** The likelihood of the other questions factorises: P(X_{−q} | σ, τ) = L_AB(σ) · L_BC(τ),
where L_AB depends only on the AB questions and L_BC only on the BC questions.
*Proof.* Given (σ, τ) the questions are independent. In an AB question, P(x | σ, τ) = Σ_y K⁻¹ p_A(x_A | y) p_B(x_B | σ(y)),
free of τ. In a BC question, P = Σ_y K⁻¹ p_B(x_B | σ(y)) p_C(x_C | τσ(y)) = Σ_u K⁻¹ p_B(x_B | u) p_C(x_C | τ(u)) (substitute
u = σ(y), a bijection, Y uniform), free of σ. A C-only question gives Σ_u K⁻¹ p_C(x_C | u) = const. ∎
**Corollary T1a.** σ and τ are a-posteriori independent; σ is informed only by AB questions, τ only by BC questions, and
C-only questions carry no map information.

**Theorem T2 (composition needs both maps) [P].** If the posterior of σ (or of τ) is uniform on S₃, the posterior of the
composite τσ is uniform, so the Bayes C-only answer is at chance, however well the other map is known.
*Proof.* By T1a the composite is the group product of independent variables; if one factor is uniform on a finite group,
the product is uniform (translation invariance of the uniform measure). ∎
**Corollary T2a (entropy).** H(τσ | X) ≥ max(H(σ | X), H(τ | X)) (entropy of a convolution of independent group variables
is at least each factor's).  [N] at β = 0.9, (nAB, nBC) = (1, 1): H(σ) 1.47, H(τ) 1.49, H(τσ) 2.16 bits.

**Proposition T3 (coverage, noise-free) [P, N].** For β = 1, a single observed (a, σ(a)) pair leaves 2 of the 6
permutations of {0,1,2}; two distinct pairs identify σ. Hence C-only decisions need ≥ 2 distinct AB value pairs and ≥ 2
distinct BC value pairs for certainty; with exactly one of either, the composite is restricted to 2 candidates.

**Numerical validation [N]** (`logs/trans/identifiability.json`, 20,000 episodes, scored question q = 0): C-only Bayes
accuracy is at chance (0.29–0.45; cells of 70–850 episodes) whenever nAB = 0 or nBC = 0 (for every count of the other type), and rises with both
(e.g. β = 0.9: (1,1) 0.43, (2,2) 0.59, (3,2) 0.72; β = 1: (2,2) 0.78, (3,2) 0.875).

**Consequence for learning.** A network that ignores C reports in BC questions (as the continuation's 4000-step model did)
loses *all* τ information (Corollary T1a), so its C-only accuracy must be at chance — it cannot "partly compose".

## 2. O/D: weighted-voting coincidence (mandate §11)

Setting: S conditionally independent sources, symmetric errors, uniform Y, K ≥ 3 candidates; w_s = log(α_s(K−1)/(1−α_s));
Bayes score of c is Σ_{s: x_s = c} w_s; discrepancy D_c = #{s : x_s ≠ c}.
NSR (no strict reversal): for every report vector x and candidates c ≠ c′, D_c < D_{c′} ⇒ score_c > score_{c′}.
COND: w(P) < w(R) for all disjoint P, R ⊆ [S] with |P| < |R|.

**Theorem OD2 [P] (closes the continuation's open "only if").** For K ≥ 3, NSR ⇔ COND.
*Proof.* (⇐) Let D_c < D_{c′}. Let R = {s : x_s = c}, P = {s : x_s = c′}; they are disjoint, |R| = S − D_c > S − D_{c′} = |P|,
so by COND w(P) < w(R), i.e. score_{c′} < score_c. (⇒) Suppose COND fails: disjoint P, R with |P| < |R| and w(P) ≥ w(R).
Take x_s = c on P, x_s = c′ on R and a third value c″ on the rest (possible since K ≥ 3). Then D_{c′} = S − |R| <
S − |P| = D_c while score_c = w(P) ≥ w(R) = score_{c′}: a violation of NSR (with c′ in the role of the lower-discrepancy
candidate). ∎
Note: the condition |P| + |R| ≤ S of the continuation's statement is automatic for disjoint sets. For K = 2 the third value is
unavailable and the "only if" needs |P| + |R| = S only [O].
**Numerical validation [N]** (`logs/od_check.json`): 1200 random weight vectors (S = 4, 5, 6 with K = 3; S = 5 with K = 4),
including near-boundary draws: COND, NSR and the argmax-in-argmin-D property coincide in every case (no mismatch); the band
w_max/w_min < (m+1)/m is sufficient only (true in 325 of the 556 COND cases).
**What this says about O/D.** Unweighted discrepancy is exactly order-compatible with Bayes iff no smaller source set can
outweigh a larger one. The weights are the source semantics (accuracies); they are *not* identified by the discrepancy
itself, so the condition cannot be checked without the Bayesian model. Copying and zero-weight (noise) sources violate the
product-form assumption or COND respectively. This is a characterisation of when a renamed vote-count is the Bayes rule,
not a new decision theory (failure F29 unchanged).

## 3. Adaptive and dependent sources (Workstream F)

**Proposition AD1 (static counts are insufficient under change points) [P, N].** In ADAPT (`tasks3.py`), the posterior
depends on *which* questions each pair agrees in, not only on how many. [N] Among episodes with equal per-group agreement
counts and equal own reports, the exact posterior at q differs by up to 0.35 (median 0.04; 2,536 classes).
*Proof sketch.* The configuration likelihood Π_q h_{r_g(q, cfg)}(A_{q,g}) depends on the position of each agreement through
the question-dependent role r_g(q, cfg); two agreement patterns with equal counts but agreements before vs after a candidate
change point receive different likelihoods. ∎
**Proposition AD2 (camouflaged copier) [P, N].** A copier pair that copies a uniform common value w.p. ρ_c = (K a_rel −
1)/(K − 1) in every question has exactly the per-question law of an honest reliable pair marginalised over Y (atlas I2);
any adaptive strategy whose per-question copying probability is constant has the same law. [N] max |Δ| of the pair law =
2·10⁻¹⁷ at ρ_c = 0.49 (α = 0.8). Neural training cannot resolve it (statistical non-identifiability).
**Proposition AD3 (triggered copier: dependence makes the trigger partly identifiable) [N].** A copier that copies only
when the reliable pair disagrees (ρ_t = 0.9) has a different JOINT law with the reliable group than a static copier with
matched marginal agreement rate (ρ_s = 0.306); exact posterior P(triggered) averages 0.66 when true and 0.34 when false;
assuming the wrong mechanism costs ≤ 0.007 accuracy (`logs/adapt/theory.json`).  The decision-relevant evidence is the
cross-group co-occurrence, which no per-group count captures.
**Observation AD4 (posterior lag) [N].** After a change point, the exact posterior mass on the currently active reliable
group dips to 0.36 at the questions adjacent to the change and recovers to ≈ 0.5 two questions away (Q = 6, one change).

## 4. Extrapolation from support behaviour (Workstream D1)

**Proposition EX1 [P].** Let S be the support of the training generator and f, g two models with f = g on S. For any
off-support set U disjoint from S and any functions h₁, h₂ on U, the models f′ = f·1_S + h₁·1_U and g′ = g·1_S + h₂·1_U still
agree on S. So without restrictions on the model class, behaviour on S constrains nothing on U. ∎ (Trivial; stated because
every predictor of extrapolation must therefore use information beyond on-support behaviour: architecture, off-support
queries, or a class restriction.)
**Realised counterexample [E].** Transformers tf_s12 and tf_s10 have natural T4 KL 0.0019 and 0.0008 (essentially
identical on S) and p_c = 0 local indices 0.30 and 0.87.
**Corollary for predictors.** A predictor that uses only (a) the model's on-support input–output function cannot beat a
constant beyond what architecture correlates with; it must use internal structure (whose off-support meaning is not
guaranteed — §2.3 of STRUCTURE_AND_EXTRAPOLATION) or off-support queries (behavioural probes, which are then partly
tautological).

## 5. Counterexamples to unifications (Workstream E5) — see RELATIONAL_COMPUTATIONAL_CLASSES.md

Listed there with evidence; each is an explicit model/input pair, not a construction on paper, unless marked.

## 6. Open obligations [O]
- OD2 for K = 2.
- AD1 quantitative: which finite-dimensional statistic of the agreement history is sufficient (here: the agreement
  vector A ∈ {0,1}^{6×3} itself is; whether any lower-dimensional statistic is, is open).
- T3 for larger alphabets: coverage needed to identify a permutation of K symbols is K − 1 distinct pairs; the effect on
  the composite under noise is computed only for K = 3.
