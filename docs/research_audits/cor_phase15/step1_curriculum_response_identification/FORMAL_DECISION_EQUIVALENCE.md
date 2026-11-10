# Formal decision equivalence: corrected Theorem C3, approximation bounds, counterexamples

Grades: [P] proved here, [N] numerically checked (`logs/dev/`, `scripts/bounds_check.py`), [E] elementary (stated for
completeness, not claimed as new). Setting: T4 decision q, K = 3 candidates, x = (x_q, x_{-q}), V = n^{(-q)},
p*(·|x) the exact posterior. A model M outputs logits ℓ_M(x) ∈ ℝ^K.

## 1. Output spaces and gauges

- **Centred logits** c(ℓ) = ℓ − (1/K)Σ_y ℓ_y. Softmax is invariant to adding a constant, so c(ℓ) is the unique gauge-fixed
  representative; c(log p) = centred log-probabilities.
- **Margins** m_{yy′}(ℓ) = ℓ_y − ℓ_{y′} = c_y − c_{y′} (a linear image of c; c is recovered from the margins).
- **Probabilities** softmax(ℓ). **Temperature-calibrated** probabilities softmax(ℓ/T).
- The Bayes reference in the same gauge is c* = c(log p*). The model's **log-odds error** is e_M(x) = c(ℓ_M(x)) − c*(x).

## 2. Four equivalences (domain D, a set of inputs; ≡ means equality for every x ∈ D)

| name | definition |
|---|---|
| classification | argmax ℓ_M(x) ≡ argmax ℓ_{M′}(x) |
| posterior | softmax ℓ_M(x) ≡ softmax ℓ_{M′}(x) |
| log-odds | c(ℓ_M(x)) ≡ c(ℓ_{M′}(x)) |
| computational (relative to a declared intervention family I and mediator set 𝓜) | for every (x, ι) ∈ I and every mediator Z ∈ 𝓜, the interventional values Z_M(ι(x)) and Z_{M′}(ι(x)) correspond under a fixed alignment, and the outputs agree in log-odds |

**Lemma 1 [P].** Posterior equivalence ⇔ log-odds equivalence (softmax is injective on centred logits). Calibrated-posterior
equivalence (softmax(ℓ_M/T_M) ≡ softmax(ℓ_{M′}/T_{M′})) ⇔ c(ℓ_M)/T_M ≡ c(ℓ_{M′})/T_{M′}: equal log-odds **up to a model-specific
scale**. Classification equivalence is strictly weaker than both.

## 3. Corrected Theorem C3

Investigation IV's C3 ("two exactly Bayes-optimal models with exact abstract intervention equivalence have identical
decision-response functions on natural support") is false if "Bayes-optimal" means optimal classification
(Counterexample 1). The correct statement:

**Theorem 1 (decision-response equality) [P].** Let D be the natural support. Suppose
(a) *posterior exactness on D:* c(ℓ_M(x)) = c*(x) for every x ∈ D (equivalently softmax ℓ_M = p*);
(b) *pointwise interchange exactness:* for a set Π of pairs (x, x′) with x, x′ ∈ D, x′_q = x_q, the abstract intervention
    do_M(V := V(x′)) applied to x yields c(ℓ_M) = c(ℓ_M(x′)) exactly, for every pair in Π (not on average);
and the same for M′. Then for every (x, x′) ∈ Π,
  Δ_M := c(ℓ_M(do(x; V(x′)))) − c(ℓ_M(x)) = c*(x′) − c*(x) = Δ_{M′}.
In particular Δ_M = Δ_{M′} = 0 on π-preserving pairs (p*(x′) = p*(x)).
*Proof.* By (b) Δ_M = c(ℓ_M(x′)) − c(ℓ_M(x)); by (a) this is c*(x′) − c*(x). ∎
The statement concerns centred logits (equivalently margins); it holds for raw probabilities as well, but **not** for
temperature-calibrated equivalence: with (a) replaced by softmax(ℓ_M/T_M) = p*, Δ_M = T_M (c*(x′) − c*(x)), which is still 0 on
π-preserving pairs but differs by the factor T_M/T_{M′} on Bayes-relevant pairs.

**Corollary 1 (the input-level identity) [P].** For a natural input-level pair (x, x′) (no abstract intervention; both in D),
  Δ_M^{input}(x, x′) = c*(x′) − c*(x) + e_M(x′) − e_M(x).
On a π-preserving pair the first term vanishes: **every input-level Bayesian-null response is exactly a difference of the
model's own log-odds errors**, Δ_M = e_M(x′) − e_M(x). Consequently the cross-model correlation of input-level null
responses is the correlation of *error increments*; it measures how alike the models' approximation-error functions are along
π-preserving directions, and nothing else.

**Approximate version (Lemma 2) [E].** If ‖e_M(x)‖_∞ ≤ η_M on the inputs involved and the abstract intervention misses the natural
counterfactual by ‖c(ℓ_M(do(x; V(x′)))) − c(ℓ_M(x′))‖_∞ ≤ ζ_M, then
  ‖Δ_M − (c*(x′) − c*(x))‖_∞ ≤ 2η_M + ζ_M,  ‖Δ_M − Δ_{M′}‖_∞ ≤ 2(η_M + η_{M′}) + ζ_M + ζ_{M′}.
On π-preserving input pairs (ζ = 0) |Δ_M| ≤ 2η_M. The bound is elementary; what matters is which notion of "approximately
Bayes" supplies η (Section 4).

## 4. Which approximation notions bound Bayesian-null responses

| notion of closeness to Bayes | bounds \|Δ_M\| on null pairs pointwise? | reason |
|---|---|---|
| sup log-odds error η | **yes**: ≤ 2η | Lemma 2 |
| pointwise probability error ‖p_M − p*‖_∞ ≤ ε | only with a floor p* ≥ p_min > ε: \|e\| ≤ log((p_min+ε)/(p_min−ε)) + same for the max term | log is unbounded at 0 |
| total variation, pointwise | as the previous row (TV ≥ ε/2 per coordinate) | |
| KL(p* ‖ p_M) ≤ κ on average | **no** (Counterexample 3) | KL controls errors only in p*-weighted average; a set of small measure may carry arbitrary errors |
| average TV, accuracy | **no** (Counterexamples 1, 3) | |
| average log-odds error E\|e\| ≤ η̄ | in expectation only, and only with a density-ratio bound: E_Π\|Δ_M\| ≤ (1 + ‖dQ′/dP‖_∞) η̄ where Q′ is the law of x′ under the edit | the edit maps P to Q′ ≠ P |

**Proposition 1 [P].** For any κ > 0 and any B > 0 there is a model with E_x KL(p*(x) ‖ p_M(x)) ≤ κ, Bayes-optimal argmax on all
of D, and |Δ_M| ≥ B on a set of π-preserving pairs of positive probability.
*Proof (construction).* Let S be a set of episodes of probability δ that are endpoints of π-preserving pairs (S has positive
probability: 81 % of decisions admit such an edit). Set ℓ_M = log p* + B·1[x ∈ S]·(e_{ŷ} − 1/K), ŷ = argmax p*: the argmax is
unchanged, Δ_M = ±B c(e_ŷ) on pairs crossing S, and KL ≤ δ·B·max(1 − p*_ŷ) → choose δ ≤ κ/B. ∎

## 5. Counterexamples

1. **Scale (classification ≠ log-odds) [P].** ℓ_M = 2 log p*: Bayes-optimal argmax everywhere, Δ_M = 2Δ* on Bayes-relevant pairs.
   It still gives Δ_M = 0 on null pairs — scale alone cannot produce null responses.
2. **Argmax-preserving count bias [P].** ℓ_M = log p* + β(n_0 + n_1 + n_2)·(e_ŷ − 1/K): perfect classification, null responses
   β·(Δ total count)·c(e_ŷ) ≠ 0 on every null edit (which always changes the total count by ±2). A model with exact Bayes
   classification can therefore have large, perfectly structured null responses; two such models with the same β have
   null responses correlated at r = 1 with no shared mechanism beyond "add a count-dependent bonus to the winner".
3. **Small KL, large null responses [P]:** Proposition 1.
4. **Average vs pointwise interchange [P].** An abstract intervention can match the natural counterfactual on 90 % of pairs (IIA 0.9)
   and differ arbitrarily on the remaining 10 %; Theorem 1 then fails on that 10 %, which may contain all π-preserving pairs.
5. **Off-support abstract values [P].** For v′ with P(x_q, v′) = 0, condition (a) is vacuous; two exactly Bayes-optimal models may
   respond differently (Investigation IV, D2).
6. **Temperature [P].** Two calibrated models with T_M ≠ T_{M′} have unequal Bayes-relevant responses and equal (zero) null responses.

## 6. Consequences used in this investigation

- The primary null-response statistic is computed on **input-level** π-preserving pairs, where Corollary 1 makes it an exact
  statement about error increments; abstract interventions are secondary (they add ζ).
- Pearson correlation of centred-logit responses is invariant to a per-model scale (temperature), so "robust to calibration" is
  automatic for that statistic; it is not invariant to non-linear calibration or to the choice of probabilities as the output
  space, which are therefore reported separately (BAYES_NULL_RESPONSE_STRUCTURE.md §2).
- Correlated null responses cannot by themselves distinguish a shared algorithm from a shared bias (Counterexample 2); the
  operator-family analysis (APPROXIMATION_VS_ALGORITHM.md) is the test that can.
