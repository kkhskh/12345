# XIII-C B1: mathematical identifiability of the decision-law parametrisations (development notes; proofs)

> **Corrections (math review 1, accepted).**
> - **T3:** under the exact weights every on-support episode has ≥ 2 groups with non-zero weight. The exact cancellation of ρ never occurs
>   on-support. The "6.3% single-weighted" figure uses a 1% weight threshold, and it is a statement about weak identification, not
>   cancellation.
> - **T2 scope:** T2 holds for ANY fixed weights w, not only the Bayes weights. It needs every group to have the same number of sources at
>   q and wrong answers to be uniform. It does not hold for the HYB/FULL families when γ ≠ 0. By T2, the parameter labelled "COMB ρ" in the
>   B5 Shapley can equally be read as "q-evidence tempering"; it does not identify a combination mechanism.
> - **Misspecification:** ρ is a property of (model, family, loss, fitting distribution).
> - **E/A1/A2:** "E is the only rule correct on all four families" is a statement against A1 and A2 only. Graded eligibility rules were not
>   fitted, and rules that agree with E on every on-support input cannot be distinguished from it at all.


Setting: the C = 1, β = 1 T4 generator (G = 3, K = 3, Q = 6, α = 0.8). At the target question q, each group has exactly two sources. Write
b = (1 − α)/2 and r = α/b (= 8). For group g and candidate y let k_g(y) ∈ {0, 1, 2} be the number of g's sources reporting y at q.
The candidate likelihood ratio used by Bayes is L_g(y) = c_{g1}(y) c_{g2}(y) / noi with c(y) = b · r^{[x = y]}, so L_g(y) = (b²/noi) r^{k_g(y)}.

Power-mean family (XIII-B, corrected nested form): log p(y) = (τ/ρ) log Σ_g w_g L_g(y)^ρ + const(x), with w from W, f.

**T1 (tempering identity).** PM(ρ, τ, w) is identical, as a function of the input, to "mix the ρ-tempered evidence L_g^ρ with weights w, then
raise the output to τ/ρ". *Proof:* (τ/ρ) log Σ w L^ρ = (τ/ρ) · log M̃(y) with M̃ = Σ w L̃, L̃ = L^ρ. ∎ (numerical max diff 1.3e-15)

**T2 (reliability identity).** For every ρ > 0, PM(ρ, τ, w) is identical to the exact-form mixture computed with a misstated per-source
reliability α′ at the target question, followed by the output exponent τ/ρ, where α′/b′ = (α/b)^ρ, b′ = (1 − α′)/2.
*Proof:* L_g(y)^ρ = (b²/noi)^ρ r^{ρ k_g(y)}. Under α′, L′_g(y) = (b′²/noi) r′^{k_g(y)} with r′ = r^ρ. Hence L_g^ρ = κ L′_g with
κ = (b²/noi)^ρ / (b′²/noi), the same for every group and candidate (every group has exactly two sources at q). Then
Σ_g w_g L_g^ρ = κ Σ_g w_g L′_g and (τ/ρ) log Σ w L^ρ = (τ/ρ) log Σ w L′ + const. ∎
(α′(0.6) = 0.635, α′(0.3) = 0.483, α′(0.9) = 0.765; numerical max diff 1.8e-15.)
Consequence: "a different combination operator (ρ < 1)" and "the correct mixture operator applied to under-confident candidate evidence at q,
then a sharpened output" are the SAME input–output function. No observational input–output experiment can distinguish them. The reliability
W from the other questions is unaffected (it enters only the weights), so the equivalent description is specifically *miscalibration of the
target-question evidence*, not of reliability in general.

**T3 (where ρ is identified at all).** If one group carries all the weight, log p = (τ/ρ) log L_g^ρ = τ log L_g: ρ cancels exactly
(numerical 2e-16). ρ is identified only by episodes where ≥ 2 groups have non-negligible weight and non-proportional L_g. Under the natural
distribution and the exact weights, 6.3% of episodes have one weighted group (w > 0.01), 88.7% two, 5.0% three.

**T4 (lapse confound, numerical).** p = (1 − λ) p_B + λ/3 is not in the PM family. The KL-best PM fit to it has ρ = 0.969 (λ = 0.01),
0.919 (λ = 0.03), 0.809 (λ = 0.10). A lapse floor therefore moves the fitted exponent downward. A fitted ρ < 1 is not by itself evidence of
a combination difference unless lapse is fitted jointly (PML family).

**T5 (output temperature).** p ∝ p_B^τ is in the family with ρ = 1; the fit recovers ρ = 1.000 for τ = 0.7 and 1.5. Pure calibration is not
confounded with ρ.

**T6 (E / A1 / A2 are separable on-support).** Under the generator, on-support edits exist that change exactly the following eligibility
indicators (counts on 6000 natural episodes):
- relational edit n_h 5 → 4 of a group agreeing at q (another all-six-agreeing group kept): changes E and A1, not A2 (n = 1032);
- relational edit n_h 4 → 3: changes A2 only (n = 1313);
- flip of agreement at q of an n_h = 5 group: changes E and A2, not A1 (n = 1308);
- flip at q of an n_h = 4 group: changes A2 only (n = 2136).

No single family separates all three. The four families jointly do: E is the only rule whose predicted response is correct on all four.

**Misspecification caveat (new, from fair_fits).** When no family member reproduces a model well, the "best" ρ depends on the scoring rule.
- uP161 s22: ρ = 0.56 under XIII-B's MSE on centred log-probs, versus 0.71 (bootstrap 0.69–0.75) under KL.
- The FULL family explains only about 20% of that model's KL from Bayes.

ρ is therefore a property of (model, family, loss), not of the model alone.
