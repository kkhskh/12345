# Formal equivalence and theorems

> **Post-review [R].** Corrections FE1–FE13 (A3 formula, B2/D3 scope, D4 completed, D5 restated, C2 sufficiency only, C3 conditional, definitions, E-Comp). Pre-review text: commit 8c6363e; adjudication: REVIEW_ADJUDICATION.md.

Grades: **[P]** proved here; **[N]** exact numerical verification (`logs/info_theory.json`); **[L]** learned-model instance
(confirmation models unless marked cal). Notation: one decision q of a T4 episode X = (X_q, X_{-q}); latent R = (g, h) ∈ 𝒞
(6 configurations: group g reliable, group h colluding); A_{k,g} = [x_{k,2g} = x_{k,2g+1}]; V = n = n^{(-q)} ∈ {0..5}³,
n_g = Σ_{k≠q} A_{k,g}; π = P(R | X_{-q}).

## 1. Objects

- **Model and site.** A transfer model M computes logits ℓ_M(X) = h_M(Z_M(X), W_M(X)), where Z_M is the residual stream at
  the input of layer 2 at a set of positions S_M(q) (the *site*) and W_M is the rest of that layer's input. Sites:
  P-count S_P(q) = {36+q}; T1 S_T(q) = {6q, …, 6q+5}.
- **Abstraction.** α_M : Z → V (decoder; multinomial logistic, fitted on natural episodes). **Implementation map** E_M : V → ℝ^{|S|×d}
  (affine, least squares from generator features to activations on natural episodes; fitted without any intervention outcome).
- **Abstract intervention.** do_M(V := v′) acts on x by Z_M(x) ↦ Z_M(x) + E_M(v′) − E_M(v(x)), all other computation recomputed.
  Its result is the **abstract response** g_M(x, v′) = ℓ_M under this intervention; g_M(x, v(x)) = ℓ_M(x).
- **Input-level intervention families** I (pairs x → x′ with x′_q = x_q): cond (X_{-q} resampled from P(X_{-q} | X_q)), single-group
  edit (one A_{k,g} toggled, k ≠ q), π-preserving edit (two components shifted equally, π unchanged), relabel (all counts unchanged).
  Each pair carries support labels: marginal support P(x′) > 0, same world P(x′ | R_true) > 0.
- **Abstract intervention families** J: natural-support (P(X_q, n′) > 0, i.e. some h with n′_h = 5 and A_{q,h} = 1),
  off-natural-support (n′ has a 5 only at groups h with A_{q,h} = 0: no generator episode has these values), π-preserving.

## 2. Equivalence notions [R: FE7: ε = the frozen thresholds; E-Int is measured as an average over pairs, not per pair; the notions are not nested — P1–P4 satisfy E-Int with lossy readout E-Info] (two models M, M′; tolerance ε; distance on decisions δ = argmax disagreement on informative cases, or
1 − Pearson r on centred-logit changes)

- **E-Info (information equivalence).** α_M(Z_M(x)) = v(x) and α_{M′}(Z_{M′}(x)) = v(x) for natural x, up to error ε.
- **E-Int(M) (intervention equivalence of one model; the commuting square).** For (x, x′) ∈ I:
  δ(g_M(x, v(x′)), ℓ_M(x′)) ≤ ε, scored above the mismatched floor (g_M(x, v(x̃′)) for an unrelated x̃′) and relative to the
  full-site patch. This is "V is a causal abstraction of M at S_M for the relational component, under I".
- **E-DR (decision-response equivalence).** For (x, v′) ∈ J: δ(g_M(x, v′) − ℓ_M(x), g_{M′}(x, v′) − ℓ_{M′}(x)) ≤ ε.
- **E-Comp (full computational equivalence)** relative to a declared mediator decomposition of h_M (here: layer-2 attention
  output and MLP output at the readout): E-DR plus equal decomposition of every abstract response across the mediators.

With the candidate's own evidence held fixed (x′_q = x_q in every family), all four notions concern only the relational
component of the decision.

## 3. Theorem A — abstract state sufficiency [P, N]

**A1.** P(X_k | R = (g,h)) = c · f(A_{k,g}) · [A_{k,h} = 1], with f(1) = (α² + (K−1)e²)/K = 0.22, f(0) = (2αe + (K−2)e²)/K = 0.0567,
e = (1−α)/(K−1), c = 1/(K·K²).
*Proof.* Only the reliable pair's likelihood depends on Y_k; summing it over Y_k ~ U(K) gives f(A_{k,g}). The colluding pair has
P = [x_a = x_b]/K (ρ = 1), the noise pair 1/K². ∎
**A2 (relational sufficiency).** π(g,h) ∝ ρ^{n_g} [n_h = 5], ρ = f(1)/f(0) = 3.882; hence π is a function of n.
*Proof.* Multiply A1 over k ≠ q and the uniform prior. ∎ [N: max |error| 1.8e-15 against the generator's exact posterior.]
**A3 (relational vs full decision).** P(Y_q | X) ∝ Σ_R π(R) P(X_q, Y_q | R) **[R: FE1** — the pre-review form Σ_R π(R) P(Y_q | X_q, R) omitted the P(X_q | R) weighting; the code used the joint and no number changes**]**: n is sufficient for the relation-dependent component,
not for the decision (X_q is also needed). [N: Bayes from (X_q, n) equals the full posterior to 1.9e-15.]
**A4 (not minimal).** π(n) = π(n′) iff {h : n_h = 5} = {h : n′_h = 5} and n_g − n_{g′} = n′_g − n′_{g′} for all pairs whose
configurations have positive weight. Of the 91 count vectors with positive probability, there are 43 distinct π (e.g.
(0,5,1) ~ (1,5,2) ~ (2,5,3) ~ (3,5,4)). [P, N]
**A5 (components are insufficient).** Bayes accuracy on 24,000 decisions [N]: (X_q, n) 0.745; (X_q, n_g) for one group
0.669–0.673; (X_q, two components) 0.733–0.739; X_q alone 0.580. No proper subset of components is sufficient.

## 4. Theorem B — distributed realization [P]

Let D : V → {0..5}⁶, D(v)_s = v_{g(s)} (each component duplicated on its pair's two tokens), and call a 6-tuple **consistent** if it
lies in D(V). Let β = (β_s) be local abstractions of the six source tokens of question q.
**B1.** On consistent tuples, r(β) = (β_0, β_2, β_4) inverts D, so the tuple carries exactly V's information, and do(V := v′)
corresponds to do(β := D(v′)). Inconsistent tuples (e.g. a write to one token of a pair) have no V-preimage: the model's
response to them is fixed by its own reconstruction map, not by V; it is an additional degree of freedom.
**B2 [R: FE2, restricted].** No single *own-group component* and no tuple of own-group components missing a group is Bayes-sufficient
(A5). (A token can carry all of n — the P-count readout does, and T2's readout decodes n at 0.93 — so the pre-review "no single
token … must read all three groups" was overbroad.)
**B3 (definitional [R: FE5]).** If M (centralized) and M′ (distributed) each satisfy E-Int under I, their abstract transition systems on V are the same
(both are the generator's V-transition under I) — equivalence of abstract *transitions* is therefore cheap, and carries no
information about the downstream maps g_M, g_{M′}.

## 5. Theorem C — what cross-model transport can and cannot identify [P]

Transport from donor D to recipient R: ΔZ_R = E_R(α̂_D(Z_D(x′))) − E_R(α̂_D(Z_D(x))) with α̂_D the donor decoder's soft one-hot.
**C1 (factorization).** If α̂_D(Z_D(·)) = e_{v(·)} on the evaluated inputs, the transported intervention equals the oracle
intervention E_R(v(x′)) − E_R(v(x)) in R, for every donor, whatever h_D is. With decoder errors,
‖ΔZ_R − ΔZ_R^{oracle}‖ ≤ ‖W_R‖ (‖α̂_D(x) − e_{v(x)}‖ + ‖α̂_D(x′) − e_{v(x′)}‖), W_R the encoder matrix.
*Proof.* E_R is linear in the one-hot features; substitute. ∎
**C2 (consequence) [R: FE8].** (Decodability of V in D) ∧ (E-Int(R)) is *sufficient* for transport success (pre-review: "equivalent"; the converse fails — P-count self-transport is exact with lossy decoders). It cannot distinguish H1 from H3:
two donors with different downstream computations but exact decoders give identical transported responses.
**C3 (Bayes-optimality and E-DR)** [R: FE10 — requires exact per-pair E-Int, which fails for π-preserving abstract interventions in T2 and partly T11; the empirical VI conclusion therefore rests jointly on the abstract and the input-level π-preserving families]. If two models are exactly Bayes-optimal on natural data and satisfy E-Int, then
g_M(x, v′) = g_{M′}(x, v′) for all natural-support (x, v′) — E-DR on natural support is implied by optimality. Hence evidence for
a *shared* downstream computation can only come from (i) the models' deviations from Bayes on natural support — in particular
their responses to π-preserving interventions, which an optimal model must ignore — and (ii) responses off natural support, on
which optimality imposes no constraint.
**C4 (limit of identification).** Two models with equal g_M on all of J are indistinguishable by every V-level experiment;
E-Comp is identifiable only relative to a declared mediator decomposition (here layer-2 attention vs MLP at the readout).

## 6. Theorem D — non-identifiability counterexamples

**D1 (same decoded counts, different causal role) [P, L].** Let M_a store one-hot V at site S and read it; let M_b store the same
one-hot V at S and a copy at S′, reading only S′. α decodes V at S in both; do(V) at S changes M_a and not M_b.
Learned instance: in P-count models question q's source tokens decode n at 0.62–0.85, but marginalising a count variable
there moves the output only 5–27 % toward the own-question posterior, against 68–83 % at the readout (confirmation models).
**D2 (agree on natural decisions, disagree under abstract intervention) [P, L].** Take φ₁(x_q, v) = Bayes and φ₂ equal to Bayes on
natural support but any other function off it (off-natural-support values exist whenever A_{q,h} = 0 for some h). The models
are identical on every natural episode and differ under do(V := v′) off support. Learned illustration only [R: FE13]: the `off_natural_support` family shows models differing off support, but their identity on
natural support is not established.
**D3 (distributed state not replaceable by one local own-group variable) [P, R: FE2].** By A5 any single component loses 0.07–0.08 Bayes accuracy;
by B1 a one-token write produces an inconsistent tuple. (Doubling a one-token edit recovers much of the effect in T2 and T10
[R: ME]; the consistent-pair requirement is clear only in T11.)
**D4 (causally effective without being uniquely necessary) [P].** Let readout = σ((V¹ + V²)/2) with saturating σ and two copies V¹,
V² of V. Interchanging V¹ alone moves the output past saturation (high IIA); ablating V¹ alone changes nothing once V²
saturates. This gives "interchange effective, ablation ineffective". The other direction [R: FE4]: readout = sign(V¹·V²) with V¹, V² ∈ {−1, 1}
copies; ablating V¹ (setting it to 0) changes the output, interchanging V¹ alone with a counterfactual where both copies flip
leaves the sign unchanged. Interchange and ablation can therefore disagree in either direction.
**D5 (high IIA by steering a correlated decision feature) [R: FE3, restated].** Pre-review claim (withdrawn): a site storing only
configuration log-odds is fitted exactly by the additive one-hot count encoder. Correct statement: an additive one-hot code that
is constant on π-classes can encode only the colluder set H (reviewer's proof), so an exact fit is impossible; the real risk is an
*approximate* fit of a π-like site by the additive count encoder, giving high IIA without representing n. Distinguisher:
decoders that separate π-equivalent count vectors. In every model with decoder accuracy ≈ 1.0 (all T models, P0) the site
distinguishes the count vectors, so D5 is excluded for them. For P2–P4 it is excluded by a direct test [R: FE12]: the
readout decodes the within-π-class rank of n at 0.87–0.91 (baseline 0.50) and the π-class only at 0.29–0.39
(`logs/review_checks/adj_pi_vs_n.json`).

## 7. Relationship to the experiments

| notion | experiment | document |
|---|---|---|
| E-Info | decoder accuracy; info_structure R² | COUNT_INFORMATION_STRUCTURE.md |
| E-Int | within-model abstract interchange vs floor/ceiling, all families | CROSS_MODEL_CAUSAL_TRANSPORT.md §1 |
| transport (C1/C2) | donor-decoded vs oracle vs floor, 30 ordered pairs | CROSS_MODEL_CAUSAL_TRANSPORT.md §2 |
| E-DR beyond Bayes (C3) | π-preserving and off-support response correlations, within vs across curricula | CROSS_MODEL_CAUSAL_TRANSPORT.md §3 |
| E-Comp (C4) | path patching at the readout — not informative: the T result is forced by the site [R: FE11, ME4] | CROSS_MODEL_CAUSAL_TRANSPORT.md §4 |
| necessity | on-support marginalisation, nulls, span ablation/restoration | NECESSITY_AND_RESTORATION.md |
