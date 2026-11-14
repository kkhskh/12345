# Independent mathematical review of `CAUSAL_ABSTRACTION_AND_MATHEMATICS.md` (Investigation XII, Workstream C)

Reviewer: independent; did not write the work under review. Scope: items (a)–(g) of the review brief.
Files read: `CAUSAL_ABSTRACTION_AND_MATHEMATICS.md`, `src/xii_bayes.py`, `src/genlib.py`, `scripts/math_xii.py`,
`logs/math_xii.json`, `scripts/disc_ident.py`; for context `../step1_native_causal_construction/MATHEMATICAL_AUDIT_AND_RESULTS.md`,
`../step1_native_causal_construction/scripts/das_decoy.py` and its logs `../step1_native_causal_construction/logs/decoy/*.json`.

All numerical checks were run with `nice -n 19`, outside the audited directory (`/tmp/xii_review/`), each under 30 s CPU.
Their full code is reproduced in the Appendix.

## Summary of verdicts

| item | claim | verdict |
|---|---|---|
| (a) | Theorem XII-1 (independent copying, e_C rule) | **correct with caveat** (minor conditions unstated; one "consequence" is wrong for G > C + 1 and one is overstated) |
| (b) | §1.3 / Theorem XII-2 (shared copying; `posterior_brute` not exact for C ≥ 2; `posterior_shared`) | **correct**, independently confirmed against sampler frequencies, including β < 1; the doc *understates* the β < 1 discrepancy |
| (c) | §1.4 (e = f(W, L); VE not input-realizable; retraction of XI §4) | **correct with caveat** (needs α ≠ 1/K; one log entry is hard-coded, not measured) |
| (d) | §1.5 (product-support caveat for A2) | **diagnosis correct; restricted statement imprecise and partly incorrect** |
| (e) | Propositions XII-3, XII-4 and their implications for XI's A1 decoy | **bounds correct (elementary); implications for XI incorrect.** M4/M5 were computed on the wrong pair distribution for the A1 run. On the distribution XI actually used, XI's A1 preservation test *does* discriminate A1 from E |
| (f) | §3 identification conditions; §4 taxonomy and rule | **overstated / one logical error** (the rule's quantifier "at least one implementation" is wrong and contradicts XI's own Theorem X1′; conditions 3 and 4 are not necessary under redundancy; §3 contradicts §2 on A1 vs E) |
| (g) | novelty | XII-1/XII-2 are elementary Bayesian computations; XII-3/XII-4 are elementary probability; the conceptual points (non-uniqueness, off-manifold interchanges, intervention-set dependence, bijective re-encoding) are standard in the causal-abstraction literature |

---

## (a) Theorem XII-1: verdict **correct with caveat**

**Proof check.** Fix a role (g, H), |H| = C, uniform prior over the `env.roles` list (which is what `genlib.sample` draws). Marginalising Y_{q'} for q' ≠ q under its uniform prior:

p(X, Y_q = y | g, H) = [Π_{q'≠q} r̄_g(q')] · r_g(q, y) · Π_{h∈H} Π_{q'} c_h(q') · (1/K²)^{Q(G−1−C)} · (1/K)

where r̄ is the per-question marginal reliable-pair likelihood, r_g(q, y) the reliable-pair likelihood at q, and c_h(q') = 1[pair h agrees at q']/K when β = 1. Then:
- Π_{q'} c_h(q') = K^{−Q} · 1[h agrees at all Q questions], and 1[agrees at all Q] = 1[n_h = Q−1 ∧ agrees at q] = e_h. Correct.
- Π_{q'≠q} r̄_g(q') · r_g(q, y) = (1/K²)^Q · W_g · L_g(y), with W_g, L_g as in `posterior_closed`. Correct.
- The remaining factor, (1/K²)^{Q(G−C)} K^{−QC}, depends on C but not on (g, H). Since C is fixed, it is role-independent. Correct.
- Summing over H ⊆ [G]∖{g} with |H| = C gives e_C(e_{−g}). Correct.

The proof is right. The M1 check agrees to ≤ 4e-15.

**Unstated conditions** (all hold in the actual setting, but they belong in the statement):
1. **β = 1 is essential.** For β < 1 the colluder factor is β·eq/K + (1−β)/K², and the product over questions no longer collapses to an indicator. The theorem says β = 1, which is fine. Note that for β < 1 the analogous formula is a polynomial in per-question agreement counts, not in e.
2. **Uniform prior over roles** is used to drop the prior. It holds for `genlib.sample`, but it is not stated.
3. **Normalisability.** The "∝" needs Σ_y Σ_g W_g L_g(y) e_C(e_{−g}) > 0. On the support this always holds: for the true reliable group g*, all C colluders have e = 1 (β = 1), so e_C(e_{−g*}) ≥ 1; and W_{g*}L_{g*}(y) > 0 for some y at any α ∈ [0, 1]. Off the support (fewer than C fully agreeing groups outside every candidate g) the formula is 0/0, and so is `posterior_brute`, which returns NaN from log 0. `esym_rule` in `math_xii.py` hides this with `np.maximum(..., 1e-300)`. State the theorem "for x in the support".
4. **α**: no restriction is needed for the theorem itself. (A restriction *is* needed for §1.4; see (c).)

**Problems in the "Consequences" paragraph.**
- "The correct C = 2 rule needs the degree-2 aggregation Π_{h≠g} e_h: an AND across groups." This holds only when G = C + 1 (e.g. G = 3, C = 2). For G = 4, C = 2, e_2(e_{−g}) = Σ_{h<h'} e_h e_{h'} is a sum of pairwise ANDs, not Π_{h≠g} e_h. Write e_C(e_{−g}).
- "The XI C = 2 failure is therefore precisely an aggregation-degree error with a correct variable." This is overstated twice:
  - By XII-2, the generator XI trained and tested on is the shared-copy one, whose Bayes rule is *not* e_C(e_{−g}). "The correct C = 2 rule" is J_H-based, so the degree-2 e-rule is itself the wrong target, by 4–6 accuracy points.
  - "With a correct variable" is an empirical claim about the network. The theorem does not establish it.

## (b) §1.3 / Theorem XII-2 / `posterior_shared`: verdict **correct** (independently confirmed); doc understates the β < 1 case

**Sampler semantics (code reading).** In `genlib.sample`, `col = rng.integers(0, K, (n, Q))` and `cop = rng.random((n, Q)) < beta` are drawn once per (episode, question) and applied to *every* colluding source, via `np.where(coll[:, gi, None] & cop, col, X[:, :, s])`. So:
- the copied value is shared across all colluder groups;
- **the copy event is also shared**: with probability β all 2C colluder sources copy the same value, and with probability 1 − β all of them keep their initial `X = rng.integers(0, K, ...)` draws. Those draws are independent and uniform, so a non-copying colluder source does report uniformly at random. That answers the brief's question.
- The module docstring ("with prob 1−beta each source reports independently uniformly") does not say that the event is shared. That is a documentation gap in genlib.

`posterior_brute` multiplies per-group factors β·eq_h/K + (1−β)/K². That is the posterior of a model with independent copy values *and* independent copy events. So it is not the exact posterior for C ≥ 2. It is exact for C = 1.

`posterior_shared` uses β·1[all 2|H| sources equal]/K + (1−β)/K^{2|H|} per question. That is exactly the likelihood of the shared-value, shared-event sampler. **Correct for all β.**

**Independent check (Appendix, `chk_b.py`).** M2's "max |Δ| ≤ 2e-15" compares two of the author's implementations of the *same* formula. So I compared against the sampler itself: empirical P(Y_q | X) over all observed X patterns in tiny environments (K = 2, Q = 2, 1.5M samples), and log-loss in the real environment (K = 3, Q = 6).

| environment | weighted sq. err, `posterior_shared` | weighted sq. err, `posterior_brute` | multinomial noise floor if exact |
|---|---|---|---|
| G3 C2 K2 Q2 β=1 | 3.86e-5 | 0.160 | 3.85e-5 |
| G3 C2 K2 Q2 β=0.6 | 1.09e-3 | 0.142 | 1.13e-3 |
| G3 C1 K2 Q2 β=0.6 | 1.17e-3 | 1.17e-3 (identical) | 1.18e-3 |

| environment (K=3, Q=6, n=1e5) | log-loss shared | log-loss brute | acc shared / mean max-p | acc brute / mean max-p |
|---|---|---|---|---|
| G3 C2 β=1 | 0.402 | 0.456 (Δ = 0.054 ± 0.0008) | 0.801 / 0.801 | 0.748 / 0.783 |
| G3 C2 β=0.6 | 0.426 | 0.885 | 0.791 / 0.792 | **0.500** / 0.649 |
| G4 C2 β=0.6 | 0.534 | 0.925 | 0.758 / 0.759 | **0.468** / 0.629 |

Results:
- `posterior_shared` sits at the noise floor and is calibrated (accuracy ≈ mean max-probability).
- `posterior_brute` is miscalibrated for C ≥ 2.
- Empirically, P(all 4 colluder sources equal at a question) = 0.6145, against 0.6148 predicted by the shared-event model. That confirms the copy event is shared.

The §1.3 table values (accuracies 0.794/0.747, 0.778/0.729, 0.768/0.707; KL 0.049, 0.050, 0.076) match `math_xii.json`.

**Caveats.**
- With β < 1 and C ≥ 2, the brute-vs-exact discrepancy is huge: brute accuracy falls to chance level, about 0.50. The doc reports only β = 1. Any past or future use of `posterior_brute` as "true Bayes" with C ≥ 2 and β < 1 would be badly wrong. I found no such use: the `gen_readers.py` β = 0.7 runs in Investigation IX all have C = 1, and XI's `generalize.py` b07 arm is C = 1. A one-line warning should be added to `genlib.posterior_brute`.
- "J_H is a relation between groups, which the per-group statistic e cannot express": correct, since J_H = Π_{h∈H} e_h · 1[the groups' common values coincide at every question].
- "Prediction (P8): … It can therefore succeed **without** the E-subspace" does not follow. A correct readout must use cross-group equality *in addition to* e, because J_H ≤ Π e_h. The theorem therefore *permits* E-free solutions but neither predicts nor implies them. Reword as "it need not be E-based in the C = 1 sense; it can also use the E-subspace plus a cross-group equality signal."

## (c) §1.4 (determinism and the VS/VE retraction): verdict **correct with caveat**

**e_g = f(W_g, L_g) on the support.**
- W_g = const · ρ_a^{n_g} ρ_d^{Q−1−n_g}, where ρ_a = K(α² + (1−α)²/(K−1)) and ρ_d is the disagreeing-pair ratio. W_g therefore determines n_g iff ρ_a ≠ ρ_d, iff α ≠ 1/K.
- L_g has the shape "one entry α², the rest b²" when the pair agrees, and "two entries αb, the rest b²" when it disagrees (b = (1−α)/(K−1)). These shapes are distinguishable iff α ≠ b, iff α ≠ 1/K.

So the claim needs **α ≠ 1/K**, which is unstated (α = 0.8 here). Check (`chk_ce.py`):
- α = 0.8: 36 distinct (W, L) keys and 0 collisions. This reproduces M3.
- α = 1/3: a single key and 35,706 collisions.

I also checked that **U_g = W_g L_g alone** (the I2 coordinate) determines e: 36 distinct values, 0 collisions. This matters because A2 in XI is applied to I2 with R = U.

**Non-realizability.** VE gives the high-level state (e_s, W_b, L_b). Input-realizability requires e_s = f(W_b, L_b) = e_b, so every informative pair is off-image. The argument is correct.

Presentation issue: `logs/math_xii.json` field `M3_frac_informative_E_pairs_input_realizable = 0.0` is **hard-coded** (`0.0 if inf_.any() else None`, line 55 of `math_xii.py`), not measured. It is a restatement of the proof, and should be labelled as such or removed.

"100% of the pairs DAS is evaluated on" holds for the E rule: `das_decoy.diag` evaluates IIA on `diff` = (e_s ≠ e_b), which is exactly the informative set. It does not hold for the training pairs, which include non-informative ones.

**Retraction of XI §4.** Justified. Under SCM semantics, do(S := s) replaces S's structural equation by definition. VE equally produces a joint value of (e, W, L) outside the joint support, although each coordinate is inside its marginal range. The same holds for VS with (S, e, W, L): S_s lies in range(S). So the two are symmetric.

The only asymmetry XI pointed to is that VS breaks an *explicit structural equation of the high-level model* (S = Σe), while VE breaks a constraint *induced through the shared parent X*. For input-realizability that distinction does not matter. XI's stronger I2-specific observation (VS requires fractional e′ in 49% of pairs) is a fact about the *I2 coordinates*, not about the legitimacy of the high-level intervention, and the corrected statement is right to keep only the projection-realizability conclusion.

Minor: the boxed "Neither is an input-realizable counterfactual" should say "for informative pairs". Non-informative VE pairs are trivially realizable by the base itself.

## (d) §1.5 (product-support caveat for XI's A2): verdict **diagnosis correct; restricted statement imprecise and partly incorrect**

**Correct.**
- The (⇒) proof in XI sets (I−P)φ(V_s, R_b) = (I−P)φ(V_b, R_b) and Pφ(V_s, R_s) = Pφ(V_s, R_b). Both require φ to be *defined* at off-image points (V_s, R_b) and the pair set to range over enough combinations.
- On T4, V = e is a function of R = (W, L), or of U alone (see (c)). The joint support is the graph of a function, which is about as far from a product as a support can be.
- XI's results do not depend on (⇒) on T4: the I2-VE and I1-VS corollaries use (⇐), and the I1-VE and I3 failures were proved separately with R_any witnesses (C5, C6). §1.5 should say this explicitly, so readers do not infer that XI results are weakened.

**Imprecise.** "(⇐) is unconditional … including on off-image pairs" is true only once the off-image target is *defined* as a(V_s) + b(R_b). For a formula-defined implementation (I1, I2) that is the natural extension, but then the statement is close to tautological. A low-level network has no canonical φ off the image, and in DAS the target is the *output* p(V_s, R_b), not a state.

**Incorrect.** "On a non-product support … the interchange then constrains the representation only on states never produced naturally." This is false. A precise version on a functional support V = f(R), with a free off-image extension:
- Projection realizability with *some* extension φ̂ (φ̂ = φ on the image) holds iff the patched state (I−P)φ(x_b) + Pφ(x_s) depends on the pair only through (V_s, R_b). Given rich pairing, that is iff **Pφ is a function of V alone on natural states**, i.e. Pφ(x) = a(V(x)).
- The other half, (I−P)φ = b(R), is vacuous because V is a function of R.
- So the interchange *does* constrain the natural representation: the projected component of every natural state must be V-measurable. What it does not constrain is the readout's behaviour at the off-image states b(R_b) + a(V_s).
- Non-informative pairs (V_s = V_b, R_s ≠ R_b) impose the further natural-state constraint P(φ(x_s) − φ(x_b)) = 0 whenever the target equals the base.

At output level the constraint is weaker still: P(z_s − z_b) need only lie in directions the downstream readout ignores. Replace the sentence with this statement, which still supports the qualitative point (interchange success says little about how the readout behaves on natural data).

The "[proved; the restriction is new in XII]" label should be withdrawn until a precise statement and proof are written.

## (e) Propositions XII-3 and XII-4: verdict **bounds correct (elementary); the implications for XI are incorrect**

**XII-3 as mathematics.**
- IIA_V(R) = E_pairs 1[â = argmax cf_V] and IIA_{V′}(R) = E 1[â = argmax cf_{V′}], with â the patched decision.
- On pairs where the argmaxes agree the indicators coincide, so |IIA_V − IIA_{V′}| ≤ P(argmax cf_V ≠ argmax cf_{V′}) = δ. Correct.
- Taking a sup over R also gives |max_R IIA_V − max_R IIA_{V′}| ≤ δ.

Conditions omitted:
1. **Both IIAs must be averaged over the same pair set.**
2. Ties in argmax must be broken identically. `cf` clamps to 1e-30, so all-zero rows become uniform with argmax 0. This is consistent across V and V′, so it is harmless here.
3. The bound does not apply to KL-based or soft IIA.

**Condition 1 is violated in XI's protocol.** `das_decoy.diag` evaluates each rule's IIA on its own `diff` set (rule value of source ≠ base). In addition, `das_decoy.sample` **stratifies bases by P_ = Σ pf when RULE == "A1"**, and by S otherwise (line 32: `key = P_ if RULE == "A1" else S`). `math_xii.py` M4/M5 rebuild only the S-stratified distribution and reuse it for A1.

This is confirmed by the logged counts:
- M5's A1 preservation set has n = 1006, while XI's actual A1 runs log `n_pres = 957`. The PERM and RANDE counts (1008, 1047) match exactly.
- My faithful reconstruction of the A1 run reproduces XI's logs (IIA set n = 5041, n_pres = 957).

Recomputed on the distribution XI actually used for the A1 decoy (`chk_ce.py`):

| quantity | doc (S-stratified) | XI's actual A1 run (P_-stratified) |
|---|---|---|
| δ(E, A1), all pairs | 0.0035 | **0.0465** |
| δ(E, A1) on the A1 IIA set (pf_s ≠ pf_b) | 0.0030 | **0.040** |
| A1-preservation pairs on which E changes | 3.9% | **32.4%** |
| … on which the Bayes decision changes | 2.5% | **19.0%** |
| preserve_KL of a *clean E mediator* = KL(cf(e_b) ‖ cf(e_s)) on A1-preservation pairs | 0.027 (not reported) | **0.292** |
| observed preserve_KL, A1-trained subspace (XI logs) | — | 0.158 / 0.164 / 0.224 |
| observed preserve_KL, E-trained subspace (XI logs, E-preservation set) | — | 0.008 / 0.008 / 0.126 |

Consequences:
1. "No interchange-accuracy comparison on that distribution can separate E from A1 by more than 0.35 points" holds for the E/PERM/RANDE distribution only. On the A1 run's own evaluation set the bound is about 4 points. The qualitative point that interchange IIA hardly separates E from A1 survives, but with δ ≈ 0.04, not 0.0035.
2. The observed gap between E-trained (0.906–0.969) and A1-trained (0.912–0.926) IIA, up to 4.3 points, exceeds the stated 0.35-point bound. The doc does not reconcile this. The gap comes from different subspaces and different pair sets, not from the targets. The bound concerns one fixed R evaluated on one pair set.
3. **XII-4's conclusion is contradicted on XI's actual test.**
   - On XI's A1-preservation pairs, a clean E mediator would change E on 32% of pairs and incur preserve_KL ≈ 0.29, *larger* than the A1-trained subspace's observed 0.16–0.22. XI's preservation test therefore *did* have power to discriminate A1 from E.
   - The sentence "XI's A1 decoy failed preservation because the A1-trained subspace carried other information, not because the test discriminated A1 from E" is not supported, and the evidence points the other way.
   - "Pass preservation on 96% of pairs" holds only for the S-stratified distribution. It also mixes an E-change criterion (96.1%) with a decision criterion (97.5%), while XI's preservation metric is a KL.
4. "XI's 'A1 false positive' was a property of the evaluation distribution before it was a property of the network": for interchange IIA this is defensible (δ ≈ 0.04 on the A1 set); for preservation it is false. The "Counterexample to 'interchange + preservation identifies the variable'" is valid as a hypothetical only on the S-stratified pair distribution. As stated ("on XI's distribution") it is wrong for the A1 run.

**XII-4 as mathematics** (preservation pairs can expose V′ only where V changes) is a correct, elementary observation.

## (f) §3 identification conditions and §4 taxonomy: verdict **overstated, with one logical error**

1. **The rule in §4(a) has the wrong quantifier.** "Failure under a class-IV intervention falsifies a proposed abstraction only if (a) the alignment class is rich enough to realize that intervention in **at least one** exact implementation of the function (XII-5: projections are not, for VS)."
   - XI's Theorem X1′ shows that **I1 = (S, A, B) realizes VS by projection** (copy the S coordinate; C1, error 3e-16). So for VS and projections, condition (a) as written *is satisfied*. The parenthetical "projections are not, for VS" is false in general; it holds only for I2.
   - Under the stated rule, a projection-VS failure *would* qualify to falsify the pooled-S algorithm, which contradicts §1.4's surviving conclusion ("failure of a projection search for S cannot falsify a pooled-S algorithm").
   - The needed condition is: the alignment class realizes the intervention in **every** implementation consistent with the hypothesis being tested, or in every implementation of the hypothesised class. Equivalently, falsification is relative to (algorithm, implementation class, alignment class).
2. **"Each necessary".** The five §3 conditions are stated as necessary for "R is supported for V over V′". Counterexamples:
   - **Condition 3 (natural-activation necessity) is not necessary** when V is redundantly encoded. Suppose R and R′ both carry V and the readout averages them. Then removing the upstream computation feeding R alone leaves the decision unchanged by compensation (cf. the "Hydra effect", McGrath et al. 2023), although R genuinely mediates V.
   - **Condition 4 (restoration through R alone) is not necessary** under the same redundancy. Ablate upstream (corrupting R and R′) and restore R only: the decision recovers only partially, although R carries V exactly.
   - Conditions 3 and 4 are necessary for "R is the *sole* mediator", which is a different claim; the target claim should be stated.
   - **Conditions 1 and 2 are necessary only within behavioural-intervention evidence.** Weight-level or circuit-reading evidence can support V over V′ without discriminating pairs. Also, condition 2's pairs (V′ equal, V different) are themselves discriminating pairs in the sense of condition 1, so the two overlap.
   - **Condition 5** (beating a source-free mean patch) is a sensible control, but "necessary" is too strong. On informative pairs where the mean patch and the counterfactual decisions differ, IIA_disc already accounts for source information.
3. **Internal inconsistency.** §3 says "variables that coincide on the support (A1 vs E on XI's distribution) are not separable at all". A1 and E do **not** coincide on the support: they differ on 0.35% (S-stratified) or 4.6% (A1-stratified) of XI's pairs, and §2's own `disc_ident.py` result separates them on discriminating pairs drawn from natural inputs. They nearly coincide *on one pair distribution*, which is a different and much weaker statement.
4. **Taxonomy (§4)** mixes levels:
   - Classes II and III are low-level interventions on network activations, class IV is high-level, and class V is target definitions rather than interventions.
   - The low-level DAS patch that is supposed to realize a class-IV intervention is itself a class-III synthetic state: z_b + P(z_s − z_b) is generally not a natural activation. The table implies class IV is the only off-image case on the high-level side, while every interchange test is off-manifold on the low-level side too.
   - Class II's "the combination is not [natural]" is right. Note also that single-component resampling (causal scrubbing) is natural per component only under the scrubbing hypothesis's independence assumptions.
5. **XII-5 (related to §4's rule) is overstated.**
   - "The network realizes part of VS through **exactly** the non-projective linear class the theorem identifies": the theorem identifies M = ½11ᵀ on the m-block, with spectrum {3/2, 0, 0} and zero on U. The learned map's spectrum {0.005, 0.005, 0.55, 0.55, 1.54, 3.83} does not resemble it.
   - A free linear map (36 parameters on a 6-dim subspace) fits any target better than a projection does. Without a control (a free linear map on a random 6-dim subspace, or trained against RANDE), KL 0.40 vs 0.83 does not show that the network uses an I2-type code.
   - "False negative of projection-DAS" is an interpretation, not a demonstration.

## (g) Novelty: what is standard

- **Theorems XII-1 and XII-2** are routine posterior computations: sum over a discrete latent role, deterministic copying makes the likelihood an indicator, and summing indicators over C-subsets gives an elementary symmetric polynomial. They are fine as task-specific lemmas, but they should not be presented as new mathematics. The *finding* that `posterior_brute` was the wrong generator's posterior is a useful bug report, not a theorem.
- **Proposition XII-3** is the inequality |E 1_A − E 1_B| ≤ P(A Δ B), and **XII-4** is a restatement of "a test has power only where the hypotheses' predictions differ". "[proved; elementary]" is accurate; the word "Proposition" oversells them.
- The **conceptual points** are standard in the causal-abstraction and interpretability literature:
  - Abstraction is relative to a chosen set of allowed interventions (Rubenstein et al. 2017, "Causal consistency of structural equation models"; Beckers & Halpern 2019, "Abstracting causal models"). The §4 taxonomy is a task-specific instance of choosing that intervention set.
  - Non-uniqueness of abstractions and invariance under bijective re-encoding of a high-level variable are discussed by Geiger et al. ("Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability").
  - Interchange-intervention and subspace-patching results can be produced by off-manifold or dormant directions, and high IIA does not identify the variable (Makelov, Lange & Nanda 2024, "Is This the Subspace You Are Looking For?"). Metric and distribution choices change patching conclusions (Zhang & Nanda 2024, "Towards Best Practices of Activation Patching").
  - The richness of the alignment class (projection vs linear vs nonlinear) changes what can be "found" (Sutter et al. 2025, "The Non-Linear Representation Dilemma").
  - Redundancy and compensation defeat necessity tests (McGrath et al. 2023, "The Hydra Effect").
  - do() on a deterministic intermediate producing jointly unrealizable states is textbook SCM semantics (Pearl).

  These should be cited as known, with XII's contribution described as the explicit task-specific instances and the corrections to the record.
- **Plausibly new and useful (task-specific):** the shared-copy correction; the δ calculation showing that a particular evaluation distribution barely discriminates E from A1 (once recomputed on the right distribution); and the discriminating-pair design in `disc_ident.py`.

---

## Prioritised list of required corrections

1. **(e, high)** Recompute M4/M5 for the A1 decoy on the pair distribution XI actually used. `das_decoy.sample` stratifies by P_ for A1, and IIA/preservation are evaluated on rule-specific `diff` sets. Then:
   - replace δ(E, A1) = 0.0035 with ≈ 0.040 (A1 IIA set) / 0.047 (all pairs);
   - replace "E changes on 3.9%" with ≈ 32%;
   - **withdraw** the claim that XI's A1 preservation failure was not discriminative: a clean E mediator would have preserve_KL ≈ 0.29 on that test, above the observed 0.16–0.22;
   - restrict the "counterexample" to the S-stratified distribution, and state XII-3's same-pair-set condition explicitly.
2. **(f, high)** Fix the §4 rule's quantifier ("at least one" → "every implementation in the hypothesised class"). Delete or correct "(XII-5: projections are not, for VS)", which contradicts I1-VS ∈ R_proj (XI Theorem X1′).
3. **(d, high)** Replace the §1.5 "restricted statement": on a functional support, projection realizability (free extension) ⟺ Pφ is V-measurable on natural states, so the natural representation *is* constrained. State explicitly that no XI result depends on A2's (⇒) on T4. Drop "[proved]" until a precise proof is written.
4. **(f, medium)** Remove "A1 vs E … not separable at all" from §3 (it contradicts §2's disc_ident result). Downgrade "each necessary" to "necessary for sole-mediator claims within behavioural intervention evidence", and give the redundancy counterexamples for conditions 3 and 4.
5. **(a, medium)** In the XII-1 consequences, replace Π_{h≠g} e_h by e_C(e_{−g}) (they are equal only when G = C + 1). Remove "precisely an aggregation-degree error with a correct variable", since the true generator's rule is J_H-based and the "correct variable" part is empirical. Add "uniform role prior; x in the support" to the theorem statement.
6. **(b, medium)** Report that for β < 1 and C ≥ 2 `posterior_brute` is grossly wrong (accuracy ≈ 0.50 vs 0.79 exact at G3C2 β=0.6). Add a warning to `genlib.posterior_brute`, and document in genlib that the copy *event* is shared across colluder groups. Reword the P8 "therefore … without the E-subspace" as a possibility, not a consequence.
7. **(c, low)** Add the condition α ≠ 1/K to the e = f(W, L) claim. Mark `M3_frac_informative_E_pairs_input_realizable` as derived, not measured (it is hard-coded). Add "for informative pairs" to the boxed non-realizability statement.
8. **(f, low)** Qualify XII-5: the learned spectrum does not match ½11ᵀ, and add a linear-map control before calling it a projection-DAS false negative. Note that DAS-patched low-level states are themselves class III.
9. **(g, low)** Label XII-1 to XII-4 as elementary or task-specific, and cite the standard sources for non-uniqueness, intervention-set dependence, off-manifold interchange illusions, alignment-class richness and redundancy.

---

## Appendix: exact code of the numerical checks

Run as `cd /tmp/xii_review && nice -n 19 python3 chk_b.py` (≈ 11 s CPU) and `nice -n 19 python3 chk_ce.py` (≈ 2 s CPU).

### `chk_b.py`: `posterior_shared` vs `posterior_brute` against the actual sampler

```python
import sys, numpy as np, itertools
D="/home/claude/12345/docs/research_audits/cor_phase15/step1_native_relational_algorithm_validation/src"; sys.path.insert(0, D)
import genlib as GL, xii_bayes as XB
np.seterr(all="ignore")
def freq_check(G,C,K,Q,beta,n,seed):
    env=GL.Env(G=G,C=C,K=K,Q=Q,beta=beta); r=np.random.default_rng(seed); d=GL.sample(env,n,r); X=d["X"]; q=np.zeros(n,int); Y=d["Y"][:,0]
    key=(X.reshape(n,-1)*(K**np.arange(X[0].size))).sum(1)
    uk,inv,cnt=np.unique(key,return_inverse=True,return_counts=True)
    emp=np.zeros((len(uk),K)); np.add.at(emp,(inv,Y),1); emp/=cnt[:,None]
    rep=np.zeros(len(uk),int); rep[inv]=np.arange(n)
    Ps=XB.posterior_shared(env,X[rep],q[rep]); Pb=GL.posterior_brute(env,X[rep],q[rep])
    w=cnt/cnt.sum()
    se_s=(w*((emp-Ps)**2).sum(1)).sum(); se_b=(w*((emp-Pb)**2).sum(1)).sum(); noise=(w*((Ps*(1-Ps)).sum(1)/cnt)).sum()
    return dict(env=repr(env),patterns=len(uk),wSE_shared=se_s,wSE_brute=se_b,expected_noise_if_exact=noise)
def logloss(G,C,beta,n,seed):
    env=GL.Env(G=G,C=C,beta=beta); r=np.random.default_rng(seed); d=GL.sample(env,n,r); X=d["X"]; q=r.integers(0,6,n); Y=d["Y"][np.arange(n),q]
    Ps=XB.posterior_shared(env,X,q); Pb=GL.posterior_brute(env,X,q); i=np.arange(n)
    ls=-np.log(Ps[i,Y]); lb=-np.log(Pb[i,Y]); dd=lb-ls
    return dict(env=repr(env),logloss_shared=ls.mean(),logloss_brute=lb.mean(),diff_mean=dd.mean(),diff_se=dd.std()/np.sqrt(n),
                acc_shared=(Ps.argmax(-1)==Y).mean(),conf_shared=Ps.max(-1).mean(),acc_brute=(Pb.argmax(-1)==Y).mean(),conf_brute=Pb.max(-1).mean())
for a in [(3,2,2,2,1.0,1_500_000,1),(3,2,2,2,0.6,1_500_000,2),(3,1,2,2,0.6,1_500_000,3)]:
    print(freq_check(*a),flush=True)
for a in [(3,2,1.0,100_000,11),(3,2,0.6,100_000,12),(4,2,0.6,100_000,13)]:
    print(logloss(*a),flush=True)
env=GL.Env(3,2,beta=0.6); r=np.random.default_rng(5); d=GL.sample(env,40000,r); X=d["X"]
roles=[env.roles[i] for i in d["ridx"]]; H=np.array([list(h) for _,h in roles])
a_=X.reshape(40000,6,3,2); s=np.stack([a_[np.arange(40000),:,H[:,0],:],a_[np.arange(40000),:,H[:,1],:]],2).reshape(40000,6,4)
alleq=(s==s[...,:1]).all(-1).mean()
print("P(all 4 colluder srcs equal)=%.4f  shared-event pred=%.4f"%(alleq,0.6+0.4/27))
```

Output (abridged): wSE shared 3.86e-5 / brute 0.160 / noise 3.85e-5 (β=1); 1.09e-3 / 0.142 / 1.13e-3 (β=0.6); C=1 identical. Log-loss G3C2 β=1: 0.402 vs 0.456. G3C2 β=0.6: 0.426 vs 0.885 (brute acc 0.500). G4C2 β=0.6: 0.534 vs 0.925 (brute acc 0.468). P(all 4 equal) = 0.6145 vs 0.6148.

### `chk_ce.py`: determinism of e, and M4/M5 on XI's actual per-rule pair distribution

```python
import sys, numpy as np
B="/home/claude/12345/docs/research_audits/cor_phase15/step1_native_relational_algorithm_validation"; sys.path.insert(0, B+"/src")
import genlib as GL
np.seterr(all="ignore")
def hl(env,X,q):
    n=len(X); ii=np.arange(n); rel,col,noi=GL._group_lik(env,X); relm=rel.mean(-1); lr=np.log(relm/noi)
    W=np.exp(lr.sum(1)-lr[ii,q]); L=rel[ii,q]/noi; cnt=GL.counts(X,env.G)[ii,q]; a_=X.reshape(n,env.Q,env.G,2)[ii,q]
    pf=(cnt==env.Q-1).astype(float); e=pf*(a_[...,0]==a_[...,1]); return W,L,e,pf
def determ(keyarr,e):
    tab={};c=0
    for k,v in zip(map(tuple,np.round(keyarr,8).reshape(-1,keyarr.shape[-1])),e.reshape(-1)):
        if k in tab and tab[k]!=v: c+=1
        tab[k]=v
    return len(tab),c
for alpha in (0.8,1/3):
    env=GL.Env(3,1,alpha=alpha); r=np.random.default_rng(7); d=GL.sample(env,20000,r); q=r.integers(0,6,20000); W,L,e,_=hl(env,d["X"],q)
    print("alpha=%.3f  (W,L):"%alpha, determ(np.concatenate([np.log(W)[...,None],np.log(L)],-1),e), " U=WL alone:", determ(np.log(W[...,None]*L),e))
import native as NV
def xi_sample(n,stream,RULE):          # faithful copy of XI das_decoy.sample, incl. key = P_ for RULE == "A1"
    env=GL.Env(3,1); rng=np.random.default_rng(stream); d=NV.Bm.sample_fast(NV.ENV,4*n,rng); X=d["X"]; q=rng.integers(0,6,len(X))
    W,L,e,pf=hl(env,X,q); S=e.sum(1); P_=pf.sum(1); key=P_ if RULE=="A1" else S
    two=np.where(key>=2)[0]; one=np.where(key==1)[0]
    bi=np.concatenate([rng.choice(two,n//2),rng.choice(one,n-n//2)]); rng.shuffle(bi)
    si=np.where(rng.random(n)<0.6,rng.choice(two,n),rng.integers(0,len(X),n)); ri=rng.integers(0,len(X),n)
    return W,L,e,pf,bi,si
def cf(W,L,x,b):
    P=np.maximum((W[b][...,None]*(x.sum(1,keepdims=True)-x)[...,None]*L[b]).sum(1),1e-30); return P/P.sum(-1,keepdims=True)
kl=lambda P,Q:(P*(np.log(P+1e-12)-np.log(Q+1e-12))).sum(-1)
for RULE in ("E","A1"):
    W,L,e,pf,bi,si=xi_sample(6000,64_500_000,RULE)
    cE=cf(W,L,e[si],bi); cA=cf(W,L,pf[si],bi); cB=cf(W,L,e[bi],bi)
    dE=np.abs(e[si]-e[bi]).sum(1)>0; dA=np.abs(pf[si]-pf[bi]).sum(1)>0; same=si==bi; dis=cE.argmax(-1)!=cA.argmax(-1)
    print(f"[{RULE}-stratified] delta_all={dis.mean():.4f} delta_on_A1diff={dis[dA].mean():.4f} delta_on_Ediff={dis[dE].mean():.4f} |A1diff|={dA.sum()} |Ediff|={dE.sum()}")
    pres=(~dA)&(~same)
    print(f"   A1-pres n={pres.sum()}: E changes={dE[pres].mean():.4f} Bayes dec changes={(cE.argmax(-1)!=cB.argmax(-1))[pres].mean():.4f} clean-E preserve_KL={kl(cB,cE)[pres].mean():.4f}")
    presE=(~dE)&(~same); print(f"   E-pres n={presE.sum()}: clean-E preserve_KL={kl(cB,cE)[presE].mean():.2e}")
    print(f"   clean-E mediator IIA vs A1 target on A1diff={(cE.argmax(-1)==cA.argmax(-1))[dA].mean():.4f}")
```

Output:
```
alpha=0.800  (W,L): (36, 0)  U=WL alone: (36, 0)
alpha=0.333  (W,L): (1, 35706)  U=WL alone: (1, 35706)
[E-stratified] delta_all=0.0035 delta_on_A1diff=0.0030 delta_on_Ediff=0.0030 |A1diff|=4992 |Ediff|=4999
   A1-pres n=1006: E changes=0.0388 Bayes dec changes=0.0249 clean-E preserve_KL=0.0266
   E-pres n=999: clean-E preserve_KL=0.00e+00
   clean-E mediator IIA vs A1 target on A1diff=0.9970
[A1-stratified] delta_all=0.0465 delta_on_A1diff=0.0397 delta_on_Ediff=0.0185 |A1diff|=5041 |Ediff|=4974
   A1-pres n=957: E changes=0.3239 Bayes dec changes=0.1902 clean-E preserve_KL=0.2924
   E-pres n=1024: clean-E preserve_KL=0.00e+00
   clean-E mediator IIA vs A1 target on A1diff=0.9603
```

The A1-stratified reconstruction reproduces XI's logged counts exactly: n = 5041 and n_pres = 957 in `../step1_native_causal_construction/logs/decoy/*_A1.json`. The E-stratified reconstruction reproduces M5's n_pres = 1006, which confirms that M4/M5 used the E-stratified distribution for the A1 decoy.

XI logged diagnostics used for comparison (trained subspace): A1 runs IIA 0.926 / 0.923 / 0.912 with preserve_KL 0.164 / 0.158 / 0.224; E runs IIA 0.969 / 0.954 / 0.906 with preserve_KL 0.008 / 0.008 / 0.126.
