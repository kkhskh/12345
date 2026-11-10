# Review R-formal: formal causal equivalence and theorem validity

Reviewer perspective: formal definitions, theorem correctness, and whether the main answer follows. Independent checks:
`reviews/checks/r_formal_checks.py` -> `logs/review_checks/r_formal_checks.out`; `reviews/checks/r_formal_baselines.py` ->
`logs/review_checks/r_formal_baselines.out`. Both use exact rational arithmetic and their own enumeration over the generator
(`gen_family.Fam()`, T4). Neither reuses `eqv.pi_of_n`.

## What checks out (independently verified)

- **A1** holds exactly: 0 mismatches over 6 configurations x 729 values of X_k. The constants are exact: f(1) = 11/50,
  f(0) = 17/300, rho = 66/17 = 3.882.
- **A2** follows from A1 and the uniform prior (pc = 1).
- **A4**: I compared the characterisation with exact equality of rational pi over all 91 x 91 ordered pairs and found 0
  mismatches. There are **43 classes**: 27 with a single colluder group (75 vectors, 9 difference values per h), 15 with two
  colluders (15 vectors), and 1 for (5,5,5). Class sizes are {1: 22, 2: 6, 3: 6, 4: 6, 5: 3}. The 12-decimal rounding in
  `info_theory.py` did not merge or split any classes.
- **B1**, **C1** (with the operator norm), **C4**, **D1** and **D2** are correct as constructions.

## Findings

**F1 (major): A3's formula is wrong as written.** A3 writes P(Y_q | X) = sum_R pi(R) P(Y_q | X_q, R). The correct mixture
weights are P(R | X) ∝ pi(R) P(X_q | R), not pi(R), so the correct statement is
P(Y_q | X) ∝ sum_R pi(R) P(X_q, Y_q | R). With the stated weights the posterior is off by up to 0.27, and the argmax changes on
3.0 % of 18,000 decisions (check output). The code (`bayes_from`, which uses the joint J) is correct, so no number in the
package is affected. *Fix:* restate A3 with the joint likelihood P(X_q, Y_q | R).

**F2 (major): B2 and D3 are overbroad.** Their own data falsify them. B2 says "no single token ... carries a Bayes-sufficient
statistic" and "a Bayes-sufficient decision must read tokens of all three groups (in T1: layer-2 attention from 36+q to S_T(q))".
This holds only under the hidden assumption that each token carries only its own-group component beta_s. A token can carry all
of n: the P-count readout is a single token that does. In T2, n decodes at 0.93 at the readout at layer-2 input
(COUNT_INFORMATION_STRUCTURE §3), so earlier layers have already aggregated across groups and a layer-2 read of S_T(q) is not
logically necessary. D3 has the same issue. *Fix:* restrict both to "no single own-group component / no tuple missing a group",
and drop "must".

**F3 (major): D5 is false as stated, and its practical worry is understated.**
- *The statement is false.* (a) log pi(g,h)/pi(g',h) is additive only for unnormalised log-weights n_g log rho, and those are
  a function of n, not of pi. (b) I computed the space of additive one-hot functionals (19-dimensional with intercept) that are
  constant on every pi class. It is 7-dimensional, and every member is a function of H = {h : n_h = 5} alone (within-H
  variation 4e-16). So no site that stores "only a function of pi" carrying more than H can be "fitted exactly by the additive
  one-hot encoder".
- *The real concern is quantitatively live for P-count.* The best per-component accuracy of decoding n from pi alone (exact
  MAP over the natural distribution) is **0.727**. Reference points: 0.376 with no information, 0.549 from H only. The P-count
  site decoders score 0.717–0.785 (P1–P4), which brackets this value; P2 (0.717/0.730) is at or below it. An additive fit to
  log pi already gives R² 0.92 (P-count readout from counts: 0.95–0.97).
- *Consequence.* The P sites are quantitatively consistent with holding roughly the minimal sufficient statistic pi, plus a
  small seed-specific remainder. That is a different variable from V, not merely "a lossy code of V".
- *Fix:* restate D5 as approximate. Report the pi-only decoding bound (0.727) next to the P decoders. Ideally, test directly
  whether P decoders separate pi-equivalent count vectors (within-class confusion matrix).

**F4 (minor): D4 is muddled and half-proved.** It says interchange works "past saturation" and that ablation changes nothing
"once V2 saturates". If V2 saturated sigma, interchanging V1 would also do nothing; the construction actually relies on a
threshold. "Either direction" is claimed but only one direction is built. The reverse direction is easy: y = sign(V1·V2) with
V in {±1}. Interchanging V1 gives IIA 0 on informative cases, while ablating V1 := 0 changes every output (check output).
*Fix:* use a threshold readout and add the product counterexample.

**F5 (minor): B3 is tautological.** The "abstract transition system" x -> x' => v(x) -> v(x') is defined by the generator and
does not involve M at all. *Fix:* say "by definition"; it should not be called a theorem.

**F6 (minor): sloppy wording in A and in the information table.**
- **Notation:** rho is reused for f(1)/f(0) = 3.88, while the generator's rho = 1 is the collusion probability.
- **A5:** insufficiency of subsets is shown by Monte Carlo accuracy (SE ≈ 0.003). It follows exactly from A4, since pi depends
  on all three components.
- **"pi minimal sufficient":** this is true for R and as a statistic uniform in X_q. For the decision at a given X_q it is not
  minimal: over the 513 possible X_q, the 43 pi classes give between **1** and 43 distinct posteriors (median 11).
- **Comparability:** COUNT_INFORMATION_STRUCTURE §1 says single components and pairs are "incomparable with each other". But
  n_g ≼ (n_g, n_g') is comparable.
- **Seed count:** "two of four P seeds ... at this level" should be four of five (P1–P4).
- **pi-preserving fraction:** "81 %" (doc) vs 0.829 (`info_theory.json`, q = 0); state which one is meant.

**F7 (major): the equivalence definitions are not precise, and they are not "nested".**
- **(a) Tolerances.** No epsilon is ever fixed. E-Info is operationalised as *per-component* accuracy, not joint recovery of V.
  Under any epsilon < 0.2, P1–P4 fail E-Info, yet the table calls them "lossy/partial".
- **(b) Quantifiers.** E-Int is written pointwise ("for (x, x') in I: delta <= epsilon"), but delta (argmax disagreement or
  1 − r) is a population average. C3 and B3 then use E-Int pointwise.
- **(c) Normalisation.** The ceiling is ill-posed: adjusted IIA reaches 1.03 (T11 single) and 1.82 (T2 cond), and T2 is
  excluded post hoc.
- **(d) Dependence on fitted maps.** g_M is a property of the pair (M, E_M) with a least-squares E_M, not of M alone.
  E-DR is therefore encoder-relative, and this should be stated.
- **(e) Not nested.** E-Int does not presuppose E-Info (it uses the encoder, not the decoder). P1–P4 satisfy E-Int (adj
  0.76–0.95) while failing E-Info, which is a direct counterexample to "four nested notions" (RESULT, "What equivalent means").
  E-DR is cross-model, while E-Int and E-Info are per-model; only E-Comp ⇒ E-DR holds by definition.
- **(f) Experiments test a different criterion.** The frozen VI criterion is *relative* (cross − within_T < −0.10 on Pearson r
  of residuals). This is not the absolute E-DR (delta <= epsilon) that the definition states.
- *Fix:* state epsilon, the averaging measure and the decoder class in each definition; drop "nested", or give the actual
  implication graph.

**F8 (major): C2's "equivalent" is only a sufficient condition, and the package's own data contradict the converse.**
- C1 gives (exact decodability) ∧ (E-Int(R)) ⇒ transport succeeds. The converse fails whenever decoder errors fall in
  directions the recipient ignores.
- *Constructed counterexample:* a donor decoder that returns a canonical member of n's pi class (per-component accuracy 0.727)
  and a pi-only (Bayes) recipient give exactly the oracle response (check output: True).
- *Learned counterexample:* P self-transport is exact (0.97–1.05) with decoders at 0.72–0.78 (CROSS_MODEL §2). Success
  without decodability.
- *Fix:* replace "is equivalent to" with "is implied by". Add that transport success identifies only the coarsening of V that
  the recipient uses.

**F9 (major): C1's error bound is correct but cannot carry the explanation it is used for.**
- **Scale of the bound.** It bounds the *activation* change, with ‖W_R‖ as an operator norm. A bound on the *response* also
  needs a Lipschitz constant of h_R on the patched region. With soft one-hots, ‖alpha_hat − e‖ can reach √2 per component, so
  for decoders at 0.72–0.78 the bound is vacuous.
- **Asymmetry.** For a fixed donor, the bound differs across recipients only through ‖W_R‖. What drives the asymmetry is the
  recipient's sensitivity J_{h_R} W_R projected onto the donor's error directions. So "P→P worse than P→T *because* P sites are
  lossy" (RESULT item 2) is not explained by the bound: the donor is the same in both cases, and the difference is a recipient
  property.
- **Reading consistent with F3/F8.** P donors err within (approximately) pi classes. T recipients respond to pi-preserving
  changes less idiosyncratically, while each P seed keeps a different seed-specific remainder.
- *Fix:* state the response-level bound with L_h, and attribute the asymmetry to the recipient.

**F10 (major): C3's hypotheses are not met, so its practical corollary is weaker than stated.**
- C3 needs (i) calibrated Bayes logits (argmax-optimality gives only argmax agreement, not equality of centred-logit
  responses) and (ii) *exact pointwise* E-Int on every natural-support (x, v').
- On (ii): an abstract do(V) leaves the correlated copies of V elsewhere in W_M unchanged. P source tokens decode n at
  0.62–0.85, so even "natural-support" abstract states are joint off-distribution states, and optimality constrains them only
  through E-Int.
- E-Int fails materially on the very family used by VI. pi-preserving adjusted IIA is 0.63–0.94 in P, −0.29 in T2 and 0.56 in
  T11. The *abstract* pi-preserving responses therefore mix model deviations from Bayes with commuting-square failure, and the
  latter is encoder-specific (it inflates idiosyncrasy).
- *Mitigation already in the data:* the encoder-free pi-preserving *input* edits reproduce the ordering (within T 0.53,
  within P 0.03, across 0.21).
- *Fix:* state C3's hypotheses explicitly. Make the input-edit family primary evidence for VI-sep, and the abstract family
  secondary.

**F11 (major): the E-Comp "not shared route" result is forced by the choice of site, not discovered.** The mediator
decomposition is at the readout. Any implementation whose site is off the readout must reach the decision through layer-2
attention: CROSS_MODEL §4 itself says "necessarily so". Hence "attention frozen: 0.00 in T vs 0.45–0.91 in P" follows from
B1 + site location and is not evidence for H3 about the V→decision map. *Fix:* present RESULT item 3(b) as a consequence of the
different site, not as evidence of different computation. A discriminating E-Comp test would compare the readout's processing
*after* V has arrived there (e.g. intervene on T's readout state after attention vs P's readout).

**F12 (critical): the main answer overstates sharing. "Instantiate the same identifiable causal relational variable (with
P-count's realisation coarser)" does not follow from the theorems and data.**
- **(a) "Identifiable" is undefined.** By D1/D5/C2 the variable is identified only up to the coarsening the site supports.
- **(b) P sites hold roughly pi, not V.** For P1–P4 the coarsening is quantitatively close to pi (F3: the pi-only bound 0.727
  sits within the decoder range 0.72–0.79). E-Int with a V-indexed encoder shows V is a valid *intervention handle*, not that V
  is the stored variable (the D5 logic).
- **(c) No single variable is shared even among P seeds.** P→P transport fails (0.28–0.66) while self-transport is exact (F8),
  so different P seeds hold *different* coarsenings.
- **(d) What the data support** is that T models (and P0) realise V exactly, while P1–P4 each realise a seed-specific
  statistic between roughly pi and V. The variable common to all six is at best around pi, the posterior over configurations.
- **(e) Sharing pi is weak evidence.** Any near-Bayes model must carry roughly pi somewhere (the C3 logic), so sharing it is
  close to a consequence of shared task performance, not evidence of a shared mechanism.
- **(f) Item 2 is not a sharing result.** By C1/C2, "shared in one direction" (T→P exact transport) is implied by T
  decodability + E-Int(P), both already established.
- *Suggested rewrite:* "Both implementations carry and causally use a relational statistic sufficient for the decision
  (approximately pi). V itself is realised exactly only in T1 models and one P seed, and the P seeds' realisations are
  seed-specific coarsenings. V serves as a common intervention handle; the downstream use differs."

**F13 (minor): D2's "learned instance" is not an instance.** D2's hypothesis is that two models are identical on natural
episodes, and the learned models are not. The off-support family only illustrates that responses there are unconstrained.
*Fix:* call it "illustration".

## Verdict table

| item | verdict | severity of issue |
|---|---|---|
| A1, A2 | correct (exact) | — |
| A3 | conclusion correct, formula wrong | major (F1) |
| A4 + 43 classes | correct (independently verified exactly) | — |
| A5 | correct; better proved via A4 | minor (F6) |
| B1 | correct | — |
| B2 | overbroad; contradicted by T2 readout | major (F2) |
| B3 | tautological | minor (F5) |
| C1 | correct at activation level; output-level bound missing | major (F9) |
| C2 | "equivalent" false; only sufficiency; learned counterexample in data | major (F8) |
| C3 | conditional on unmet exact hypotheses | major (F10) |
| C4 | correct, near-definitional | — |
| D1, D2 | correct; D2 "learned instance" mislabelled | minor (F13) |
| D3 | overbroad (same as B2) | major (F2) |
| D4 | imprecise; reverse direction missing (counterexample given) | minor (F4) |
| D5 | false as stated; practical risk larger than stated | major (F3) |
| definitions E-Info/E-Int/E-DR/E-Comp | epsilon, measure, quantifiers unspecified; not nested | major (F7) |
| E-Comp cross-model result | forced by site choice | major (F11) |
| main answer "same identifiable variable" | overreaches; supported only near the pi level | **critical** (F12) |
| Theorem C applied to transport asymmetry | explanation not supported by the bound | major (F9) |
