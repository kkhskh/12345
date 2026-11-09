# R1 — Formal mathematics, counterexample construction, positive-theory defence

Scope: THEOREMS_AND_COUNTEREXAMPLES.md (T1–T3, OD2, AD1–AD4, EX1), RELATIONAL_COMPUTATIONAL_CLASSES.md (RC and its
tables), TRANS_COMPOSITION_RESEARCH.md §1, ADAPTIVE_SOURCE_THEORY.md §2; code `src/tasks3.py`,
`../step1_foundational_atlas_continuation/src/tasks_ext.py` (`sample_trans`, `trans_joint`), `scripts/trans_identifiability.py`,
`scripts/od_check.py`, `scripts/adapt_theory.py`; logs `logs/trans/identifiability.json`, `logs/od_check.json`,
`logs/adapt/theory.json`, `logs/trans/restricted_bayes.json`.
Independent checks: two small numpy scripts (exact enumeration, < 1 min each) kept outside the package, at
`/tmp/r1/chk1.py` and `/tmp/r1/chk2.py`. Their outputs are quoted below.

Summary verdict: T1, T2, T2a and OD2 (K ≥ 3) are correct, ties included. T3 has two false sub-claims. AD2's
"any adaptive strategy" clause is false, and the package's own AD3 contradicts it. The consequence drawn from T1 only holds
under the product prior, so it fails for the holdout run's training distribution. RC is not circular, but it is
under-specified, and its "TRANS: no" verdict is too broad. EX1 is correct, but its corollary goes beyond what EX1 proves.

---

## Findings

### F1 (confirmation) — T1 holds under the actual masking/noise model; BC questions carry no σ information
Checked against `tasks_ext.sample_trans` / `trans_joint`. A masked report contributes likelihood 1 (`f` returns 1 when x = MASK).
In a BC question A is masked, so the B and C copies depend on Y only through u = σ(Y). With Y uniform and σ a bijection,
Σ_y K⁻¹ p_B(x_B|σ(y)) p_C(x_C|τσ(y)) = Σ_u K⁻¹ p_B(x_B|u) p_C(x_C|τ(u)), which does not involve σ. Two further points hold.
First, the question type is a deterministic function of the mask pattern and is drawn independently of (σ, τ). Second, each
duplicate report is drawn independently (`noisy` is called once per source).
Numerical check: under the product prior, the leave-q-out posterior over the 36 pairs equals the outer product of its marginals
up to 8.9·10⁻¹⁶ (2,000 episodes). T2, its proof (independent factors, one of them uniform on a finite group, give a uniform
product) and T2a (H(XY) ≥ H(XY | X) = H(Y) for independent X, Y) are correct. The entropy values 1.47 / 1.49 / 2.16 match the log.

### F2 (major) — T1a and the "Consequence for learning" depend on the product prior, but they are applied without that condition
T1a ("σ and τ a-posteriori independent; σ informed only by AB") requires the prior to factorise. The holdout run trains on
`TRAIN_PAIRS` ((s_i + t_i) mod 3 ≠ 0). That prior couples σ and τ, so BC questions do carry σ information through τ, and
a learner without τ evidence is not at chance. Exact check on stream 9,950,000 (6,000 episodes of training pairs), with the C
reports of BC questions masked:

| prior | AB | BC | C-only |
|---|---|---|---|
| product, full input | 0.950 | 0.661 | 0.493 |
| product, no C in BC | 0.950 | 0.645 | **0.328** |
| restricted, full input | 0.958 | 0.741 | 0.591 |
| restricted, no C in BC | 0.952 | 0.643 | **0.429** |

The sentence "a network that ignores C reports in BC questions … must be at chance" is therefore true only for the product
prior. The masking criterion for the holdout model (τ evidence removed: 0.562 → 0.416) should be compared with **0.429**,
not with chance. On that comparison, the 0.416 shows the model losing all τ use when τ evidence is removed, which is
consistent with the package. TRANS §5 alludes to this ("composite partly predictable from σ alone"), but THEOREMS §1 states no
scope. Fix: add "under independent priors on σ and τ" to T1a, T2 and the Consequence paragraph, and report 0.429 as the
σ-only reference for the restricted prior.

### F3 (major) — T3 contains two false statements, labelled [P, N]
(a) "With exactly one of either, the composite is restricted to 2 candidates." This is false when both map types have
exactly one value pair and those pairs are not chained (the B value of the AB pair differs from the B value of the BC pair).
Exhaustive enumeration over all (a→b, b′→c) gives **4** composite candidates when b ≠ b′, and 2 when b = b′. Concrete case:
σ(0)=1 and τ(0)=2 give the composites {(0,2,1), id, (1,2,0), (1,0,2)}. The package's own log shows the same thing. At
β = 1, (nAB, nBC) = (1,1), the composite entropy is 1.68 bits, which is above the 1 bit that 2 candidates would allow.
(b) "C-only decisions need ≥ 2 distinct AB and ≥ 2 distinct BC value pairs for certainty." This is false. Certainty
requires only the preimage (τσ)⁻¹(x_C), not the whole composite. Take AB pair 0→0, BC pair 0→0 and a scored x_C = 0. Every
consistent composite maps 0 to 0, so Y = 0 with certainty. The log's C-only accuracy of 0.568 at β = 1, (1,1) (above the 0.5
that pure 2-way ambiguity gives) is consistent with this. The first sentence of T3 (one pair leaves 2 permutations, two
distinct pairs identify the map) is correct. The same claim is repeated in MAIN_RESEARCH_SYNTHESIS item 21 and TRANS §1.

### F4 (confirmation, with one wording defect) — OD2 is correct for K ≥ 3, ties and the third value included
(⇐): R = {x = c} and P = {x = c′} are disjoint with |R| > |P|, and COND (strict) gives score_c > score_c′.
(⇒): with x = c on P, c′ on R and c″ elsewhere, score_c = w(P) and score_c′ = w(R) exactly, because the c″ block
contributes to neither. Then D_c′ < D_c while score_c′ ≤ score_c, which violates strict NSR. Equality is handled correctly,
because NSR demands a strict score inequality. COND with P = ∅ forces every w_s > 0 (α_s > 1/K), which is consistent with the
sampler. The confusion counts in `od_check.json` re-add correctly (COND true in 556 of 1,200 cases; band true in 325).
Defect: the name "no strict reversal" and the `od_check.py` docstring ("equivalently Bayes never strictly prefers a candidate
with strictly larger D") describe the **weak** property, D_c < D_c′ ⇒ score_c ≥ score_c′. That property is not equivalent to
the NSR defined here. It corresponds to weak COND (w(P) ≤ w(R)), which the same proof establishes. The docstring should be
corrected, and both versions stated. (minor)

### F5 (minor) — The K = 2 open obligation for OD2 is trivially closable and should not be listed as [O]
For K = 2 every report vector splits [S] into two complementary sets. The same two-line proof therefore gives NSR ⇔ COND
restricted to complementary pairs (|P| + |R| = S). Full COND is **not** necessary when K = 2. Weights (10, 4, 4, 4) satisfy
NSR for K = 2 (verified exhaustively) but violate COND (10 ≥ 4 + 4). For K = 3 the same weights violate NSR at
x = (0,1,1,2): scores (10, 8, 4), D = (3, 2, 3). This is a clean demonstration that the third value is essential.
Also: the "near-boundary draws" in `od_check` lie near the *band* boundary, not the COND/NSR boundary. They therefore do not
stress the equivalence specifically, although the proof makes the numerical check redundant anyway.

### F6 (critical for AD2 as stated) — "any adaptive strategy whose per-question copying probability is constant has the same law" is false
The first clause is correct. Honest-pair and static-copier laws are exchangeable on [K]², each fixed by its agreement
rate, and both are i.i.d. across questions. So matching ρ_c = 0.49 gives identical episode laws for the pair (2·10⁻¹⁷
confirmed). The second clause fails in two ways.
- **History-dependent copier.** Let the copy indicator be a stationary Markov chain with P(copy_q) = 0.49 in every question
  and persistence λ. Then P(agree_q ∧ agree_{q+1}) is 0.4911 at λ = 0.5 and 0.5245 at λ = 0.8, against 0.4356 for the
  honest pair. The pair's own episode law differs, so the copier is detectable in principle.
- **Copier that reacts to other sources.** Let the copier copy w.p. 0.742 only when the reliable pair agrees. Its
  per-question copy probability is the constant 0.49, and its own agreement rate equals the honest 0.66. But its joint law
  with the reliable pair is P(both agree) = 0.547, against 0.436 under independence. The package's own AD3 triggered copier
  also has a constant marginal copying probability ((1 − a_rel)·ρ_t = 0.306 in every question), and AD3 shows that its law
  differs. AD2 and AD3 contradict each other as written.
Correct statement: copy indicators i.i.d. Bernoulli(ρ_c), independent of Y, of the other sources and of the past, with the
copied value uniform and independent of the other sources. Scope must be stated as well: non-identifiability holds when the
copier's partner hypotheses involve only one Y-correlated group. With two honest groups, the cross-group co-occurrence
under Y separates an honest pair from even a static camouflaged copier. ADAPTIVE §2(iii) ("no estimator, hence no network")
inherits this over-generalisation.

### F7 (minor) — AD1: correct, but the proof is a sketch and the reported number understates the effect
Under ADAPT the per-question likelihood depends on X only through the agreement indicators, because noise is uniform and the
reliable pair's marginal depends only on agreement. So the within-class spread in `adapt_theory.py` is purely positional, and
the existence claim is witnessed numerically. That is adequate, but it should be labelled [N]-witnessed rather than [P].
"Differs by up to 0.35" is the maximum deviation from the class mean. The maximum pairwise difference inside a class can be up
to about twice that, so the reported value is conservative. The sufficient statistic named in §6 is (A, own reports X_q),
not A alone.

### F8 (minor) — AD4 is not a "lag"
The leave-q-out posterior is a smoothing posterior: it uses questions on both sides of q. The dip is symmetric: 0.361 at
q − c = −1, 0.364 at 0, 0.447 at −2 and 0.446 at +1. "After a change point … posterior lag" (THEOREMS) and "decision lag"
(ADAPTIVE §2) describe a filtering delay that is not present. The correct description is change-point localisation
ambiguity at the boundary questions.

### F9 (major) — EX1's corollary dismisses on-support predictors too broadly
EX1 is correct and trivial: no constraint holds over an unrestricted function class. It does not imply that "a predictor
using only the on-support input–output function cannot beat a constant beyond what architecture correlates with". The models
that actually occur are produced by a fixed architecture and optimiser, and that is exactly a class restriction, which EX1
explicitly exempts. Within such a population, fine on-support features (residual KL pattern, calibration, behaviour on
low-probability support points) may correlate with extrapolation. The realised pair (tf_s12, tf_s10) is one counterexample to
*determination*. It is not evidence of zero *statistical* predictive power, and the two models' KLs to Bayes still differ by a
factor of 2.4. Downgrade the corollary to "on-support behaviour cannot determine extrapolation; whether it predicts it within a
trained-model population is empirical".

### F10 (major) — RC is not circular, but it is under-specified, and "TRANS: no" goes beyond what was tested
Non-circularity: R is fixed from the generator before M is examined, condition 2 is tested against M's own counterfactual,
and condition 4's sign comes from the generator. That part is acceptable. Three defects remain.
(a) **No threshold.** "Interchange sufficiency ≥ a stated level" is never stated. T4 seed 12 at 0.47 is counted as "yes",
while the holdout run's input-level 0.64 / 0.63 counts only as "input-level only". The membership verdicts are therefore not
reproducible from the definition.
(b) **Type inconsistency.** Condition 1 requires R to be "a function of the reports", but for TRANS the table lists R = σ, τ.
These are latent parameters, not functions of X. R would have to be the co-occurrence statistics or the posterior over (σ, τ).
(c) **Wrong test for condition 2, and the verdict is too broad.** Condition 2 requires the mediator to reproduce *M's*
counterfactual decision. The TRANS-full "no" is justified by "C-only decisions do not follow map interventions (≈ chance)",
which is a following-Bayes test (condition 4). More importantly, RC membership is a property of (M, R), not of the hardest
question type. The full and aux models solve BC questions at 0.640 / 0.639 against Bayes 0.660. That requires σ⁻¹ inferred
from AB co-occurrences in *other* questions, which is a cross-question relation-mediated computation, and σ is decodable at
its ceiling. With R = σ (one-hop), these models plausibly satisfy RC. That was simply not tested. "No learned TRANS model
satisfied RC" and "fails … for TRANS because no member exists" (§6) should read: "no TRANS model was shown to satisfy RC with
the *composite* as relation; one-hop σ-mediation is untested". The holdout run's "internal mediator not localised" is also an
absence of evidence, not a non-membership result.

### F11 (minor) — Numerical-claim wording
"C-only Bayes accuracy is at chance (0.29–0.45)" mixes an exact fact (T2: exactly 1/3 in expectation) with Monte-Carlo
cells as small as about 27 C-only episodes, which is the source of the 0.45. State the exact value and note that the spread
is sampling noise. The AD3 cost "≤ 0.007" is a difference between two point estimates on 48k scored questions with no
interval given. It is plausible, but it is not an upper bound.

---

## Positive-theory defence (what survives)
- T1, T2, T2a are correct and give the right diagnosis for the full and aux runs: zero τ use implies exactly chance on C-only
  questions under the product prior.
- OD2 is a correct, complete characterisation for K ≥ 3, and K = 2 is a one-line extension (F5). Its interpretive caveat
  ("weights not identified by D") is appropriate.
- AD2's static core and AD3 are correct, and together they give a sharp boundary: per-pair statistics cannot detect
  copying, but cross-group dependence partly can. They should be restated so that they agree with each other (F6).
- RC's restrictive form, and its use of counterexamples (decodable but not mediating; salient but dispensable), is sound
  methodology. The defects are in specification (threshold, type of R) and in how the TRANS row was scored, not in the idea.
