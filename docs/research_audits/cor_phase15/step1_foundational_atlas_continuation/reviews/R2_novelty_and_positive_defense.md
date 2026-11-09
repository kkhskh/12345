# Review R2: established theory, novelty, and positive mathematical defense

Reviewer role: adversarial, two perspectives.
- (A) Established theory and novelty: closest prior work for each claim, a verdict (known / instance / possibly new), and whether a simpler explanation is enough.
- (B) Positive defense: valid restricted results that the package rejects, understates or mislabels, and the strongest claim each one can defensibly make.

Files read:
- BASELINES_AND_NOVELTY.md
- THEOREMS_COUNTEREXAMPLES_AND_LIMITS.md
- PP1_CALIBRATION_AND_INTERVENTION_THEOREMS.md
- COMPUTATIONAL_CLASSES_AND_EQUIVALENCE.md
- REPRESENTATION_AND_CAUSAL_IDENTIFICATION.md
- RELATIONAL_GENERATIVE_LANDSCAPE.md
- COR_TEN_FAILURE_MATRIX.md
- PROSPECTIVE_PREDICTIONS.md
- `../step1_relational_computation_atlas/BASELINES_AND_NOVELTY_AUDIT.md` (as context)

Computation: one exact enumeration over 3^6 = 729 report vectors (`/tmp/od_check.py`, run at nice 19, under a second, not kept in the package). Nothing else was run.

Reference status:
- **[v]** verified by a web search in this review.
- **[m]** cited from memory; the bibliographic detail should be checked before it is cited.

Severity: **critical** means a headline claim is false or unsupported as stated; **major** means a material overclaim, a missing baseline, or a missing key reference; **minor** means a wording, labelling or scope issue.

---

## Part A: established theory and novelty

### A1 (critical). The OD1 "iff" characterisation is false in the "only if" direction

**Claim.**
- THEOREMS §5: "the bridge holds iff the source semantics are trivial".
- COR_TEN F3: "sharpened into an iff-style statement".

**Refutation (exact).** OD1's own proof shows the Bayes log-score of candidate c is Σ_{s: x_s = c} w_s, with w_s = log(α_s(K − 1)/(1 − α_s)). Let A = {s : x_s ≠ c} and B = {s : x_s ≠ c′}.
- The discrepancy order D_c < D_{c′} agrees with the Bayes order whenever w(A \ B) < w(B \ A) holds for every pair of disjoint source sets with |A \ B| < |B \ A|.
- A sufficient condition is w_max/w_min < (m + 1)/m, where m = ⌊(S − 1)/2⌋. For S = 6 this is a ratio below 3/2.
- Unequal accuracies inside that band therefore give an *exact* bridge.

**Enumeration** (K = 3, S = 6, every report vector, checking whether the Bayes argmax lies in the arg-min-D set and whether every strict D order is a strict Bayes order):

| per-source accuracies | w_max/w_min | argmax violations | order violations |
|---|---|---|---|
| all 0.8 | 1.00 | 0/729 | 0 |
| 0.80, 0.80, 0.82, 0.82, 0.85, 0.85 | 1.17 | **0/729** | **0** |
| 0.78, 0.78, 0.85, 0.85, 0.90, 0.90 | 1.48 | **0/729** | **0** |
| 0.9, 0.9, 0.7, 0.7, 0.55, 0.55 (the package's case) | 3.23 | 54/729 | 114 |

So heterogeneous reliability, which is a non-trivial semantics, can coexist with an exact O/D bridge. The package's single unequal-α test point (0.9 / 0.7 / 0.55) lies outside the band, which is why the error went unnoticed.

**Prior art.** This is the classical question of when a weighted-majority rule coincides with simple majority (or plurality). Optimal rules are weighted majority with log-odds weights:
- Nitzan and Paroush 1982, Int. Econ. Rev. 23(2):289–297 [v];
- Shapley and Grofman 1984, Public Choice 43:329–343 [m];
- the polychotomous (K-ary) case: Ben-Yashar and Paroush 2001, Social Choice and Welfare 18(4):737–746 [v].

Whether two weight vectors induce the same rule is a threshold-logic or weighted-voting-game representation question (Isbell; Taylor and Zwicker, *Simple Games*, 1999 [m]).

**Required fix.** Withdraw "iff trivial semantics". The correct characterisation for conditionally independent sources with symmetric errors is combinatorial:

> The D-ranking equals the Bayes ranking on all report patterns iff, for all disjoint source sets P and Q with |P| < |Q| and |P| + |Q| ≤ S, w(P) < w(Q).

Copying and uninformative roles break the bridge for a different reason: zero weights, and dependence that violates the product form. That part of the determination stands.

### A2 (major). C4 does not quantitatively explain PP1's 0.996+ sign agreement; the explanation is the empirical C1 coverage

**Claim.** PP1 §4: "W ≈ 1 here, so C4 reduces to 'natural KL ≈ 10⁻³ plus a margin tail'. That explains the 0.996+ agreement *mechanistically*". Also COR_TEN F1, and K1.

**Check with the package's own numbers** (ε ≈ 0.0015, W ≈ 1.2):
- C3/C4 need τ < π ≤ 0.5.
- The off-calibration term Wε/τ² is about 0.045 at τ = 0.2, and is never below 0.0072 for any admissible τ. A 0.4% failure rate is therefore not reachable by the bound at all.
- At τ = 0.2, b_τ = 2τ/(π − τ) ≥ 1.33. The margin-tail term then covers every edit with |Δm| ≤ 2.67, which is a large part of a scored set defined by |Δm| > 0.5.

C4 is valid but **quantitatively vacuous** on PP1. The package itself reports the symptom: the C2/C3 certificates cover only 66–92% of edits.

**What explains the 0.996 is C1 coverage (98.6–99.7%).** C1 is an identity: Δm̂ = Δm + e(x′) − e(x). So "C1 never violated in 10 models [E]" is a code sanity check, not evidence. Report it as such.

**Simpler explanation.** The models are *pointwise* close to Bayes on almost all endpoints. That is stronger than average KL closeness, and Markov cannot recover it from average KL. "Sign agreement is not evidence about the computation" is still correct: pointwise Bayes-closeness is implementation-independent.

**Prior art.**
- Importance-weighted covariate-shift bounds: Shimodaira 2000, J. Stat. Plan. Inference 90:227–244 [m]; Cortes, Mansour and Mohri 2010, NeurIPS, "Learning bounds for importance weighting" [v]. These use Rényi-divergence (moment) conditions instead of a sup bound W, and give tighter tails.
- C1 is the logit-margin certificate familiar from margin-based robustness; the "known in principle" verdict is right.

### A3 (major). The relative-count "abstraction" test is explained by T1 KL-closeness, and the K2 counterexample is invalid

**Claims.**
- COMPUTATIONAL_CLASSES §2: "an abstraction property no input–output accuracy test checks".
- THEOREMS §6 K2: "a model that is KL-exact on T1 need not be in 𝒦_rel. Take Bayes plus a perturbation supported on uniformly shifted inputs of measure zero under T1."

**Refutation.** T1 has no colluder. Reliable and noise sources give every value positive probability (PP1 §1 states this for those roles). So **every** input, including every uniformly shifted episode, has positive T1 probability. There is no measure-zero shifted set.
- KL-exactness on T1 therefore implies equality with T1-Bayes everywhere, and hence membership in 𝒦_rel.
- K2's counterexample does not exist for T1.
- K2's positive part ("a model in 𝒦_rel has no response to uniform shifts") is true by definition, since a uniform shift preserves relative counts. It is a tautology, not a theorem.

**Simpler explanation.** The test is an input–output test *on T1-support inputs*. The observed shift-KL (0.0004–0.0007) is below the models' own natural KL (about 10⁻³). The invariance ratios of 0.02–0.04 are what any T1-Bayes-close model gives, so no abstraction-specific property is shown.

**Prior art.** Shift-invariance of normalised exponential-family likelihoods (softmax invariance to a common additive term): textbook. Causal-abstraction framing (Geiger et al.) is not needed.

The T4-on-T1 violations are off-T4-support behaviour (PP2), as the package says.

### A4 (major). R1 is the quotient-game step of the Owen value applied to merged clones

**Closest prior work.**
- Owen 1977: the Owen value's union totals equal the Shapley value of the quotient game [m; the property is standard, see e.g. arXiv:2504.14230 [v]].
- Splitting and merging non-proofness of the Shapley value: Knudsen and Østerdal 2012, Int. J. Game Theory 41(4):763–774 [v].
- Group Shapley in ML attribution: Jullum, Redelmeier and Aas 2021, arXiv:2106.12228 (groupShapley) [v].
- Duplicate or correlated features splitting credit: Kumar et al. 2020, ICML, "Problems with Shapley-value-based explanations" [m].
- The gauge part is the GL(d_h) symmetry (da Silva et al. 2025, already cited). Net2Net (Chen et al. 2016 [m]) supplies the split.

**Verdict.** **Known in principle; the proof is immediate.** In the seed-10 check no two original heads share a pattern, so class Shapley equals head Shapley, and the split clones are simply merged back. Add Owen, Knudsen–Østerdal and groupShapley to BASELINES. The package's own verdict ("known in principle") is right; the missing piece is the citations.

### A5 (major). V1 is Nitzan–Paroush log-odds weighting with a copier counted once

The threshold (K − 1)a_c/(1 − a_c) > ((K − 1)α/(1 − α))² is a one-line corollary of optimal weighted majority (Nitzan and Paroush 1982 [v]; Ben-Yashar and Paroush 2001 for K-ary [v]), with a deterministic copy contributing a single vote.
- Dong et al. 2009 (PVLDB 2(1)) gives the vote discounting for copiers.
- Correlated-vote Condorcet literature: Ladha 1992, Am. J. Pol. Sci. 36(3) [m]; Berg 1993, Public Choice 75 [m].

**Verdict: known.** The package says so. Add the social-choice references, which are closer than Dong et al. for the *threshold* form.

### A6 (major). The PP2 direction has an architectural explanation the package does not cite

**Simpler explanation.** A DeepSets readout ρ(h_q, m, X_q) sees the episode only through a pooled summary m. Trained only on T4 ("exactly one colluder, which always agrees"), the cheapest pooled code is "colluder = argmax agreement". Carried off support, that is the global plug-in.

Per-question readout attention in Transformers can act locally. The "probe picks the top-agreement group" signature (1.00 for every DeepSets model) is what that code looks like from the inside. The −0.89 Spearman correlation across all 10 models is driven by the 3 DeepSets models; it is −0.71 within Transformers (n = 7).

**Prior art.**
- How aggregation choice and architecture fix extrapolation: Xu et al. 2021, ICLR, arXiv:2009.11848 [v].
- Set-function representation limits: Wagstaff et al. 2019, ICML [m].
- Pretraining-prior Bayes estimators: Raventós et al. 2023, already cited.
- ICL as Bayesian under a prior: Panwar et al. 2024, ICLR [m].

**Verdict.** The direction is an **instance** of architecture-determined extrapolation. The only open content is the *within-Transformer* spread (0.31–0.87). P-A1 is correctly framed around beating B1, the architecture indicator.

### A7 (major). The P-A predictors use off-support inputs, so "T4-derived inputs only" is misleading

A1 (CCP_3) and A2 (LPI_deg) are computed on T4 episodes with the colluder's agreement broken in 3 questions. Under ρ = 1 such episodes are outside S2, and usually outside T4 support. So P-A predicts one off-support behaviour from another off-support probe.
- A2 (development MAE 0.041) is close to a proxy of the target: the same local-versus-global preference, under a milder shift.
- A pass for A2 would show behavioural transfer between two shifts, not structural prediction.
- A1 is the only predictor of interest, and development shows A1 ≈ B1.

Do not edit the frozen text. In the outcome section, state that the predictors' inputs are off-support.

### A8 (major). The "misleading relation" decision failure lacks the T4-Bayes baseline

COR_TEN §1 reports 0.63–0.65 disagreement with the true answer when an honest group agrees in all 6 questions on p_c = 0 episodes.
- Those episodes are inside T4 support (P-A3 says so), so T4-Bayes itself discounts the honest group.
- The likely simpler explanation is that the models equal T4-Bayes there (KL ≈ 10⁻³), and the disagreement is prior mismatch rather than a model or mechanism failure.
- Report the T4-Bayes disagreement rate on the same episodes. Without it, the row in the six-level table ("improves correctness: no") reads as a model property.

**Prior art.** This is the paradox of unanimity (Gunn et al. 2016, Proc. R. Soc. A [m; already in the atlas list]) under prior shift.

### A9 (minor). The count-boundary reversal (0.111) is the paradox of unanimity

The 5 ↔ 6 boundary, where full agreement becomes possible only for the colluder, is Gunn et al. 2016 in this family. Cite it at F2 and in the decision ledger.

### A10 (minor). "S needs (O)" is the standard cross-annotator identifiability of latent-class models

Cross-agreement between conditionally independent informative sources identifies their accuracies:
- triplet / method-of-moments: Fu et al. 2020, ICML, arXiv:2002.11955 [v];
- spectral Dawid–Skene: Zhang, Chen, Zhou and Jordan 2014, NeurIPS [m].

So counts alone not being sufficient when two answer-dependent groups exist is expected. **Known in principle.** Add the references.

### A11 (minor). The seed-12 necessity/sufficiency dissociation is standard

The "instance" verdict is right. Add the formal counterpart: probabilities of necessity versus sufficiency (Tian and Pearl 2000, Ann. Math. AI 28 [m]). For interchange on a jointly-read variable producing inconsistent states, cite Mueller 2024, "Missed causes and ambiguous effects" [m], alongside Makelov et al. 2024.

**Overclaim.** "Class C is therefore not a different operation from A′. It is A′ plus a parallel rereading channel" is bold-faced as a conclusion. Its key premise, the joint tag + content interchange, was not run. Mark it as a hypothesis, consistent with the "Status" paragraph that follows it.

### A12 (minor). Calibration ≠ ranking (C5): prior art and a missing witness

- Temperature scaling preserves the argmax: Guo et al. 2017, ICML [m].
- Calibration/refinement decomposition: Murphy 1973; DeGroot and Fienberg 1983 [m].

"ECE is not *sufficient*" is asserted but not witnessed by TEMP or CX1, both of which have high ECE. The trivial witness: the constant marginal predictor has ECE = 0 and Δm̂ ≡ 0. Add it.

### A13 (minor). Smaller scope and consistency issues

- **ρ → 1 singularity.** "The configuration posterior is continuous in ρ on [0, 1]" holds only for episodes with at least one configuration of positive likelihood at ρ = 1. Many episodes sampled at ρ = 0.6 have no fully agreeing group; for them the ρ = 1 posterior is 0/0, and only the ρ ↑ 1 limit exists. State how the 52% flip at ρ = 1.0 was computed (limit, or a clip).
- **Labelling of F1.** RELATIONAL_GENERATIVE_LANDSCAPE §2.3 calls the overlap result "COR failure F1: a fixed relation grammar". COR_TEN F1 is "missing noncircular causal bridge". Reconcile the labels.
- **OD1 prior.** OD1 also needs a uniform prior over Y. With a non-uniform prior the bridge fails even for identical sources. Add the assumption.
- **Verdicts confirmed.** The hidden-pairing ≤ 1/3 bound, misspecified overlap (White 1982), U failing for mixtures (a sum of quasi-concave functions need not be quasi-concave), and the ρ* line (label switching) are correctly labelled known or known in principle.

---

## Part B: positive mathematical defense

### B1 (major). OD1 has a stronger valid theorem than the package states (companion to A1)

**Strongest defensible restricted claim:**
- **OD1⁺.** For conditionally independent sources with symmetric K-ary errors and a uniform prior on Y, discrepancy ranking equals Bayes ranking exactly iff the log-odds weights satisfy the disjoint-set condition in A1. A sufficient condition is w_max/w_min < (m + 1)/m, m = ⌊(S − 1)/2⌋.
- **OD1″.** With a shared but *unknown* accuracy α and any prior supported on (1/K, 1], cross-question information still cannot change the ranking. The predictive score ∫ f(α | other questions) · α^{S − D_c} e(α)^{D_c} dα is strictly decreasing in D_c. So the result does not need α to be known.

This is a real if small "if and only if" theorem, replacing the false one. The determination "O/D fails for copying and uninformative roles" survives. "O/D fails for any heterogeneity" does not.

### B2 (major). The interior support rule (1.000) is very probably a theorem, not just an empirical regularity

COR_TEN §1 offers "inside the interior of the T4 count range, the support rule's sign holds for every scored edit (1.000)" as an empirical [N] rule.

**Proof sketch.**
1. Under T4 (ρ = 1), a group whose total within-group agreement is ≤ Q − 1 before and after the edit has zero colluder probability at both endpoints.
2. Its configuration likelihood ratio, reliable versus noise, is multiplied by (a_rel/a_n)·((1 − a_n)/(1 − a_rel)) > 1 when agreement rises (a_rel = 0.66 > a_n = 1/3 at α = 0.8).
3. So P(g reliable) rises, and by Theorem D the margin of g's own candidate rises.

**Strongest defensible restricted claim.** Under T4, for edits that keep the edited group below full agreement, the generator-free support rule is a *guaranteed* sign rule. At the full-agreement boundary it provably fails, by the paradox of unanimity. This gives F2 a proved region with a proved counterexample at its edge, which is stronger than an empirical 1.000 versus 0.111.

The authors should check the proof against the exact definition of "own candidate" used in the ledger.

### B3 (minor). I2 is a theorem, labelled only [N]

Two facts suffice:
- under symmetric errors and Y marginalised, a reliable pair's per-question law is "agree with probability a_rel, value uniform; else an ordered distinct pair, uniform";
- the colluder pair's law is the same with agreement ρ + (1 − ρ)/K.

Equating the two agreement rates gives ρ* and exact invariance of the episode law under the role swap. Label I2 [P] (as the atlas proved), with [N] as confirmation.

### B4 (minor). K2 can be replaced by a true and stronger statement (companion to A3)

**Strongest defensible restricted claim.** On a full-support generator such as T1, KL-exactness implies every invariance of the Bayes predictor, including 𝒦_rel.

**Quantitative version.** Let W_s be the density ratio of shifted to natural T1 episodes. Then E_shift[KL(Bayes ‖ model)] ≤ W_s · ε, and uniform-shift KL can be compared directly against that budget. This is an honest "abstraction certificate" in the style of C4. The genuine abstraction question only arises for generators with *partial* support (T4), where the package already locates it (PP2).

### B5 (minor). PP1: state the defensible claim at the right strength (companion to A2)

The package's self-assessment ("not evidence about the computation") is correct. It slightly undersells one checkable, implementation-independent fact.

**Strongest defensible claim.**
- C1 is a per-edit certificate computable from the model's own margin errors.
- In 10/10 models it certifies 98.6–99.7% of scored edits, and every sign failure is uncertified.
- That is a sharper and *falsifiable* account than C4.

A non-vacuous distribution-level version would need tail information beyond Markov: the empirical distribution of t(x), or a Rényi-moment condition as in Cortes et al. 2010.

### B6 (minor). R1 and L1 together are a legitimate small result

R1 is known in principle (A4), but the package is right to keep it. The *pairing* with L1 is clean:
- R1: invariance under pattern-preserving reparametrisations;
- L1: no pattern-sensitive head attribution can be invariant under pattern-changing, function-preserving ones.

L1 is stated informally. It can be made a proposition: any attribution invariant under the full function-preserving group is a function of the computed function plus the site, so it cannot separate heads with distinct patterns that compute the same total.

R1 also extends at once to the Owen value, which allocates within classes so that clones get equal halves. That gives a two-level consistent attribution, not only class totals.

### B7 (minor). The net-novelty statement is accurate, with one item understated

"Nothing ... is a new *mathematical* theory" is correct after A1–A5. Two items are reasonably described as possibly new observations and could be stated as such without overclaiming:
- the S1/S2/S3 support taxonomy for evaluating input-level interventions (PP1 §1), together with the concrete finding that the transformer sign errors concentrate in S1 \ S2;
- the quantitative necessity (0.91) versus sufficiency (0.47) dissociation of one identified subspace across seeds.

Both are small. Neither should be inflated.

---

## Summary of required changes (no frozen text to be edited)

1. Withdraw the OD1 "iff trivial semantics" and the F3 "iff-style" status. Replace them with OD1⁺/OD1″ (A1, B1).
2. Replace "C4 explains PP1 mechanistically" with "pointwise closeness, as certified by C1, explains PP1; C4 is valid but vacuous at these ε". Reword "C1 never violated [E]" (A2, B5).
3. Withdraw the K2 counterexample and the "abstraction property no input–output test checks" claim for T1. Optionally add B4 (A3).
4. Add the T4-Bayes baseline to the misleading-relation row (A8).
5. In the P-A outcome, disclose that A1 and A2 use off-support inputs (A7).
6. Promote the interior support rule and I2 to [P], after checking the proofs (B2, B3).
7. Add the missing prior art: Nitzan–Paroush 1982; Ben-Yashar–Paroush 2001; Shapley–Grofman 1984; Owen 1977; Knudsen–Østerdal 2012; Jullum et al. 2021; Cortes–Mansour–Mohri 2010; Xu et al. 2021; Fu et al. 2020; Gunn et al. 2016 (at F2); Guo et al. 2017; Tian–Pearl 2000.
