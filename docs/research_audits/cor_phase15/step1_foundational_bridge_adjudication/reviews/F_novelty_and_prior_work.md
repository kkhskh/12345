# Review F: novelty and prior work

Reviewer F (adversarial). Read: `CAUSAL_BRIDGE_MAIN_ARGUMENT.md`, `proofs/PROOFS.md`. No package file was modified except this one. The literature search was web-based and is not exhaustive. Where I could not confirm something, I say so.

## Bottom line

Every mathematical result reduces to established work. SS is residual-stream linearity. NC is the final-LN "norm/temperature channel" (Stolfo et al. 2024; Gurnee et al. 2024), and Corollary CI-5 is algebraically the Rushing and Nanda (2024) LayerNorm decomposition. CI is the weighted-mean update, which is the finite-sample influence function of a mean. RB.1–RB.5 are textbook additive conjoint measurement. RB.6 ("prospective forcing") is the *necessary preference relation* of robust ordinal regression (Greco, Mousseau, Słowiński 2008), computed by the same LP.

Three things look new, and they are applications rather than theorems:
1. Treating "consistency has a causal role" as an ordinal *nonremovable-interaction* question over canonical Transformer variables.
2. The exact NC flip criterion, including the limit toward the bias ranking `κ`.
3. The specific falsifiable prediction that sub-consensus supportive evidence lowers the margin (dilution), set against additive attribution.

The package itself mostly concedes this (§8, §10: "the mathematics of each link is known"). Two gaps remain: it misses the Rushing and Nanda identity for CI-5, and it misses the robust-ordinal-regression precedent for RB.6.

---

## Theorem SS: same-cut sum screening

- **Closest prior work:** Elhage et al. 2021, *A Mathematical Framework for Transformer Circuits*: https://transformer-circuits.pub/2021/framework/. The residual stream is a sum of component writes, and heads in a layer act independently and additively. Every downstream function reads only the sum.
- **What is known:** (i) and (ii) follow directly from the definition of an additive cut, which is the basis of DLA and path patching. (iii) says a non-sum statistic is causally inert given the sum. That is the standard fact that a function of `S` is constant on the fibers of `S`.
- **What may be new:** Nothing mathematical. Its use as a negative result against "local agreement among same-layer components" mechanisms is a legitimate and useful framing for this project.
- **Verdict: KNOWN.**

## Theorem NC: final-LN norm channel

- **Closest prior work:**
  - Stolfo et al. 2024, *Confidence Regulation Neurons in Language Models* (NeurIPS 2024): https://arxiv.org/abs/2406.16254. Entropy neurons write onto an *effective* null space of `W_U` (bottom singular directions), raise the residual norm, and through final LN rescale logits. The paper reports "little impact on the relative ranking of the tokens". It does not discuss `β_f` or a flip criterion. Its token-frequency neurons are a separate mechanism (writing along a log-frequency direction), not the norm channel.
  - Gurnee et al. 2024, *Universal Neurons in GPT2 Language Models* (TMLR): https://arxiv.org/abs/2401.12181. The entropy neuron's output is mostly orthogonal to `W_U`, it raises the final LN scale, and the paper compares the effect to "manually increasing the temperature".
  - Rushing and Nanda 2024, *Explorations of Self-Repair in Language Models* (ICML 2024, PMLR 235): https://arxiv.org/abs/2402.15390. With LN folded in, logit ∝ ⟨resid/S, L⟩. They decompose `Δlogit = (S/S' − 1)·logit + (S/S')·ΔDE`. This is **algebraically identical to Corollary CI-5** (`Δm = a(1/σ' − 1/σ) + Δa/σ'`) after rearrangement. The package cites Rushing and Nanda only for "the mechanism" and does not note the identity.
  - Cancedda 2024, *Spectral Filters, Dark Signals, and Attention Sinks* (ACL 2024): https://aclanthology.org/2024.acl-long.263.pdf. Residual "dark signals" lie in the bottom of the `W_U` spectrum. This is related background for NC.6 and the effective null space.
- **What is known:**
  - NC.1: `m = a/σ + κ` is the LN formula.
  - NC.2: null-space edits change only `σ`, which is the Stolfo/Gurnee mechanism.
  - NC.3: all margins share one `τ = 1/σ`. This is the temperature analogy in exact form, and single crossing is trivial for affine functions of `τ`.
  - NC.6: no exact null space when `|V| ≥ d`. Stolfo et al. already work with the effective null space for this reason.
- **What may be new:**
  - NC.4 and NC.5: the exact reachable set `[σ_min, ∞)` and the iff flip criterion (`aκ < 0`, `σ* > σ_min`).
  - Explicitly naming the large-norm limit as the bias ranking `κ = Δu^Tβ_f + Δb_U`. I found no source that states that norm-only edits can flip pairs, and only toward the `β_f` ranking. It is elementary, but I did not find it written down.
  - The "writer alignment" reading (`dσ² = Σ c̃_k^T c̃_l`) is the norm expansion that Rushing and Nanda use informally ("heads align with the residual stream").
- **Verdict: KNOWN_IN_SUBSTANCE.** The flip criterion and the `κ` limit are **POSSIBLY_NEW** as minor formal corollaries. CI-5 is **KNOWN** (Rushing and Nanda 2024).

## Theorem CI: softmax consensus identity

- **Closest prior work:**
  - The weighted-mean update identity is textbook: incremental and weighted mean formulas, and `∂M/∂w_j = (v_j − M)/Z`.
  - The sign law in CI.3 is the finite-sample sensitivity curve (Tukey) and the influence function of the mean, `IF(x; mean, F) = x − μ` (Hampel 1974, JASA 69:383–393). Adding a point moves the mean toward itself, which is exactly `sign(v' − M)`.
- **Attention-specific background:**
  - Attention output as a convex combination, analyzed with value norms: Kobayashi et al. 2020 (EMNLP), https://aclanthology.org/2020.emnlp-main.574.
  - Attention sinks as low-value "no-op" mass that deliberately dilutes: Xiao et al. 2024 (ICLR), https://arxiv.org/abs/2309.17453.
  - Qualitative "attention dilution" with added tokens: e.g. Mudarisov et al. 2025, https://arxiv.org/abs/2508.17821. That paper gives bounds only, not the per-token sign law.
- **What is known:** CI.1–CI.4 entirely. The package admits that CI's sign is redundant with weight-space sensitivity (§8).
- **What may be new:** Reading `sign(v'_j − M)` as the decision-relevant "source vs. consensus" relation. Also, contrasting it with additive DLA as a falsifiable prediction: supportive-but-sub-consensus evidence *lowers* the margin. I found no paper that states this as a sign law for candidate margins, though practitioners widely know the phenomenon informally (dilution, distractors). Construction D is a hand-built existence example, and its novelty is presentational.
- **Verdict: KNOWN** (math). **NEW_APPLICATION** (dilution sign prediction for decision margins), with the caveat that it is untested and conditional on A4.

## Theorem RB: ordinal relation-blindness

- **Citation check:**
  - Scott, D. (1964), "Measurement structures and linear inequalities", *J. Math. Psych.* 1, 233–247, doi:10.1016/0022-2496(64)90002-1: https://portal.mardi4nfdi.de/wiki/Publication:2394875. **Correct.** Scott proves that finite additive representability is equivalent to the cancellation conditions, using linear-inequality alternatives. Calling the LP in RB.1 "Scott's criterion" is a fair paraphrase; the max-`t` box LP is a standard reformulation, not Scott's literal statement.
  - Luce and Tukey (1964, *J. Math. Psych.* 1:1–27) and Krantz, Luce, Suppes, and Tversky (1971): standard and correctly cited. I did not fetch them; I checked them against secondary sources.
  - Loftus (1978), "On interpretation of interactions", *Memory & Cognition* 6(3), 312–319: **correct.**
  - Wagenmakers, Krypotos, Criss, and Iverson (2012), *Memory & Cognition* 40(2), 145–160: https://memolab.syr.edu/wp-content/uploads/2021/04/WagenmakersKrypotosCrissIverson_inpress.pdf. **Correct for the 2×2 claim.** The paper states that "the only interactions that are nonremovable are the ones that cross over", and it discusses only 2×2 designs. The package uses it only for RB.4, which is correct.
- **RB.1–RB.5:** RB.3 (independence ⇒ no crossover), RB.4 (2×2: crossover ⇔ nonremovable), and RB.5 (a double-cancellation violation without any independence violation in 3×3) are textbook conjoint measurement (Krantz et al. 1971, ch. 6 and 9; Michell 1990). I checked the RB.5 table: it is monotone in both factors, and double cancellation fails (6 ≥ 5 and 8 ≥ 7.5, but 6.5 < 7). **Verdict: KNOWN.**
- **RB.6 "prospective forcing":**
  - **Closest prior work:** Greco, Mousseau, and Słowiński 2008, "Ordinal regression revisited: Multiple criteria ranking using a set of additive value functions", *EJOR* 191, 416–436 (RePEc page range; the PDF self-reports 415–435), doi:10.1016/j.ejor.2007.08.013: https://www.lamsade.dauphine.fr/mcda/biblio/PDF/GMS-EJOR2008.pdf. Their *necessary preference relation* `a ≿^N b` holds iff every additive value function compatible with the reference preferences ranks `a ≥ b`. It is computed by an LP that minimizes `U(a) − U(b)` subject to the reference constraints with margin `ε`. That is RB.6. The only differences are that UTA-GMS imposes monotone marginals and RB.6 uses separation 1 in place of `ε`, which is the same up to scaling.
  - **Related:** Dunn and James 2003, "Signed difference analysis: Theory and application", *J. Math. Psych.* 47(4), 389–416, doi:10.1016/S0022-2496(03)00049-X: https://eres.library.adelaide.edu.au/dspace/handle/2440/31812. It derives testable predictions under monotonicity alone, with no parameter estimation and no functional form, and connects them to conjoint measurement and state-trace analysis. The "metric-free, link-free, no fitted map" selling point is exactly SDA's selling point. Also related: Bamber (1979) on state-trace analysis, and cancellation-axiom testing in psychometrics (Karabatsos 2001; the CRAN package `ConjointChecks`).
  - **Verdict: KNOWN** (it is the necessary relation of robust ordinal regression). The package does not cite this precedent and presents the name "prospective forcing" as its own. It should cite Greco et al. 2008 and Dunn and James 2003.
- **Has conjoint measurement or removable-interaction reasoning been applied to NN interpretability?** The ML feature-interaction literature measures interaction on a fixed output scale:
  - NID, Tsang et al. ICLR 2018: https://arxiv.org/abs/1705.04977
  - Shapley-Taylor, Sundararajan et al. ICML 2020
  - Shapley values vs. GAMs, Bordt and von Luxburg AISTATS 2023: https://proceedings.mlr.press/v206/bordt23a.html
  - the 2026 "quadrilateral loss" additivity work: https://arxiv.org/abs/2607.20201

  None that I found uses ordinal or monotone-link removability (conjoint cancellation, crossover) as the criterion. Searches for "conjoint measurement" combined with neural network or language model interpretability returned nothing relevant. A related idea exists in biostatistics as *qualitative interaction* (crossover), which is scale-invariant by construction (Gail and Simon 1985, *Biometrics* 41:361–372). **Verdict: NEW_APPLICATION.** I am moderately confident; the search was not exhaustive, and the scale-dependence of interaction measures (logit vs. probability) is commonly acknowledged in XAI.

## Overall framing (consistency → relation-sensitive aggregation; causal role = nonremovable interaction)

- **Causal abstraction:** Geiger et al., "Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability", *JMLR* 26 (2025): https://www.jmlr.org/papers/v26/23-0058.html. **Citation correct.** "Head = weighted-mean aggregator" fits there as a high-level model, as the package concedes (§8). Causal abstraction has no notion of a scale-invariant, nonremovable-interaction criterion for a relation's causal role, so that part of the framing is not subsumed.
- **Interaction literature:** Defining "has a causal role" as "not representable as `ψ(f(a) + g(b))`" is Luce–Tukey additivity, which psychology calls removability (Loftus; Wagenmakers et al.; Sternberg-style additive-factors logic) and biostatistics calls qualitative interaction (Gail and Simon). Carrying it over to Transformer evidence sources is new as an application, not as a concept.
- **Verdict: NEW_APPLICATION** of a KNOWN concept.

## Other citations

- **Sutter et al. 2025**, "The Non-Linear Representation Dilemma" (NeurIPS 2025): https://arxiv.org/abs/2507.08802. Their Theorem 1 does concern unrestricted (non-linear, bijective) alignment maps, under Assumptions 1–5. **Correct.**
- **Stolfo et al. "report" an effective null space** (§9): **correct.** The reported null-space dimension is k=12 for GPT-2 Small.
- **GPT-2 has an LN-final bias and no unembedding bias** (§9): correct as an architecture fact. I did not re-verify it from weights.

## Citation problems

1. **Prop. FIB, "the fiber condition of Geiger et al. 2025": attribution UNVERIFIED and likely wrong.** I fetched the JMLR PDF in two passes. Neither the word "fiber" nor a statement of the form "Ψ determines R iff R is constant on Ψ's fibers" was found. The only nearby wording is the §3.5.1 description of a partition `τ`. The claim itself is elementary set theory and needs no citation. Drop the attribution or give a page and section.
2. **Corollary CI-5 is presented without noting that it is Rushing and Nanda's (2024) LN decomposition.** This is an omission, not a miscitation.
3. **RB.6 is presented as unnamed and new.** It is the necessary relation of Greco et al. 2008. This is an omission.
4. **The RB.1 label "Scott 1964 criterion" is acceptable but loose.** Scott gives a cancellation-condition characterization via linear inequalities; the LP form is a standard derivative. This is minor.
5. **The Loftus (1978) clause in the §8 table** ("non-crossover four-term interactions are removable") is true only for 2×2 tables. RB.5 itself shows the converse fails in 3×3. The table cell should say "2×2". This is minor.

## Summary table

| Result | Closest prior | Verdict |
|---|---|---|
| SS | Elhage et al. 2021 | KNOWN |
| NC.1–3, NC.6 | Stolfo et al. 2024; Gurnee et al. 2024 | KNOWN_IN_SUBSTANCE |
| NC.4–5 (reachable set, flip iff, `κ` limit) | none found | POSSIBLY_NEW (minor corollary) |
| CI.1–4 | weighted-mean update; Hampel influence function of the mean | KNOWN |
| CI-5 | Rushing and Nanda 2024, identical decomposition | KNOWN |
| CI as dilution sign prediction for margins | attention-dilution and sink literature (qualitative) | NEW_APPLICATION (untested, conditional on A4) |
| RB.1–5 | Scott 1964; Luce and Tukey 1964; Krantz et al. 1971; Loftus 1978; Wagenmakers et al. 2012 | KNOWN |
| RB.6 prospective forcing | Greco, Mousseau, Słowiński 2008 (necessary relation); Dunn and James 2003 (SDA) | KNOWN |
| Conjoint/removability applied to NN interpretability | none found; Gail and Simon 1985 analogue | NEW_APPLICATION (moderate confidence) |
| Framing vs. causal abstraction | Geiger et al. 2025 | NEW_APPLICATION |
| Construction D, Construction LN, D′ | hand-built existence examples | NEW_APPLICATION (illustrative only) |
| FIB | elementary; Geiger "fiber" attribution not found | KNOWN; citation UNVERIFIED |
