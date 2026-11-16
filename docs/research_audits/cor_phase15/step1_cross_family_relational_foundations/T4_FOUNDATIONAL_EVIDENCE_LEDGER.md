# T4 foundational evidence ledger, from the latent-reliability investigation to XIV Stage I

COR Phase 1.5, Step 1 (provisional ≈ 25%, unchanged). This ledger is the reference against which XIV's cross-family claims are judged.

**How the ledger is built.**
- Each claim is stated in its **latest corrected form**, never the most favourable earlier one.
- Earlier formulations that were withdrawn are recorded in the "history" column and in §3, not hidden.
- Pointers are to package directories under `docs/research_audits/cor_phase15/` (abbreviated `pkg/FILE`).
- Statements are checked against the cited primary documents. The clean-extraction verification (step 5) re-checks every pointer exists.

**Status classes.**
- **PROVED**: mathematically proved under stated assumptions.
- **NUM**: independently numerically verified.
- **CAUSAL**: causally supported in the tested neural systems.
- **BEHAV**: behaviourally or correlationally supported.
- **COHORT**: confirmed only on a limited model cohort.
- **EXPL**: exploratory.
- **NONDIAG**: non-diagnostic after corrected controls.
- **FALSIFIED/RETRACTED**.
- **UNRESOLVED**.

**The task family.**
- **Episode.** Q = 6 questions and K = 3 candidates. Six sources sit in three fixed pairs.
- **Configuration.** A hidden configuration is shared by all questions of an episode. One pair is reliable: each source equals Y with probability α. In T4, a second pair *colludes*: both of its sources report a common value c ~ U(K), independent of Y, with copying probability ρ = 1. The third pair is noise.
- **Variants.**
  - T1 has no colluder.
  - The native line (Investigations IX–XIII-E) uses the native environment (`ENV T4, α = 0.8`). It has C ∈ {1, 2} copying groups and β-partial copying in Investigation XIII's mathematics.

## 1. Mathematical results

| # | claim (current form) | status | evidence | history |
|---|---|---|---|---|
| M1 | Within-pair agreement counts across questions are an exact sufficient statistic for the configuration (T1/T4, fixed pairing). One episode-level posterior P(R \| X) answers every question. | PROVED + NUM | `step1_latent_reliability/MAIN_ANSWER.md` §1 (Theorems S, M; checked to 1e−14) | stable |
| M2 | Global vote histograms are insufficient (T1 Bayes on histograms 0.636 vs 0.743). Insufficiency does not imply that a separate agreement *variable* is necessary. | PROVED + NUM | same | stable |
| M3 | In T4 the agreement → reliability map is **non-monotone**: perfect agreement indicates the colluder. "Trust the most coherent group" scores 0.359. | PROVED + NUM | same | stable |
| M4 | **Exact versus partial copying.** For C = 1, a fixed three-component per-group statistic with pooled readout is sufficient at every β. For C = 2 with shared copying, per-group statistics are insufficient: at β = 1 on 5.5% of episodes, and broadly at β < 1, where the product rule's decision is off on 37%. Any factorised representation needs ≥ 729 features per group. | PROVED + NUM | `step1_relational_decision_computation/MAIN_RESEARCH_SYNTHESIS.md` §1.1 | review 1 corrected two descriptions |
| M5 | **Limits of per-group summaries.** Any mixture-form weighting that depends only on each group's own count gives the R1, R2 and R3 perfect-agreement configurations the same update sign. The correct R1 update needs the cross-group eligibility factor m_g (perfect *and* agreeing at q). | PROVED | `step1_r1_causal_origin_and_selective_rescue/MAIN_CAUSAL_RESULT.md`; Proposition C in `step1_learned_relational_computation` | the corollary was corrected after review (VII) |
| M6 | **Pooled S versus per-group m is unidentifiable by realisable inputs** in T4 (C = 1, β = 1): (e, S) ↔ (e, m) is bijective on the support. | PROVED | XIII §1.2 | Investigation X's Theorem X1 proof sketch was invalid; corrected and partly retracted in XI (VS realisable by a linear map, by no projection) |
| M7 | The exact posterior (`posterior_shared`) equals a brute-force latent sum over all 531,441 inputs of a reduced environment (error ≤ 3.3e−16). | NUM | XIII §1.1 | stable |
| M8 | **Mathematical equivalence ≠ neural causal equivalence.** Transport through a decodable variable V works whenever the donor's state determines V (Theorem C), so successful transport is not evidence of a shared downstream computation. | PROVED | `step1_cross_implementation_causal_equivalence/CAUSAL_EQUIVALENCE_RESULT.md` item 3 | stable |
| M9 | **Null responses** on Bayes-null pairs are exactly increments of the model's own log-odds error. Their correlation measures similarity of approximation errors and cannot identify an algorithm. | PROVED (elementary) | `step1_curriculum_response_identification/MAIN_RESULT.md` item 4 | stable |
| M10 | Ideal-mechanism references (XIII-E C1): several causal metrics have non-trivial ideal values (unrelated donor 0.72, DAS excess over source-free 0.28, ablation damage 0.35, restore 0.46, SRCFREE own-share 0.40–0.50, decoy rows ≈ null ≈ 0). | NUM | `step1_reroll_and_critical_period_validation/CAUSAL_METRIC_AND_SUBSPACE_AUDIT.md` C1 | new in XIII-E; the references assume own slope 1 (review A) |
| M11 | **(new, XIV) Relational invisibility in T4.** Every single source's report sequence is i.i.d. uniform under every role (reliable, colluder, noise), so it is independent of the configuration. The trust variable has no single-source (order-1) component. | PROVED | CROSS_FAMILY_MATHEMATICAL_STRUCTURE.md §2 | new |

## 2. Neural and causal findings

### 2.1 Curriculum-era models (T1 → T4 Transformers; Investigations I–VII, atlas, continuation, deepening)

| # | claim (current form) | status | evidence | history |
|---|---|---|---|---|
| N1 | T1 from scratch reaches Bayes using last-layer **reliability-gated attention routing**: readout-q attention to each group's q-sources ∝ P(group reliable). Seeds 0–2; patch IIA 0.84–0.85. | CAUSAL, COHORT (3 seeds) | `step1_latent_reliability/MAIN_ANSWER.md` §2(a) [R] | the frozen primary answer was PARTIAL; the explicit readout mediator (Type A) failed in every model |
| N2 | **T4 from scratch was not learned (0/3).** T4 is learned only after a T1 curriculum; that was post hoc, then replicated on seeds 10–13: routing 3/4, causal variable 3/4, all criteria jointly 2/4. | BEHAV + CAUSAL, COHORT | `step1_latent_reliability/MAIN_ANSWER.md` header | seed 12 used a different implementation; the distributed key/query form was not seen in new seeds |
| N3 | The reliability tag is low-dimensional and relational (a group-indexed function of agreement counts; linear in raw reports R² ≤ 0.04). Its location is seed-dependent: source-token keys, or split between keys and the readout query. | CAUSAL (seeds 0, 1), EXPL (seed 2) | same §2(b) | — |
| N4 | **No shared mechanism beyond near-Bayes behaviour** across 10 T4 solvers. Intervention signs agree with Bayes because the models are pointwise accurate; off-support (p_c = 0) the implementations diverge. | BEHAV | `step1_foundational_atlas_continuation/MAIN_FOUNDATIONAL_RESULT.md` §1 | the distribution-level theorem C4 is valid but vacuous |
| N5 | **The curriculum transfers via agreement counts.** P-count pretraining transfers 88% of the curriculum gain; P-local transfers nothing. | BEHAV | same | timing of feature existence not established [R3] |
| N6 | Both curricula carry a **causally used additive one-hot agreement-count variable**, at different sites and in different codes. T1: slot-specific source-token directions (IIA 0.66–0.82). P-count: a shared readout code (IIA 0.60–0.72). "The same variable" is not established. | CAUSAL, COHORT | `step1_relational_mechanism_deepening/MAIN_RESEARCH_SYNTHESIS.md` §1 [R: CA1] | the pre-review "different, non-count variable" for T1 was an artefact and is withdrawn |
| N7 | **Pre-registered cross-implementation equivalence failed (Outcome C)**, on transport fidelity and necessity-specificity. A shared identifiable variable V = n^(−q) with a shared intervention handle exists. Its realisations differ, and the V → decision map is not shared beyond what Bayes requires. | FALSIFIED (frozen equivalence) + CAUSAL (V handle) | `step1_cross_implementation_causal_equivalence/CAUSAL_EQUIVALENCE_RESULT.md` | the pathway and transport-asymmetry evidence was withdrawn after review |
| N8 | The Investigation IV curriculum contrast in null responses **did not replicate** (Δ 0.149 vs 0.45; CI includes 0). T1 null-response coherence emerges late, as models converge. Null responses behave as approximation errors. | FALSIFIED (replication) | `step1_curriculum_response_identification/MAIN_RESULT.md` | the "nuisance route" claim was withdrawn |
| N9 | **Perfect-agreement boundary (R1 regime).** With frozen pretrained layers 0–1, T1 parents move the candidate up (+0.19) and P-count parents move it down (−0.51). Bayes-corrected difference +0.70, p = 7.8e−5, 16 new parents. This is a structured, curriculum-dependent approximation error, not a distinct algorithm. | COHORT (frozen, 16 parents) | `step1_perfect_agreement_boundary/MAIN_RESULT.md` | P1(a) downgraded; causal summary weakened [R1–R5] |
| N10 | The R1 failure needs the cross-group conjunction (M5). **Supplying the conjunction precomputed** raises the R1 response on 8/8 new lineages (\|E_R1\| 0.72 → 0.30). Supplying the counts, which contain the ingredients, makes it worse (7/8). | CAUSAL (behavioural rescue), COHORT | `step1_r1_causal_origin_and_selective_rescue/MAIN_CAUSAL_RESULT.md` | "Bayes level" withdrawn; called a behavioural rescue by a precomputed conjunction |

### 2.2 Relational readers and native models (Investigations VIII–XIII-E)

| # | claim (current form) | status | evidence | history |
|---|---|---|---|---|
| N11 | The cross-group eligibility relation is **learnable from raw evidence by relational readers** (group attention, relation networks; 6/6 new seeds). Concatenation MLPs fail at the same budget. | COHORT (frozen) | `step1_learned_relational_computation/MAIN_RESEARCH_RESULT.md` 1 | — |
| N12 | **Learning goes through a separable phase first.** Every learner without m first adopts the P-count sign (9/9 runs at 1k steps). | COHORT (frozen) | same 2 | tested by XIV X1 as a cross-family prediction |
| N13 | **Decodable ≠ used.** Failed MLPs decode m at 0.99, better than the successful readers (0.83–0.87). | COHORT (frozen) | same 4 | — |
| N14 | **The P-count R1 failure is not an information failure.** Probes on frozen states recover every count and report, which gives exact Bayes; the transfer still fails R1 4/4. Pretraining is bimodal (incomplete parents). | COHORT (frozen) | `step1_native_relational_mechanisms/MAIN_RESEARCH_SYNTHESIS.md` 1–2 | "incompleteness explains R1" refuted |
| N15 | Native Transformers can learn the cross-group operation (4/4 independent parents). | COHORT (frozen) | same 4 | — |
| N16 | Independently trained native repairs are **behaviourally the same algorithm** (closest to the Bayes eligibility rule, distance 0.11–0.27). A per-group E variable is interchangeable at the downstream input (IIA 0.85–0.97); pooled S interchanges only partly. | COHORT (frozen, n = 3 parents) | `step1_causal_equivalence/MAIN_RESEARCH_SYNTHESIS.md` | the F3 interpretation was downgraded in XI (M6) |
| N17 | The conjunction (perfectness ∧ agreement-at-q) is formed in **layer 1**. Removing it makes the decision as if ineligible; restoring the 6-d E subspace restores it. | CAUSAL, COHORT | `step1_native_causal_construction/MAIN_RESEARCH_SYNTHESIS.md` 2 | — |
| N18 | **Interchange accuracy gave a confirmed false positive.** A subspace trained for the wrong rule A1 reaches IIA 0.83–0.93; the preservation diagnostic exposed it. | COHORT (frozen) | same 4 | methodological; carried into XIV's toolkit |
| N19 | **Two organisations.** In *early-readout* models, candidate evidence reaches the readout at layer 0 and relational evidence is pooled into the readout by layer 1. In *mixture* models, evidence stays at source tokens, relational evidence is routed into the q source tokens by layer-1 attention, and layer-2 readout attention acts as mixture weights. The relational effect starts at the edited group's layer-0 tokens in every model (0.84–0.95). | CAUSAL (dev + conf on fine-tuning replicates) | `step1_relational_decision_computation_followup/MAIN_RESEARCH_SYNTHESIS.md` 1; XIII §1.3 | "pre-assembled" corrected to early-readout |
| N20 | **Same weighting variables, different combination operator.** Both classes respond like the E-based rule. A nested power mean gives ρ ≈ 0.54–0.62 (early) versus 0.87–0.98 (mixture). The D1/D2 edits do not separate E from the threshold-4 rule. | BEHAV + COHORT | XIII-B 2 | "E-like" scoped (review 3) |
| N21 | Mixture-head attention rows act causally as mixture weights (R² 0.915–0.933 conf; a uniform row 0.86–0.88). "Attention ≈ Bayes responsibilities" is an **observational** match only. | CAUSAL (row) + BEHAV (match) | XIII §1.3 | — |
| N22 | **XIII-C: "the readout map W selects the organisation" — FALSIFIED in XIII-D** (W-only cells 4/4 mixture; W22 effect context-dependent). The organisation is an interaction-dependent function of parent × (L, W). | FALSIFIED | `step1_causal_algorithm_selection_validation/MAIN_RESEARCH_SYNTHESIS.md` §1 | — |
| N23 | **XIII-E re-roll.** A pooled within-parent W component transfers across contexts (R1-W, p = 0.006). Per parent p = 0.086 (uP160) and 0.018 (uP161), carried mainly by uP161, and no explanatory statistic was found. L transfer was not detected (R1-L), within a bound that assumes homogeneous transfer. The W − L contrast is untested. | COHORT (2 parents), EXPL (decomposition) | `step1_reroll_and_critical_period_validation/REROLL_VS_STRUCTURE.md` + `reviews/ADJUDICATION.md` | "identifiable structure" withdrawn |
| N24 | **Early-training sensitivity.** A 100-step route block redirects the organisation 6/6. Matched noise at the layer-2 input does not change the route class (0/6). The XIII-E sham was only a code-path identity check. Late blocks also leave lasting change (R4 falsified), so there is no closing critical period. On M, the late flip partly reverses (exploratory, 2 runs). | COHORT (1 parent, 3 data orders) | `step1_reroll_and_critical_period_validation/EARLY_TRAINING_CAUSALITY.md` | XIII-D "early critical period" downgraded; "difference of degree" withdrawn |
| N25 | **XIV Stage I.** Off-route sham, location-matched noise, persistence: R-S, R-N, R-P. | see T4_FINAL_CAUSAL_CONTROLS.md | `logs/score_I.json` | filled from the frozen scorer, see §4 |

### 2.3 Measurement and causal-identification results (T4)

| # | claim (current form) | status | evidence |
|---|---|---|---|
| C1 | Discriminating metrics: DAS sufficiency (ideal 1 / null 0), reverse-swap necessity (0 / 1), true row interchange (1 / 0). | NUM + CAUSAL | XIII-E C1 |
| C2 | **Non-diagnostic**: the unrelated-donor control, the DAS excess over the source-free patch, ablation damage, the restore test, the SRCFREE own-share (P6's criterion ill-posed), the SHUFD decoy rows, and null preservation as a specificity test. | NONDIAG | XIII-E C1 / C5 |
| C3 | "Decodable but not used" (XIII-D P3c) is **retracted** in that wording. The decoder filters decode m (R² 0.77–0.86) but carry about 1/10 of the signal variance, so they do not transmit. | RETRACTED → NUM | XIII-E C2, R6 confirmed |
| C4 | Low-dimensional relational sufficiency survives; **"3-dimensional carrier" is withdrawn**. 1–2 stable directions suffice, the number is model-specific, and DAS subspaces are not unique (seed overlap 0.42/0.69). | NUM + FALSIFIED (R5) | XIII-E C3–C4 |
| C5 | "m-specific carrier" is **not established**: decoy pairs change neither m nor the posterior. | UNRESOLVED | XIII-E adjudication A13 |
| C6 | No **unique natural eligibility subspace** was identified (C4, N18). | UNRESOLVED | XI, XIII-E |
| C7 | Patching evidence is model-site-specific, and mean versus resample ablation can reverse single-head results (GATE, deepening). | NUM (method) | `step1_relational_mechanism_deepening/GATE_FULL_CAUSAL_CIRCUIT.md` header |

## 3. How the explanations evolved (the failures in sequence)

1. **Reliability as an explicit readout variable (Type A).** Failed in every model; replaced by routing (latent reliability).
2. **"T4 is learnable."** Only after a curriculum; from scratch 0/3.
3. **"A shared mechanism across T4 solvers."** Only the behaviour is shared, implied by accuracy (continuation C1, C4).
4. **"Cross-curriculum causal equivalence."** Frozen outcome C, failed (cross-implementation).
5. **"Curriculum-specific downstream computation."** Did not replicate (curriculum response).
6. **"The R1 error is an information failure."** Refuted (IX). It is a reader-class limitation (M5, N10, N14).
7. **"Pooled versus per-group".** Unidentifiable (M6). Theorem X1 partly retracted (XI).
8. **"Interchange accuracy identifies the variable."** False positive (N18).
9. **"The readout map selects the algorithm" (XIII-C).** Falsified (XIII-D).
10. **"Early critical period" (XIII-D).** Downgraded (XIII-E R4 falsified); controls inadequate. Revisited in XIV Stage I.
11. **"3-d m-specific carrier" and "decodable but not used".** Withdrawn or retracted (XIII-E).
12. **The O/D framework.** Never established. Unweighted discrepancy is Bayes-order-compatible only under a weighted-voting coincidence, and fails for copying sources (continuation; deepening).

## 4. Stage I additions (XIV)

These are filled from `logs/score_I.json` after the frozen scorer ran; see T4_FINAL_CAUSAL_CONTROLS.md.

## 5. What is suitable for cross-family testing

**Carried forward:**
- M1/M11 (relational invisibility);
- M5 (per-group summaries are insufficient when the decision needs a cross-group conjunction);
- N12 (separable phase first);
- N13/C3 (presence ≠ use, with the variance caveat);
- C1/C2 (the corrected metric toolkit and ideal references).

**Not carried forward as general claims:**
- the specific layer and head layouts (N3, N19), which are seed-specific even within T4;
- the combination-operator split (N20), which XIV shows to be T4-specific (non-identifiable in GATE);
- initialisation and organisation claims (N22–N24), which concern one parent.
