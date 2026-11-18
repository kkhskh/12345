# The original COR obstruction/discrepancy (O/D) mathematics and positional reuse

> **Post-review corrections** (`reviews/ADJUDICATION.md`; agent reviews, not human peer review). Case D (silent) is endorsed by Reviewer A. **W1:** Proposition S is false as stated. Its gate-flip column was computed over the evidence sources only, so 0 was guaranteed. Argmin D over all six sources changes on 51–54% of gate flips but tracks the gate only at F_G 0.14–0.23. **Corrected S′:** every source-level O_c or D_c with single-source germs is a quadratic form, hence an order-≤2 score, and an exact LP over all 567 uniquely-decided inputs shows no order-≤2 score reproduces the GATE Bayes decisions at arity 2 or 3. **F2:** the count form of D is an anchored (relative) instance, not Definition 3 itself, which gives a variance. **F3:** the 'order-1 optimum' column is the NLL-optimal rule, and an accuracy-optimal weighted vote reaches 0.705 in 2:0.9:0.55:0.05. **F4:** proposition numbering mixes the two PDFs; the relative extension obstruction is Proposition 6 in the original and Proposition 5 in the Update. Update Propositions 6 (local validity) and 7 (superposition non-uniqueness) were omitted; both reinforce silence. **W2:** see Case C(ii). Text withdrawn after review is kept in place and marked *[withdrawn, Wn]* so the record is preserved.


**Stage 1 of Investigation XVII.** This determination was written and committed before any XVII neural-mechanism result was inspected or used to choose an interpretation. At the time of writing, only the following existed:
- the architecture's forward code;
- dense deterministic replays of existing XVI runs (in progress; not yet analysed);
- a scan of XVI pooled trajectories for partial-acquisition windows.

**Evidence labels:** [proof], [num] independent numerical verification, [dev], [frozen], [expl], [spec].

## 1. Sources (read, not modified)

| source | content |
|---|---|
| `docs/notes/Causal_Obstruction_Response_theory.pdf`, `..._Update.pdf` §5 | original definitions 1–2, propositions 1–7 |
| `docs/math_map_v0.md` §2; `docs/theory_mvp_theorem_set.md` §0–1 | main objects; energy–distance theorem |
| `docs/research_audits/cor_phase1/02_formal_definitions.md`, `03_…`, `06_…`, `10_counterexample_catalog.md` | type checks, O ≠ D, failed causal bridge, counterexamples |
| `cor_phase15/overnight_reconstruction/final/MATHEMATICAL_RECONSTRUCTION.md` | the Phase 1.5 status of the equations |
| `cor_phase15/step1_foundational_atlas_continuation/COR_TEN_FAILURE_MATRIX.md` §2, `scripts/od_bridge.py` | source-level discrepancy bridge (OD1, OD1′) and its failures |

## 2. The original objects, in the original notation

For an input x and a candidate contrast c = (a, b):

| object | definition |
|---|---|
| **computation complex** (Def. 1) | K_x = (V_x, E_x), a directed graph of internal sites v = (ℓ, t, q): layer, **token position**, component type. Edges encode "computational, causal, or learned compatibility relations". The graph is chosen by the analyst. |
| **stalks and germs** (Def. 2) | F_c(v) ≅ ℝ^{r_v}. The local decision germ is s_c(v; x) = P^c_v φ_v(x), where φ_v is the activation at the site and P^c_v a candidate-specific projection, probe, feature map or logit-lens contrast (fitted or chosen) |
| **compatibility maps and coboundary** | ρ^c_{u→v}: F_c(u) → F_c(v); (d_c s)(e: u → v) = ρ^c_{u→v} s_c(u) − s_c(v) |
| **residual energy** | O_c(x) = ‖d_c s_c‖²_W = s^T L s, with L = d^T W d |
| **distance to compatibility** | D_c(x) = min over g ∈ ker d_c of ‖s_c − g‖²_M |
| **propositions** | energy–distance equivalence, λ⁺_min D ≤ O ≤ λ_max D (proved); minimum-energy repair under linear response; obstruction–margin stability; **non-identifiability** ("multiple low-overlap compatibility structures may induce similar obstruction-response predictions"); local affine decision boundary; **relative extension obstruction** Obs(s_U) ∈ H¹(K, U; F) for a partial section on a subcomplex U |

**Source-level form used in Phase 1.5.** D_c(x_q) is the number of answer-reporting sources whose report differs from candidate c (`od_bridge.py`). It is the simplest instance: one vertex per source, a scalar stalk, and identity compatibility.

## 3. What O/D measures, and what it assumes

- **What it measures.** The linear consistency of analyst-assigned, candidate-specific local evidence states across a chosen set of internal sites, **for one fixed input and one candidate pair** [definition].
  - It is not a decision rule.
  - It does not rank decisions: Phase 1 showed O ≠ D in general, that D does not rank decisions, and that D = 0 does not imply a correct decision.
  - It is not a model of training.
- **Inputs it requires:** the model's activations φ_v(x), plus the analyst's graph, ranks, probes P, maps ρ, and metrics M and W. Phase 1 established that with P = 0, O = D = 0 for every input. So the content is in those choices [proof, Phase 1].
- **Invariances.**
  - The definitions state none.
  - Phase 1's gauge audit (`04_gauge_and_identifiability.md`) concerns basis changes within stalks.
  - **There is no axiom that ρ is shared or equivariant across token positions.** ρ is indexed by edge, and nothing ties ρ_{(ℓ,t,q)→(ℓ′,t′,q′)} to the map at shifted positions [definition].
- **Position.** Positions enter only as vertex labels t. The framework neither assumes nor forbids position-dependent processing [definition].
- **Parameter sharing.** Not mentioned. O/D is defined on activations, not weights [definition].
- **Neural computation.** O/D describes compatibility of states that a computation produced. It does not say how a network implements a computation, nor how the computation is acquired [definition; Phase 1 verdict: no causal bridge].
- **Transfer between question positions.** Nothing in the definitions or propositions refers to moving a relational operation between positions [definition].

## 4. Source-level discrepancy cannot express any GATE decision [proof + num]

**Proposition S** *[withdrawn as stated, W1; replaced by S′ below]*.
- D_c(x) = Σ_s w_s [x_s ≠ c] is a sum of single-source terms, for any weights, including the "repaired" weighted vote.
- So argmin_c D_c is an order-1 rule.
- Every rule of order ≤ a ignores the gate in every GATE condition (XVI, verification V5 and V6), so argmin D never responds to a gate flip.

**Exact over all 729 inputs per condition** (`scripts/xvii_od_check.py` → `logs/xvii_od_check.json`):

| condition | argmin D accuracy (evidence sources) | argmin D (all six sources) | argmax of NLL-optimal order-1 rule (F3) | gate-free optimum | Bayes | gate flip changes argmin D (evidence sources only; 0 by construction, W1) |
|---|---|---|---|---|---|---|
| 2:0.9:0.9:1/3 | 0.738 | 0.608 | 0.791 | 0.791 | 0.900 | 0.000 |
| 2:0.9:0.9:0.05 | 0.791 | 0.618 | 0.872 | 0.872 | 0.900 | 0.000 |
| 3:0.9:0.9:1/3 | 0.738 | 0.621 | 0.791 | 0.791 | 0.900 | 0.000 |
| 3:0.9:0.9:0.05 | 0.791 | 0.621 | 0.872 | 0.872 | 0.900 | 0.000 |
| 2:0.9:0.55:0.05 | 0.681 | 0.558 | 0.668 | 0.705 | 0.725 | 0.000 |

So the GATE relation, the object of every XVII experiment, lies **outside** what source-level O/D can represent. That is the same failure class Phase 1.5 recorded for zero-weight and copying sources (OD1′: the bridge holds only under a weighted-voting coincidence). Here it is total: every competent GATE model is, by definition, doing something source-level D cannot.

## 5. Determination: which case holds

| case | holds? | reason |
|---|---|---|
| **A.** O/D predicts uniform, reusable relational computation | **No** | No invariance or sharing axiom exists on ρ, P or the graph. A prediction of position-independent computation would require adding an equivariance constraint, which would be new mathematics |
| **B.** O/D distinguishes relational computation from positional access | **No** | The framework has no notion of "access" or of acquisition. Proposition 6 (relative extension obstruction) gives *language* for "a partial section on a subcomplex U does not extend", which one could apply to "acquired slots" against "unacquired slots". But every substantive ingredient (which vertices, which stalks, which ρ) would have to be supplied by the analyst from the neural results. That is the circularity Phase 1 warned against. Doing it here would be a new hypothesis inspired by COR, not an O/D prediction |
| **C.** O/D predicts a more limited property | **Only two limited, negative properties** | (i) Source-level O/D cannot represent the GATE decision (corrected S′, W1). (ii) *[overstated, W2: an equivariance axiom gives only invariance across (θ, θ_π), not reuse within one trained network]* If an analyst built an O/D complex whose ρ, P and metrics respect the architecture's exact slot-permutation symmetry (COMPETING_NEURAL_MECHANISMS.md, Theorem P), the resulting O and D would be slot-permutation invariant. That is a consequence of the analyst's choice plus the architecture, and it says nothing about which slot acquires first |
| **D.** O/D is silent on positional reuse and internal access | **Yes, for the primary question** | Nothing in the definitions, propositions or Phase 1–1.5 results bears on whether a network implements a relational operation once and reuses it across positions, or how the positions come to use it |

**Determination: Case D for positional reuse and access, with the limited negative properties of Case C.** P6 (an O/D prediction) is therefore **not frozen**. No XVII neural result will be described as confirming or falsifying O/D.

## 6. Historical counterexamples and their bearing

| counterexample (source) | bearing on XVII |
|---|---|
| P = 0 gives O = D = 0 for all inputs (Phase 1, Attack F) | O/D values are only as meaningful as the probes. A probe-based "shared relational state" would inherit this |
| O ≠ D; D does not rank decisions; zero D ≠ correct (Phase 1; overnight reconstruction) | No O/D quantity can serve as the decision-level measure. XVII uses the exact Bayes decisions and gate-flip following instead |
| No causal bridge: alignment ≠ causal influence; ρ is not a functor (Phase 1 §3, file 06) | A fitted cross-slot alignment map would be the same object. XVII therefore does not fit alignment maps on confirmation data (mandate §7.1) |
| Unweighted D fails for zero-weight and copying sources (atlas continuation) | GATE adds a total failure: an order-(a + 1) relation is invisible to any additive discrepancy (Proposition S) |
| Unrestricted alignments manufacture agreement (Sutter et al. 2025, Theorem 1; overnight reconstruction) | Cross-slot "shared direction" claims must use fixed, not fitted, correspondences. XVII uses only identity correspondences between slots (same layer, same token role), which the architecture's symmetry makes canonical |

## 7. Classifying the claims XVII could make

| kind of claim | example | status |
|---|---|---|
| Original O/D prediction | — | none exists for this question |
| Mathematically justified restricted version | Proposition S′ (any source-level O_c or D_c is an order-≤2 score; exact LP: no order-≤2 score reproduces GATE Bayes decisions at a = 2, 3) | Review A, `reviews/scratch_A/lp.py`; original S withdrawn (W1) |
| New hypothesis inspired by COR | "slot acquisition is a relative-extension obstruction" | not pursued: it would need the circular analyst choices described above |
| Standard Bayesian explanation | the ideal GATE decision is the same function at every slot (slots are exchangeable) | proof, trivial |
| Ordinary neural conditional computation | shared weights conditioned on the question embedding E_q | the object of XVII's experiments |

**Consequence for COR.** XVII is an investigation of neural relational computation that COR's programme motivated. It is **not a test of the O/D framework**. Its results can neither validate nor refute O/D, which is silent here.
