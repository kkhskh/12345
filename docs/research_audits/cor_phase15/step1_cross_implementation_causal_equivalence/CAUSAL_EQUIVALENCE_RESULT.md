# Is there a shared causal relational variable across the P-count and T1 implementations?

**Post-review version** (pre-review text: commit 8c6363e; corrections **[R: id]** per REVIEW_ADJUDICATION.md). Base commit
8c31051; freeze commit 579a7a2. Grades: [P] proved, [N] exact numerics, [C] confirmed under a frozen criterion, [F] frozen
criterion failed, [E] exploratory, post hoc or descriptive.

## Direct answer

**Pre-registered outcome: C — the proposed equivalence fails.** Two frozen requirements failed: transport fidelity was not
independent of the recipient's curriculum for P-count donors (II(a)), which by itself excludes Outcomes A and B **[R: ST1]**, and
the necessity-specificity criterion (IV(a): P4 0.677 < 0.70; T11's norm-matched rotated control 0.386 > 0.30). The
Bayes-tracking criterion (P2 0.486 < 0.5) and the training-objective site prediction (V) also failed.

**What the six confirmation models do show [E — descriptive synthesis after the frozen verdict, not a reclassification]:**

1. **A shared, identifiable relational variable with a shared intervention handle.** V = n^{(-q)} (leave-q agreement counts) is
   defined from the generator and is sufficient for the relational component of the decision [P]. Abstract-variable
   interventions on V at each model's main site reproduce natural counterfactual decisions far above mismatched floors and
   matched-capacity wrong-variable nulls (cond adjusted 0.73–0.95 including T2 after the own-evidence correction [R: ME1];
   single-group 0.84–1.03; Prediction I [C]). Count-preserving edits act outside V in both [C: III]. Marginalising V moves every
   model 68–88 % of the way to the own-question posterior against 1–2 % for the wrong-variable null [mostly C; F at P4, T11
   rotated].
2. **Different realisations of that variable.** T1: an exact, slot-bound tuple of own-group counts on question q's source tokens
   (decoding ≥ 0.998), plus a partial copy at the readout. P-count: an additive code at the readout that is lossy on its own
   (0.72–0.79 per component) but is completed by question q's source tokens (union decoding 0.87–0.97) **[R: ME5]**; it holds more
   than the posterior π (within-π-class rank decodable at 0.87–0.91) and is not π-respecting **[R: FE12]**. Neither main site is
   complete (union ceilings T 0.94–0.96, P 0.77–0.86) **[R: ME3]**.
3. **Transport through V works whenever the donor's state determines V.** With union decoders every tested P donor transports
   into P and T recipients at 0.81–0.98, and T donors at 0.98–1.01 **[R: ME5]**. By Theorem C this is expected from decodability
   plus intervention validity and is **not** evidence of a shared downstream computation [P].
4. **Not shared: the V→decision map beyond what Bayes requires.** For count changes to which an optimal model must not respond
   (π-preserving), the three T1 parents share part of their response (mean r 0.38), the P-count seeds do not share it with each
   other (−0.07) or with T1 (0.10) [C: VI-sep, these six models; model-level permutation p = 0.05, the minimum attainable
   **[R: ST3]**]. The input-level version without any encoder gives the same pattern (0.53 / 0.03 / 0.21). Bayes-tracking of the
   map differs (r 0.49–0.72 P-count, 0.73–0.82 T1).
5. **Withdrawn after review:** the pathway (attention vs MLP) evidence for different downstream computations — it is forced by the
   site choice **[R: ME4, FE11]**; the transport asymmetry — a decoder-site artefact **[R: ME5]**; "no collateral damage" for P —
   the rotated control damages P3/P4 accuracy **[R: ME2]**.

**Answer to the final directive.** On these models the two curricula instantiate the same identifiable, causally used relational
variable — differently distributed over token sites — and transport of that variable between them is exact once it is fully
decoded. They do **not** share the computation from that variable to the decision beyond their common approximation of Bayes:
the T1 parents share structured deviations from Bayes, the P-count seeds each have their own. Shared relational information and a
shared intervention handle therefore do not imply a shared causal mechanism of the decision; with three models per curriculum
this is a statement about these models, at the edge of model-level statistical resolution.

## Equivalence notions (FORMAL_EQUIVALENCE_AND_THEOREMS.md; tolerances = the frozen thresholds [R: FE7])

| notion | P-count | T1 | across |
|---|---|---|---|
| E-Info: V recoverable from the site | readout lossy; readout + q's sources 0.87–0.97 | exact | yes, with union sites |
| E-Int (average commuting square, cond/single) | yes | yes | — |
| transport through V | 0.81–0.98 (union decoders) | 0.98–1.01 | follows decodability (Theorem C) |
| E-DR on natural support | tracks Bayes 0.49–0.72 | 0.73–0.82 | only via Bayes |
| E-DR beyond Bayes (π-preserving) | seeds idiosyncratic | partly shared | **no** (six models) |
| E-Comp | not evaluable with this mediator split [R: FE11] | | |

## Hypotheses (these six models)

| | verdict |
|---|---|
| H1 shared variable with shared downstream use | variable: supported; downstream beyond Bayes: not supported |
| H2 different projections needing reconstruction | rejected: both carry V once the site is taken as readout + q's sources; transport is then exact |
| H3 shared information, different V→decision computations | supported for the beyond-Bayes part of the map |
| H4 intervention artefact | largely rejected (floors, nulls, natural activation statistics, marginalisation toward the own posterior); residuals: T11 rotated control, T2 own-evidence leakage (corrected) |

## Site and objective

Moving P-count supervision from the readout to a source token did not produce a localised source-site count variable (Prediction V
failed; one of three seeds failed pretraining); the test could not detect a variable that moved but stayed distributed (S-model
union ceilings 0.82–0.99) **[R: ME]**. Why T1 places its tuple at source tokens remains unexplained.

## Documents

FROZEN_PREDICTIONS.md · FORMAL_EQUIVALENCE_AND_THEOREMS.md · COUNT_INFORMATION_STRUCTURE.md · CROSS_MODEL_CAUSAL_TRANSPORT.md ·
NECESSITY_AND_RESTORATION.md · TRAINING_OBJECTIVE_SITE_TEST.md · COUNTEREXAMPLES_AND_LIMITATIONS.md · REVIEW_ADJUDICATION.md ·
REPRODUCTION_README.md.
