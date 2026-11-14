# Open COR foundational map after Investigation XIII

Phase 1.5 Step 1 (~25%). This map is **not** a phase closure. It contains no thesis, does not state that COR is validated, and draws no
link to O/D. Each entry says what is established, with its evidence class and its development or confirmation status; what is limited;
and what remains open.

## Established (within T4, these architectures, these training protocols)

| # | statement | evidence | status |
|---|---|---|---|
| E1 | `posterior_shared` is the exact posterior of the generator for C ∈ {1, 2} and β ∈ {1, 0.6}. The independent-copying posterior is wrong for C = 2 | exhaustive enumeration (531,441 inputs) and calibration | mathematics; verified |
| E2 | A fixed three-component per-group statistic with a pooled readout suffices for C = 1 at every β. The hard (S − e_g) form holds only at β = 1. For C = 2 (G = 3) the fixed per-group statistics do not suffice: a cross-group statistic is needed on the S = 3 not-all-agree slice (5.5%) at β = 1, and broadly at β < 1 (≥ 729 features per group for any factorised representation) | closed forms, rank argument, counterexample | mathematics; independent review 1 found 2 description errors, both corrected |
| E3 | Pooled-S and per-group-m factorizations cannot be distinguished by realizable inputs (bijective on the support) | support enumeration | mathematics (identifiability limit) |
| E4 | Layer-2 solutions come in two classes: an attention mixture, and a decision pre-assembled before layer 2 | direct-logit attribution, 21 dev models | dev, descriptive |
| E5 | In strong mixture-class models, readout attention rows are causally used as mixture weights. Their resemblance to E-based Bayes responsibilities (incl. on E≠A1 donors) is observational; an exploratory factorial check favours E over A1 | class II | dev 4/4; **confirmed 2/2** (P1); sensitivity: falsified if the borderline uP164 is included |
| E6 | Mixture-head attention rows are an eligibility-specific, head-specific, partial natural mediator | class I × II factorial, DiD | dev 4/4; **confirmed 2/2** (P2); partial (projection 0.25–0.33) |
| E7 | Matched-budget DAS did not recover E (subspaces overlap 4–5/6), and mean patching is non-specific against matched decoys | class III, stratified pairs | dev; mean-patch non-specificity confirmed (P3c); **P8 falsified**: E-trained subspaces are even less E-specific than predicted |
| E8 | Frozen C = 1 features suffice for an adapted (attention-only, MLP-only) or fresh layer 2 to reach the exact C = 2 posterior (5/5); necessity untested. Readout support for S = 3 is graded (2/5) | class V adaptation | local layer-2 repair **5/5**; readout pattern class-dependent (**P5 falsified**) |

## Limited or negative

- Mediation through attention rows is **partial** (projection 0.12–0.38 of the total effect). The rest goes through value, MLP and other
  routes and is not decomposed.
- The **pre-assembled class** is not dissected. Where and how its decision forms before layer 2 is unknown.
- The **query/key decomposition** of the mixture scores fits only partially (R² 0.29–0.52).
- **Restoration** through a distinct pathway with a discriminating prediction was not frozen. It is unresolved.
- **Heterogeneity**: the class appears parent-determined. Only one gated cohort exists, so its drivers are open.
- **Pretraining incompleteness** is group-level. The gated continuation remedy rescued 3 of 4 stalled parents (P161, P163, P165) and
  failed for P162.
- **Cross-family comparison** (GATE): not run in XIII, for compute. It is optional and secondary in the mandate.

## Missing controls (from review 3), not run in XIII

- A size-matched decoy head.
- DAS trained on balanced E≠A1 pairs.
- Resample and partial layer-1 ablations, with the behavioural residual.
- A P5 necessity test: adapt layers 0–1 with layer 2 frozen.
- More than one fine-tune seed per fresh parent.
- A frozen version of the E-vs-A1 transplant test inside the factorial.

## Open questions, ordered by how directly they follow from XIII

1. How does the pre-assembled class compute the mixture: in layer-1 MLP, in layer-1 attention, or in another form?
2. Is the attention mixture the unique minimal solution in a 1-layer readout, or one of several? This could be tested with
   architecture-constrained fresh training.
3. Can a restoration test with a discriminating prediction (E vs A1) be designed through a pathway distinct from the attention row?
4. The C = 2 repair is local to layer 2, and resid2 already supports S = 3 decisions. Which layer-2 components carry the new
   combination rule after adaptation, and do they reuse the mixture heads?
5. Do the results hold in β < 1 regimes, where per-group sufficiency holds but the hard eligibility form fails? This is a direct test of
   whether the network learned "eligibility" or "graded colluder evidence".
6. Generalization beyond T4 (other relational tasks, other families) is entirely open. XIII offers no evidence on it.

## Relation to other parts of the program (stated, not argued)

- **T4.** All XIII results are T4 results (G = 3, Q = 6, K = 3, α = 0.8), plus generator variants with C = 2 and β < 1 in the
  mathematics and in the adaptation experiments.
- **GATE / TRANS families.** Not examined in XIII. Whether an attention mixture over source reliability appears there is open.
- **Adaptive sources and unknown groups.** Not examined. The sufficiency analysis suggests that any dependence between colluders that
  is shared across questions or groups will require cross-group statistics [spec].
- **Original O/D.** **No connection is drawn.** Nothing in XIII bears on it.
- **Future causal-abstraction work.** XIII's methodological lessons are the obvious inputs:
  - DAS non-identifiability under correlated variables;
  - natural-mediator tests with discriminating predictions;
  - scorer preflight against null units;
  - strict separation of intervention classes.
