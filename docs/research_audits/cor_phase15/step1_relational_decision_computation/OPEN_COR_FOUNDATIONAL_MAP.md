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
| E5 | In mixture-class models, inserting natural donor attention rows changes the output as the donor's E-based Bayes responsibilities predict, including where E and A1 differ | class II | dev; confirmation: see FROZEN_PROTOCOL §6 (P1) |
| E6 | Mixture-head attention rows are an eligibility-specific, head-specific, partial natural mediator | class I × II factorial, DiD | dev; confirmation: P2 |
| E7 | Matched-budget DAS is non-identifying between E and A1, and mean patching is non-specific against matched decoys | class III, stratified pairs | dev; confirmation: P3c, P8 |
| E8 | The C = 1 → C = 2 repair is local to layer 2: attention-only or MLP-only adaptation suffices. Frozen C = 1 resid2 already supports most S = 3 decisions linearly | class V adaptation | dev; confirmation: P5 |

## Limited or negative

- Mediation through attention rows is **partial** (projection 0.12–0.38 of the total effect). The rest goes through value, MLP and other
  routes and is not decomposed.
- The **pre-assembled class** is not dissected. Where and how its decision forms before layer 2 is unknown.
- The **query/key decomposition** of the mixture scores fits only partially (R² 0.29–0.52).
- **Restoration** through a distinct pathway with a discriminating prediction was not frozen. It is unresolved.
- **Heterogeneity**: the class appears parent-determined. Only one gated cohort exists, so its drivers are open.
- **Pretraining incompleteness** is group-level. The gated continuation remedy helps only partly (P161, P162 needed continuation; see
  FROZEN_PROTOCOL).
- **Cross-family comparison** (GATE): not run in XIII, for compute. It is optional and secondary in the mandate.

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
