Single source of truth for scorer (scripts/score_13b.py) and document table (scripts/render_predictions_13b.py). Thresholds were set on DEVELOPMENT models only (XIII seed-20 cohort uP160-165 and the earlier pools). Confirmation units: the NEW seed-21 fine-tunes of uP160, uP161, uP163, uP164, uP165, analysed on NEW streams (79_000_000+) and the NEW evaluation set logs/eval_13b.npz (stream 21_400_000). These are FINE-TUNING replicates of already-pretrained parents (unit of independence = fine-tuning seed); they are not independent pretraining replicates.

Pass rule: an arm passes if its test holds on >= ceil(0.75 n) of its units (n >= min_units, else INCONCLUSIVE); a prediction is CONFIRMED iff all arms pass; an arm is DIAGNOSTIC only if its null was computed on every unit and failed on every unit.

### Q1: Same abstract computation in every solution class: the input-output map is a posterior MIXTURE over 'which group is reliable' built on E-based eligibility, not a product-of-experts pooling and not the perfect-only (A1) rule

- **Competing explanations:** (i) the classes implement different algorithms, e.g. log-linear pooling of per-group evidence (POE) in the early-readout class; (ii) some models use perfectness-only eligibility (A1).
- **Procedure:** fit_family.py (MIX vs POE families, held-out R^2 on 3000 episodes, stream 79_100_000); algorithm_discrimination.py on eval_13b (on-support D1/D2 eligibility edits).
- **Unit:** each seed-21 fine-tune (min_units 3)
- **Interpretation:** Pass: both classes compute the same function family (equivalence at the input-output level only; says nothing about implementation). Fail of arm a: some model is not a posterior mixture; fail of arm b: some model is not E-based.

| arm | statement | test | null |
|---|---|---|---|
| a | MIX test R^2 >= 0.97 and exceeds POE by >= 0.10 | `{ff:MIX.R2_test} >= 0.97 and {ff:MIX.R2_test} - {ff:POE.R2_test} >= 0.10` | `{ff:POE.R2_test} >= 0.97 and {ff:POE.R2_test} - {ff:MIX.R2_test} >= 0.10` |
| b | on D1 and D2 the response is closest to the E-based rule: distance to B <= 0.35 and to A1 >= 0.8 | `{ad:D1.B} <= 0.35 and {ad:D2.B} <= 0.35 and {ad:D1.A1} >= 0.8 and {ad:D2.A1} >= 0.8` | `{ad:D1.A1} <= 0.35 and {ad:D2.A1} <= 0.35 and {ad:D1.B} >= 0.8 and {ad:D2.B} >= 0.8` |

### Q2: The solution class is inherited from the pretrained parent: a new fine-tuning seed reproduces the parent's class signature (early candidate readout vs late mixture readout)

- **Competing explanations:** class is set by fine-tuning initialisation / optimisation noise (then about half of the replicates would switch).
- **Procedure:** class_features.py (stream 79_000_000): lik_RO1 = availability of candidate evidence at the readout token after layer 0 (donor resid1 transplant fidelity on likelihood pairs); mix_mass = max layer-2 readout-token q-mass.
- **Unit:** parent (one new seed each); uP164 is reported but not predicted (min_units 3); parents ['P160', 'P161', 'P163', 'P165']
- **Interpretation:** Pass: the class is a property of the pretrained representation. Fail: class depends on fine-tuning seed.

| arm | statement | test | null |
|---|---|---|---|
| a | early parents (P160, P163): lik_RO1 >= 0.7 and mix_mass < 0.5; late parents (P161, P165): lik_RO1 <= 0.2 and mix_mass >= 0.8 | `({exp:x} == 'early' and {cf:lik_RO1} >= 0.7 and {cf:mix_mass} < 0.5) or ({exp:x} == 'late' and {cf:lik_RO1} <= 0.2 and {cf:mix_mass} >= 0.8)` | `({exp:x} == 'late' and {cf:lik_RO1} >= 0.7 and {cf:mix_mass} < 0.5) or ({exp:x} == 'early' and {cf:lik_RO1} <= 0.2 and {cf:mix_mass} >= 0.8)` |

### Q3: Routing law: the relational effect originates in layer 0 at the edited group and reaches the decision through layer-1 attention into the q source tokens in late-readout models but NOT in early-readout models

- **Competing explanations:** (i) the same route in both classes (location-only difference after layer 1); (ii) no consistent routing.
- **Procedure:** class_features.py route measures on relational pairs (edit at q' != q): transfer of the edited group's layer-0 outputs; route_Q = share removed by freezing layer-1 attention output at the 6 q tokens; units grouped by their OWN measured lik_RO1.
- **Unit:** each seed-21 fine-tune (min_units 1)
- **Interpretation:** Pass: computational location of relational evidence differs categorically with the candidate-readout signature.

| arm | statement | test | null |
|---|---|---|---|
| a | relational transfer from the edited group's layer-0 outputs >= 0.8 (all units) | `{cf:transfer} >= 0.8` | `{cf:transfer} <= 0.2` |
| b | late-readout units (lik_RO1 <= 0.2): route_Q >= 0.4 (units: `{cf:lik_RO1} <= 0.2`) | `{cf:route_Q} >= 0.4` | `{cf:route_Q} <= 0.3` |
| c | early-readout units (lik_RO1 >= 0.7): route_Q <= 0.3 and route_R >= 0.2 (units: `{cf:lik_RO1} >= 0.7`) | `{cf:route_Q} <= 0.3 and {cf:route_R} >= 0.2` | `{cf:route_Q} >= 0.4` |

### Q4: In models with a mixture head, the mixture-head attention rows are a specific natural mediator of purely relational evidence: they survive magnitude/contribution-matched decoys, and restore the effect through an independent pathway after the layer-1 route is removed, with an E-vs-A1 discriminating restoration

- **Competing explanations:** (i) any head row-swap of matched magnitude produces the effect (non-specific); (ii) the effect after removal cannot be restored through attention rows (rows are downstream readouts only); (iii) the restored variable is A1, not E.
- **Procedure:** l2_components.py (stream 79_200_000), relational families EA (e and perfectness flip) and P0 (perfectness flips, e stays 0); decoys LOW/TOP/MAG/PERM/MEAN; removal = layer-1 attention output at q tokens from the base cell; restoration = intact donor mixture rows.
- **Unit:** late-readout seed-21 fine-tunes (own lik_RO1 <= 0.2) with a strict mixture head (relpairs.mixture_heads) (min_units 2); unit filter `len({lc:mixture_heads}) >= 1 and {cf:lik_RO1} <= 0.2`
- **Interpretation:** Pass: specific, restorable, E-based natural mediator in mixture-class models. Fail of b: rows are not specific; fail of c: no independent-pathway restoration; fail of d: E/A1 not discriminated.

| arm | statement | test | null |
|---|---|---|---|
| a | Shapley share of the mixture rows in the EA family >= 0.5 | `{lc:EA.shapley.R} >= 0.5` | none |
| b | mixture rows alone proj >= 0.6 while the contribution-matched (TOP) and magnitude-matched (MAG) decoy rows give proj <= 0.3 | `{lc:EA.decoys.MIX.proj} >= 0.6 and {lc:EA.decoys.TOP.proj} <= 0.3 and {lc:EA.decoys.MAG.proj} <= 0.3` | `{lc:EA.decoys.PERM.fid_E.R2} >= 0.6` |
| c | removal of the layer-1 route removes >= 0.4 of the effect; intact mixture rows restore proj >= 0.85 with fid_E R^2 >= 0.85; no decoy restoration (LOW/TOP/MAG) exceeds the removed level by > 0.15 | `{lc:EA.restoration.removal_fraction_removed} >= 0.4 and {lc:EA.restoration.RESTORE_MIX.proj} >= 0.85 and {lc:EA.restoration.RESTORE_MIX.fid_E.R2} >= 0.85 and max({lc:EA.restoration.RESTORE_LOW.proj}, {lc:EA.restoration.RESTORE_TOP.proj}, {lc:EA.restoration.RESTORE_MAG.proj}) <= {lc:EA.restoration.REMOVED_only.proj} + 0.15` | `{lc:EA.restoration.RESTORE_PERM.proj} >= 0.85 and {lc:EA.restoration.RESTORE_PERM.fid_E.R2} >= 0.85` |
| d | E vs A1 (P0 family): transplanted and restoring mixture rows follow E: fid_E - fid_A1 >= 0.5 for both | `{lc:P0.decoys.MIX.fid_E.R2} - {lc:P0.decoys.MIX.fid_A1.R2} >= 0.5 and {lc:P0.restoration.RESTORE_MIX.fid_E.R2} - {lc:P0.restoration.RESTORE_MIX.fid_A1.R2} >= 0.5` | `{lc:P0.decoys.MIX.fid_A1.R2} - {lc:P0.decoys.MIX.fid_E.R2} >= 0.5` |
| e | an unrelated likelihood-edit effect is preserved under removal + restoration (slope in [0.8, 1.2]) | `0.8 <= {lc:EA.preservation_likelihood_edit.fid_rr_to_intact.slope} <= 1.2` | none |

### Q5: Balanced E-vs-A1 DAS: an E-specific causal abstraction can be learned, but the apparent A1 abstraction is fully reproduced by source-free patches (A1 is not identified as a mediated variable)

- **Competing explanations:** (i) DAS identifies both E and A1 equally (non-identifying); (ii) DAS identifies neither.
- **Procedure:** das_bal.py: 50% discriminating training pairs, identical budgets for E, A1 and RANDE (decoy) subspaces; held-out discriminating pairs (DB_EVAL 79_400_000); source-free and no-op controls.
- **Unit:** each seed-21 fine-tune (min_units 3)
- **Interpretation:** Pass: balanced identification separates E (source-dependent, specific) from A1 (source-free artefact). Fail of a: no E-specific subspace; fail of c: A1 genuinely identified beyond source-free.

| arm | statement | test | null |
|---|---|---|---|
| a | E-trained subspace: fid_E R^2 >= 0.4 and fid_E - fid_A1 >= 0.3 | `{db:trained_E.fid_E.R2} >= 0.4 and {db:trained_E.fid_E.R2} - {db:trained_E.fid_A1.R2} >= 0.3` | `{db:trained_RANDE.fid_E.R2} >= 0.4` |
| b | the E effect needs source information: source-free E-subspace patch fid_E R^2 <= 0.2 | `{db:sourcefree_E.fid_E.R2} <= 0.2` | none |
| c | the A1-trained subspace fits A1 no better (<= +0.1 R^2) than the best source-free or decoy patch | `{db:trained_A1.fid_A1.R2} - max({db:sourcefree_A1.fid_A1.R2}, {db:sourcefree_RANDE.fid_A1.R2}, {db:trained_RANDE.fid_A1.R2}) <= 0.1` | none |

### Q6: Two-colluder repair under partial copying (beta = 0.6): adapting layers 0-1 with layer 2 FROZEN repairs the decision and uses the cross-question identity relation; whether a layer-2-only repair can use that relation depends on the class (early-readout yes, late-readout no)

- **Competing explanations:** (i) layer-2 adaptation is necessary (early01 fails); (ii) every repair ignores the cross-question relation (J-pair slope ~ 0) and uses only the local relation at q; (iii) layer-2 repair uses J regardless of class.
- **Procedure:** adapt2.py beta = 0.6 (AD_STREAM 79_500_000), conditions full (layer 2 + readout on frozen resid2) and early01 (layers 0-1 trainable, layer 2 + readout frozen), 1000 steps each; J-pairs: one group's values relabelled at every q' != q (per-group statistics and values at q unchanged); slope of the observed on the exact change.
- **Unit:** each seed-21 fine-tune (min_units 1)
- **Interpretation:** Pass: the J computation is not confined to layer 2; access to J for a final-layer repair is conditioned by the class's earlier routing.

| arm | statement | test | null |
|---|---|---|---|
| a | early01 (layer 2 frozen): acc >= exact-Bayes acc - 0.03 and J-pair slope >= 0.6 (all units) | `{a2:early01.acc} >= {a2:bayes_acc} - 0.03 and {a2:early01.Jpair.slope} >= 0.6` | `{a2:native.Jpair.slope} >= 0.6` |
| b | early-readout units (lik_RO1 >= 0.7): layer-2 full repair uses J (slope >= 0.2) (units: `{cf:lik_RO1} >= 0.7`) | `{a2:full.Jpair.slope} >= 0.2` | `{a2:native.Jpair.slope} >= 0.2` |
| c | late-readout units (lik_RO1 <= 0.2): layer-2 full repair does NOT use J (slope <= 0.1) and stays below the local-relation ceiling (acc <= 0.768) (units: `{cf:lik_RO1} <= 0.2`) | `{a2:full.Jpair.slope} <= 0.1 and {a2:full.acc} <= 0.768` | `{a2:early01.Jpair.slope} <= 0.1` |
