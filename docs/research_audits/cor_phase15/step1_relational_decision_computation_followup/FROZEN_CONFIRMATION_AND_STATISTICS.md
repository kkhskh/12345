# Frozen confirmation and statistics (Workstream D)

Investigation XIII-B, Phase 1.5 Step 1.

## 1. What is being confirmed, and what kind of replication it is

**Units.** New fine-tuning seeds (seed 21, `native_train.py uP<p> full_c1 4000 21 label logs/eval_conf13.npz`, the same protocol as
XIII's seed 20) of the five complete XIII parents uP160, uP161, uP163, uP164 and uP165.

**What these replicates test.** They are **fine-tuning replicates** of already-pretrained parents, so they test fine-tuning variability.
They are **not** independent pretraining replicates and are not counted as such. No new pretraining cohort was run.

**Fresh inputs.** The replicates were launched first (23:29Z and 01:05Z). Their outputs were not opened before the freeze. Confirmation
analyses use new streams (79_000_000 and above) and a new evaluation set (`logs/eval_13b.npz`, stream 21_400_000).

**Development data.** Everything in the other four documents comes from XIII's seed-20 models and earlier pools. Thresholds were set there.

## 2. Freeze

- **Frozen.** The predictions Q1–Q6, scorer and preflight are in `logs/predictions_13b.json`, `scripts/score_13b.py` and
  `scripts/preflight_13b.py`. The table below is rendered from the JSON by `scripts/render_predictions_13b.py`.
- **Commit.** Freeze commit `d7e74a4`, pushed at 02:12:53Z, one second before the confirmation lanes started. Hashes are in
  `logs/FREEZE_13B_HASHES.txt`. Integrity review 2 verified all 24 hashes and found no frozen file modified.
- **Post-freeze addendum AQ1.** Registered in commit `5c7e67f` (02:39:20Z), before its seed-21 run started (02:39:26Z). Its aggregation
  rule was added in commit `e624787` before its output was opened.
- **Reporting rules RB1–RB5.** Logged in COMPUTE_LOG after integrity review 2, before any confirmation output was opened, and implemented
  in `scripts/report_13b.py`.
- **No outcome blindness is claimed.** The author designed the predictions with development results in view.

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


## 3. Scorer preflight

`scripts/preflight_13b.py` scores the frozen scorer on six author-written null scenarios. Each targets the predictions it should defeat:
NOOP, UNIFORM, POE (log-linear pooling), A1 (perfect-only rule), SEEDSWITCH and DECOYALL.

Result: **OK**. No targeted prediction is confirmed by any single null unit. Arm-level passes on null units are
listed in `logs/preflight_13b.json`.

Limitations, from integrity review 2:
- The null values are numbers written by the author, so this checks the scorer's logic, not measurement.
- **Q1a cannot rule out normalised log-linear pooling.** Its POE family is handicapped (math review 1), so Q1a is non-diagnostic for that.
- Arms Q4e, Q5b, Q5c and Q6c also pass on some null scenarios and are therefore non-diagnostic on their own.


## 4. Confirmation results (seed-21 fine-tunes; `logs/score_13b_conf13b.json`, `logs/report_13b.json`)

No confirmation job crashed, and every exit code was 0. No unit was dropped for missing data.

### 4.1 Per-unit measurements

| unit (seed 21) | lik_RO1 | max q-mass / share | strict heads | route_Q / route_R | MIX / POE R² | power-mean ρ (AQ1) | D1 dist B / A1 |
|---|---|---|---|---|---|---|---|
| uP160 | **0.85** | 0.17 / 0.07 | – | 0.11 / 0.33 | 0.984 / 0.835 | **0.57** | 0.25 / 1.06 |
| uP161 | 0.01 | 0.97 / 0.62 | 1, 3 | **0.38** / 0.61 | 0.994 / 0.822 | **0.87** | 0.12 / 1.03 |
| uP163 | **0.37** (hybrid; seed 20: 0.82) | 0.91 / 0.56 | 1 | 0.25 / 0.55 | 0.990 / 0.832 | 0.69 | 0.18 / 1.06 |
| uP164 | **0.82** (seed 20: 0.52) | 0.28 / 0.03 | – | 0.05 / 0.30 | 0.970 / 0.833 | **0.54** | 0.28 / 1.08 |
| uP165 | 0.05 | 0.97 / 0.38 | 1, 3 | 0.54 / 0.39 | 0.992 / 0.818 | **0.87** | 0.11 / 1.05 |

**Mixture-head models** (l2_components, EA family; restoration columns are projections):

| unit | Shapley R / V / O / M / K | rows alone: MIX / TOP / MAG / PERM / MEAN | removed fraction | removed only → restored with mixture rows | decoy restorations LOW / TOP / MAG / PERM / MEAN | P0 rows fid E / A1 |
|---|---|---|---|---|---|---|
| uP161 | **0.88** / 0.02 / −0.01 / 0.09 / 0.02 | 0.99 / −0.03 / −0.04 / 0.69 / 0.68 | **0.23** | 0.77 → 1.01 | 0.77 / 0.77 / 0.77 / 0.70 / 0.69 | 0.87 / 0.04 |
| uP165 | **0.85** / 0.02 / 0.04 / 0.08 / 0.01 | 0.94 / 0.03 / 0.05 / 0.59 / 0.59 | 0.62 | 0.38 → 0.97 | 0.38 / 0.39 / 0.41 / 0.63 / 0.63 | 0.89 / 0.04 |
| uP163 (hybrid; not a Q4 unit) | 0.48 / 0.00 / 0.16 / 0.19 / 0.17 | 0.48 / 0.00 / 0.01 / 0.55 / 0.39 | 0.26 | 0.74 → 0.95 | 0.79 / 0.74 / 0.74 / 0.85 / 0.78 | 0.67 / 0.01 |

**Balanced DAS** (fid_E / fid_A1):

| unit | E-trained | E subspace, source-free | A1-trained (fid_A1) | A1 subspace, source-free (fid_A1) |
|---|---|---|---|---|
| uP160 | 0.38 / 0.06 | 0.01 | 0.69 | 0.84 |
| uP161 | 0.77 / 0.01 | 0.01 | 0.73 | 0.84 |
| uP163 | 0.54 / 0.02 | 0.01 | 0.68 | 0.87 |
| uP164 | 0.13 / 0.08 | 0.01 | 0.67 | 0.78 |
| uP165 | 0.57 / 0.03 | 0.01 | 0.61 | 0.85 |

**Two-colluder repair at β = 0.6** (acc / J-pair slope; exact Bayes acc 0.796):

| unit | layer-2 full repair | layers 0–1 only, layer 2 frozen |
|---|---|---|
| uP160 | 0.780 / **0.53** | 0.793 / **0.95** |
| uP161 | 0.725 / **−0.00** | 0.791 / **0.95** |
| uP163 | 0.744 / 0.11 | 0.793 / 0.91 |
| uP164 | 0.781 / **0.63** | 0.798 / 0.91 |
| uP165 | 0.728 / **−0.00** | 0.792 / 0.87 |

### 4.2 Verdicts

| prediction | frozen verdict | arms (k/n) | reported under RB1–RB3 |
|---|---|---|---|
| Q1 same function family; E not A1 | **confirmed** | a 5/5, b 5/5 | confirmed. **Q1a is non-diagnostic** against normalised log-linear pooling; Q1b (E vs A1) diagnostic |
| Q2 class inherited from parent | **confirmed** | a 3/4 (uP163 failed) | confirmed at the 75% rule, but **one of four predicted parents changed class** with a new seed. Low power: about 0.31 chance of passing under random switching |
| Q3 routing law | **falsified** | a 5/5, b 1/2 (uP161 0.38 < 0.4), c 2/2 | not confirmed (arm b falsified) |
| Q4 specific, restorable, E-based mediator | **falsified** | a 2/2, b 2/2, c 1/2 (uP161: removal 0.23 < 0.4), d 2/2, e 2/2 | not confirmed (arm c). Including PERM/MEAN restorations (RB4) also fails c in uP165 |
| Q5 balanced DAS: E identified, A1 not | **falsified** | a 3/5 (uP160 0.38, uP164 0.13 fail), b 5/5, c 5/5 | not confirmed (arm a). b and c are non-diagnostic alone |
| Q6 two-colluder repair | **confirmed** | a 5/5, b 2/2, c 2/2 | confirmed. Q6c is non-diagnostic alone (it passes on the no-op null), so the evidence rests on a and b |
| AQ1 addendum (post-freeze, registered before evaluation): ρ vs lik_RO1 | **holds** | early 0.57 / 0.54 ≤ 0.70; late 0.87 / 0.87 ≥ 0.85; corr −0.99 (n = 5) | exploratory status |

**Counts** (frozen Q1–Q6):

| count | value | notes |
|---|---|---|
| confirmed | 3 | Q1, Q2, Q6 |
| confirmed with every arm strictly diagnostic | **1** | Q2 |
| confirmed with a non-diagnostic arm | 2 | Q1 (arm a); Q6 (arm c) |
| falsified | 3 | Q3, Q4, Q5 |
| inconclusive | 0 | |
| protocol deviations | 0 | scheduling-only extra lanes K3 and K4 |
| addendum | AQ1 holds | |

The predictions are **not independent**: four depend on the same lik_RO1 measurement, and three bear on E versus A1.

### 4.3 What the falsifications show

The corrections from review 3 govern. **Q3b and Q4c are a single event**: the route-Q necessity of uP161 seed 21, within stream-to-stream
noise. **Q5a reflects site coverage and target choice.** Neither is evidence against E. The das_bal "NOOP" control is not a no-op (QR of a
1e-9 matrix), so it is a second random control.


- **Q3 and Q4c (uP161, seed 21).**
  - This replicate of a mixture parent still has strong mixture heads: rows Shapley 0.88, matched decoys about 0, P0 E/A1 0.87/0.04.
  - But layer-1 relational evidence reaches the readout token more (route_R 0.61) and the q tokens less (route_Q 0.38) than in seed 20
    (0.28 / 0.70).
  - So removing route Q removes only 23%, and restoration from that level is not tested. **Routing proportions vary with the fine-tuning
    seed even within the mixture class.** The frozen dichotomy on route_Q was too sharp.
  - The specificity part of Q4 (arms a, b, d) held in both units.
- **Q5a (early-readout uP160 and uP164).** The balanced E-trained subspace at the q-token site reproduces E counterfactuals weakly (0.38,
  0.13) in early-readout models but well (0.54–0.77) in late and hybrid ones. Arm c, the A1 null, held in all five.
  - The likeliest explanation is **site**: early-readout models carry relational evidence into the *readout token* via layer 1, and the
    learned subspace is shared with the q tokens.
  - This is a limit of where E was looked for. It is not evidence that early models use A1: they are E-like behaviourally (Q1b) and A1
    patches are fully explained by the source-free null.
- **Q2 (uP163).** A parent that was early-readout under seed 20 became a hybrid under seed 21 (lik_RO1 0.37, a mixture head with mass
  0.91). The unpredicted uP164 went from hybrid (0.52) to early (0.82).
  - **The class is largely but not wholly inherited.** Hybrid and boundary parents can move with the fine-tuning seed.
  - This supports a continuum with an attracting early-readout cluster, not a fixed binary class.

### 4.4 Statistics

- Units are fine-tuning replicates, with n ≤ 5 per prediction and n = 2 for several filtered arms. No confidence intervals are claimed for
  parent-level means at such n; per-unit values are reported instead.
- The episode-level estimates use thousands of episodes per unit; for example the J-pair slopes use 1,500 pairs.
- **Joint development + confirmation evidence for the strongest structural relation.** Across 23 models (18 development and 5
  confirmation), corr(lik_RO1, ρ) is −0.90 (development) and −0.99 (confirmation).

## 5. XIII's original verdicts (preserved, not re-scored)

| prediction | XIII verdict |
|---|---|
| P1–P4, P7 | met the frozen criteria on 2 strong-mixture parents |
| P5, P8 | falsified |
| P6 | exploratory, n = 1 |

Sensitivity: including uP164 as mixture-class would falsify P1 and P2.

XIII-B does not re-score XIII. It tests new hypotheses that XIII's results raised, on new fine-tunes.

## 6. Deviations, execution incidents and unfinished items

- **Confirmation.** No crash and no non-zero exit code. The extra lanes K3 and K4 are scheduling-only: identical lines with claim keys.
- **Development OOM kills (rc = 137)**, from memory pressure on 8 GB with up to 8 concurrent jobs. All are logged in COMPUTE_LOG and were
  requeued or dropped:
  - l2_components (3 models in one process): split by model;
  - l1_graded uP164;
  - das_bal for uP160 and uP163, twice each. **Never completed in development**, so Q5's thresholds came from late-readout models only;
  - adapt2 for uP160 (early01 rerun), uP165 (rerun) and uP161 β = 1 (twice);
  - decode_J for the uP161 early01 checkpoint (twice);
  - traj_resp (completed on the third attempt).
- **Development runs dropped for time.**
  - adapt2 β = 0.6 for uP164 in development;
  - adapt2 β = 1 for uP165, uP163 and uP164;
  - path_chain over the full development pool, superseded by class_features over 26 models.
- **Two-colluder necessity at β = 1.** See JOINT §C4 for whether the β = 1 layers-0–1 arm (uP161, uP160) finished.
- **Learning transition (D4, exploratory; `logs/traj_resp_dev.json`, six seed-20 models).**
  - At each checkpoint, the input-intervention response to W-only and to e-changing relational edits was measured.
  - Both slopes rise **together** (t50 equal in 5/6 models; e-change first in uP164). This happens at or after the KL halfway point and at
    or before the behavioural D1/D2 gap (t50 of the gap 813–2686).
  - So eligibility weighting does not emerge before reliability weighting, and the "A2 → B gap" is a late, small-magnitude readout of a
    transition already visible in the weighting responses.
  - These responses are behavioural (class I), not internal. **No internal predictor of the transition is established**, which preserves
    XIII's negative result.
- **GATE cross-family comparison:** not run, for compute (OPEN_COR_FOUNDATIONAL_QUESTIONS).
