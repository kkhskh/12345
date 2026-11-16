# Adjudication of the three agent reviews (Investigation XIII-D)

The three reviews were written by separately initialised agents. They are agent reviews, not external human peer review. Each reviewer had the actual code, raw results and checkpoints, and each wrote its review without modifying package files.

- **Review 1 (mathematical):** 22 findings.
- **Review 2 (mechanistic):** 14 findings.
- **Review 3 (experimental integrity):** 12 findings.

No frozen verdict was changed. The 13 verdicts (7 confirmed, 6 falsified) are reproduced byte-for-byte by the frozen scorers (review 3). What changed is the *meaning* attached to several verdicts, and those changes are listed in MAIN_RESEARCH_SYNTHESIS.md under "Post-review corrections".

Dispositions: **A** = accepted and corrected; **P** = partly accepted; **N** = noted, with no change needed.

## Corrections that change scientific meaning

| # | finding | disposition | change |
|---|---|---|---|
| R1-1 (critical) | SRCF is an unrelated-donor control at the same site; an ideal carrier gives 0.72 | A | The "readout token holds the effect" reading and the "~0.6 of the effect" limit are withdrawn. DAS − SRCF is compared with an ideal value of 0.28. P3a passes at about the ideal value. |
| R2-F1 (critical) | "Mixture default; early needs the diagonal W and L" is not separated from re-roll; base rates 7 early / 6 mixture / 4 hybrid; Fisher p = 0.69 | A | Withdrawn. Restated as: the organisation is an interaction-dependent function of parent × (L, W), and D does not change the class. The per-parent "which component decides" table is replaced by a description of the specific cells tested. |
| R2-F3 | "Mixture is an attractor at every stage" contradicts F1c/F1d; F1f is a stable dual-route hybrid | A | Withdrawn. F1f is described as a stable hybrid. |
| R2-F4 | Late-block asymmetry is confounded with severity (F1e loss reaches chance) | A | The asymmetry is reported as confounded. A severity-matched test is listed as not done. |
| R2-F5, R1-10 | "Decided within 100 steps" overreaches; no sham mask; < 100-step windows untested | A | Now "sensitive to route availability in the first 100 steps (≈ 1.2% of the learning-rate mass)" and "no recovery window ≥ 100 steps". |
| R2-F6 | P2c is n = 1 on the pooled measure; group-resolved presence is reduced (0.78 vs 0.99) | A | Both stated. |
| R2-F7, R1-5 | "Decodable ≠ used" is a filter-versus-pattern artefact; decoder decodability never measured | A | P3c is reread as "pattern ≠ filter". The model-property claim is withdrawn. |
| R2-F8 | Carrier not unique (ENC, CCD, DAS_DECOY are equally necessary) | A | Restated as "a shared low-dimensional relational channel". |
| R2-F9 | K/Q split traced only for DAS | A | Only the joint K+Q share is claimed. |
| R1-2 | P4b's ablation-damage logic is invalid; an ideal carrier gives KL 0.347 | A | P4b is declared non-informative, although it passes as scored. |
| R1-3 | The restore reference is 0.46, not 1 | A | The non-separability claim is withdrawn. |
| R1-4 | The E1 specificity metric is ill-posed (unnormalised own_share; ideal value 0.40–0.50; same tokens) | A | The P6 falsification is reported as falsifying an ill-posed criterion. The token-location reading is withdrawn. |
| R1-7 | SHUFD_vs_borrowed has no power (an ideal carrier scores 0.003) | A | The "no generic signal" claim is dropped. The arm is declared uninformative. |
| R1-8, R2-F11 (E4) | P7 builds the KL damage into its success criterion; SYM is weights-only; number errors | A | P7 is now "cannot exclude H_pos". Numbers corrected: SYM2 −7.2% / −4.8%; PERM 3.4–5.0×. |
| R1-9, R2-F11 (E3) | P8 rules out redundancy only; blocking is off-distribution; BLOCK_R masks own sources | A | P8 is restated as non-redundancy. "Interdependent" is described as blocking-specific. |
| R2-F11 (E1) | Rule margins mirror the model's own behaviour (XIII-C M6 persists) | A | The E1 claim is restated as transmitting the model's own eligibility response. |
| R2-F10, R1-11/12 | "L1 cannot" holds only at lr 1e-3; depth alternative; P5b query/key labels untraced | A | L1 is limited to lr 1e-3, and a post-review exploratory L1only run at lr 3e-3 was added. Depth is listed as an untested alternative. The P5b labels are now "readout-token / source-token supply". |
| R2-F2 | uP163 W20 counter-example; L22/W21/D22 missing from the table | A | Both added. |
| R2-F12, R1-14, R3-11 | uP164 L-only / W-only labels swapped; class read at step 1211 | A | Labels corrected in the reports, with an erratum for the frozen text. The step-1211 caveats are stated. |

## Integrity corrections (review 3)

| # | finding | disposition |
|---|---|---|
| R3-1 | Undisclosed interim scoring of freeze 2 (19:15 and 20:00); P5b drew on the interim P3 result | A. Disclosed in FROZEN (event 7) and in COMPUTE_LOG. |
| R3-2 | The sensitivity script wrote into `logs/` | A. The script now uses a temporary copy; the score-file hashes were verified unchanged. |
| R3-3 | F1b misreported; falsified predictions recast as support | A. F1b's failing criteria are stated correctly. Post-hoc readings are labelled as such, and no falsified ID is cited as support. |
| R3-4 | F3a scored while descriptive arms were still running | A. Re-scored after L0attn and L0mlp finished. |
| R3-5, R1-13 | Effect-size intervals pooled development units; n = 2–3 intervals equal the range; no sampling error | A / P. The table now covers confirmation units only and is described as ranges. The B5 analysis has a bootstrap. Sampling error for P2a, P4 and P6 is limited to the stated binomial SE for P2a (≈ 0.011); no per-episode bootstrap was run. |
| R3-6 | Stray lane-line list incomplete; A-C3/A-C4 misattributed | A. Corrected in FROZEN event 4. |
| R3-7 | C-C1 launched before its declaration | A. Corrected in the chronology. |
| R3-8, R2-F14 | P1 references measured on a different stream | A. Disclosed; the verdict margins make it immaterial. |
| R3-9 | `queue_f3a.txt` changed after freeze; retry queues unhashed | A. See `logs/FREEZE_ADDENDUM_HASHES.txt`. |
| R3-10 | Determinism claim stretched | A. Narrowed to the training script. |
| R3-11 | Overlap and labelling errata | A. FROZEN event 8. |
| R3-12 | P5b is uncalibrated | N. Already stated. |

## Minor findings

- **R1-15 (step-0 R² measures random readout alignment): A.** The statement now reads "no readout-aligned candidate evidence at step 0".
- **R1-16 (transform attribution too narrow): A.**
- **R1-17 (code defects: posterior NaN; site has 448 features): A.** Disclosed. The posterior row is excluded.
- **R1-18 (null metric lax, not compared with the model's own null error): N.** Recorded as a limitation.
- **R1-19 (reverse test misdescribed): A.**
- **R1-20 to R1-22 (wording, pointers, factorial overstatements): A.**
- **R2-F13 (P2a reactivation partial; cap miss within noise): A.**
- **R2-F14 (D moves continuous lik by up to 0.18): A.**

## Items not resolved by new experiments

Each of these is stated as a limitation:

- sham-mask and < 100-step mask controls;
- a severity-matched late block;
- random re-draws of W and L to estimate P(early | change);
- continuing the uP164 cells to step 4000;
- a decoder-decodability measurement and a variance-matched random control;
- principal angles between the sufficient subspaces;
- traces for non-DAS subspaces and for the supply arms;
- the "layer 1 + extra block" depth arm;
- resample-ablation of the routes;
- patch-minus-behaviour margins for E1;
- varying k.
