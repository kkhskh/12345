# Information access, attention competition and training disturbance (Experiment A)

**Status.** COR Phase 1.5, Step 1 (≈ 25%, unchanged).

**Labels:**
- [proof]: a mathematical result;
- [dev]: development data, D = 801, one run per cell;
- [frozen]: the frozen scorer's output;
- [post-hoc]: analysis written after the frozen output;
- [interp]: interpretation.

**Numbers and design.** All confirmation numbers are in FROZEN_PREDICTIONS_AND_RESULTS.md and `logs/score_XVIII.json`. The design is in PRE_EXPERIMENT_IDENTIFIABILITY_REVIEW.md §10.

## 1. What the XIV block does, mathematically [proof]

A −1e9 mask on edge set T removes three things at once during the window:
- **(F)** T's forward contribution;
- **(R)** the routing gradient on T's logits;
- **(V)** the value gradient through T in that head.

**Results that removed non-discriminating designs:**
- **L1.** Boosting the whole competing set in a row is identical to suppressing T, by softmax shift invariance.
- **L2.** Graded suppression moves information, share and gradient together. It cannot separate the hypotheses.
- **Pre-review A.** Under gsg, which stops the gradient on T's logits, the complement logits still carry part of the recruitment signal: 0.5–0.7 on own0, measured.

## 2. Arms and what each one sets

| arm | forward through T | content | routing gradient on T | value gradient through T | share of T |
|---|---|---|---|---|---|
| block | none | — | none | none | 0 |
| **vrg** | full mass | destroyed (another episode's values, stop-gradient) | yes | none | free |
| **gsg** | intact | real | none on T's logits | yes | free |
| **gsv** | intact | real | none | none | free |
| **sha-hi-rolled** | clamped to the paired baseline's per-head trajectory | destroyed | yes | none | = baseline's |
| **sha-lo-real** | clamped to the paired baseline's step-0 share | real | yes | yes | = initial |
| gof b = 2.3 | reduced (odds × 0.1) | real | scaled | scaled | reduced |

## 3. Results on the confirmation cohort

uP160, E and M bases, D = 811–813, measured on a fresh stream.

### 3.1 The frozen outcome

**Arm bands.** Every arm-pooled band is INCONCLUSIVE: only 2 of 6 pairs are evaluable. The behavioural gate in the evaluability rule did not transfer to the fresh stream.

**Hypothesis verdicts** [frozen]:
- **CONTRADICTED (6):** H1d, Hm_window, H2g_R, H2g_RV, H4, H1d_plus_Hm.
- **INCONCLUSIVE (2):** Hm_release, H1d_plus_H2g.
- With 2 pairs, no hypothesis could reach CONSISTENT.
- Under the literal reading of the clamp β rule (Reviewer D, D1), Hm_window and H1d_plus_Hm become INCONCLUSIVE.

### 3.2 Per pair, under the pre-registered sensitivity analysis applied in full [sens; 5 evaluable pairs]

| arm | E (3 pairs) | M (2 evaluable) | reading (Reviewer C §3.6) |
|---|---|---|---|
| vrg | F, F, F (φ 0.99–1.04) | F, F | Destroying T's forward content (value gradient stopped) **suffices**. |
| sha-hi-rolled | F, F, F | F, F | The same, with T's forward share held on the baseline's trajectory. Not an independent replication: its release state and immediate Δloss match vrg's. It adds only the Hm_window test. |
| sha-lo-real | N, N, N (φ −0.03 to 0.03) | N, N (M\|813 also N; non-evaluable) | Holding the forward share at its initial level, **while T's logits and values keep learning**, does **not** suffice. |
| gsv | F, F, F (φ 0.81–0.85) | F, F (φ 0.98–1.02) | Removing all gradient through T suffices fully on M. On E it switches route necessity, just above the band edge, **without** mixture-like behaviour (ρ_L 0.57–0.61, against 0.73–0.87 for block and vrg). |
| gsg | F, P, F (φ 0.65–0.89) | P, P (φ 0.41–0.43; M\|813 0.36) | On E it switches necessity in 2/3 pairs (behaviour ρ_L 0.46–0.54, not mixture-like). On M it gives a hybrid. The measured difference from gsv on M is the value gradient through T. |
| gof b = 2.3 | F, F, F | F, F | A ≈ 90% forward-share reduction that also scales both gradients (a near-block) suffices. **Attention-share restriction of this size does select.** |

### 3.3 Mediator: latent own-question share at release (step 100)

| arm | φ_share | outcome |
|---|---|---|
| vrg, sha-hi-rolled | 0.86–0.94 (block-like) | switch |
| gsv, gof | 0.99–1.03 | switch |
| sha-lo-real | E −0.20 to −0.31 (above the baseline's); M 0.07–0.08 | no switch |
| gsg E | 0.49–0.55 | necessity switches in 2/3 pairs |
| gsg M | 0.82–0.85 | hybrid |

**gsg dissociates from the latent-share mediator on both bases, in opposite directions** (Reviewer C, C5).
- **E, robust:** T's share recovers after release, and gsg's busiest head exceeds the baseline's, yet the early route ends unnecessary.
- **M, marginal:** the denominator A_base − A_block is 0.031–0.034.

No attention statistic at release (mean share, max-head share, within-route concentration) separates all arms (Reviewer C §3.3). On uP163 M the denominator is 0.0048, so it is uninformative.

## 4. What is distinguished, and what is not

**Excluded** [frozen on 2 pairs, reinforced by sens on 5; within this design's contrasts]:

| hypothesis | evidence |
|---|---|
| Hm_window: the forward-share *trajectory during the window* as the cause | Only within a small contrast: forward-share difference ≤ 0.10 on E and ≤ 0.03 on M, confined to steps ≈ 60–99. This excludes "the share must grow in the window", not "the share must be substantial". sha-lo-real does not test a latent-logit race: its latent share at release is at or above the baseline's. |
| H1d: forward-content absence as a *necessary* condition | gsv and gsg are never N. |
| H2g_R, H2g_RV as *necessary* conditions | vrg switches with a routing gradient present. The potency wording of the dose rule (§10.11) concerns sufficiency and does not apply to this necessity failure. |
| H4 | gsv on M; sha-hi-rolled. |
| H1d_plus_Hm | the sha arms are F and N, not P. |
| Immediate-size H3 on M | XIV data and the in-cohort vrg Δloss of 0.011–0.013. Ledger N24 still stands. |

**Not excluded:**
- **Hm_release** (latent share at release) and **H1d_plus_H2g**. Each is contradicted only in the sensitivity analysis, by gsg, in one parent, at thin margins.
- **Trajectory-loss H3 on E.** The window Δloss orders the N arm below every F/P arm. This is confounded with the outcome.
- **Immediate-size H3 on E.** It is excluded only against a cross-cohort threshold (sens).

**No matched non-targeted disturbance arm** was run.

**Descriptions that fit the data** [interp; none pre-registered in this form; Reviewer C §3.4]:

| | description | status |
|---|---|---|
| D0 | Mine: "the organisation switches when T receives no useful learning signal" | **Withdrawn as a finding.** Taken literally, gsg on E (useful value signal plus share recruitment, yet a switch) and fst [dev] contradict it. Read loosely, it is circular. |
| D1 | Over-determination: content removal and signal removal are each nearly sufficient; neither is necessary; a forward-share clamp with continued learning is not sufficient | A description, not a mechanism. |
| D2 | A direct logit-recruitment race | Fits E fully. On M it needs the value-gradient component. |
| D3 | A latent-share mediator | Strained by gsg on both bases. |
| D4 | Value learning by downstream consumers | Strained by gsg on E. |

**What would discriminate them** (not run): vsg; a gradient-only rolled-content arm; gsg with complement-logit recruitment also stopped; attention logs to step 1211.

## 5. Matching (mandate A5)

No single quantity can be matched across all arms. What each arm matches, and what it gives up:

| arm | matches | gives up |
|---|---|---|
| vrg | attention mass exactly | immediate Δloss E 0.08–0.11, M 0.01 |
| sha-hi-rolled | the baseline's per-head forward share | the same Δloss as vrg |
| sha-lo-real | the initial forward share | — (latent share free) |
| gsg / gsv | the forward pass at equal parameters (by construction, so **not** evidence about disturbance) | — |

**Realised window Δloss on E** (Reviewer C §4): sha-lo-real +0.025; gsg +0.030–0.033; sha-hi-rolled +0.045–0.048; gsv +0.050–0.054; vrg +0.056–0.060; block and gof +0.061–0.067. On M every arm is ≤ 0.010.

**Gradient energy over the window** is within about 5% across arms on each base.
