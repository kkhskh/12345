# Reviews and adjudication (XVIII)

All four reviewers are agents. **This is not human peer review.**

Design review and results review were separate activities (mandate §12):
- **Reviewers A and B** reviewed the design **before** the freeze, in two rounds each.
- **Reviewers C and D** review the results **after** scoring.

The full texts are in `reviews/`. Every objection and decision is kept.

## Part 1. Pre-experiment design review (before the freeze, commit `1547ec9`)

| round | reviewer | verdict | file |
|---|---|---|---|
| 1 | A (mathematics, causal identification) | **REJECT (redesign)** | `reviews/PRE_REVIEW_A_MATH_IDENTIFIABILITY.md` |
| 1 | B (design, alternatives, novelty) | APPROVE WITH REQUIRED CHANGES (R1–R10) | `reviews/PRE_REVIEW_B_DESIGN_NOVELTY.md` |
| 2 | A, on design v2 | APPROVE WITH REQUIRED CHANGES (A1–A5) | addendum in the same file |
| 2 | B, on design v2 | APPROVE WITH REQUIRED CHANGES (fixes 1–6) | addendum in the same file |

### What the design review changed (decisions in PRE_EXPERIMENT_IDENTIFIABILITY_REVIEW.md §9–§10)

**1. Arms removed because they were non-discriminating, or their result was guaranteed in advance:**
- **fst (forward-only block).** Its forward pass equals the block's and its push towards T collapses within the window. A flip was close to guaranteed (A2).
- **The whole-complement diversion.** By softmax shift invariance it is identical to suppression (L1).
- **Graded suppression as a discriminating arm.** Every component scales with the same share (L2), so it was kept only as a secondary arm for mandate P1.

**2. Arms added because the original factorial could not interpret its own dev pattern:**
- **gsv.** All learning signal through T is removed. This is the missing cell, and the attention analogue of Kleinman et al.'s gradient-only deficit.
- **vrg.** Rolled content with stop-gradient replaces vrs, which had trained values across episodes.
- **The adaptive share clamp** (sha-hi-rolled, sha-lo-real; sha-hi-real as a dev validity cell). It separates content from the share used during the window.

**3. Edge set.** T = own0 on E. The ro0 set confounded gsg on E and includes 30 edges outside the route. Dev confirmed that an own0 block flips E.

**4. Outcome.**
- φ_nec (log test-time necessity) between the paired anchors, with F/P/N bands.
- Unanimity of three diagnostics is required for F or N.
- Competence failures are counted as missing, not as "no flip".
- A run counts as stable only if its band is the same at steps 1000 and 1211. P or unstable runs are continued to step 2546.
- One symmetric per-pair verdict rule applies to every hypothesis, with per-base verdicts as well.
- A pre-registered sensitivity analysis drops the behavioural gate.

**5. Hypotheses.**
- H2g is split into H2g_R (routing gradient) and H2g_RV (all learning signal).
- Hm is split into Hm_window (the forward share during the window) and Hm_release (the latent share at release).
- New row H1d_plus_H2g.
- Declared equivalences: §10.9 items 1–11.

**6. Diagnostics.**
- **Lapse-aware behavioural fit.** Plain ρ is confounded with lapse-type error.
- **rq1 relabelled.** It is a layer-1 readout pooling route, not a third route.
- **H3 on E.** Judged on the immediate loss change at fixed parameters, not on the difference between window trajectories.

**7. Solver fix.** The share clamp's Newton solver diverged on far targets. It now uses a warm start: 144 cases, 0 failures.

**8. Dose.** A rule was written before the T = 200 dev runs. Those runs then triggered it: gsv(200) is F on both bases, so H2g is read as a potency hypothesis.

**9. Novelty.** The broad phenomenon is established (Kleinman et al. 2024; Chen et al. 2024; and others). Only the forward/learning-signal decomposition is a candidate contribution (PRIOR_WORK_AND_NOVELTY.md).

**Adopted versus not adopted.** Every required change was adopted. The recommended but non-blocking items were not:
- xbo;
- vsg;
- a 4000-step horizon for all runs;
- an Adam-state swap;
- a parameter transplant.

Each omission is stated as a limitation in §10.2 and §10.9.

## Part 2. Post-confirmation review

| reviewer | remit | file | outcome |
|---|---|---|---|
| C | mechanistic interpretation | `reviews/REVIEW_C_MECHANISM.md` | 15 required corrections (C1–C15) |
| D | integrity and reproduction | `reviews/REVIEW_D_INTEGRITY.md` | core integrity holds; one major finding (D1), nine minor (D2–D10) |

### What Reviewer D verified
- **Freeze timing.** Freeze commit 19:54:40Z, stage 1 start 19:54:44Z. Of the 47 frozen files, only `h3_immediate18.py` changed afterwards: the disclosed output-path fix.
- **H3 file.** Recomputed byte-identically.
- **Blinding.** File access times show no read of any confirmation output before the scorer at 00:22:12Z. The method cannot detect reads between appends.
- **Completeness.** 48/48 confirmation runs, 22 screening runs and 12 parent runs, all rc = 0, each measured once on the correct stream.
- **Scorer.** It reproduces its output; byte-identical except a set-ordering field. It refuses tampered input in 6 cases.
- **Replay.** Bitwise to step 150 for a confirmation baseline and a vrg run.

### Adjudication
Every finding is **accepted**. Corrections are applied in the reports and listed in MAIN "Post-review corrections". The decisions that change claims:

| finding | decision | effect |
|---|---|---|
| C1 | accepted | The narrow claim "restricting a route's attention share does not change the organisation" is **withdrawn**: graded suppression (odds × 0.1) switches 5/5. It is replaced by "holding T's *forward* share at its initial level while its logits and values keep learning does not change it". |
| C2 / C15 | accepted | The unifying "no useful learning signal" description is **withdrawn as a finding**. It is essentially H1d_plus_H2g with its failing gsg prediction relaxed after the data. Taken literally it is contradicted by gsg on E and by fst [dev]; otherwise it is circular. The alternatives D1–D4 are listed. The supported result is the descriptive pattern of Reviewer C §3.6. |
| C3 / C4 | accepted | **Frozen verdicts:** 6 CONTRADICTED (H1d, Hm_window, H2g_R, H2g_RV, H4, H1d_plus_Hm) and 2 INCONCLUSIVE, on n = 2 pairs. CONSISTENT was unreachable with so few pairs. Hm_release and H1d_plus_H2g are **not excluded**: they are contradicted only in the sensitivity analysis, by one arm (gsg) in one parent, at thin margins. |
| D1 | accepted, with an interpretation | §10.6 states the β rule ambiguously. Its source, Reviewer A §3.3, applies it to the validity cell sha-hi-real, which had 0.00 in development and passed. Read literally against every E sha run (0.10–0.13), the E sha arms would be uninterpretable. Then Hm_window and H1d_plus_Hm move from CONTRADICTED to INCONCLUSIVE in the frozen analysis; post-hoc verdicts are unchanged. **Both readings are reported.** |
| C5 | accepted | gsg dissociates from the latent-share mediator on **both** bases, in opposite directions. On E the share recovers after release, yet the route ends unnecessary (robust). On M the dissociation is marginal. MAIN Q14's "observationally equivalent" is corrected. |
| C6 | accepted | "gsg and gsv leave the loss identical" is withdrawn as H3 evidence; it is true by construction. **Trajectory-loss H3 is not excluded on E.** Immediate-size H3 is excluded on M (in-cohort plus XIV, with N24 attached) and on E post-hoc only, against a threshold from another cohort and edge set. Gradient magnitude does not track the outcome. No matched non-targeted disturbance arm was run in XVIII. |
| C7 | accepted | On E, gsg and gsv are "F" by structural and causal diagnostics only; their behaviour is not mixture-like (ρ_L 0.46–0.61). The wording becomes "switches route necessity on E; switches fully on M". MAIN Q6's "complete double dissociation on every run" is withdrawn. |
| C8 / D5 / D6 | accepted | Counts corrected: 5/5 evaluable, 2/2 evaluable, and 4 untestable candidate parents. The block statement is no longer circular: "the block switches structural + causal class in 6/6 pairs; 5/6 also pass competence". |
| C9 | accepted | Dose rule. The H2g contradictions are *necessity* failures: vrg and sha-hi-rolled switch while a routing gradient is present. The potency wording of §10.11 concerns sufficiency; it is quoted and applies to gsg and gsv. |
| C10 | accepted | Frozen per-parent rule on uP163: vrg MIXED, gsv MIXED, sha-hi-rolled F, sha-lo-real N, block F. "The pattern matches uP160's" is withdrawn. The uP163 M mediator denominator is 0.0048, which makes it uninformative. uP163 M competence is marginal (KL ≤ 0.043). The natural-hybrid statement is weakened. |
| C11 | accepted | sha-hi-rolled is not an independent replication of vrg: the release states are near-identical. Its only added information is the Hm_window test. The forward-share contrast of the share arms is small (≤ 0.10 on E, ≤ 0.03 on M, in steps ≈ 60–99). sha-lo-real does not test a latent-logit race. |
| C12–C14 | accepted | Odds × 0.1 is a near-block in mechanism. The M\|813 block also fails the accuracy criterion. The stability readout is reported for the doubtful runs first. |
| D2 | accepted | Streams 95_400_000 and 95_500_000 were used once before, by XIII-E on 2026-11-16, for development checkpoints only. There is no leakage. "Never used before" is corrected. |
| D3 | accepted, disclosed | Training data depend only on D, and the new layer's initialisation only on L/W. The uP163 M base therefore shares the new-layer initialisation and data order with uP160 M\|812; only the pretrained parent differs (a paired-by-design contrast). Screening at D = 811 shares training data with confirmation D = 811. |
| D4 | accepted | Post-fix hash of `h3_immediate18.py` recorded in COMPUTE_LOG. The deleted `./23` was the pre-freeze development output: the bug predates the freeze. |
| D7, D8 | accepted, noted as limitations | The scorer accepts records outside the design and does not validate the H3 file's contents. The preflight lacks CONSISTENT cases for H2g and H4. The logs of the two interrupted screening runs were overwritten at resume. |
| D9, D10 | accepted | Placeholders filled, reports committed, wording fixed. REVIEWS_AND_ADJUDICATION.md was first committed after the results; its Part 1 records reviews that were committed before the freeze. |
