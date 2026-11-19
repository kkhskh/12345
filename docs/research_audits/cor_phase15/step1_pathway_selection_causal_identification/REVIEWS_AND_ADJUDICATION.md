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

(Reviewers C, mechanistic interpretation, and D, integrity and reproduction, after scoring. Added below.)
