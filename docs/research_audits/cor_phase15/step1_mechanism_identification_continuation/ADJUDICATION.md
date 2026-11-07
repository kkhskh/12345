# Adjudication of the seven-role review (§19)

## Process

- **What was reviewed:** commit `d1aa97d`, after the strongest mechanism had been written.
- **Who reviewed it:** seven separate subagents, one per role.
- **Reports:** `reviews/R1_mathematical.md`, `R2_causal.md`, `R3_transformer.md`, `R4_representation.md`, `R5_baseline.md`, `R6_counterexample.md` and `R7_scientific_defense.md`. Their scripts are in `reviews/*scripts*/` and `reviews/R4_checks/`.
- **Disposition:** every finding is listed below. "Accepted" means the claim was corrected in the package.
- **New post-review code:**
  - `src/mediator_sign_test.py`
  - `src/learned_review_checks.py`
  - fixes in `src/cprime.py`, `src/cprime_regions.py`, `src/program_b.py` and `src/program_c.py`

**No reviewer approved any theorem that did not exist at review time.** The mechanisms added after review (the mediator sign test S1–S3, T-RF′, and N2) are flagged in `OPEN_RESEARCH_OBLIGATIONS.md` as requiring a fresh review.

## Fatal and critical findings

| # | finding | reviewer(s) | disposition | what changed |
|---|---|---|---|---|
| 1 | X4, the C2 `q_R` patch and block-1 ablation do not distinguish C′ from a twin with an additive (per-source AND) gate. The twin has the same decisions (100%), the same relation-blind pathway, the same X4 flips and the same C2 shares. The package's own relation-free twin also "passes" X4. | R2 FATAL-1, R5 H1, R7 | **accepted** | X4 is now described as evidence that **the block-1 pathway is necessary**, not that a relation is used. A **mediator-level sign test** was added (`src/mediator_sign_test.py`). S1: C′'s gate feature has the candidate-symmetric pattern (++, −− > 0; +−, −+ < 0) on 500/500 random draws; the twin's has it on 0/500. S2: interchanging R's block-1 output from (−a, −b) into (a, −b) routes C′ to the sources (1.000) but the twin to the prior (1.000). These tests separate the two. |
| 2 | In learned models, the "agreement-probe interchange transfers 100% vs 0% random" result measures alignment with the readout. In T1, \|cos(probe, readout)\| = 0.97–0.99, and readout-direction interchange transfers 1.0 in T1, T3 and T4 alike. Upstream, the T1 transfer is only 0–0.5. | R2 FATAL-2, R4 F4, R5 H3 | **accepted** | Interchange-as-evidence-of-use is **withdrawn** for the final site, and the "decodable but not used" claim for T3 and T4 is withdrawn (`logs/learned_review_checks.txt`, K1/K2). |
| 3 | The relation-unit criterion is not specific. It flags 50–83% of units; it implements a non-literal definition (ab and sign(ab) regressors, two-way ANOVA instead of backfitting); random same-size sets give similar ablation effects; the result depends on the threshold. | R4 F1/F3, R2 MAJOR-6 | **accepted** | Unit-level identification in learned models is **withdrawn as a positive result**. The criterion is reported as a failed identification procedure, in its own right a §22 counterexample-type result. The undeclared deviations are now declared as D5–D7 in the preregistration. |

## Major findings

| # | finding | reviewer(s) | disposition | what changed |
|---|---|---|---|---|
| 4 | T-C′1* in `CONDITIONAL_CAUSAL_THEOREMS.md` drops the `G ≥ 0` hypothesis (657 of 2,167 points violate the statement as written). | R1 F1 | accepted | Hypothesis restored. The original theorem in `CPRIME_…` §3 was correct. |
| 5 | B-TM's final-LN clause is false: the σ of the final LN depends on routing. | R1 F2, R3 MAJOR-1, R6 | accepted | The clause is removed. Measured: 2,980/3,000 at default settings, and 2,960/3,000 with nonzero values at A. Without final LN: 3,000/3,000, including a **nontrivial** test with nonzero values at A. T-C4 conclusion 2 is restricted accordingly. |
| 6 | "Every relation-gated sign rule needs a dead band" is false: `min(a, b)` implements the rule exactly. | R1 F3, R5 H2, R6 | accepted | T-CO is restated: a continuous margin that implements T must **vanish on the positive axes**. A nonzero margin there forces a dead band. C′'s bands come from its design (`m(a, 0) ≠ 0`). The historical C′1 was false *for C′*; its form was not impossible. |
| 7 | The ReLU unequal-weight cone is missing its upper bound, `t < (r+1)/(r−1)`. | R1 F4 | accepted | `MLP_NORMALIZATION_THEOREMS.md` §4 corrected. |
| 8 | "Only candidate-symmetric rules force a cross-source interaction" is false on two counts. (a) The K = 3 rule "agree → that candidate, else the unpicked one" is LP-infeasible for additive scores. (b) Shared LN couples sources, so a single-coordinate reader computes XNOR. | R2 MAJOR-5, R1 F5 | accepted | T-RF is scoped to the **fixed-prior deferral family**. Counterexample (a) is recorded, and (b) is added as Proposition N2: "no unit reads two sources" does not mean relation-free in a pre-norm Transformer. |
| 9 | T-RF's "behaviour never shows relation use" holds only at the decision level. C′'s graded margins have 357/435 order reversals; both twins have 0. | R2 MAJOR-4, R7 1 | accepted | Added **T-RF′** (single cancellation): a margin of the form `h(f(a)+g(b))` with h increasing has 0 reversals. The graded output is therefore a behavioural test of cross-source interaction, though not of *agreement* specifically. F4 is re-rated. |
| 10 | The C2 claim "values carry 1.8%" depends on which source is flipped, and `q_B`'s 0 share is forced by the symmetric flip. | R2 MAJOR-3, R3 | accepted | C2 is rerun with three families (`logs/program_c.txt`): values carry 0.018 (b-flip), 0.909 (a-flip) and 0.753 (asymmetric). `q_R` alone is sufficient (1.0) in all three. The shares are not additive. "Entirely through the query route" is withdrawn. |
| 11 | The corollary to T-B-IMP is too broad. With per-source nonlinear features before the score, one bilinear score detects agreement exactly. | R3 MAJOR-2 | accepted | Corollary scoped to linear encodings. Check: a one-hot argmax feature plus one bilinear score gives 1.000 (B1b). |
| 12 | "T1 relation computed redundantly via attention" is contradicted by the logs. | R3 MAJOR-3 | accepted | Withdrawn. The three mean-ablation-robust cases are method artefacts: resample ablation gives 0.50–0.69 in TF1 (K3). |
| 13 | The relation-free twin matches C′ only on [−3, 3]². | R6 | accepted | Claim scoped to [−3, 3]². |
| 14 | The ablation method matters: mean vs resample ablation, and non-flagged controls. | R2 MAJOR-6 | accepted | K3 reported for all models. |
| 15 | B-add is over-specified (p enters both f and g). | R5 M1 | accepted | Stricter baselines added (K5): B-add3, `f(a)+g(b)+h(p)`, scores T3 0.996 and T5b 0.979. **B-signs**, linear on per-source sign one-hots plus the prior, scores **1.000 on T3 and T5b**. |
| 16 | Residual-basis rotation *is* applicable to the learned models. | R3, R4 F6 | accepted | The doc statement is corrected. The reviewer's exact implementation (`reviews/R4_checks/c4_resid_rotation.py`, logit change ≤ 8e−6, residual-site results unchanged) is cited. The "intrinsic because untransformed" argument is withdrawn as circular. |
| 17 | The T-RF "new" label conflicts with known threshold logic; T-B-IMP is convexity; T-CO is continuity. | R5 M5 | accepted | Relabelled as known mathematics applied to this question. |
| 18 | The T4 learned conclusion was wrong. The criterion misses odd interactions such as `ab(a−b)`. | R4 F5, and post-review K4 | accepted | K4 graded test: T4 margins have 28–337 reversing row pairs, up to 1.1 SD. **The learned T4 models use a cross-source interaction** even though the T4 decision rule is additively representable. Corrected throughout. |
| 19 | The T3 graded test was described as "essentially additive". | R7 2 | accepted, with nuance | K4 gives T3 reversals of at most 0.07 SD (TF1/TF2) and 0.25 SD (TF2-AO). Consistent with `h(f(a)+g(b))` up to small distortions. |
| 20 | Sign-prediction rates depend on the unit count, ignore the final-LN mean, and the TF2 rates are lower. | R4 F7, R5 L3, R2 minor | accepted | TF2 rates (71–75%) reported next to TF1. Caveats added. |

## Minor findings (all accepted unless noted)

**Mathematics (R1).**

- F6: the interval arithmetic was not fully outward-rounded. Fixed: relative padding on every operation, and constants enclosed. All E3/E6 results reproduce exactly.
- F7: the conflict-ratio region was not fully covered, and Q−+ was not run. Fixed with covering boxes; both quadrants are now certified.
- F8: "the decision depends on relation and aggregate" needed a witness. Added: (0.07, 0.07) and (0.2202, 0.02296) have equal `G = 0.2001` and margins −0.159 vs +0.0008.
- F9: the B-IMP proof needed a measure argument (Fubini over `z_B`). Added.
- F10: the mixed-sign cross-talk case has two failure cones. Corrected.
- F11: ray uniqueness and the sweeps are numerical only. Labelled.
- F12: T-R is a supremum as a → ∞. The bounded-domain value within |a| ≤ 3 is 1.376 (reviewer). Added.
- F13: hypothesis slips (γ > 0 for P3; "generically" for P4). Fixed.

**Transformer (R3).**

- TF2-AO wording: routing is necessary (frozen patterns → chance), and LN is per-position. Corrected.
- C3 "redundant" variant: not function-preserving (Δm up to 0.33). Corrected.
- C3 "competing" variant: equals a gain rescale `6(1−c)`. Corrected.
- C4 bypass check used `mid[0]` instead of `resid_in[1]`. Fixed; outcome unchanged (107 points, 0 violations).
- No-op line in `ablate_dir`. Removed.
- Downstream MLP is not a scalar φ(m). Wording corrected.
- `qk3` description: B attends to A in layer 1, and the stated "/3" factor was wrong. Corrected.
- B3 was trivially true. A nontrivial test was added.
- The B1 binary check was tautological. Removed.
- C2 manifold grid: 61 × 61, as now stated.

**Causal (R2), minor.**

- The C′ reachability limit is borrowed from A: labelled numeric for C′.
- "The only difference is the relation feature": corrected.
- The tied argmax in B4: already corrected before review.
- MAIN cited the wrong control: corrected.

**Counterexample (R6), minor.**

- C′ conflict band beyond 3: E6b added (b* → −0.039 at a = 50). Rectangles are claimed only on [τ, 3].
- T2 "sum function" holds only on the training range: scoped.
- `closed_form` overflow at λ ≥ 300: fixed (shifted exponent).
- Exact zeros below 1e−16: noted.

**Representation (R4), minor.**

- F8: deviations. D5–D7 declared. The D4 timing is stated exactly; see the preregistration.
- F9: split invariance is trivial by construction. Acknowledged.
- F10: K = 3 agreement is on noisy observations (differs from latent agreement on 34% of draws), and `μ_s`/`Y_s` were not recorded. Noted in `SOURCE_SEMANTICS_AND_IDENTIFICATION.md`.
- F11: the gain-vs-cross-talk split depends on parameterization. Noted.

**Baseline (R5), low.**

- B-MLP parameter counts: noted.
- B-add T4, trained 4× longer: 0.978 (reviewer). Noted.
- Population gauge invariance holds by construction. Acknowledged.

**Scientific defense (R7).**

- **Accepted:** T-RF′ and the F4 re-rating; B's confidence modulation reported as a restricted positive under the B-TM class; the engineered F1 bridge is now complete with S1/S2; the preregistered held-out sign prediction (33/35, T1/TF1) as the prospective element; X4 not called prospective; T1 called a learnability control only.
- **Rejected:** none.

## Net effect on the headline

**Strengthened:**

- The engineered identification chain for C′ is now complete. The mediator sign test (S1/S2) separates a relation gate from an additive gate with identical behaviour, identical relation-blind pathway and identical X4 and C2 results.
- The graded-output test (T-RF′) gives a behavioural signature of cross-source interaction.

**Weakened or withdrawn:**

- unit-level identification in learned models;
- interchange-based "use" claims;
- "T4 learned models use no relation";
- "every gated rule needs a dead band";
- B-TM with final LN;
- "only candidate-symmetric rules need interaction";
- the C2 single-path claim.

**Unchanged:** no predictive advantage over relation-free baselines (F29).
