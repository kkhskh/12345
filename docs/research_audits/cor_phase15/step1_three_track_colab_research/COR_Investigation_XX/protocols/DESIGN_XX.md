# Investigation XX — scientific design for the pre-compute review gate (v2.1, after review rounds 1 and 2; see reviews/ADJUDICATION_v1.md and ADJUDICATION_v2.md)

Status: Phase 1.5, Step 1 (≈ 25% provisional). Nothing has been run on Colab. Development runs (labelled `dev/`, parent uP160,
data order D801) were done on CPU in the build session only to check that each planned test can discriminate; they are not
evidence and never enter a scorer. The single source of every constant is `cor_research/protocol.py`, written to `protocols/*/*.json`.

## 0. What XIX left open (project doc `claude/xix_result_summary.md`)
XIX (one parent uP160, one fresh-layer seed per base, 3 data orders) established at state level that the 100-step training-window
selection is carried by step-100 parameters, not Adam moments; E: the layer-0 state is sufficient and necessary; M: L0+L1 and
L0+L2 sufficient, L0 alone not; non-targeted disturbances do not reproduce it; on M, removing V+O learning through the blocked
route suffices. Claim 2 (a different functional rule) was not established: "implementation reorganisation with a precision
difference". Next steps named: a routing-gradient stop on E, a routing-state transplant on M, transport to other parents.

## 1. Track A — causal generalisation across independent pretrained parents
**Question.** Do the XIX state-level results hold at the level of *independent pretrained parents*?

**Parents (A1).** Preregistered ordered list per preset (`PRESETS[*].parents`): the historical XIII cohort uP161–165 (vendored,
pretrained by the same recipe; uP160 is the XIX development parent and is excluded from every verdict) and fresh parents
nP3001–nP3012 pretrained in the run by the unchanged recipe (P-count, 3000 steps; resumable wrapper verified bitwise equal to the
historical function). Pretraining validity is frozen before screening: aux_acc ≥ 0.95 (uP162 = 0.779 fails; it is reported).

**Eligibility (decided before any intervention).** Screening: seeds 20–27 (L = W = s), data orders 901 AND 902 (v2), baseline only,
to step 1211. All 8 seeds always run (no optional stopping). Unit (parent, base) eligible iff some seed gives a competent, causally
pure baseline of the base class at 1211 on both data orders; the unit seed is the lowest such seed. Only screening records whose jobs are
registered done with matching checkpoint / measurement / code hashes are used. Fresh parents come first; uP161–165 are tagged
previously observed (XVIII screened them and ran arms on uP163), and every verdict is also reported on fresh parents only. The unit of
replication is the (parent, selected-init) unit: between-"parent" variance includes init-seed variance (stated limitation). The denominator (every preregistered parent, its validity,
every screening record, every unit decision) is reported in `eligibility.json` and RESULTS. The eligibility file is written once
and an attempt to change it is an integrity failure.

**Replication (A2).** Per eligible unit, fresh data orders D = 2001… (4 in Standard, 6 in Extended/High). Data order is matched
across arms within a pair. Parents are the statistical unit; pairs are nested (two-stage bootstrap, variance components).

**Arms (A3/A4)** — chosen to separate explanations, not a factorial:
| arm | separates |
|---|---|
| fullB, fullI (anchors) | the effect itself; anchors re-entered through the composer (bitwise check vs native states) |
| paramI_optB / paramB_optI | parameters vs Adam moments (optimizer-state-only control) |
| BfromI_L0, IfromB_L0 | E: layer-0 sufficiency / necessity (forward and reverse transplant) |
| BfromI_L0L1, BfromI_L0L2, IfromB_L0L2 | M: component combinations, reverse restoration |
| randnorm_L0L2 | parameter-norm control (B + random direction of the same per-group norm as I − B) |
| vsgo | V+O learning through the blocked route removed; forward and routing gradient intact |
| rsg | all routing gradient of the target layer's readout rows removed (graph-verified). On M this freezes layer-2 WQ/WK (zero gradient: disclosed by the audit) = a replication of XIX frzp-M (exploratory XA2); on E: HA8E (low-confidence) |
| BfromVSGO_QK / BfromVSGO_nonQK / BfromVSGO_L2QK (new, M) | QK weights (all layers) vs non-QK weights (vs layer-2 QK only, exploratory) learned under vsgo, transplanted into B at step 100; evaluated only where the pair's vsgo arm is I-like. Interpretation: QK I-like & nonQK B-like → the vsgo effect is carried by QK weights (note: vsgo also changes layer-0/1 QK gradients on M, so this is QK-weights vs non-QK-weights, not pure routing vs value); QK B-like & nonQK I-like → carried by value/output/MLP state; both I-like → redundant carriers; both B-like → neither alone (interaction) |
| lbl | matched non-targeted loss disturbance (label noise); valid only if its step-100 KL excess ≥ 0.5 × the block's (else CONTROL_INVALID) |
High preset adds QK/OV-only reverse transplants and a 0.5 interpolation.

**Outcomes.** Primary at step 4000 (full historical schedule, continuation from step 100); secondary at the historical horizon 1211;
state compatibility: composed states measured at step 100, continuations at 150 and 300 (INCOMPATIBLE flag for reverse transplants);
Extended: anchors continued 4000 → 12000 with a restarted warmup-cosine at peak 1e-4 AND a full-LR (1e-3) positive control; wording
limited to "class unchanged under this continuation".
Measurement: XIX primitives (KL to exact Bayes, Bayes-argmax accuracy, necessity of the own0 and own2 routes on 1500 natural
episodes of a fresh stream); labels recomputed by the scorer.

**Frozen hypotheses** (`protocols/frozen_hypotheses/track_a_hypotheses.json`): HA1 switch; HA2 parameters carry; HA3 moments do not;
HA4a (replication anchor) / HA4b E layer-0 sufficient / necessary; HA5a–d M combinations (HA5c weak, not headlined); HA6a/b norm and
label-noise controls nil; HA7 M vsgo suffices (replication); HA10a/b (new, M) routing state from vsgo suffices / non-QK state does not;
HA8E E rsg partial with HYBRID class (low confidence; CONTRA is the informative outcome; 1211 co-primary); HA9 persistence 1211 → 4000 with
competence. Rules (v2.1): pair statuses as in protocol.py; parent SUPPORTS iff SUPPORTED ≥ ⌈n/2⌉ and CONTRA ≤ max(1, ⌊n/4⌋), CONTRADICTS iff
CONTRA ≥ ⌈n/2⌉ and CONTRA > SUPPORTED; cross-parent GENERALIZES iff SUPPORTS ≥ max(n − 1, ⌈0.8 n⌉) (i.e. every parent at n = 4, all but one at
n ≥ 5) and no CONTRADICTS, FAILS_TO_GENERALIZE iff CONTRADICTS ≥ max(2, ⌈n/3⌉), else NOT_ESTABLISHED; INCONCLUSIVE with < 4 evaluable
parents. Operating characteristics (tests/test_verdict_oc.py): with every pair evaluable P(GEN | strong effect) ≥ 0.97 at 4–12 parents,
P(FAILS) = 0, P(GEN | null) = 0, non-decreasing in n_D; one contradicting null parent among 8: P(GEN) = 0.075. Stated limits: at 4 parents ×
4 data orders with 80% of pairs evaluable P(GEN | strong effect) = 0.42 (INCONCLUSIVE 0.55): the Standard preset is underpowered when pairs
are lost (Extended, 6 data orders: ≈ 0.90); a silently null parent (mostly MIDDLE) can be the single allowed exception (listed).
INCOMPATIBLE (any transplant; step 100 vs source windows and steps 150 / 300 vs anchors; threshold 0.10 uncalibrated, 0.05 / 0.20 reported).
Mediation verdicts are conditional on the switch (GENERALIZES_CONDITIONAL unless HA1 GENERALIZES on the same base); EPS sensitivity
0.003 / 0.03; phi_E and phi_M reported; nec SEs recorded.

**Design-gate answers (A).**
1. Guaranteed by construction? HA1 is not (the block is temporary; later training could restore the route). HA4a (E layer-0
   sufficiency) is close to tautological (XIX said so); kept as a replication anchor, not as a discovery. HA2/HA3 are not guaranteed.
2. Literature: gradient-only deficits from early training windows replicate Kleinman, Achille & Soatto (ICLR 2024); the parent-level
   generalisation of a route-selection effect in this task is not established anywhere.
3. Competing explanations: (a) parameter-state mediation (HA2, transplants), (b) optimizer history (HA3), (c) generic disturbance
   (HA6), (d) routing-gradient vs value-learning (HA7, HA8M/E) — they predict different arm outcomes.
4. Metric: phi against the pair's own anchors at the same step, plus the class condition; discriminates I-like vs B-like.
5. Controls implemented: norm-matched random, label noise, optimizer-only, reverse transplants, bitwise anchor re-entry.
6/7. A positive cross-parent verdict shows parent-level generalisation of the named mediation; a FAILS verdict weakens it directly.
8. Inconclusive is explicit (missing / noncompetent / anchors not evaluable are counted, never dropped).

## 2. Track B — different algorithms or the same rule? (v2)
**Units.** Confirmation anchors (fullB, fullI). **Precision matching (per parent, outcome-blind).** κ_p = the largest step-4000 KL (selection
stream) among the parent's competent, pure anchors (floor 0.015); each anchor's matched snapshot = the earliest snapshot with KL ≤ κ_p (no
class-dependent censoring by construction; matched steps reported by class; censoring gap > 20 points ⇒ NON_IDENTIFIABLE). Second level:
step 4000 unmatched; a classification is headlined only if both levels agree.
**Candidates (mandate B1)** with an on-support equivalence table (`t4.equivalence_table`, 12k natural episodes; decision disagreement with
Bayes): local 0.24, noelig 0.62, nocount 0.09, maprole (full role MAP, tie-averaged; ties 20%) 0.14, softrole (geometric pooling over roles)
0.009 — calibration-only, listed not tested, Bayes(α' = 0.7) 0.023 (precision null); historical nested power-mean with lapse (descriptive);
the historical threshold-4 rule could not be located and is not reconstructed.
**Tests (v2.1).** All rule decisions are tie-aware (tie-aware disagreement with Bayes: local 0.127, noelig 0.530, nocount / maprole / softrole /
Bayes(α' = 0.7) 0.000 — the latter are calibration- or tie-level and are not used for directed tests). HB_c (PRIMARY, decision level,
calibration-invariant): per parent the average of the within-unit contrast (class + intervention history; different data orders in both
terms) and the same-D cross-unit contrast (history-matched), which cancels history; parents with one base are flagged history-confounded,
and a difference seen only there is "FUNCTIONAL_DIFFERENCE (class and intervention history not separable)". HB_c_dir (local, noelig):
precision-adjusted directed statistic d_k = P(k's decision) − P(third option), Holm, native and induced contrasts must agree in sign. HB_d: descriptive locus profile (other-question
interchange from the hybrid run; base-run null; normalised IIA). HB_e: precision (expected positive; not a discovery). HB_a / HB_b:
descriptive only (precision family with α' profiled; parametric-precision null).
**Classification (frozen).** INSUFFICIENT_UNITS (< 4 parents with both classes); B1-FUNCTIONAL (HB_c difference AND a rule-directed
difference; the internal mechanism is NOT identified by this design); FUNCTIONAL_DIFFERENCE_RULE_UNIDENTIFIED; B2-LOCUS (TOST equivalence
AND a locus difference: "functionally equivalent; different location of unidentified content"; +B3 flag when HB_e); B3 (equivalence AND
HB_e only); EQUIVALENT_NO_DIFFERENCE_DETECTED; NON_IDENTIFIABLE (underpowered) when neither a difference nor equivalence is shown.
The functional status must agree between the matched and final levels; locus is read at the matched level, precision at the final level.
Fewer than 6 parents ⇒ "indicative". Operating characteristics on the real discriminating set with dev-calibrated error rates
(tests/test_trackb_oc.py): precision-only → B3 (10/10), directed 0.15 → B1-FUNCTIONAL (10/10), history-only → no B1, at 4 and 6 parents.
Equivalence is achievable only if the between-parent SD of HB_c is ≲ 0.01–0.02 (unknown; one dev parent).
**Deferred (stated):** a cross-model interchange / abstract-variable alignment test of "the same sufficient statistic" (review A M11,
review B MAJ-6). Without it B1 can only be functional and B2 only a locus statement.

## 3. Track C — O/D mathematics (independent; CPU) (v2)
Original O/D from the Phase-1 definitions; repaired O/D from the project note (its script `od_repair_check.py` could not be found and is
NOT reconstructed; an independent implementation re-derives every claim). Results (dev run of the v2 code, `dev/track_c/track_c_math_v2.json`):
- original: T1 D = (ds)ᵀ(dM⁻¹dᵀ)⁺(ds); T2 W* ⇒ O ≡ D (checked against the direct projection); T3 evenness (with the odd-margin premise);
  T4 P = 0 ⇒ O = D = 0. Source-level instantiation on GATE: an exact order-1 LP with a rational Farkas certificate shows no order-1 additive
  rule reproduces the Bayes decisions ⇒ **C4**; germ-level definition ⇒ **C5 / underdetermined** (Attack F, metric arbitrariness).
- repaired on GATE (729 inputs × 5 conditions): analytic proof (E_h = −log factor, observed gate, uniform prior ⇒ D_c = −log p(x|c) + const)
  ⇒ R1, R3, R5 and factor deletion = Bayes with the source marginalised out of the joint (independent reference, max error 1e-16) ⇒ **C2**,
  scope: observed selector, each source in one factor, uniform prior. Order: exact LP certificates (orders ≤ a infeasible, a+1 feasible).
  M1 (O is a function of the posterior) holds on GATE only (restricted C3; false in general). Decisive gate flips: 216 ordered = 108
  unordered (the note's 162 is not reproduced). Latent-marginalisation change mass 0.111 / 0.017 reproduces the note (ties excluded).
- T4: nonzero interaction coefficients on 18 input coordinates ⇒ no additive complex with hyperedges of order < 18 reproduces T4's exact Bayes
  LOG-ODDS (decision-level representability not addressed); independently recomputed at 40 digits by reviewer B.
- neural prediction: **C5 (reviewer-adjudicated)**; candidates considered and rejected: FD destinations (= Bayes without the factor),
  O-tie sensitivity (GATE-only function of the posterior), soft-gate expected-energy pooling (mean-field vs exact mixture, standard; not in the
  note; needs an unstated bridge; GATE only), T4 latent-role expected energy (calibration-level), T4 additive complex (not representable).
No neural O/D experiment is run.

## 4. Integrity and execution
Registry with one record per job, never replaced; atomic checkpoints; exact resume (parameters, AdamW state, data RNG, loss history);
freeze verified before any job; 20 integrity checks; planted tests for every scorer; measured benchmark → presets' GPU-hours.
GPU determinism is run-to-run on the same GPU model and stack only (fp32 matmuls forced; verified on the user's runtime by the preflight
resume/determinism test); CPU is bitwise (tested). Run identity (device, GPU model, versions, flags, freeze) is recorded per run root and a
change is refused unless deliberately allowed (recorded). The runner refuses scientific presets without a passing preflight.
