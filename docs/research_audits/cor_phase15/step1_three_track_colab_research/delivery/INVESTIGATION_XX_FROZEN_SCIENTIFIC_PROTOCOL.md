# COR Investigation XX — frozen scientific protocol (delivery copy)

Freeze: `protocols/FREEZE.json`, freeze_sha256 **f1b92d934921e55b09bfd693e436f81be0cdc53dea2a78bb382267cc0adb441d**, frozen 2026-11-20T08:06:39Z, 93 files.
This document is generated from the frozen sources; the machine-readable protocol (protocols/*/*.json, generated from cor_research/protocol.py) is authoritative.
Phase 1.5, Step 1 (≈ 25% provisional). No result exists until the package is run. Reviews are AI-agent reviews (reviews/), not human peer review.

## Frozen Track A hypotheses

| id | base | arm | expected | step | role | statement |
|---|---|---|---|---|---|---|
| HA1 | EM | fullI | switch | 4000 | primary | Blocking the base's own route for the first 100 steps switches the organisation: the I anchor is competent and of the opposite causal class at 4000. |
| HA2 | EM | paramI_optB | I_like | 4000 | mediation | The step-100 parameters carry the selection (I parameters with B moments -> I-like). |
| HA3 | EM | paramB_optI | B_like | 4000 | mediation | Adam moments alone do not carry it (B parameters with I moments -> B-like). |
| HA4a | E | BfromI_L0 | I_like | 4000 | replication_anchor | E: the I layer-0 state is sufficient (near-tautological; replication anchor, not a finding). |
| HA4b | E | IfromB_L0 | B_like | 4000 | mediation | E: the I layer-0 state is necessary (restoring B's layer 0 prevents the switch); INCOMPATIBLE pairs are not evidence. |
| HA5a | M | BfromI_L0L1 | I_like | 4000 | mediation | M: L0+L1 sufficient. |
| HA5b | M | BfromI_L0L2 | I_like | 4000 | mediation | M: L0+L2 sufficient. |
| HA5c | M | BfromI_L0 | not_I_like | 4000 | weak_negative | M: L0 alone is not sufficient (weak negative claim; not headlined). |
| HA5d | M | IfromB_L0L2 | B_like | 4000 | mediation | M: restoring B's L0+L2 prevents the switch; INCOMPATIBLE pairs are not evidence. |
| HA6a | EM | randnorm | B_like | 4000 | control | A random displacement of the sufficiency groups with matched per-group norm does not reproduce the effect. |
| HA6b | EM | lbl | B_like | 4000 | control | Label-noise loss disturbance (valid strength only) does not reproduce the effect. |
| HA7 | M | vsgo | I_like | 4000 | replication | M: removing V+O learning through the blocked route suffices (XIX replication). |
| HA10a | M | BfromVSGO_QK | I_like | 4000 | discriminating | M (new): the QK WEIGHTS (all layers) learned under vsgo suffice. This is a QK-weights vs non-QK-weights test, not a pure routing-vs-value test (vsgo changes the layer-0/1 QK gradients on M; review A r2 N1). Evaluated only in pairs whose vsgo arm is I-like. Interpretation table: DESIGN §1. |
| HA10b | M | BfromVSGO_nonQK | B_like | 4000 | discriminating | M (new): the non-QK weights learned under vsgo do not suffice (complement of HA10a; only in pairs whose vsgo arm is I-like). |
| HA8E | E | rsg | partial | 4000 (+1211) | low_confidence | E: removing the routing gradient of the layer-0 readout rows (all edges) gives a PARTIAL, HYBRID-class outcome. Basis: one dev run (uP160, D801, step 1211). Overlaps XIX E_P5_frzp_partial; SUPPORTED distinguishes little, a CONTRA outcome is the informative one. |
| HA9 | EM | fullB,fullI | persist | 4000 | primary | Persistence: each anchor's causal class at 1211 equals its class at 4000 and it is competent at 4000 (>= 80% of evaluable pairs per parent). |

Exploratory (no verdict):
- XA1 (E, vsgo): E: vsgo effect (no directional prediction in XIX; distribution reported).
- XA2 (M, rsg): M: rsg = replication of XIX M_P5_frzp_nil plus an upstream-gradient change (review A B2: not discriminating).
- XA3 (EM, *): Every hypothesis re-evaluated at the historical horizon 1211 (secondary).
- XA4 (EM, fullB,fullI): Stability 4000 -> 12000: low-LR (peak 1e-4) and full-LR positive control (peak 1e-3); wording limited to 'class unchanged under this continuation' unless the full-LR control moves some organisation.
- XA5 (EM, *): Unconditional sensitivity: over every pair with a native B anchor, the arm's class and its concordance with the same pair's fullI class.
- XA6 (E, IfromB_L0_QK,IfromB_L0_OV,half_L0L2): High-preset extra arms: exploratory, no frozen prediction.
- XA7 (M, BfromVSGO_L2QK): M: layer-2-QK-only transplant from the vsgo window (localises HA10a; exploratory).

## Arms

| arm | kind | bases | role |
|---|---|---|---|
| fullB | transplant | EM | anchor (B state re-entered through the composer; bitwise check) |
| fullI | transplant | EM | anchor (I state) |
| paramI_optB | transplant | EM | parameters without moments |
| paramB_optI | transplant | EM | moments without parameters (optimizer-state-only control) |
| BfromI_L0 | transplant | EM | forward transplant, layer 0 |
| IfromB_L0 | transplant | E | reverse transplant (restore B's layer 0 into I) |
| BfromI_L0L1 | transplant | M | component combination |
| BfromI_L0L2 | transplant | M | component combination |
| IfromB_L0L2 | transplant | M | reverse combination |
| randnorm | perturb | EM | parameter-norm control: B + random direction with ||I-B|| per group, groups = the base's sufficiency groups (review A M6) |
| BfromVSGO_QK | transplant | M | routing-state-only transplant: QK of every layer from the vsgo window state (review A B2 option b) |
| BfromVSGO_nonQK | transplant | M | complement: every non-QK group from the vsgo window state |
| BfromVSGO_L2QK | transplant | M | layer-2 QK only from the vsgo window state (review A r2 N1) |
| vsgo | window | EM | value/output-gradient intervention through T (routing gradient intact) |
| rsg | window | EM | readout-row routing-gradient removal at the target layer (on M = layer-2 QK frozen + upstream gradient change: a replication, review A B2) |
| lbl | window | EM | matched non-targeted disturbance (validity: step-100 KL excess >= 0.5 x the block's) |
| IfromB_L0_QK | transplant | E | EXPLORATORY: reverse transplant, layer-0 QK only |
| IfromB_L0_OV | transplant | E | EXPLORATORY: reverse transplant, layer-0 OV only |
| half_L0L2 | transplant | EM | EXPLORATORY: interpolation 0.5 on L0+L2 |

## Track B (frozen)

```json
{
 "units": "confirmation anchors fullB and fullI; class = causal class at the analysed step (competent, pure)",
 "matching": {
  "selection_stream": 91430000,
  "n": 3000,
  "steps": "every saved snapshot in [150, 4000]",
  "rule": "per parent, kappa_p = the largest KL at step 4000 among the parent's anchors that are competent and pure at 4000 (floored at 0.015); each anchor's matched snapshot is the EARLIEST snapshot whose selection-stream KL <= kappa_p (every such anchor reaches it by 4000: no class-dependent censoring by construction); matched step, LR and step difference are reported per class (review A M9)"
 },
 "sets": {
  "nat_stream": 91400000,
  "disc_stream": 91410000,
  "pair_stream": 91420000,
  "n_nat": 12000,
  "n_disc_pool": 40000,
  "n_pairs": 3000
 },
 "candidates": "bayes; local; noelig; nocount; maprole (full role MAP, tie-averaged); softrole (geometric pooling over roles: listed, calibration-only); historical nested power-mean with lapse (xviiim.rho_fit_lapse; descriptive); threshold-4 rule: NOT LOCATED in the vendored code, not reconstructed",
 "thresholds": {
  "delta_dis": 0.02,
  "delta_dir": 0.03,
  "delta_w": 0.03,
  "delta_oqi": 0.1,
  "delta_kl": 0.005,
  "min_parents_both": 4,
  "indicative_below_parents": 6,
  "alpha": 0.05
 },
 "tests": {
  "HB_c": "PRIMARY functional test (decision level, calibration-invariant, on-support tie-aware discriminating set). Per parent: the within-unit contrast (different data orders in both terms: between = fullB(D1) vs fullI(D2), within = same class, D1 != D2) measures class + intervention history; the same-D cross-unit contrast (parents with both bases; history-matched) measures class only; the parent statistic is their average when both exist (history cancels, review A r2 N-B2), else the within-unit contrast with the parent flagged history-confounded. If every parent is confounded, a difference is reported as FUNCTIONAL_DIFFERENCE (class and intervention history not separable), never as B1.",
  "HB_c_dir": "rule-directed (v2.1, tie-aware): for k in {local, noelig} (nocount / maprole differences are exact ties), on the inputs where k's UNIQUE decision differs from Bayes's: per model d_k = P(decision = k's) - P(decision = the third option) (zero in expectation for precision-only errors); per parent mean_EARLY - mean_MIXTURE; Holm across the two rules on t-test p; B1 requires the native-vs-native and induced contrasts to have the SAME sign (missing -> not satisfied)",
  "HB_a": "DESCRIPTIVE ONLY: rule-mixture weights over the precision family F0' (tau, eps, alpha' free), KL-adjusted (within-class regression to kappa_p)",
  "HB_b": "DESCRIPTIVE ONLY: deviation above the parametric-precision null Bayes(alpha') at the model's KL and above the iid logit-noise null",
  "HB_d": "DESCRIPTIVE LOCUS profile: other-question interchange from the HYBRID run (source-matched; base-run null must reproduce the base), IIA normalised by the model's own following of Bayes(x_h); EARLY - MIXTURE at the layer-2 input. It shows where cross-question evidence is read, not what is computed.",
  "HB_e": "precision: KL at 4000, mean_p(E - M); expected positive from development (dev dKL ~ 0.015 > delta_kl): a '+B3' flag is not a discovery",
  "equivalence": "TOST at parent level: functional equivalence iff the upper 95% bound of HB_c < delta_dis AND for every rule the 90% CI of HB_c_dir lies within +-delta_dir (t-interval on parent means; exact sign-flip p reported)"
 },
 "classification": [
  "INSUFFICIENT_UNITS if < 4 parents have >= 1 evaluable model of each class (labels 'indicative' below 6)",
  "B1-FUNCTIONAL (different decision functions; internal causal mechanism NOT identified by this design): HB_c difference (lower 95% bound > delta_dis, sign-flip p reported) AND some rule-directed HB_c_dir difference (Holm) with the same sign in the native-vs-native and induced contrasts",
  "FUNCTIONAL_DIFFERENCE_RULE_UNIDENTIFIED: HB_c difference without a rule-directed difference",
  "B2-LOCUS (functionally equivalent; different location of unidentified content): equivalence AND |HB_d| > delta_oqi (sign-consistent)",
  "B3 (same rule, different precision): equivalence AND not B2-LOCUS AND HB_e; '+B3' flag added to B2-LOCUS when HB_e",
  "EQUIVALENT_NO_DIFFERENCE_DETECTED: equivalence and neither HB_d nor HB_e",
  "NON_IDENTIFIABLE (underpowered): neither a functional difference nor equivalence"
 ],
 "headline_rule": "v2.1 (review A r2 N-B1): the FUNCTIONAL status (difference / equivalence / neither) must agree between the matched and the final levels (else NON_IDENTIFIABLE); the locus (HB_d) is read at the matched level and the precision flag (HB_e) at the final level (matching removes the precision difference by construction)",
 "frozen_prediction": "From development (uP160 XIX anchors, n = 2 + 2; development only): predicted B2-LOCUS (+B3 flag): no functional difference at matched precision, equivalence achieved if >= 6 parents; HB_d EARLY > MIXTURE at the layer-2 input; HB_e positive (expected, not a discovery)."
}
```

## Track C (frozen)

```json
{
 "code": "cor_research/mathematics (od_original, od_repaired, t4_order, run_track_c)",
 "neural_test": null,
 "classification": "cor_research.mathematics.run_track_c.classify (predeclared; C5 is reviewer-adjudicated with the candidate list)",
 "rule": "no neural O/D experiment is run: every candidate neural prediction considered (factor-deletion destinations, O-tie sensitivity, soft-gate expected-energy pooling, T4 latent-role expected energy, T4 additive complex) coincides with exact Bayes on the declared model, is a standard mean-field/log-linear pooling distinction, needs an unstated bridge assumption, or is not representable (review B \u00a73.6, C-MAJ-3)"
}
```

## Rules, thresholds, streams, presets

```json
{
 "SCREEN": {
  "seeds": [
   20,
   21,
   22,
   23,
   24,
   25,
   26,
   27
  ],
  "D": 901,
  "D2": 902,
  "stop": 1211,
  "stream": 91000000,
  "all_seeds_always_run": true,
  "unit_rule": "unit (parent, base) is ELIGIBLE iff some seed s (L = W = s) gives a competent, causally pure baseline of the base class at step 1211 on BOTH screening data orders 901 and 902 (base E: EARLY, base M: MIXTURE); the unit seed is the lowest such s. Only screening records whose measurement job is registered done with matching checkpoint/code hashes and the frozen stream are used. A screening run or measurement that fails 3 attempts makes THAT SEED non-qualifying (recorded as failed); the parent is still screened."
 },
 "COMP": {
  "KL": 0.06,
  "acc": 0.8
 },
 "CLASS": {
  "E_hi": 0.3,
  "E_lo_M": 0.1,
  "M_hi": 0.3,
  "M_lo_E": 0.15
 },
 "CONF_D": [
  2001,
  2002,
  2003,
  2004,
  2005,
  2006
 ],
 "PILOT_D": [
  2901
 ],
 "STREAMS": {
  "measure_primary": 91100000,
  "measure_window": 91150000,
  "measure_secondary": 91200000
 },
 "STEPS": {
  "window": 100,
  "horizon_hist": 1211,
  "primary": 4000,
  "stability_end": 12000,
  "stability_lr": 0.0001,
  "stability_lr_hi": 0.001,
  "early_checks": [
   150,
   300
  ]
 },
 "BAND": {
  "I_like": [
   0.75,
   9.99
  ],
  "B_like": [
   -9.99,
   0.25
  ],
  "partial": [
   0.25,
   0.75
  ]
 },
 "DELTA": 0.1,
 "EPS": 0.01,
 "EPS_SENS": [
  0.003,
  0.03
 ],
 "MIN_PAIRS_PER_PARENT": 3,
 "MIN_PARENTS": 4,
 "PARENT_RULE": {
  "support_frac": 0.5,
  "max_contra": 1,
  "max_contra_frac": 0.25,
  "contra_frac": 0.5,
  "contra_rule": "C >= ceil(n/2) and C > S (review A r2 N4)"
 },
 "CROSS_RULE": {
  "gen_frac": 0.8,
  "gen_exceptions": 1,
  "fail_min": 2,
  "fail_frac": 0.3333333333333333
 },
 "VALIDITY": {
  "lbl_excess_ratio_min": 0.5,
  "incompatible_kl_excess": 0.1,
  "incompatible_sensitivity": [
   0.05,
   0.2
  ],
  "incompatible_rule": "a composed state is INCOMPATIBLE if its KL at step 100 (composed state) exceeds both window states' KL, or at step 150 / 300 exceeds both anchors' KL, by more than 0.10 (uncalibrated; flag rates and 0.05 / 0.20 sensitivity reported). A supporting outcome of a flagged transplant (any transplant arm) is relabelled INCOMPATIBLE (not evidence)"
 },
 "PARENT_VALIDITY": {
  "metric": "pretraining eval aux_acc (train_tasks.evaluate('pcount', seed=41_000_000))",
  "min_aux_acc": 0.95,
  "note": "below threshold -> INVALID_PRETRAINING (reported); a failed pretraining job -> FAILED_PRETRAINING (reported). Known: uP162 = 0.779 (threshold set knowing this; review A M3)."
 }
}
```

```json
{
 "smoke": {
  "label": "SMOKE: software test only (CPU-feasible, shortened schedules). NEVER evidence; every verdict is forced to TECHNICAL_ONLY.",
  "parents": [
   "uP160"
  ],
  "n_D": 1,
  "stability": true,
  "claims_can": "code paths execute",
  "claims_cannot": "anything scientific"
 },
 "pilot": {
  "label": "Pilot: end-to-end technical validation on separate material (parent nP3901, data order 2901). NOT confirmation evidence.",
  "parents": [
   "nP3901"
  ],
  "n_D": 1,
  "stability": false,
  "claims_can": "the pipeline runs end to end on this runtime; measured throughput; storage",
  "claims_cannot": "any scientific claim (verdicts forced to TECHNICAL_ONLY)"
 },
 "standard": {
  "label": "Standard research",
  "parents": [
   "nP3001",
   "nP3002",
   "nP3003",
   "nP3004",
   "uP161",
   "uP162",
   "uP163",
   "uP164",
   "uP165"
  ],
  "n_D": 4,
  "stability": false,
  "claims_can": "parent-level Track A verdicts if >= 4 parents are evaluable per base (fresh-only verdicts need >= 4 fresh parents); Track B indicative (< 6 parents)",
  "claims_cannot": "persistence beyond 4000 steps; generalisation to other architectures/tasks"
 },
 "extended": {
  "label": "Extended research",
  "parents": [
   "nP3001",
   "nP3002",
   "nP3003",
   "nP3004",
   "nP3005",
   "nP3006",
   "nP3007",
   "nP3008",
   "uP161",
   "uP162",
   "uP163",
   "uP164",
   "uP165"
  ],
  "n_D": 6,
  "stability": true,
  "claims_can": "as standard, with 6 data orders per unit, fresh-only verdicts, and the stability continuations (low-LR + full-LR control) for anchors",
  "claims_cannot": "other architectures/tasks"
 },
 "high": {
  "label": "High-compute research",
  "parents": [
   "nP3001",
   "nP3002",
   "nP3003",
   "nP3004",
   "nP3005",
   "nP3006",
   "nP3007",
   "nP3008",
   "nP3009",
   "nP3010",
   "nP3011",
   "nP3012",
   "uP161",
   "uP162",
   "uP163",
   "uP164",
   "uP165"
  ],
  "n_D": 6,
  "stability": "all_arms",
  "claims_can": "as extended, plus exploratory QK/OV splits and interpolation, stability for every arm",
  "claims_cannot": "other architectures/tasks (no validated alternative architecture exists in COR)"
 }
}
```

---

## Design and design-gate answers (protocols/DESIGN_XX.md, verbatim)

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

