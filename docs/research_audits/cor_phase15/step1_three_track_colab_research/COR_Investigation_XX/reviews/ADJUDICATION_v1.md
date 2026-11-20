# Adjudication of the pre-run design reviews (round 1)

The three reviewers are AI agents (Claude), not human peer reviewers. Each was given the v1 design and code, worked independently and
had authority to reject tests. Their reviews are `PRE_RUN_REVIEW_A_causal_identification_v1.md`, `PRE_RUN_REVIEW_B_mathematics_functional_v1.md`
and `PRE_RUN_REVIEW_C_reproducibility_integrity_v1.md`. The corrected design (v2) is `protocols/DESIGN_XX.md`. As the mandate requires
("any corrected design must be reviewed again"), it goes to a second review round before the freeze.

Decision codes: **FIXED** (changed as required, or with an equivalent change explained below), **DOCUMENTED** (accepted as a limitation
and stated in the design and reports), **DEFERRED** (accepted as valid but not implemented in XX, with the reason given), **DECLINED** (not adopted, with the reason given).

## Review A — causal identification (verdict: approve with required changes; 5 blockers)

| id | issue | decision | where |
|---|---|---|---|
| A-B1 | Verdict rules non-monotone; FAILS without contradictions | FIXED. The parent supports iff SUPPORTED ≥ ⌈0.5·n⌉ and CONTRA ≤ max(1, ⌊0.25·n⌋). FAILS requires CONTRADICTS ≥ max(2, ⌈n/3⌉). NOT_ESTABLISHED is the outcome otherwise. The reviewer's suggested parent rule (≥ 0.75 supported) was simulated: it still failed its own criterion (P(GEN) fell from 0.82 to 0.62 going from 4 to 6 data orders), so the rule was recalibrated. Operating characteristics are preregistered in `tests/test_verdict_oc.py` (PASS at 4/6/8/12 parents × 4/6 D) | protocol.PARENT_RULE/CROSS_RULE; track_a.parent_verdict/cross_parent |
| A-B2 | HA8M (rsg on M) not discriminating (layer-2 QK frozen; = XIX frzp) | FIXED, option (a) plus option (b). HA8M is demoted to exploratory XA2 and labelled as a replication. New discriminating arms on M: **BfromVSGO_QK** (routing state from the vsgo window), frozen as HA10a, I-like, with the competing value-state prediction stated; and its complement **BfromVSGO_nonQK** (HA10b). The graph audit now reports the tensors whose gradient is exactly zero (rsg on M → layer-2 WQ/WK) | protocol ARMS/HYPOTHESES_A; graph_audit |
| A-B2(b2) | Projected T-share routing stop (prsg) | DEFERRED to XXI. It needs a new custom-gradient operator, and its graph audit must establish what it removes. BfromVSGO_QK answers the M question first; on E, HA8E stays low-confidence | DESIGN §1 |
| A-B3 | B2 is preordained (no equivalence test; HB_d implied by the class definition) | FIXED. B2 and B3 now require equivalence (TOST at parent level). Otherwise the outcome is NON_IDENTIFIABLE (underpowered). B1 requires HB_c. HB_d is a descriptive locus profile only (the B2 label is "B2-LOCUS: different location of unidentified content"). B1 is renamed **B1-FUNCTIONAL** (mechanism not identified) | track_b.score_level; protocol.TRACK_B |
| A-B4 | HB_c biased (between = same D, within = different D) | FIXED. The primary contrast uses different data orders in both terms and the same unit. A same-D 2×2 cross-unit secondary is added. Planted test: a pure shared-D effect gives no difference | track_b; tests |
| A-B5 | HB_d donor contaminated (donor run, not hybrid run) | FIXED. The donor is now the hybrid run x_h (the donor run is kept as the labelled "unmatched" secondary). A base-run null must reproduce the base logits (max difference reported). IIA is normalised by the model's own following of Bayes(x_h) | functional.interchange |
| A-M1 | Mediation conditional on switch | FIXED. The label is mandatory. A mediation GENERALIZES becomes GENERALIZES_CONDITIONAL unless HA1 GENERALIZES on the same base. A denominator chain is given per hypothesis, and an unconditional sensitivity analysis (XA5) is added | track_a |
| A-M2 | Single screening data order | FIXED. The unit seed must be native on both screening data orders, D901 and D902 | protocol.SCREEN; plan.eligibility |
| A-M3 | Historical cohort not blind | FIXED. uP161–165 are tagged previously_observed and fresh parents come first in every list. Every verdict is also computed on fresh parents only. A headline GENERALIZES is withdrawn if the fresh-only verdict is FAILS, and is annotated if fewer than 4 fresh parents are evaluable | track_a.score_all |
| A-M4 | Parent confounded with selected init seed | DOCUMENTED. The claim is restated as "independent (parent, selected-init) units", and variance components are renamed between-unit. A second qualifying seed per unit is DEFERRED for compute reasons | DESIGN §1; track_a |
| A-M5 | phi noise near zero necessity | FIXED. Episode-bootstrap SEs of nec_E and nec_M are recorded (dev: nec_M SE ≈ 0.0008, so the floor 0.01 is > 2 SE). EPS sensitivity at 0.003 and 0.03 is reported. phi_E and phi_M are reported separately | measure; track_a |
| A-M6 | lbl validity check dropped; randnorm groups mismatched on E | FIXED. Step-100 window states are measured on a dedicated stream. CONTROL_INVALID applies when lbl's KL excess is < 0.5 × the block's. randnorm uses the base's sufficiency groups (E: L0; M: L0+L2) | plan; track_a.flags_for |
| A-M7 | No compatibility checks for composed states | FIXED. Composed states are measured at step 100 and every continuation at steps 150 and 300 (both kept after pruning). INCOMPATIBLE applies when the KL excess over both anchors exceeds 0.10. A B-like reverse-transplant outcome is then not counted as necessity | plan; track_a |
| A-M8 | False precision with 4–8 parents; multiplicity | FIXED. Inference uses t-intervals on parent means, with exact sign-flip p reported. Track B rule tests use Holm correction. Classifications with fewer than 6 parents are labelled "indicative" | track_b; track_a |
| A-M9 | Matched precision confounds training time; class-dependent censoring | FIXED. Matching is per parent (κ_p = the largest step-4000 KL among competent pure anchors; matched step = the earliest snapshot with KL ≤ κ_p, so there is no class-dependent censoring by construction). Matched steps are reported by class. If the censoring gap exceeds 20 points the result is NON_IDENTIFIABLE. Headlining requires the matched and final levels to agree | trackb_run; track_b |
| A-M10 | Class confounded with intervention history | FIXED. Native-vs-native and induced contrasts are computed per rule, and B1 requires them not to have opposite signs | track_b |
| A-M11 | HB_d is a locus test | FIXED (accepted). HB_d is descriptive and never part of B1. A cross-model interchange is DEFERRED (it needs a learned alignment between models); B1 is therefore "B1-FUNCTIONAL, mechanism not identified" | track_b; DESIGN §2 |
| A-M12 | Stability continuation near-guaranteed | FIXED. A full-LR (peak 1e-3) positive control is added for anchors. The wording is limited to "class unchanged under this continuation" | protocol; plan |
| A-M13 | HA8E mis-specified | FIXED. "Partial" now requires the HYBRID class. Step 1211 is co-primary. The design states that SUPPORTED distinguishes little and CONTRA is the informative outcome | protocol; track_a |
| A-m1 | FAILED_PRETRAINING missing | FIXED | plan.eligibility |
| A-m2, m3, m6 | HA5c weak; HA4a replication anchor; High arms exploratory | FIXED (role field; XA6) | protocol |
| A-m4, m5, m7, m8, m9 | competence-loss rate; HA9 competence; var_comp descriptive; HB_b descriptive; zero-gradient report | FIXED | track_a; track_b; graph_audit |

## Review B — mathematics and functional equivalence (verdict: approve with required changes; Track B classification rejected)

| id | issue | decision | where |
|---|---|---|---|
| B-BLK-1 | "No difference" = failure to reject | FIXED (TOST equivalence; NON_IDENTIFIABLE (underpowered); planted test) | track_b; tests |
| B-BLK-2 | Parametric imprecision read as a rule mixture | FIXED. The precision family F0′ now has α′ profiled on a grid. HB_a is descriptive only and removed from B1. B1 requires the decision-level HB_c and a rule-directed HB_c_dir | functional; track_b |
| B-MAJ-1 | HB_c unbalanced and non-directional | FIXED (with review A B4's design as primary and the reviewer's same-D cross-unit design as secondary; HB_c_dir added) | track_b |
| B-MAJ-2 | iid noise null unfair | FIXED. A parametric-precision null Bayes(α_null) at the model's KL is added. Both nulls are descriptive | functional |
| B-MAJ-3 | OQI donor not source-matched | FIXED (= A-B5) | functional |
| B-MAJ-4 | Historical models and the B1 equivalence table missing | FIXED, partly. Added: the historical nested power-mean with lapse (xviiim.rho_fit_lapse, descriptive), softrole (calibration-only), and `t4.equivalence_table` (decision disagreement of every candidate with Bayes and with each other). The threshold-4 rule could **not be located** in the vendored code and is **not reconstructed** (mandate C1 / user instruction) | t4; functional |
| B-MAJ-5 | maprole definition and position ties | FIXED. maprole is now the full role MAP over role assignments, with ties averaged (tie fraction 0.20 on natural episodes) | t4.role_rules |
| B-MAJ-6 | No positive test of the same sufficient statistic | DOCUMENTED and DEFERRED. B2 is labelled "B2-LOCUS: functionally equivalent; different location of unidentified content". An abstract-variable interchange with a learned alignment is deferred to XXI | track_b; DESIGN §2 |
| B-MAJ-7 | Thresholds not justified | FIXED. `tests/test_trackb_oc.py` gives power and false-positive rates at 4 and 8 parents through the real scorer (P(B1 \| null) = 0; HB_c effects of 0.045 are detected at 4 and 8 parents; 0.015 is NON_IDENTIFIABLE). delta_dis = 0.02 is the minimum effect of interest: 2% of on-support discriminating decisions, about twice the per-model noise floor assumed in the simulation | tests; protocol |
| C-MAJ-1 | T5 guaranteed; C4 mis-scoped | FIXED. C4 (source-level instantiation) now rests on the exact order-1 LP with a rational Farkas certificate, verified in every condition. The germ-level definition is C5/underdetermined. T5 is descriptive | od_original; od_repaired; run_track_c |
| C-MAJ-2 | C2 needs a proof and an independent M2 reference | FIXED (analytic proof and scope in od_repaired; M2 against the marginalised joint: max error 1e-16; M1 removed from C2) | od_repaired; run_track_c |
| C-MAJ-3 | C5 designer-enumerated | FIXED. C5 is marked "reviewer-adjudicated", and all five candidates are listed with their rejection reasons, including the reviewer's soft-gate expected-energy candidate | run_track_c |
| C-MAJ-4 | T4 order theorem scope | FIXED (log-odds scope; trivial completion; latent roles = factor-graph marginal inference) | t4_order |
| C-MIN-1..6, B-MIN-1..4 | T2 vs D_direct; T3 premise; LS claim; 216 ordered / 108 unordered; high-precision certificate; protocol filename; HB_c calibration text; censoring by class; sign-flip; HB_e expected | FIXED. The reviewer's 40-digit script is included verbatim (attributed) in `tests/independent/` | various |

## Review C — reproducibility and integrity (verdict: reject in the v1 state; approve after B1–B3, M1–M11)

| id | issue | decision | where |
|---|---|---|---|
| C-B1 | No freeze, manifests or tests | FIXED. Missing tests are written. The freeze and manifests are created after round 2 (`scripts/freeze.py`, `scripts/build_package.py`), and the freeze hash is recorded in the delivery documents | scripts |
| C-B2 | No run identity; TF32; mixed environments | FIXED. `RUN_IDENTITY.json` covers device, GPU model, package versions, XLA flags, matmul precision and freeze hash. It is checked at every setup and before every run, and a mismatch is refused unless deliberately allowed (the change is then recorded). GPU runs set JAX_DEFAULT_MATMUL_PRECISION=highest | env.identity; pipeline.check_identity |
| C-B3 | Notebook interrupt duplicates jobs | FIXED. On interrupt the runner terminates its child process groups and records the attempts as interrupted. A 'running' record whose process is still alive on this machine is never reset | runner v2 |
| C-M1 | Eligibility reads unverified screening records | FIXED (registry status, checkpoint, measurement and code hashes, and stream are all checked; mismatch raises) | plan._verified_screen |
| C-M2 | Stray files adopted | FIXED (any pre-existing file in an unregistered job's directory → blocked) | runner.register |
| C-M3 | Non-atomic writes; truncated outputs marked done | FIXED. opt_step / final.json / curve / run_config / window_loss / stop_done / composer outputs are now written atomically, with the completion marker last. Outputs are validated (npz/pkl/json load) before a job is marked done | harness_xx; compose_xx; runner.validate |
| C-M4 | No refusal on wrong deps / Python / failed audits | FIXED. The preflight `pass` flag (freeze, deps + Python ≥ 3.12, graph audit, planted tests, verdict OC, runtime resume + determinism test, storage) is required by `pipeline.run` for every scientific preset. The notebook raises on a pip failure | pipeline |
| C-M5 | GPU determinism/resume untested | FIXED. `scripts/resume_test.py` runs on the user's runtime during preflight (CPU result here: bitwise resume and rerun identical) | scripts |
| C-M6 | Report completeness only over registered jobs | FIXED. Completeness also requires that no planned job is unregistered and that all three tracks are present. Integrity failures appear in every RESULTS banner. Verdicts are marked PROVISIONAL when incomplete | report |
| C-M7 | Scorer hash coverage | FIXED. Measurement code now includes env.py and models/load.py. Parent checkpoint hash, measurement-job status and hash, and snapshot hashes for every training checkpoint are checked. The Track B code hash covers trackb_run + functional + t4 + measurement code | measure; track_a; track_b; runner |
| C-M8 | Pilot uses confirmation material | FIXED (pilot: parent nP3901, data order 2901) | protocol |
| C-M9 | Vacuous integrity checks | FIXED (NOT_YET when nothing exists; 13 compares the frozen file; 18 also cross-checks registered pairs; 12 includes planned-but-unregistered jobs) | integrity v2 |
| C-M10 | Planted tests incomplete | FIXED. 31 planted cases now cover eligibility immutability, runner rules and integrity check 14 | tests |
| C-M11 | Storage; Drive rescans | FIXED. Preflight refuses if the estimated storage exceeds 90% of free space. The runner registry is in-memory, write-through, and replans only when a job finishes | pipeline; runner |
