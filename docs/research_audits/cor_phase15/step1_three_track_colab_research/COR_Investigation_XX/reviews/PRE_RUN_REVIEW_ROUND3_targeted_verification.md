# Pre-run review, round 3: targeted verification of the ADJUDICATION_v2 fixes

Reviewer: an independent AI agent (not a human peer reviewer). It covers specialties A, B and C, but only for the round-2 fixes. Date: 2026-11-20.

## Verdict: **FREEZE BLOCKED**

The Track B and Track A science fixes are correct. All four simulated worlds come out as intended (§3). Three defects in frozen-path code block the freeze. Each one is a small edit:

- **X1:** the planted-test gate cannot fail.
- **X2:** the planner crashes on a terminal screening failure.
- **X3:** mixed identity is recorded but never reported.

After these edits, re-run the three tests and re-freeze. **Another review round is not needed.**

## 1. Blockers (exact required changes)

**X1. Regression (from the "tests write to `$COR_TEST_OUT`" minor fix): `tests/test_scorers_planted.py` always exits 0.**
- The last line ends `... default=str)   # never into the package; sys.exit(1 if fails else 0)`. The `sys.exit` sits inside the comment, so it never runs.
- This file has no other `sys.exit`. `pipeline.preflight` (`res["planted_tests"] = r.returncode == 0`) and `scripts/clean_extraction_test.py` therefore pass even when a planted test fails.
- *Change:*
  - Put `sys.exit(1 if fails else 0)` on its own line after the `json.dump`.
  - Update the docstring, which still says the results go to "tests/planted_results.json".
  - Recommended: one negative check that a forced failure gives rc = 1.

**X2. `cor_research/jobs/plan.py::eligibility` raises `TypeError` after a terminal screening failure.**
- When a seed's screening job has failed 3 times, `recs = "FAILED"; break` leaves only the inner (D) loop. The outer `if recs is None: break` does not fire.
- The next seed's verified record then runs `recs[f"{s}:D{D}"] = ...` on a string. Reproduced: seed 20/D901 failed ×3, all other seeds done → `TypeError: 'str' object does not support item assignment`.
- `plan()` calls `eligibility()`, so the runner stops at `register(planner())`.
- *Change:* use `if recs is None or recs == "FAILED": break` after the inner loop.
- *Also decide and freeze:* should one seed's terminal failure make the whole parent `SCREENING_FAILED`? Under the frozen unit rule ("some seed s qualifies"; `all_seeds_always_run=True`), the alternative is to treat that seed as non-qualifying and record it as such. Choose one and state it in `SCREEN.unit_rule`.

**X3. C-N2 is only partly done: mixed identity is recorded but never surfaced.**
- `check_identity` sets `RUN_IDENTITY.json["mixed_identity"]` and promises that "results will be labelled MIXED-IDENTITY".
- But `report.py`, `integrity.py` and the scorers never read `RUN_IDENTITY.json`. This was C r2 N2(d), and C made it a condition of approval.
- *Minimum change (in `report.py`, a frozen file):* when `RUN_IDENTITY.json["changes"]` is non-empty, the banner must list every change and label every headline verdict MIXED-IDENTITY.
- *For the per-pair `MIXED_STACK` split (attempts already carry `identity_sha256`):* either implement it, or mark it DEFERRED in the adjudication. Do not leave it marked FIXED.

## 2. ADJUDICATION_v2 rows

| id | status | evidence / note |
|---|---|---|
| A-N-B1 | verified | `track_b.score`: the matched and final functional statuses must agree; the locus is read at matched and the precision flag at final. Sim (i) gives B3 and B2-LOCUS+B3 (10/10 at 6 and 4 parents) |
| A-N-B2 | verified | `hbc = ½(prim+sec)` with both bases, otherwise flagged; all flagged → "not separable"; B1 needs `same_sign is True`. Sims (ii) and (iii) are correct |
| B-N1 | verified | `t4.decide` gives −1 for ties; `DIR_RULES=(local,noelig)`; `discriminating` uses unique decisions only. I recomputed unique-decision disagreement with Bayes: local 0.127, noelig 0.530, all others 0.000, matching the adjudication |
| C-N1 | verified | `freeze.py` stores `TBR.code_hashes()` (41 entries, the same producer the battery uses); `freeze_sha256` covers the files and both hash sets. The planted acceptance test against the real FREEZE.json that the reviewer asked for (N1(b)) was not added (minor) |
| C-N2 | **partly verified** | Verified: the freeze change is a hard stop; PREFLIGHT, RESUME_TEST and the benchmark are superseded; `run()` requires a preflight under the current identity; `identity_sha256` is stored per attempt; strictness covers GPU model, py major.minor, packages, nvidia-*, flags, precision and freeze. Missing: X3. The harness `resume_log` still records only `resumed_at` (acceptable, because every resume is a new attempt with its identity) |
| A-N1 | verified | `BfromVSGO_L2QK` is in ARMSET and in XA7; HA10a text relabelled; every non-window arm has a measured step-100 composed state (`design()["tps"]`) |
| A-N2 | verified | HA10a/b gate on the vsgo pair being I_like → `CONDITION_NOT_MET`; counted and excluded from the parent's evaluable pairs |
| A-N3 | documented (partial) | The 0.42 figure is in DESIGN_XX.md §64 but not in COMPUTE_BUDGET.md, contrary to the adjudication (minor) |
| A-N4 | verified | Contradiction rule is `C ≥ ⌈n/2⌉ and C > S`; OC re-run: one null parent P(GEN) = 0.075, silent null 0.868 |
| A-N5 | verified | `flags_for` covers every transplant/perturb arm except the anchors; uses step-100 TP KL versus the source windows plus steps 150 and 300; sensitivity at 0.05 and 0.20 in `score_all` (the monkeypatch works: same module globals) |
| B-N2 | verified | `dstat` = P(rule k's decision) − P(third option) on unique-rule ≠ Bayes inputs; ≈ 0 for iid and for structured precision errors in the sims |
| B-N3 | verified | `test_trackb_oc.py` re-run: precision-only gives B3 1.0; directed gives B1 1.0; history-only gives B3 1.0 (no B1). PASS |
| B-N4 | verified | `equivalence_table`: tie-aware, with tie rates and categories |
| C-N3 | **regression** | FAILED_PRETRAINING is no longer frozen (verified). The new SCREENING_FAILED path crashes (X2) |
| C-N4 | verified | `harness_xx.py`: step npz → opt_step → resume.pkl, each atomic; `resume_log` uses `_aj` |
| C-N5 | verified | `decisions_sha256` written by the battery and checked in `track_b.level`. Minor: the check is skipped when the field is absent; make it fail-closed |
| C-N6 | verified | nvidia-* versions in `identity()`; `JAX_PLATFORMS=cuda` in GPU mode |
| C-N7 | verified | Pass requires checks 01 and 03; resume test re-run when the identity differs. `graph_audit.json` stays cached across identity changes (minor) |
| C-N8 | verified | `budget.py` applies the 1.5× margin; a missing measurement gives `ok=False`; the trash note appears in the preflight output only |
| C-N9 | verified | tests, fixtures and both requirements files are in the freeze set (92 files); clean-extraction script present; build records commit and dirty count. `protocols/DESIGN_XX.md` and the notebooks are not hashed (minor) |
| minor: graph-audit vsgo check | verified | Executed on CPU (uP160): all 17 checks pass, including `vsgo_forward_identical` and `vsgo_routing_gradient_intact` for E and M |
| minor: report KeyError | not independently verified | The original defect was not located; the Track B report fields match the scorer output keys |
| minor: tests write outside the package | verified, with regression X1 | All three tests left the package unchanged when run with `COR_TEST_OUT=/tmp/rev3` |
| minor: hash every vendored .py | verified | `measure.code_hashes` globs `legacy/**/*.py` |
| minor: α grid / A1–A2–B split | verified | Grid 0.45–0.95; fit on A1, select α* on A2, refit on A, evaluate on B. The null bisection range starts at 0.40, below the grid (negligible) |
| minor: HB_c_dir text vs code | verified | `protocol.TRACK_B.tests["HB_c_dir"]` matches the code |
| documented: one runner / orphans | partial | The heartbeat lock is in COLAB_SETUP.md; the orphan-process / restart-runtime advice was not found in the user docs (minor) |

## 3. Tests and simulations (CPU; scratch under /tmp/rev3)

**Test re-runs:**
- `test_scorers_planted.py`: 31/31 PASS. Note that its rc is meaningless because of X1.
- `test_verdict_oc.py`: PASS.
- `test_trackb_oc.py`: PASS.

**Simulations.** All go through the real `track_b.score` with both levels as the pipeline produces them:
- Real tie-aware discriminating set: 18 671 inputs from a pool of 40 000.
- Both bases, 4 data orders, 10 replicates.
- Matched level: KL 0.015/0.015 and error rates 1.4%/1.4%.
- Final level: KL 0.030/0.015 and error rates 2.6%/1.4%.

| world | 6 parents | 4 parents | intended? |
|---|---|---|---|
| (i) true B3, iid errors | B3 1.0 | B3 [indicative] 1.0 | yes |
| (i) frozen prediction B2-LOCUS+B3 (OQI_2 0.5 vs 0.2) | B2-LOCUS+B3 1.0 | same [indicative] | yes |
| (i′) the same two, structured errors (lowest-margin inputs flip to the Bayes runner-up, nested) | B3 / B2-LOCUS+B3 1.0 | same | yes |
| (ii) history only (fullI arm 5% systematic flips), both bases | B3 1.0 (no B1, matched status EQUIVALENT) | same | yes |
| (ii′) history only, directed toward `local` (15% of S_local) | B3 1.0 (DIRECTED_DIFF false) | same | yes |
| (iii) E-only parents, real class effect toward `local` 0.15 | FUNCTIONAL_DIFFERENCE (class and intervention history not separable) 1.0 | same | yes |
| (iv) control: both bases, the same class effect | B1-FUNCTIONAL 1.0 | same | yes |

**Notes (not blocking):**
- With structured precision errors, the final-level HB_c is ≈ +0.0025 for a 1.2-point error-rate gap. Final-level equivalence (upper bound < δ_dis = 0.02) therefore holds at development-like gaps. A B3 headline would turn into NON_IDENTIFIABLE only if the final error gap were roughly 8× larger.
- When only some parents are history-confounded, they share one t-interval with the unconfounded parents. B1 is still protected by the native/induced sign requirement.
