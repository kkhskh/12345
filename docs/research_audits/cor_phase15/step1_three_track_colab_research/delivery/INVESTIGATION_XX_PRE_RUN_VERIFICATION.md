# COR Investigation XX — pre-run verification report

**Scope.** This is software validation done on **CPU only** in the build environment (Linux, Python 3.13, jax/jaxlib 0.12.0.1, optax 0.2.8, numpy 2.5.3, scipy 1.18.1, 2 cores). **No GPU was available here, and no Colab GPU was tested.** No confirmation experiment was run, and nothing below is scientific evidence. The GPU path is checked on the user's runtime, before any scientific job, by the preflight's runtime resume/determinism test, graph audit and benchmark (`cor_research/pipeline.preflight`).

## 1. Delivered artefacts

| item | value |
|---|---|
| ZIP | `cor_inv20_colab_three_track_research.zip` (single archive, 168 files, ≈ 11.5 MB) |
| freeze | `protocols/FREEZE.json`, freeze_sha256 `f1b92d934921e55b09bfd693e436f81be0cdc53dea2a78bb382267cc0adb441d`, 93 files plus the measurement and Track B code-hash sets |
| freeze history | one re-freeze, recorded in `protocols/FREEZE_HISTORY.json` (see below) |
| source | branch `cor-phase15-step1-three_track_colab_research`, commit `d4e69a2`. The manifest records 5 uncommitted package-subtree changes at build time: the generated manifests, the re-freeze and the fixed test script, committed after the build. |

The re-freeze happened after the first clean-extraction run exposed a path bug in `scripts/clean_extraction_test.py`. That run's notebook-03 step failed because the script read the notebook by a relative path from the wrong directory. No package code changed in the re-freeze.

## 2. Clean-environment test

**Procedure** (`scripts/clean_extraction_test.py`). The ZIP was extracted into an empty directory, and every check ran from the extracted copy only. The checks:
- the package manifest;
- the vendored checkpoint hashes;
- the dependency pins and the Python version;
- the freeze.

| step (build 2, final ZIP) | result | seconds |
|---|---|---|
| ZIP integrity (`testzip`) | PASS | — |
| package manifest, vendored checkpoints, dependencies, freeze (integrity checks 01, 02, 03, 05) | PASS | 1 |
| notebooks: nbformat validation and every imported module / pipeline function exists | PASS | 0.2 |
| `tests/test_pretrain.py`: resumable pretraining wrapper is bitwise equal to the historical `train_tasks.train`, including a crash plus resume | PASS | 47 |
| `tests/test_tags.py`: the planner's tag rules equal the harness's for every window mode and base | PASS | 64 |
| `tests/test_scorers_planted.py`: 31 planted cases (see §3) | PASS | 53 |
| `tests/test_verdict_oc.py`: Track A verdict operating characteristics | PASS | 7 |
| `tests/test_trackb_oc.py`: Track B operating characteristics on the real discriminating set | PASS | 31 |
| `tests/test_track_c.py`: Track C reproduces the reference classification and exact values | PASS | 13 |
| notebook 03 (Track C) executed end to end with nbclient (CPU) | PASS | 16 |

**Complete small-scale technical pipeline.** The `smoke` preset was run inside the extracted copy of build 1, whose package code is identical to build 2 apart from the test script. It is a software test only: shortened schedules on the development parent uP160. It ran 99 jobs through the real planner, runner, harness, composer, measurement, Track B selection / matching / batteries, scorers, report generator and integrity checks. All 99 jobs finished, plus 1 deliberately failing job, in 1 650 s. Checks:
- **Exact resume:** a continuation killed after its step-150 snapshot and resumed is bitwise identical to an uninterrupted run (`resume_log: resumed_at 150`).
- **Failed job:** the deliberately failing job is recorded as failed and never replaced.
- **Track A scoring:** Track A is scored with full provenance (no PROVENANCE_FAIL), and every verdict is forced to TECHNICAL_ONLY.
- **Track B:** forced to TECHNICAL_ONLY.
- **Report:** the report generator refuses to claim completion while a job is not done.
- **Integrity checks:**
  - **PASS (15):** 01, 02, 03, 04, 05, 08, 09, 10, 13, 14, 16, 17, 18, 19, 20.
  - **FAIL (1):** 12, run completion. This is expected, because of the deliberate failure.
  - **NOT_YET (4):** 06, 07, 11, 15. These need the preflight and Track C, which the smoke preset does not run.

**Runtime resume/determinism test** (`scripts/resume_test.py`, CPU). Three runs: uninterrupted, killed and resumed, and uninterrupted again. Resume was bitwise identical and the rerun deterministic (exit codes A 0, B 3 → 0, C 0).

**Intervention graph audit** (`cor_research/interventions/graph_audit.py`, CPU). All 17 checks pass:
- rsg: forward identical, with zero routing gradient on the target and complement edges of the readout rows. On M it zeroes layer-2 WQ/WK, which is disclosed.
- vsgo: forward identical up to float32 re-association (1.4e-6), with routing gradient intact.
- gsg: target logits zero, complement nonzero.
- block: changes the forward pass.
- lbl: forward identical.

The pre-review (v1) smoke report is preserved in `dev/smoke_v1/` as a record of the v1 code.

## 3. Planted scorer tests (31 cases, all PASS)

**Track A**
- confirmation, falsification, inconclusive (too few parents);
- missing inputs counted and not dropped;
- incompetent arms classified;
- a band change keeps evaluability (the XVIII lesson);
- lbl CONTROL_INVALID;
- reverse-transplant INCOMPATIBLE;
- wrong checkpoint hash and wrong code hash refused through the real loader;
- consistent inputs accepted;
- the conditional label requires HA1.

**Track B**
- E-only functional difference reported as history-confounded;
- B2-LOCUS, B3, and underpowered → NON_IDENTIFIABLE (not B2);
- the shared-data-order null gives no difference;
- insufficient and incompetent units;
- missing input listed;
- wrong hash refused through the loader.

**Track C**
- reference classification;
- a planted distinctive prediction is detected;
- a planted order-1 representability is detected.

**Infrastructure**
- eligibility is immutable;
- the runner keeps failed jobs, refuses definition changes and blocks unregistered outputs;
- integrity check 14 detects a corrupted checkpoint.

## 4. Operating characteristics (preregistered)

**Track A** (4 000 simulated campaigns per cell):
- **Strong effect (pS = 0.9, pC = 0.02), every pair evaluable:** GENERALIZES ≥ 0.97 at 4–12 parents × 4 or 6 data orders.
- **Null:** P(GENERALIZES) = 0.
- **False failure:** P(FAILS) = 0.
- **One contradicting null parent among 8:** P(GENERALIZES) = 0.075.
- **Stated limits:** at 4 parents × 4 data orders with 80% of pairs evaluable, P(GENERALIZES) = 0.42. A silently null parent can be the single allowed exception (P = 0.87).

**Track B** (real tie-aware discriminating set, 9 353 inputs; error rates calibrated to development):

| world | outcome |
|---|---|
| precision difference only | B3 (10/10) |
| directed effect 0.15 | B1-FUNCTIONAL (10/10) |
| history-only effect | no B1 |

Each holds at 4 and 6 parents. The round-3 reviewer additionally simulated the frozen-prediction world (B2-LOCUS+B3, 10/10) and the E-only real-effect world (history-confounded label).

## 5. Design gate (mandate §2)

Every review below is an AI-agent review, not human peer review. The reviews and adjudications are inside the ZIP under `reviews/`.

| round | reviewers | outcome |
|---|---|---|
| 1 | three independent agents: causal identification; mathematics and functional equivalence; reproducibility and integrity | approve-with-changes / approve-with-changes / reject. Rejected: the rsg-on-M "new" test, the Track B classification and the verdict rules. Required: equivalence testing, a balanced HB_c, a source-matched interchange, fail-closed provenance, run identity, and more. Adjudication: `ADJUDICATION_v1.md` |
| 2 | the same three specialties re-reviewed v2 | new blockers: B3 unreachable, history confound, tie artefacts, a Track B hash that could never match, the identity override. Adjudication: `ADJUDICATION_v2.md` |
| 3 | targeted verification of the round-2 fixes, with simulations | FREEZE BLOCKED on three code defects, all fixed and re-tested (`ADJUDICATION_ROUND3.md`). The reviewer requested no further round |

Deferred and stated in the design:
- a projected routing-gradient stop (prsg);
- a cross-model / abstract-variable interchange, without which B1 can only be "functional" and B2 only a locus statement;
- a second initialisation seed per unit;
- a per-pair split for mixed GPU identities.

## 6. Not verified here (honest limits)

- **Any GPU execution.** Nothing was run on a GPU. The jax-cuda12-plugin and jax-cuda12-pjrt 0.12.0.1 wheels exist for Python 3.12 (a `pip download` check), but the full `jax[cuda12]` dependency resolution, including the nvidia-* wheels, was not completed here (disk space).
- **Colab-specific behaviour:** Drive FUSE atomic renames, Drive trash on pruning, runtime restarts after pip installs.
- **Notebooks 00, 01, 02 and 04 were not executed end to end:** they need the freeze, a GPU-length run and Drive. The same pipeline functions were exercised through the smoke run, and notebook 03 executed.
- **Measured GPU throughput, and therefore GPU-hours,** is unknown until the preflight benchmark runs on the user's runtime. COMPUTE_BUDGET.md gives exact job counts and storage estimates only.
- **The repaired-O/D companion script `od_repair_check.py` could not be found.** It was not reconstructed; Track C is an independent re-implementation. The historical threshold-4 rule was likewise not located and not reconstructed.
