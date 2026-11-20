# Pre-run review C — reproducibility and experimental integrity (v1)

**Reviewer:** an AI agent (Claude) acting as an independent pre-experiment reviewer. This is an automated agent review, not a human peer review. I did not build the package.
**Date:** 2026-11-20 (UTC). **Scope:** COR Investigation XX package as found at
`docs/research_audits/cor_phase15/step1_three_track_colab_research/COR_Investigation_XX`, before any Colab confirmation compute.
**Method:** I read DESIGN_XX.md, env/pipeline/jobs/training/scoring/verification/reporting modules, scripts, the five generated notebooks, the tests, the requirements files, `harness_xx.py`, `compose_xx.py`, and MANDATE_XX.txt §§7, 8, 9, 11–16. I ran only short CPU checks: pure tag and count computations from `plan.py` and `budget.py`, a `pip download` of the numpy pin for Python 3.11 and 3.12, file-size reads of the `dev/` snapshots, and read-only inspection of `/tmp/smk`. I modified no package file except this review.
**What I could not do:** I had no GPU, so every GPU statement below is a code-reading inference that still needs testing on a GPU. The smoke run at `/tmp/smk` was restarted from an empty root at about 06:06 UTC while I was reviewing. Before the restart I saw 61 registry records (40 done, 12 pending, 9 failed). After it I saw 60 records (8 done, 2 running, 50 pending). I could not see why the earlier run failed. Its result should still appear in the pre-delivery verification report (mandate §15), not only the result of the rerun.

---

## Summary verdict: **REJECT for confirmation in the current state. APPROVE WITH REQUIRED CHANGES once B1–B3 and M1–M11 are fixed.**

Several parts are sound and well designed:
- one registry record per job, never deleted;
- a check that refuses a job whose definition changed after registration;
- failed jobs stay failed and are never silently replaced;
- atomic snapshots and `resume.pkl` in the harness;
- an exact data-RNG and AdamW resume, bitwise on CPU in the smoke test and in `tests/test_pretrain.py`;
- measurement records that refuse to be overwritten;
- the Track A scorer fails closed on a checkpoint-hash or code-hash mismatch;
- TECHNICAL_ONLY is forced for smoke and pilot;
- evaluability is independent of the hypothesis bands.

Three defects would still make the confirmation results unverifiable or not replayable, so I reject the current state:

1. **The frozen state that reviewers approve does not exist yet** (no `FREEZE.json`, no `manifests/`), and nothing binds a run root to it.
2. **Nothing binds a trajectory to one hardware and numeric stack.** Colab routinely changes the GPU model, falls back to CPU, and changes CUDA library versions between sessions. The package lets one trajectory, or the two anchors of one pair, run on different stacks without detection.
3. **The documented "safe to interrupt, re-run the cell" workflow** leaves orphan jobs running. The runner then resets their registry records and launches the same jobs again.

---

## Objections

### BLOCKERS

**B1. The frozen artefacts and manifests are missing, so what the reviewers approve is not pinned down.**
- `protocols/FREEZE.json` does not exist. As a result `runner.verify_freeze` (`cor_research/jobs/runner.py:197-201`) returns False and `Runner.run` refuses to start (fail-closed, good). It also means no hash fixes what the reviewers are approving.
- `manifests/` does not exist: no `PACKAGE_MANIFEST.json`, `CHECKPOINTS.json` or `SMOKE_REPORT.json`, and no script generates them. Consequences:
  - integrity check 01 always FAILs;
  - check 03 **passes without checking anything**: `ck = {}`, so `bad = []` and `n_vendored = 0` (`cor_research/verification/integrity.py:32-38`);
  - check 15 FAILs, because nothing in the Colab path writes `preflight/SMOKE_REPORT.json` (the smoke writes `<root>/SMOKE_REPORT.json`, `scripts/smoke_pipeline.py:46`).
- Code refers to `tests/test_tags.py` (`plan.py:18`) and `tests/test_track_c.py` (`smoke_pipeline.py:17`), but neither file exists. The claim that window tags are "verified against the harness" is therefore untested.

*Required change:*
- (a) Add `scripts/make_manifests.py`. It must write `PACKAGE_MANIFEST.json` (every file in the package with its sha256, plus the source commit), `CHECKPOINTS.json` (every vendored `.npz` with its sha256: the `uP16x` parents, the XVIII ft states and `eval_conf13.npz`) and `SMOKE_REPORT.json`.
- (b) In `integrity.py:32-38`, make 03 FAIL when `CHECKPOINTS.json` is missing or empty. Also make it FAIL when a fresh parent with a `pre.json` has no `lineage.json`; at present such parents are silently skipped (`if L:`).
- (c) Add the two missing tests, or remove the claims that cite them.
- (d) Create the freeze **after** every review-required change has been made. Record the resulting `freeze_sha256` in `protocols/DESIGN_XX.md` §4 and in the review adjudication, so that the approval refers to an exact hash.

**B2. Nothing binds a run root or a trajectory to one hardware, numeric stack and code version.**
- *Where it happens:*
  - `env.configure` (`cor_research/env.py:27-38`) picks GPU or CPU automatically on every session.
  - `pipeline.setup` (`cor_research/pipeline.py:13-31`) only prints a warning when no GPU is visible.
  - The harness writes `run_config.json` once, at the first start (`harness_xx.py:282`), and records nothing on resume (`harness_xx.py:274`).
  - Runner attempts record only `device="gpu"`, not the GPU model (`runner.py:246`).
- *Consequences:*
  - A trajectory can be started on an A100, resumed on a T4, and finished on a CPU after the GPU quota runs out.
  - The B and I anchors of one pair can run on different stacks.
  - A new ZIP, or a `freeze.py --force`, can change the code half-way through a run root. `FREEZE_HISTORY.json` sits in the ephemeral `/content` copy and is lost when the runtime resets.
  - `requirements-gpu.txt` does not pin the `nvidia-*` CUDA wheels (cuBLAS, cuDNN), so the "same software stack" that the determinism claim (`env.py` docstring, DESIGN §4) depends on is not actually fixed.
  - Default float32 matmul precision on GPU is not set. On Ampere and later GPUs XLA may use TF32 by default, while a T4 cannot. This must be verified on Colab hardware.
  - The effect under study is a bistable selection during training, so a change of numeric stack is a potential confound and makes any replay impossible. This breaks mandate §7.4 ("must not silently create a new … trajectory"; "provide a verified continuation procedure").
- *Required change:*
  - (a) At the first `setup()`, write `RUN_ROOT/RUN_IDENTITY.json` containing: preset, protocol version, `freeze_sha256`, `PACKAGE_MANIFEST` hash, device class, GPU name, driver, the versions of jax, jaxlib, the cuda plugin and the `nvidia-*` wheels, and XLA flags. On every later `setup()` and `Runner.run`, refuse to continue if any of these differs. The only exception is an explicit `ALLOW_STACK_CHANGE=True`, which must be recorded as a protocol deviation and shown in every RESULTS banner.
  - (b) Refuse CPU for the pilot, standard, extended and high presets (`pipeline.py:30`: raise instead of warn).
  - (c) In `harness_xx.py`, append to `resume_log.json` on **every** start and resume: devices, GPU name, `XLA_FLAGS`, jax version and code hashes. Refuse to resume if these differ from `run_config.json`.
  - (d) Set `jax_default_matmul_precision="highest"` (or an explicit equivalent) in `env.configure` for both GPU and CPU, and record it in `run_config.json`. If that is judged too slow, record the precision and make it part of `RUN_IDENTITY`.
  - (e) Pin the `nvidia-*` wheels in a new `environment_lock.json` / `requirements-gpu.txt` (mandate §14 lists `environment_lock.json`).
  - (f) Record the GPU name in each runner attempt (`runner.py:246`).

**B3. The documented "Safe to interrupt; re-run this cell" workflow starts duplicate jobs.**
- `notebooks/00…ipynb` cell 12 says interrupting is safe. A KeyboardInterrupt in `Runner.run` reaches the `finally` block, which removes `runner.lock` (`runner.py:296-298`). The worker threads are daemon threads in the still-alive kernel, and the `Popen` children are never terminated, so jobs keep running.
- On the re-run, the lock is gone. `reset_stale_running` (`runner.py:301-308`) sets **every** `running` record back to `pending` without checking whether its process is still alive. Its own docstring says "whose process no longer exists", but no such check is made. The same jobs are then launched a second time.
- Result: two writers on the same `ft/<tag>/` directory (resume.pkl, run_config.json and the non-atomic resume_log.json), two threads doing read-modify-write on the same registry record under *different* `threading.Lock`s, concurrency above the benchmarked safe level (GPU out-of-memory), and an attempt history in the registry that no longer reflects what ran.
- The lock is also a non-exclusive read-then-write with a 120 s staleness window (`runner.py:269-270`). Across two Colab VMs sharing Drive, FUSE caching can make a live heartbeat look stale.
- *Required change:*
  - (a) Store `pid`, `boot_id` (`/proc/sys/kernel/random/boot_id`) and a per-runtime UUID in each `running` record when it is launched (`runner.py:288`). `reset_stale_running` may reset a record only if its boot_id differs from the current one, or if the pid is dead.
  - (b) Start children with `start_new_session=True`. On KeyboardInterrupt or any other exception in `run()`, send SIGTERM to every child process group, wait for them, and record the attempts as `interrupted`.
  - (c) In `harness_xx.py`, after `OUT` is created (line 96), take an exclusive per-run lock `OUT/.lock` with `os.open(..., O_CREAT|O_EXCL)`, containing pid, boot_id and runtime UUID. Refuse to run if a live owner holds it.
  - (d) Make the runner lock exclusive (`O_EXCL`, owner UUID) and re-read it before each launch.

### MAJOR

**M1. The irrevocable eligibility decision reads screening measurements with no provenance check.**
`plan.eligibility` (`cor_research/jobs/plan.py:76-78`) loads any file at the screening measurement path. It never checks that the `mscr:` job is registered as `done`, that its output hash matches, that `ckpt_sha256` equals the `scr:` job's registered output hash, that `code_sha256` equals the frozen measure hashes, or that `stream` equals `SCREEN.stream`. Even a measurement whose job is `blocked` ("unregistered outputs present") would decide eligibility, and eligibility is then written once and can never be revised.
*Required change:* use only records that pass the same provenance checks as `track_a.load_inputs` (`track_a.py:44-56`), plus a registry status/hash check on the `mscr:` job itself. Any failure must raise rather than skip. Apply the same check to `parent_valid` (`plan.py:49-52`): `pre.json` must match `lineage.json` or `CHECKPOINTS.json`.

**M2. Unregistered partial outputs are adopted through the harness's resume path.**
`Runner.register` marks a job `blocked` only when **all** of its declared outputs already exist (`runner.py:229`). A job directory with a stray `resume.pkl` or `step*.npz` and no registry record is registered as `pending`. The harness then resumes from those foreign files (`harness_xx.py:270-274`). Examples: a run root reused after the registry was deleted, files copied from `dev/`, or a directory left by an older package version.
Separately, the harness exits 0 without validating anything when `final.json` exists ("already complete", line 253) or when `resume.pkl` reaches STOP ("already at stop", line 275).
*Required change:* in `register()`, mark a new job `blocked` if **any** file exists under any of its output directories (`ft/<tag>/`, `parents/zP*/`, `trackB/<tag>*`). In the harness, record the sha256 of `step{k}.npz` inside `resume.pkl`, and on resume verify that it matches and that `run_config.json` (tag, env, argv, code hashes) matches the current invocation. Otherwise exit non-zero.

**M3. Several job outputs and completion markers are written non-atomically, and Drive sync can reorder files.**
- *Non-atomic writes:*
  - in the harness: `opt_step{k}.pkl` (`harness_xx.py:306`), which is a declared output of window jobs and the input to every composition and branch; `final.json` (`:312`), which is the completion marker; `stop_done.json` (`:309`); `run_config.json` (`:282`); `resume_log.json` (`:274`); `window_loss.json`;
  - in the composer: `step100.npz`, `opt_step100.pkl` and `provenance.json` (`compose_xx.py:65-71`);
  - in Track B: the decision `.npz` files (`trackb_run.py:41`).
- *Failure case:* a kill during the `opt_step100.pkl` write leaves `resume.pkl` at step 100. The restarted job takes the "already at stop" exit with code 0, the runner hashes the truncated pickle and marks the job **done**, check 14 then passes (the hash matches the corrupt file), and every downstream composition fails permanently.
- *Drive reordering:* on Google Drive FUSE, `os.replace` is atomic locally, but uploads are asynchronous and may arrive out of order. After a runtime kill, `resume.pkl` can refer to a `step{k}.npz` that never reached Drive. The resume then fails loudly, but on every retry.
- *Required change:*
  - (a) Write all of these files via tmp + `fsync` + `os.replace`.
  - (b) Keep `resume.prev.pkl`. On resume, if the referenced snapshot is missing or its hash does not match, fall back to the previous consistent state and log the fallback.
  - (c) In the composer, write all three files to a temp directory and rename the directory last.
  - (d) In `Runner.run`, when stopping for the budget or an interrupt, call `google.colab.drive.flush_and_unmount()` or an equivalent sync, or at least document the delay before Drive is consistent.

**M4. The confirmation run is gated only on the freeze, not on the other preflight checks.**
`pipeline.run` (`pipeline.py:76-83`) and `Runner.run` check only `verify_freeze`. Confirmation can therefore start with:
- wrong or unpinned dependencies: if pip fails in notebook cell 4, the error is printed and execution continues with Colab's preinstalled jax;
- Python < 3.12: I confirmed that `numpy==2.5.3` has no 3.11 wheel and does have a 3.12 wheel;
- integrity check 03 failing;
- a failing graph audit;
- failing planted tests;
- no benchmark.
*Required change:*
- `Runner.run`, for every preset except smoke, must refuse unless `preflight/PREFLIGHT.json` shows: freeze ok, `graph_audit_all_pass`, planted tests rc 0, integrity checks 02 and 03 PASS, `sys.version_info >= (3, 12)`, a matching `RUN_IDENTITY` (B2), and the GPU replay test passing (M5).
- Notebook cell 4 must `raise` when pip fails.
- Notebook cell 2 must assert the Python version.

**M5. GPU determinism and GPU resume are asserted but never tested, and check 15 does not look at the confirmation runs.**
The only resume and bitwise tests run on CPU. Check 15 (`integrity.py:78-80`) passes on the smoke report alone and only *lists* the resumed confirmation runs. Mandate §15 asks that the GPU path be tested, and §7.4 asks for a verified continuation procedure.
*Required change:* add a preflight step on the actual GPU, using the dev parent uP160 and a reserved data order. It should run two identical 300-step harness runs, plus one run killed with `FX_DIE_AFTER` and resumed, and compare every `step*.npz` hash. Store the result as `preflight/gpu_replay.json`. Check 15 must FAIL unless `gpu_replay.json` passes for the run root's recorded stack, **and** every resumed confirmation run's `resume_log` shows the same stack as its `run_config` (from B2).

**M6. Completion and integrity reporting can overstate.**
- (a) `report.generate` decides `complete` from *registered* jobs only (`cor_research/reporting/report.py:29`). With notebook 01 (`tracks='A'`), a `phases=` filter, or a stop before later phases were planned, it says "All registered jobs are done" although Track B jobs were never registered. The Track B scorer then reports `INSUFFICIENT_UNITS` as if it were a classification.
- (b) The module docstring of `integrity.py` (lines 1-3) says FAILs are "printed at the top of every RESULTS file via the manifest". In fact `pipeline.report` runs `RP.generate` **before** `IN.run` (`pipeline.py:100`), and the generator never reads the integrity results.
- (c) When the run is incomplete, the verdicts still carry `[FROZEN-CONFIRMED]` / `[FROZEN-FALSIFIED]` labels (`report.py:14, 51`), with only an INCOMPLETE banner.
- *Required change:*
  - Compute `complete` against `plan(preset, tracks="ABC")`, with no phase filter, run on the current root, plus a check that every eligible unit has every arm.
  - Run the integrity checks first and print any FAIL or NOT_YET in every file's banner.
  - When the run is incomplete or any integrity check FAILs, replace the frozen labels with `[INTERIM — NOT A FROZEN VERDICT]`.

**M7. Scorer provenance hash sets are incomplete.**
- *Track A* (`cor_research/scoring/measure.py:10-14`): the measure code hashes omit `cor_research/models/load.py` and `cor_research/env.py`, both on the measurement path. `parent_sha256` is recorded but never compared with `CHECKPOINTS.json` or `lineage.json`. `track_a.load_inputs` (`track_a.py:44-56`) does not check that the `meas:` job is `done` or that the measurement file's hash equals its registry output hash, so an unregistered measurement file is accepted as long as the checkpoint hash matches.
- *Track B:* `trackb_code_sha256` covers only `trackb_run.py` (`scripts/freeze.py:18`; `trackb_run.py:73`), not `functional.py`, `generators/t4.py`, `models/load.py` or `measure.py`. The matched-precision snapshot is usually not a registered output, so its hash check falls back to hashing the file on disk (`track_b.py:27-29`), which only checks the file against itself.
- *Required change:*
  - Hash every module on each scorer's import path, as the harness already does with `_deps`.
  - Compare `parent_sha256`.
  - Require registry `done` and a matching hash for each measurement and Track B record.
  - In `Runner._run` (`runner.py:255-256`), when a training job is done, hash **every** `step*.npz` and `opt_step*.pkl` in its directory into `output_hashes`.

**M8. The pilot preset runs on confirmation units and reveals their outcomes.**
- *What overlaps:* `PRESETS["pilot"]` (`cor_research/protocol.py:136-138`) uses confirmation parent `nP3001`, screening seeds 20–23, and `CONF_D[:1] = 2001`. I verified that the pilot tags, and the job IDs that contain them, are identical to the standard preset's tags (for example, `nP3001_fx_L20_W20_D2001_br100_E2001_fullB` in both).
- *What the user sees:* the pilot is the notebook default. It therefore shows the user confirmation-pair outcomes for HA1, HA2 and HA8 (classes, phi, Track B `classification_if_not_pilot`) before the standard run, while `freeze.py --force` remains available.
- *Consequences:* this is a path to unblinding (mandate §9: "Do not adjust thresholds after examining confirmation outcomes"). If the two presets share a run root, the standard run would also silently adopt the pilot's jobs, and its eligibility recomputation would raise a confusing error, because pilot screening (4 seeds) differs from standard screening (8 seeds).
- *Required change:* give the pilot reserved, non-confirmation identities, for example parent `nP3901` and data order `D 3901`, recorded in the protocol as pilot-only. Add these to the dev/pilot exclusion set used by integrity check 04. Alternatively, the pilot must run after the freeze, with its scores blinded (only technical fields written). Add a `RUN_IDENTITY` preset check (see B2).

**M9. Several integrity checks check nothing or always pass.**
- *Checks that pass without checking anything:*
  - 03: see B1.
  - 04 (`integrity.py:39-44`): passes with an empty registry. It does not check confirmation streams against the dev streams (91_900_000 and 20_000_001, per `dev/DEV_NOTES_XX.md:3`), pilot or benchmark identities, or unit seeds.
- *Checks that cannot fail by construction:*
  - 13 (`:75`): compares protocol constants with literals.
  - 18 (`:88-90`): uses the same `PL.design()` that the scorer enumerates, so it cannot fail.
  - 17, second clause (`:87`).
- *Checks that are too weak:*
  - 16 (`:81-83`): re-runs the same code on the same inputs, for Track A at step 4000 only.
  - 09 (`:59-61`): checks the optimizer state only for fullB, not for fullI, paramI_optB or paramB_optI.
  - 10: checks only that a seed exists and `FX_LBL_P == "0.5"`.
  - 19: Track A only.
- *Required change:*
  - 04: return NOT_YET when the registry is empty, and add the stream, seed and pilot comparisons.
  - 18: count pairs independently from `eligibility.json × CONF_D × arms`, and require one registered `cont:` and `meas:` job for each.
  - 16: also re-score Track B and step 1211 in a fresh subprocess, using the frozen code hashes.
  - 09: verify the moment source for every transplant arm against `opt` in `ARMS`.
  - 13: verify the competence and class labels in `eligibility.json` against the recorded numbers.
  - 19: cover Track B and Track C.

**M10. The planted tests miss refusal paths and the components that decide outcomes.**
- `B_wrong_hash_refused` (`tests/test_scorers_planted.py:100`) passes a ready-made `problems` list to the scorer, so it can never fail. `track_b.load_units` is never run.
- There are no planted tests for:
  - Track B code-hash refusal;
  - `plan.eligibility` (lowest qualifying seed, all seeds required, refusal to change a written record);
  - `Runner.register` (unregistered outputs become `blocked`; a changed definition raises);
  - integrity check 14 or 05 (a tampered checkpoint or protocol must FAIL).
- *Required change:* add these tests, and run them in preflight as part of the M4 gate.

**M11. Storage growth on Drive is not checked before the run.**
- *Estimates* from `budget.counts` × 0.613 MB per snapshot (measured from the `dev/` snapshots): about 15.8 GB for standard, 39.5 GB for extended and 105 GB for high.
- *Not included in those estimates:*
  - `resume.pkl`, about 1.2 MB per run (`dev/runs/ft/*/resume.pkl`); the harness never deletes it after completion;
  - window `opt_step100.pkl` files.
- *Context:* free Drive is 15 GB and shared with other services. When Drive fills during the run, the jobs that follow fail.
- *Load on Drive:* `Runner.run` calls `load_all()` over every registry JSON (about 3,000 for standard) plus a full `plan()` / `eligibility()` rescan every 2 s (`runner.py:277, 295`), all over FUSE.
- *Required change:*
  - In preflight, compare free space with the preset estimate and refuse below 1.3× (`pipeline.py:56-73`).
  - Include resume and optimizer files in `budget.counts`.
  - Delete `resume.pkl` once `final.json` or `stop_done.json` is written atomically, keeping its hash in `run_config` or the registry.
  - Cache the registry in memory, and rescan no more often than about every 30 s.

### MINOR

- **m1.** Track B silently skips a unit in HB_c when its decision `.npz` is missing (`track_b.py:41, 72`). Count and report it as MISSING.
- **m2.** The rule that `eligibility.json` is written once is enforced only in code (`plan.py:85-88`). Also append each parent's record hash to an append-only `eligibility_log.jsonl` and to the registry.
- **m3.** The interrupted attempt loses its start time, log, device and pid (`runner.py:307`). Append the attempt record at launch and finalise it at the end.
- **m4.** The window-B tag has no base (`plan.py:30`). Two units with the same parent and seed but different bases (possible through `forced_units` or `max_units`) would share a directory. At plan time, assert that no two jobs share an output directory.
- **m5.** The `math:trackC` job is exempt from the unregistered-output rule (`runner.py:229`), and `pipeline.track_c()` writes the same file outside the registry (`pipeline.py:47-53`). Run Track C only through the registry, or give the notebook's output its own path.
- **m6.** The smoke preset uses confirmation data order D2001 (with dev parent uP160). This does no harm, but give it a reserved data order so that smoke output never shares a name with confirmation data.
- **m7.** `benchmark.json` is never re-measured when the GPU changes (`pipeline.py:65`), so concurrency chosen on an A100 is reused on a T4. This is subsumed by B2.
- **m8.** `FREEZE.json` does not cover `requirements*.txt`, `tests/*.py` or the vendored `.npz` files (`scripts/freeze.py:9-13`); the `.npz` files rely only on the missing `CHECKPOINTS.json`. Add the first two to the freeze. Nothing the user must edit is frozen: the notebooks are outside the freeze and only the config cell needs editing, which is correct.
- **m9.** No tool resolves `blocked` jobs. Document the manual procedure and require a recorded justification.
- **m10.** When `resume.pkl` is missing the harness restarts from step 0 without logging it (`harness_xx.py:254-275`). Log every start in `resume_log.json`.
- **m11.** Notebook `PACKAGE_ZIP` uses the mandated ZIP name, which is fine. Notebook 01 (`tracks='A'`) is the trigger for M6(a).

---

## Answers to the specific questions

| Question | Finding |
|---|---|
| Exact resume after a disconnect during training | Exact on CPU (tested). On GPU it is untested (M5) and the stack is unbound (B2). Out-of-order Drive sync can leave the job stuck (M3). |
| Resume during composition | Recomposition is deterministic. A half-written `provenance.json` causes permanent failure (M3). |
| Resume during measurement | Output is atomic and refuses to be overwritten. OK. |
| Registry records left as 'running' | Reset without a liveness check (B3). |
| Runner lock and heartbeat | Non-exclusive, removed on interrupt, too weak across Drive (B3). |
| Drive FUSE and `os.replace` | Atomic locally; durability and ordering on Drive are not guaranteed (M3). |
| Can a restart silently start a new random trajectory? | Not from new seeds (all seeds are deterministic). It can, however, silently continue on a different numeric stack (B2), resume from foreign unregistered state (M2), or restart from 0 without logging it (m10). |
| Silent replacement of failed runs | None. Failed jobs stay failed; retries append attempts (good). |
| Adoption of unregistered outputs | Partly prevented. Gaps: partial directories (M2), eligibility (M1), measurement records (M7), Track C (m5). |
| Freeze verified before confirmation | Yes, at `Runner.run`, but `FREEZE.json` is missing (B1), nothing binds the run root to it (B2), and the other preflight checks are not gated (M4). |
| Files that affect results but are not frozen | `requirements*.txt` and the vendored `.npz` files (m8, B1). Nothing that the user must edit is frozen. |
| Determinism claims | Honestly worded ("same GPU model and stack only; not bitwise equal to CPU"), but neither tested on GPU nor enforced (B2, M5). |
| Dependency pins and Python version | The pins exist on PyPI; numpy 2.5.3 has no 3.11 wheel (confirmed). The Python version is never checked, a pip failure is not fatal, and the CUDA wheels are unpinned (M4, B2). |
| The 20 integrity checks | 03 currently passes with nothing checked. 18 and 13 cannot fail by construction. 15 does not examine the confirmation runs. 04 passes with an empty registry (M9, B1, M5). |
| Scorer fail-closed provenance | Track A is good, but its hash set is incomplete. Track B code hash and snapshot hashes are weak (M7). |
| Can development and confirmation data mix? | The streams, data orders and dev parent are separated. The pilot uses confirmation units (M8). The smoke uses D2001 on the dev parent (m6). |
| Planted tests | Track A is good. Track B wrong-hash test cannot fail. Eligibility, runner and integrity checks are untested (M10). |
| Could the smoke preset be mistaken for evidence? | Unlikely: forced TECHNICAL_ONLY, dev parent, a run root per preset, and banners. |
| Report refuses completion when jobs are incomplete | Only over registered jobs; integrity FAILs are not shown (M6). |
| Job-ID and tag collisions | No collisions between arms or bases in the scientific presets. Pilot and standard are identical by construction (M8). A latent B-window collision exists (m4). Two presets in one root: fail-closed, but only by a crash (M8, B2). |
| Storage on Drive | Standard alone exceeds 15 GB; no free-space check (M11). |
| Notebook edits beyond the config cell | None required. Pip failure is not fatal (M4). The "safe to interrupt" claim is false (B3). |

---

## Per-component decision table

| Component | Decision | Conditions |
|---|---|---|
| `protocols/` + `scripts/freeze.py` | REJECT (until B1) | Create the freeze after the fixes and record its hash in the review; add requirements and tests (m8). |
| `manifests/` (absent) | REJECT | B1. |
| `cor_research/env.py` (device, flags) | APPROVE WITH REQUIRED CHANGES | B2 (d): precision and identity. |
| `cor_research/pipeline.py` | APPROVE WITH REQUIRED CHANGES | B2 (a, b), M4, M6 (b), M11. |
| `jobs/plan.py` | APPROVE WITH REQUIRED CHANGES | M1, m4; pilot identities (M8). |
| `jobs/runner.py` (registry, lock, resume) | REJECT (until B3) | B3, M2, M7 (output hashes), m3, M11 (polling). |
| `jobs/benchmark.py`, `jobs/budget.py` | APPROVE WITH REQUIRED CHANGES | M11 storage terms; re-benchmark on a stack change (m7). |
| `training/pretrain.py` | APPROVE | Atomic and exact on CPU. Bind the stack (B2). |
| `harness_xx.py` (resume, snapshots, tag rule) | APPROVE WITH REQUIRED CHANGES | M2, M3, B2 (c), B3 (c), m10. The tag rule and RNG restoration are sound. |
| `compose_xx.py` | APPROVE WITH REQUIRED CHANGES | M3 (c). The refusal to overwrite is good. |
| `scoring/measure.py` | APPROVE WITH REQUIRED CHANGES | M7 (hash set, parent hash). |
| `scoring/track_a.py` | APPROVE WITH REQUIRED CHANGES | M7 (measurement-job status and hash). Logic and evaluability are sound. |
| `scoring/track_b.py`, `trackb_run.py` | APPROVE WITH REQUIRED CHANGES | M7, m1. |
| `verification/integrity.py` | REJECT (until M9, B1) | Checks 03, 04, 13, 15, 16, 18 as specified. |
| `reporting/report.py` | APPROVE WITH REQUIRED CHANGES | M6. |
| `scripts/smoke_pipeline.py` | APPROVE WITH REQUIRED CHANGES | Write its report where check 15 reads it; publish the failing earlier smoke run; give the smoke its own data order (m6). |
| `scripts/make_review_bundle.py` | APPROVE | Does not claim that any review took place. |
| `scripts/make_notebooks.py` / notebooks | APPROVE WITH REQUIRED CHANGES | Make the pip failure fatal and check the Python version (M4); remove or fix "safe to interrupt" (B3). Only the config cell needs editing. |
| `tests/` | APPROVE WITH REQUIRED CHANGES | M10; the missing `test_tags.py` / `test_track_c.py` (B1). |
| `requirements*.txt` | APPROVE WITH REQUIRED CHANGES | Pin the `nvidia-*` wheels / `environment_lock.json` (B2 (e)); enforce Python ≥ 3.12 (M4). |
| Presets (smoke / pilot / standard / extended / high) | smoke APPROVE; pilot REJECT (M8); others APPROVE WITH REQUIRED CHANGES | Storage (M11). |
| Track C execution path | APPROVE WITH REQUIRED CHANGES | m5. |

*End of review C (v1). Written by an AI agent reviewer.*
