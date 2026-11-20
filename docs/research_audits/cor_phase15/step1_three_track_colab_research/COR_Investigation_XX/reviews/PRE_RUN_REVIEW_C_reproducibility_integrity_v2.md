# Pre-run review C — reproducibility and experimental integrity (v2, second round)

**Reviewer:** an AI agent (Claude) acting as an independent pre-experiment reviewer for reproducibility and integrity. This is an automated agent review, not a human peer review. I did not build the package, and I did not write the round-1 review (`PRE_RUN_REVIEW_C_reproducibility_integrity_v1.md`). I read that review and judged it against the v2 code.

**Date:** 2026-11-20 (UTC).

**Scope:** the v2 package state, i.e. the round-1 responses in `reviews/ADJUDICATION_v1.md` and the design in `protocols/DESIGN_XX.md` (v2), checked against the actual code.

**What I read:**
- `cor_research/jobs/{runner,plan,budget}.py`
- `cor_research/{env,pipeline,protocol}.py`
- `cor_research/scoring/{measure,track_a,track_b,trackb_run,functional}.py`
- `cor_research/verification/integrity.py`
- `cor_research/reporting/report.py`
- `cor_research/training/pretrain.py`
- `harness_xx.py`, `compose_xx.py`
- `scripts/{freeze,build_package,resume_test,smoke_pipeline}.py`
- all five notebooks
- `tests/*`
- `COLAB_SETUP.md`, the requirements files and `environment_lock.json`

**Checks I ran:** only short CPU checks, outside the package, in scratch directories:
- (i) I imported `trackb_run` and `measure` and compared their code-hash key sets with what `scripts/freeze.py` stores;
- (ii) I listed the modules actually imported on the measurement path;
- (iii) I ran `budget.counts` for standard and extended to size storage.

I inspected the running smoke at `/tmp/smk2` read-only. I modified no file except this review.

**What I could not do:** I had no GPU, so every GPU or Colab statement below is a code-reading inference and is marked as such. `FREEZE.json` and `manifests/` intentionally do not exist yet, so I reviewed the freeze procedure, not a freeze.

---

## Verdict: **REJECT in the current v2 state. APPROVE WITH REQUIRED CHANGES once N1–N2 (blockers) and N3–N9 (major) are fixed and the fixes are verified (§4).**

The v2 corrections are substantial and mostly well made:
- the in-memory, write-through registry;
- process-group termination on interrupt;
- liveness checks using boot_id and pid;
- blocking of unregistered pre-existing output directories;
- output validation before `done`;
- per-snapshot hashes, kept also for pruned files;
- atomic harness and composer writes, with the completion marker written last;
- a provenance-checked eligibility decision;
- pilot material separated from confirmation material;
- a run identity checked at setup and before every run;
- full-fp32 matmul precision on GPU;
- a preflight `pass` gate;
- a runtime resume and determinism test;
- report completeness computed against the plan, with an integrity banner.

Two new defects would still make confirmation results unscorable or unverifiable:

1. **N1.** The frozen Track B code hash that `scripts/freeze.py` writes can never equal the hash that Track B records carry. Every Track B score on a scientific preset would be `PROVENANCE_FAIL`, and the smoke cannot detect this, because it bypasses the frozen hashes. I confirmed this numerically.
2. **N2.** `ALLOW_IDENTITY_CHANGE=True` is the only way forward when Colab hands out a different GPU type or driver, which is routine. It also allows a different *freeze* (code and protocol) to continue the same run root. After such a change, nothing records which trajectory segment ran on which stack. The cached preflight results from the old stack keep the gate open, and the reports, contrary to the error message, do not label anything as mixed-identity.

---

## 1. Round-1 items: status in the v2 code

R = resolved, P = partially resolved, N = not resolved.

| id | round-1 requirement (short) | status | evidence (v2 code) and what remains |
|---|---|---|---|
| **B1** | freeze, manifests, missing tests; check 03 vacuous | **P** (pending by design) | `tests/test_tags.py` and `tests/test_track_c.py` now exist. Check 03 FAILs on an empty `CHECKPOINTS.json` (`integrity.py:39`). The freeze and manifests are deferred until after this round (see N9). A fresh parent without `lineage.json` is still skipped silently (`integrity.py:38`, `if L:`). This is low risk, because `pretrain.py:48-51` writes the lineage before `pre.json`, but it should still FAIL. |
| **B2** | run identity; refuse CPU; harness per-start stack log; fp32 matmul; pin `nvidia-*`; GPU name per attempt | **P** | Done: `RUN_IDENTITY.json` with refusal (`pipeline.py:130-141`), and `JAX_DEFAULT_MATMUL_PRECISION=highest` (`env.py:39`). Not done: (b) CPU is still only a warning for scientific presets (`pipeline.py:125`); (c) `resume_log` records only `{resumed_at}` (`harness_xx.py:277`), with no devices, flags or code; (e) the `nvidia-*` wheels are neither pinned nor in `identity()` (`env.py:86`, `requirements-gpu.txt`); (f) attempts record only `COR_DEVICE` (`runner.py:108`). The escape hatch also reopens B2 (N2), and the stack can still drift undetected (N6). |
| **B3** | pid/boot_id liveness; process groups and SIGTERM on interrupt; per-run harness lock; exclusive runner lock | **P** (R for the single-VM notebook flow) | Done: (a) `runner.py:65-68, 108-113, 137-146`; (b) `runner.py:111, 180-186`. Not done: (c) the per-run `O_EXCL` lock in the harness; (d) the runner lock is still a read-then-write file with a 120 s staleness window, and it is written only after the first `register(planner())` (`runner.py:152-178`). See N10 and N11 for new side effects. |
| **M1** | eligibility provenance | **R** (one gap) | `plan._verified_screen` (`plan.py:66-75`) checks job status, the checkpoint hash against the `scr:` output hash, the measurement hash against the `mscr:` hash, the stream and the frozen code, and raises on mismatch. `parent_valid` (`plan.py:50-53`) still reads `pre.json` without a hash check against `lineage.json` or `CHECKPOINTS.json`. That is minor, because check 03 covers it, but check 03 does not gate (N7). FAILED_PRETRAINING introduces a new defect: N3. |
| **M2** | stray partial outputs adopted; harness resumes foreign state | **P** | `runner.register` blocks any pre-existing file in an unregistered training or compose directory (`runner.py:88-92`), and there is a planted test. Not done: the harness still neither records the `step{k}.npz` hash in `resume.pkl` nor verifies `run_config.json` against the invocation on resume (`harness_xx.py:273-277`). This is acceptable now that the registry blocks foreign directories; I list it as a recommendation only. |
| **M3** | atomic writes; `resume.prev` fallback; composer temp dir; Drive flush | **P** | Atomic now: `step*.npz`, `opt_step*.pkl`, `resume.pkl`, `final.json` (last), `curve`, `run_config`, `window_loss` and `stop_done` (`harness_xx.py:80-82, 300, 307-316`); composer files with provenance last (`compose_xx.py:65-71`); Track B npz and json (`trackb_run.py:56-58`). Outputs are validated before `done` (`runner.py:46-54, 119`). Remaining: `resume_log.json` is still non-atomic (N12). The write order `resume.pkl` → `opt_step` creates a permanent-failure window (N4). There is no `resume.prev` fallback and no Drive flush on stop. |
| **M4** | gate confirmation on the full preflight | **P** | `pipeline.run` refuses without `PREFLIGHT.json` `pass` (`pipeline.py:204-207`). The notebook raises on a pip failure. Python ≥ 3.12 is part of check 02. Not done: check 03 (vendored parents) and check 01 (the manifest, which is the only binding of `tests/` and `requirements*`) are printed but **not** in `pass` (`pipeline.py:193-194`). `pass` is not bound to the identity, freeze or preset that produced it. The resume test and benchmark are cached forever per run root (`pipeline.py:176-185`). See N7. |
| **M5** | GPU determinism and resume test; check 15 must cover confirmation resumes | **P** | `scripts/resume_test.py` runs the real harness on the runtime: A, B killed and resumed, and C. Good. It runs once per run root and never again (cached), so a new VM, driver or GPU after an allowed change is never retested (N7). Check 15 (`integrity.py:89-91`) still passes on the resume test alone and only lists confirmation resumes, which carry no stack record (B2(c)). |
| **M6** | completeness via plan; integrity banner; no frozen labels while incomplete | **P** (mostly R) | Completeness = no non-done registered job, no planned-but-unregistered job, and all three tracks present (`report.py:30-34`). Two passes put integrity FAILs in the banner (`pipeline.py:226`, `report.py:35, 40`). Remaining: the `[FROZEN-CONFIRMED]` / `[FROZEN-FALSIFIED]` labels still appear next to an INCOMPLETE/PROVISIONAL banner (`report.py:14, 51`), and identity changes are never reported (part of N2). |
| **M7** | complete scorer hash sets; parent hash; registry status and hash; all snapshot hashes | **P** | Done: `env.py` and `models/load.py` added to the measurement hash (`measure.py:15-18`); parent hash compared (`track_a.py:48-50`); measurement-job status and hash checked (`track_a.py:45`); `snapshot_hashes` for every step file (`runner.py:124-126`); Track B checkpoint hashes checked against the registry snapshot hashes (`track_b.py:26-28`). Not done: **the frozen Track B hash set is wrong (N1)**; the Track B decision `.npz` files that feed the primary HB_c are unhashed (N5); composed (`tp`) checkpoints are not checked against the registry (`track_a.py:60`, `train_jid=None`); the measurement hash omits `nnlib.py` and about ten other legacy modules that are actually imported (N18). |
| **M8** | pilot on confirmation material | **R** | Pilot = parent `nP3901` and data order 2901 (`protocol.py:13, 175`). Check 04 flags pilot parents in a non-pilot run (`integrity.py:46`). `TECHNICAL_ONLY` is forced for the pilot. |
| **M9** | vacuous or weak integrity checks | **P** | Fixed: 03 (empty manifest), 04 (NOT_YET when empty, D-overlap, pilot, stream uniqueness), 12 (planned-but-unregistered), 18 (registered pairs absent from the scores). Unchanged: 09 checks only fullB moments (`integrity.py:66-70`); 10 is the same weak check; 15 (see M5); 16 re-runs Track A in-process with the same code, with no Track B or step-1211 re-scoring (`:95`); 19 covers Track A only (`:105-108`); 04 still does not compare against the dev streams (91_900_000, 20_000_001). |
| **M10** | planted tests for refusal paths | **P** | Added: eligibility immutability (two cases), runner failed-kept / definition-change / unregistered-blocked, check 14, and a Track B checkpoint-hash refusal through the real loader. Missing: Track B **code-hash** acceptance against the *frozen* hash (it would have caught N1); `_verified_screen` refusal; lowest qualifying seed and all-seeds-required; check 05 on a tampered protocol; the interrupt/alive path; FAILED_PRETRAINING across retries (it would have caught N3). |
| **M11** | storage check; resume/opt in the estimate; delete `resume.pkl`; no Drive rescans | **P** | The in-memory registry with replanning only when a job finishes is resolved (`runner.py:74-76, 160-162`). A preflight storage check exists (`pipeline.py:187-188`). Remaining: the estimate omits `resume.pkl`, the `opt_step` pickles and composed states, and ignores the Drive trash. It passes vacuously when the benchmark has no `snapshot_mb`. `resume.pkl` is never deleted. See N8. |
| m1 | Track B silently skips a missing decision npz | **P** | `np.load` now raises (fail-closed). But a battery level whose snapshot is missing is written as `missing=True` (`trackb_run.py:54`) and then scored as "censored_or_missing" (`track_b.py:27, 31`): see N16. |
| m2 | append-only eligibility log | **N** | No `eligibility_log.jsonl`. |
| m3 | record the attempt at launch | **R** | An in-progress attempt with start, pid, boot_id and log is saved at launch (`runner.py:108-113`). |
| m4 | output-directory collisions | **R** (by blocking) | No collision is possible in the scientific presets: the E and M seeds differ by the eligibility construction. A collision would be `blocked`. No explicit assertion was added. |
| m5 | Track C written outside the registry | **N** | `pipeline.track_c()` still writes `track_c/track_c_math.json`, which the registry job `math:trackC` later overwrites. `math` is exempt from the pre-existing-output rule (`runner.py:88`). |
| m6 | smoke uses D2001 | **N** | `protocol.py:170` still has `D_list=[2001]`. Harmless, but unchanged. |
| m7 | re-benchmark on a GPU change | **N** | `benchmark.json` is cached per run root (`pipeline.py:183`). See N7. |
| m8 | freeze `tests/` and `requirements*` | **N** | `freeze.py:9-13` still covers only protocols, configurations, `cor_research/**/*.py` and `scripts/*.py`. `tests/` gates the preflight but is bound only by the non-gating check 01. See N7 and N9. |
| m9 | procedure for blocked jobs | **N** | Not documented. |
| m10 | log every start, including restart from 0 | **N** | `harness_xx.py` logs only resumes and discarded resumes. |
| m11 | — | n/a | — |

---

## 2. New objections (introduced by the v2 corrections, or not visible in v1)

### BLOCKERS

**N1. The frozen Track B code hash can never match, so all of Track B fails provenance in every scientific run.**
- *What the two sides record:*
  - `scripts/freeze.py:18` stores `trackb_code_sha256 = {"trackb_run.py": …}`.
  - Every battery record carries `trackb_run.code_hashes()` (`trackb_run.py:58, 61`). That is the full measurement set, plus `functional.py`, `t4.py` and `trackb_run.py`: 14 keys.
- *Why it always fails:* `track_b.load_units` compares the two dicts for equality (`track_b.py:25`).
- *My check:* I imported both on CPU. The key sets are `['env.py', 'ftlib.py', 'functional.py', 'genlib.py', 'maskfwd.py', 'metrics.py', 'models/load.py', 'native.py', 'relpairs.py', 'scoring/measure.py', 't4.py', 'trackb_run.py', 'xix_score.py', 'xviiim.py']` against `['trackb_run.py']`, and `equal: False`.
- *Why it is a blocker:*
  - The smoke cannot see the defect, because it passes `frozen_code=TBR.code_hashes()`, taken from the current code (`smoke_pipeline.py:107`).
  - It would surface only after the GPU-hours are spent.
  - Fixing it then means re-freezing, which changes `freeze_sha256`, so the run identity refuses the run root (or forces N2's escape hatch).
- The adjudication claim (C-M7: "the Track B code hash covers trackb_run + functional + t4 + measurement code") is true of the records and false of the freeze.
- *Required change:*
  - (a) In `scripts/freeze.py:18`, write `trackb_code_sha256=TBR.code_hashes()`, importing `cor_research.scoring.trackb_run as TBR`, i.e. the very function the records use. Do the same for any future hash set: the freeze must call the producer's own function, never re-list files.
  - (b) Add a planted test to `tests/test_scorers_planted.py`: a battery record with `code_sha256=TBR.code_hashes()` must be accepted against `FREEZE.json["trackb_code_sha256"]`, and likewise for a measurement record against `measure_code_sha256`. It must run against the real `FREEZE.json` once that exists.
  - (c) Run the post-freeze clean-extraction smoke with the hashes read from `FREEZE.json` (see N9(c)), not with the current-code hashes.

**N2. The identity escape hatch also lets the code change mid-run, and leaves trajectories without stack provenance.**
- *What the escape hatch allows:*
  - `check_identity(allow_change=True)` (`pipeline.py:139-141`) accepts *any* difference, including `freeze_sha256`, i.e. a different frozen protocol or code continuing the same confirmation run root. Round-1 B2 named this case explicitly ("a new ZIP … can change the code half-way through a run root").
  - It then overwrites `identity` with the new stack.
- *Why the user will need it:* Colab routinely changes the GPU type and the driver between sessions. Driver version and Python patch version are part of the identity, so even a same-model VM on a newer image is refused. The user is therefore pushed towards `ALLOW_IDENTITY_CHANGE=True`.
- *What is then lost or stale:*
  - Nothing records which segment of which trajectory ran under which identity. Attempts record only `COR_DEVICE` (`runner.py:108`), and the harness `resume_log` records only the step (`harness_xx.py:277`).
  - `PREFLIGHT.json`, `RESUME_TEST.json` and `benchmark.json` from the *old* stack stay valid, because they are cached and not bound to the identity (`pipeline.py:176-185, 204-206`). The determinism and resume claims are therefore unverified for the new stack, and the concurrency comes from a different GPU.
  - The refusal message promises that "results are then labelled as mixed-identity", but `report.py` never reads `RUN_IDENTITY.json`.
- *Consequence:* the effect under study is a bistable selection during training, so a mid-trajectory change of numeric stack is a potential confound. Without per-segment records it can be neither excluded nor tested in a sensitivity analysis.
- *Required change:*
  - (a) In `pipeline.py:138-139`, split the identity into **immutable** fields and **changeable** fields:
    - immutable: `freeze_sha256` and the preset; any change raises, with no override;
    - changeable with `ALLOW_IDENTITY_CHANGE`: device, GPU model, packages, flags, precision;
    - recorded but not refused: driver version, Python patch version, which are only logged in each attempt.
  - (b) `identity_sha` = the sha256 of the canonical identity JSON. Store it, with the GPU name and the driver, in every attempt (`runner.py:108`). In `harness_xx.py:270-277`, append to `resume_log.json` on **every** start and resume: `jax.devices()`, `jax.default_backend()`, `XLA_FLAGS`, `JAX_DEFAULT_MATMUL_PRECISION`, the jax/jaxlib/plugin versions and `_deps`.
  - (c) On any accepted change, `check_identity` must invalidate the cached preflight: move `preflight/PREFLIGHT.json`, `RESUME_TEST.json` and `benchmark.json` into `preflight/superseded/<utc>/`. `run()` must then require a new passing preflight whose recorded `identity_sha` equals the current one (store `identity_sha` in `PREFLIGHT.json`, `pipeline.py:195`, and compare it at `pipeline.py:205-207`).
  - (d) Scorers: a pair is `MIXED_STACK` if any of its training jobs (its window, composition sources and continuation) ran under more than one `identity_sha`. Report the headline verdicts both with and without `MIXED_STACK` pairs. `report.py:36-43` must print a banner listing every identity change from `RUN_IDENTITY.json["changes"]`.

### MAJOR

**N3. FAILED_PRETRAINING makes the planner raise on a retry, and it freezes a transient failure for good.**
- *The defect:*
  - `plan.py:90` writes `dict(status="FAILED_PRETRAINING", attempts=len(pj["attempts"]), units={})` as soon as the `pre:` job is `failed` once.
  - Eligibility records are immutable (`plan.py:107-108`).
  - With `RETRY_FAILED=True` (offered in the configuration cell), a second failed attempt changes `attempts`, and a successful retry changes the status to SCREENED. Either way the planner raises "eligibility … changed after it was recorded: integrity failure".
- *Consequences:*
  - The planner runs inside `Runner.run` (`runner.py:160`), so the whole runner aborts at every start from then on, and the run root cannot be continued without hand-editing.
  - Without a retry, a single transient failure (Drive I/O, GPU OOM at too high a concurrency) excludes a fresh parent permanently.
- *Required change:*
  - In `plan.py:88-91`, record FAILED_PRETRAINING only when the job is `failed` **and** its attempts are exhausted (the runner's limit is `len(attempts) >= 3`, `runner.py:166`; expose it as one constant in `protocol.py`), or when the run is final (report time).
  - Remove `attempts` from the immutable record; keep it in the registry only.
  - Add a planted test: a failure, then a retry, then success, must not raise.

**N4. The write order in the harness makes a window or stability job fail permanently after a kill in a short window.**
- *The defect:*
  - At a snapshot step the harness writes `step{k}.npz`, then `resume.pkl`, and only then `opt_step{k}.pkl` (`harness_xx.py:307-309`).
  - Kill the process between the last two writes for a window job (STOP=100, outputs `step100.npz` and `opt_step100.pkl`). On the next run it resumes at 100, exits 0 through "already at stop" (`:278`), and the runner marks it `failed` because `opt_step100.pkl` is missing (`runner.py:119, 133`).
  - Every retry repeats this, so the job can never complete.
  - The same happens to a stability-arm continuation at `FX_SAVEOPT_AT=4000`: the resume enters an empty loop, writes `final.json`, and `opt_step4000.pkl` is missing.
  - Every pair depending on that window is then permanently MISSING.
- *Required change:*
  - In `harness_xx.py:307-309`, write `opt_step{k}.pkl` **before** `resume.pkl`. The two pickles have identical content, so the cost is nil.
  - Additionally, on "already at stop" or "already complete", regenerate a missing requested `opt_step{k}.pkl` from `resume.pkl` when `resume.pkl["step"] == k`.

**N5. The primary Track B statistic reads unverified files.**
- *The defect:*
  - HB_c and HB_c_dir (primary) are computed from `trackB/<tag>_<level>.npz` (`track_b.py:33`).
  - These files are not declared outputs of the `fbbat:` job (`plan.py:196`), so they have no registry hash.
  - Their hashes are not in the battery JSON (`trackb_run.py:56-57`).
  - `load_units` therefore checks the JSON and the checkpoint, but not the decisions that are actually scored.
- *Required change:*
  - In `trackb_run.py:57`, store `decisions_sha256` for each level in `rec["analyses"][tag]`.
  - In `plan.py:196`, declare both npz files as outputs of `fbbat:`. For a censored level, write a placeholder, so that the output list is stable.
  - In `track_b.level()` / `load_units` (`track_b.py:22-28`), refuse with PROVENANCE_FAIL when the npz hash differs.
  - Also verify `fbsel:` and `fbmatch:`: each job must be done and its output hash must match. Recompute `matched_step` from the selection records (it is cheap and deterministic) and compare it with `trackB/match/<p>.json`.

**N6. The numeric stack can still drift, or fall back to CPU, without detection.**
- *Unpinned CUDA libraries:* `requirements-gpu.txt` pins `jax[cuda12]`, the plugin and pjrt, but the `nvidia-cublas/cudnn/cuda-*` wheels come in with lower bounds. Every new Colab VM reinstalls them, possibly at newer versions. `identity()` (`env.py:86`) does not list them, so cuBLAS/cuDNN changes between sessions pass the identity check. This is round-1 B2(e), still open.
- *Silent CPU fallback:* in GPU mode `env.configure` does not set `JAX_PLATFORMS` (`env.py:37-40`). If the CUDA plugin fails to initialise in a child, JAX falls back to CPU with only a warning. That run then uses multi-threaded Eigen and none of the CPU determinism flags, while the identity, which is computed from `nvidia-smi` in the parent, says `gpu`. (This is code reading; the fallback behaviour should be confirmed on Colab.)
- *Unverified precision:* `JAX_DEFAULT_MATMUL_PRECISION=highest` is set but never verified on the hardware.
- *Required change:*
  - (a) Pin the `nvidia-*` wheels that the resolver installs: record them from one clean Colab install into `environment_lock.json`, and use `pip install -c constraints-gpu.txt`. Add every installed `nvidia-*` distribution to `identity()["packages"]` (iterate `importlib.metadata.distributions()` and match names starting with `nvidia-`).
  - (b) Set `os.environ["JAX_PLATFORMS"] = "cuda"` in the GPU branch of `env.configure` (`env.py:38`), so that a failed GPU is an error, not a fallback. In the harness (`harness_xx.py:283`) and in `measure.py`, assert that `jax.default_backend()` is `"gpu"` when `COR_DEVICE == "gpu"` and `"cpu"` when `COR_DEVICE == "cpu"`.
  - (c) In `scripts/resume_test.py`, add a precision probe: a 512×512 float32 matmul in JAX against float64 numpy. Require a relative error ≤ 1e-6; TF32 gives about 1e-3. Record the result in `RESUME_TEST.json` and require it in `pass`.
  - (d) In `pipeline.py:125`, raise for any non-smoke preset when `dev == "cpu"` (round-1 B2(b)).

**N7. The preflight gate is incomplete and becomes stale.**
- *What `pass` leaves out* (`pipeline.py:193-194`):
  - check 03, the vendored historical parents uP161–165, which are confirmation material;
  - check 01, the package manifest. It is the *only* thing that binds `tests/` (which `pass` itself depends on) and `requirements*.txt`, since neither is in the freeze (m8).
- *Staleness:*
  - `pass` is not bound to the identity, freeze or preset that produced it.
  - The graph audit, the resume test and the benchmark are computed once per run root and reused forever (`pipeline.py:176, 180, 183`). A new VM (new driver or cuBLAS) or an allowed GPU change is never retested.
  - No test checks that a later session reproduces an earlier session's bits, and that is the actual Colab resume scenario.
- *Required change:*
  - (a) Add `res["package"] == "PASS" and res["checkpoints"] == "PASS"` to `pass`.
  - (b) Store `identity_sha`, `freeze_sha256` and `preset` in `PREFLIGHT.json`, and require that they match in `run()` (`pipeline.py:205-207`).
  - (c) Re-run `resume_test.py` at **every** preflight; it is about 4 × 300 steps. Compare its snapshot hashes with the **first** `RESUME_TEST.json` of the run root, kept as `preflight/RESUME_TEST_first.json`. Any difference is a cross-session nondeterminism FAIL, unless the identity has changed (and then N2 applies). Record every session's result in `preflight/resume_history.jsonl`.
  - (d) Re-benchmark when `identity_sha` changes (m7).

**N8. The storage check underestimates by about 40 % or more, ignores the Drive trash, and can pass vacuously.**
- *What I measured:* `budget.counts` gives 11,776 snapshots for standard (about 7.2 GB at 0.613 MB each) and 38,976 for extended (about 23.9 GB).
- *What the estimate leaves out* (`budget.py:30`):
  - `resume.pkl` in every training job (about 1.2 MB each, never deleted);
  - `opt_step*.pkl` of windows and compositions;
  - composed `step100.npz`.

  These add about 2.9 GB for standard and about 7.1 GB for extended, by my estimate from the counts: 1,216 and 3,216 training jobs; 576 and 1,296 compositions.
- *The Drive trash:* on Colab, files deleted from the mounted Drive normally go to the Drive trash and keep counting against the quota until the trash is emptied. This is widely reported and should be verified on Colab. If it holds, the pruning at `runner.py:127-131` frees nothing on Drive, so the 22 extra snapshots of every pruned continuation still count.
- *Vacuous pass:* if the benchmark produced no `snapshot_mb`, `storage_gb` is `None` and the check passes (`budget.py:45`, `pipeline.py:188`).
- *Required change:*
  - (a) Add `resume.pkl`, `opt_step`, composition and Track B decision sizes to `counts` / `estimate`.
  - (b) Count pruned snapshots as fully present, unless the pipeline empties the trash. Document in `COLAB_SETUP.md` §6 that pruned files stay in the Drive trash.
  - (c) Make `ok=False` when `need is None` for a scientific preset.
  - (d) Delete `resume.pkl` after `final.json` or `stop_done.json` (keeping its hash in the registry), as required in round 1 (M11).

**N9. The freeze and package procedure (`scripts/freeze.py`, `scripts/build_package.py`) is sound in outline but is not yet sufficient for an approval.**

The ordering is right: freeze, then manifests (`CHECKPOINTS.json`, `PACKAGE_MANIFEST.json`, which include `FREEZE.json`), then the ZIP. The build refuses to run without a verified freeze, and the run identity binds `freeze_sha256` to the run root. The gaps:
- (i) `freeze_sha256` hashes only `files` (`freeze.py:19`), not `measure_code_sha256` / `trackb_code_sha256`, so those fields can change without changing the approved hash.
- (ii) `tests/`, `requirements*.txt` and `environment_lock.json` are not frozen (m8), although the preflight gate executes `tests/`.
- (iii) No script does the announced clean-extraction test.
- (iv) The `build_package.py` docstring promises "regenerate protocol JSON and notebooks", but the code does not do it. If it did, it would have to happen *before* the freeze. Also, nothing checks that `protocols/*/*.json` equals `make_protocols.py`'s output from `protocol.py`.
- (v) `source_commit` is the HEAD of the enclosing repository (`/home/claude/12345`), whose working tree may be dirty, so it does not identify the content.
- (vi) This review requires code changes, so a freeze made directly after them would freeze code that no reviewer has seen.

*Required change:*
- (a) `freeze_sha256` = the sha256 of the canonical JSON of the whole `FREEZE.json` body except `frozen_at_utc`.
- (b) Add `tests/*.py`, `tests/fixtures/*`, `requirements*.txt` and `environment_lock.json` to `freeze.files()`. Move the test result files out of `tests/` (see N14).
- (c) Add `scripts/verify_zip.py`. It must:
  - extract the ZIP into an empty directory;
  - run `verify_freeze`, then integrity checks 01, 03 and 05;
  - check that the protocol JSON equals the `make_protocols` output (in a temporary directory);
  - run the smoke **with the frozen hashes from `FREEZE.json`**, including Track A and Track B scoring that must not be PROVENANCE_FAIL;
  - write `VERIFY_ZIP.json` with the ZIP sha256.
- (d) Record `git status --porcelain` (dirty or not) next to `source_commit` (`build_package.py:38`).
- (e) Before freezing, have a reviewer verify the N1–N9 changes (a targeted diff check, not a full round). Then record `freeze_sha256` and the ZIP sha256 in `ADJUDICATION_v2.md` and `DESIGN_XX.md` §4.

### MINOR

**N10. After a kernel crash, orphan jobs block the runner, and the message gives the wrong remedy.**
- *What happens:*
  - Children run in their own session (`runner.py:111`), so they survive a kernel crash or a "Restart session". The boot id stays the same.
  - `reset_stale_running` therefore refuses ("wait for it or restart the runtime", `runner.py:142`), but restarting the session does not kill the orphans.
  - pid reuse after a kernel restart can also make `alive()` (`runner.py:65-68`) report a dead job as alive.
- The behaviour is safe (fail-closed), but the user has no remedy.
- *Required change:*
  - In `alive()`, also require that `/proc/<pid>/cmdline` contains the job's harness or script path and that the process start time is ≥ the attempt start.
  - Provide `Runner.terminate_orphans()`, which sends SIGTERM to the recorded process groups that pass that test; call it from the notebook on request.
  - Fix the message to say "terminate them with `PL.terminate_orphans()` or wait".

**N11. The runner lock is still not exclusive.**
- *The defect:*
  - It is a read-then-write file (`runner.py:152-153`), first written only after `register(planner())` (`runner.py:160, 178`), which can take minutes over Drive.
  - Two notebooks on different VMs (the boot ids differ, so `alive()` is false) can both pass the check, and each then resets the other's running records as stale.
- *Required change:*
  - Write the lock before `reset_stale_running`, with an owner UUID.
  - Re-read it before every launch, and stop if the owner differs.
  - Document in `COLAB_SETUP.md` §7 that one run root must never be opened from two Colab sessions at the same time.

**N12. `resume_log.json` is still written with a non-atomic `json.dump`.**
- *Where:* `harness_xx.py:271, 277` (the round-1 M3 list included it).
- *Effect:* a truncated file makes `json.load` raise on every later resume, so the job fails at every retry.
- *Required change:* use `_aj`.

**N13. Every session re-runs all 20 integrity checks, twice, over the whole Drive run root.**
- *Where:* `verify_package` (`pipeline.py:146`) and `preflight` (`pipeline.py:174`) call the full `IN.run`.
- *What each call does:*
  - check 14 hashes every registered output; for standard this is gigabytes over FUSE;
  - check 16 re-scores Track A;
  - check 12 runs the planner, whose `eligibility()` *writes* `eligibility.json` (`plan.py:110`).

  The notebook asks the user to "run all cells again" after each disconnect, so every session pays this cost.
- *Required change:* add `IN.run(preset, only=("01", "02", "03", "05"))` for the package checks used by these two calls.

**N14. The tests write their result files into the package.**
- *Where:* `tests/test_scorers_planted.py:132`, `tests/test_verdict_oc.py:34`, `tests/test_trackb_oc.py:48`.
- *Effect:* each preflight rewrites shipped, manifest-hashed files. Any byte difference (float formatting, a different Python) turns check 01 into FAIL in every RESULTS banner.
- *Required change:* write the results to `$COR_RUN_ROOT/preflight/` (default: a temporary directory), and ship the build-time results as `tests/fixtures/*_build.json`.

**N15. `pipeline.run(..., require_freeze=False)` silently bypasses both the freeze and the preflight gate for scientific presets.**
- *Where:* `pipeline.py:204`.
- *Required change:* raise for any non-smoke preset when `require_freeze=False`, or record `gate_bypassed=True` in every attempt and in the report banner.

**N16. Track B conflates a lost anchor snapshot with censoring.**
- *Where:* `trackb_run.py:54` writes `missing=True`, and `track_b.level` returns "censored_or_missing" (`track_b.py:27`).
- *Why it matters:* anchors are never pruned, so a missing snapshot is lost data, not censoring, and it would distort the censoring-by-class statistic.
- *Required change:* the battery exits non-zero when the snapshot is missing (the job is then `failed`).

**N17. Composed states are not checked against the registry in Track A.**
- *Where:* `track_a.py:60` passes `train_jid=None` for the `mtp:` records.
- *Required change:* pass `f"tp:{key}"` and check `ft/{tp}/step100.npz` against its output hash.

**N18. The measurement hash set misses modules that are actually imported.**
- *What I found:*
  - `measure.MEAS_FILES` (`measure.py:11-12`) lists `step1_cross_family_relational_foundations/src/nnlib.py`, which does not exist there and is silently skipped.
  - The `nnlib.py` actually loaded is `step1_r1_causal_origin_and_selective_rescue/src/nnlib.py`.
  - Also unhashed but imported: `evalrun.py`, `r1lib.py`, `gen_family.py`, `fwd3.py`, `eqv.py`, `cri.py`, `bayes.py`, `ckpt.py`, `nets.py`, `boundary.py` and `cor_research/protocol.py`. I found these by listing `sys.modules` after importing the measurement path on CPU.
- This is mitigated, because all of them are in the freeze and the identity binds the freeze.
- *Required change:* compute `code_hashes()` from every module in `sys.modules` whose file lies under `PKG`, after the imports, as the harness does with `_deps`. A listed file that does not exist must raise.

**N19. The documentation does not tell the user about the identity binding.**
- *Where:* `COLAB_SETUP.md:1-3` says "L4, A100, T4 all work".
- *Required change:* add "Choose one GPU type for the whole run root and keep it. A run root started on one type refuses another (see `ALLOW_IDENTITY_CHANGE` and its consequences)." Also document the procedure for blocked jobs (m9).

**N20. Integrity checks that are still weak (carried over from M9).**
- *Required change:*
  - 09: verify the moment source of every transplant arm against `ARMS[arm]["opt"]` (`integrity.py:66-70`).
  - 15: require that every confirmation `resume_log` entry carries the same `identity_sha` as its `run_config` (after N2(b)).
  - 16: re-score Track B and step 1211 in a fresh subprocess.
  - 19: cover Track B and Track C.
  - Replace the frozen labels with `[INTERIM — NOT A FROZEN VERDICT]` while incomplete (`report.py:14, 51`).

---

## 3. Specific questions from the request

| Question | Finding |
|---|---|
| Interrupt handler | Correct for a single interrupt on one VM: process groups get SIGTERM, threads are joined, records become pending with the attempt kept, and the next run resumes exactly. A second interrupt during the up-to-120 s join, or a kernel crash, leaves orphans that block safely, but the remedy is wrong (N10). |
| In-memory registry vs files written by other processes | Within one runner it is consistent and write-through. The planner's `eligibility()` and the scorers re-read from disk (`load_all`), which is correct. The remaining risk is a second runner on another VM, because the lock is not exclusive (N11). Integrity and report run `plan()`, which can write `eligibility.json` (N13). |
| Pruning vs Track B | Safe. Anchors (fullB, fullI) and stability arms are never pruned (`plan.py:170`). The kept steps of pruned arms (150, 300, 1211, 4000) are exactly the measured steps, and hashes of the pruned files are kept. On Drive, pruning probably frees no quota (N8). |
| Identity strictness on Colab | Too strict where it does not matter (driver, Python patch), too loose where it does (`nvidia-*` wheels, a silent CPU fallback), and the override is too broad (it also covers the freeze), with no per-trajectory record (N2, N6). |
| Preflight on every new run root | The resume test, graph audit and benchmark are cached per run root, so there is no repeated cost, but they also become stale (N7). The planted tests and verdict OC run at every preflight and write into the package (N14). The full integrity run twice per session is the real cost (N13). |
| Freeze procedure | Sound in order, but it needs N1, N9 (a–e) and a verification of the fixes before the freeze is made. |
| Smoke `/tmp/smk2` (read-only) | At my last look (about 07:07 UTC) 39 of 99 registered jobs were done, 2 running, 58 pending, and none failed or blocked. There was no `SMOKE_REPORT.json` yet. Windows, compositions, measurements and continuations ran through the new runner with in-progress attempts carrying pid and boot_id. By construction this smoke cannot detect N1 (it uses the current-code hashes) or N4 (it never kills between the two pickle writes). Its final report must be checked before the freeze. |

## 4. Conditions for approval

1. Fix N1–N9 as specified. N10–N20 should be fixed too; any that are not must be listed in the adjudication with a reason.
2. Add the planted tests named in N1(b), N3 and M10, and run them in the preflight.
3. Before freezing, have a reviewer do a targeted verification of the diff.
4. Freeze, build, then run `scripts/verify_zip.py` (N9(c)) on a clean extraction. It must include a smoke whose Track A and Track B scores use the frozen hashes and are not PROVENANCE_FAIL.
5. Record `freeze_sha256` and the ZIP sha256 in `ADJUDICATION_v2.md` and `DESIGN_XX.md` §4.
6. Report the final `/tmp/smk2/SMOKE_REPORT.json` (and the result of the first, failed v1 smoke, as round 1 asked) in the pre-delivery verification document.

*End of review C (v2). Written by an AI agent reviewer (Claude); not a human peer review.*
