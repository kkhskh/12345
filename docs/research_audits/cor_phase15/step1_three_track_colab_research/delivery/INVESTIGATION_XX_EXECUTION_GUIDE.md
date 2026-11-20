# COR Investigation XX — execution guide

Phase 1.5, Step 1 (≈ 25% provisional). This package holds a frozen, reviewed design and the code to run it on Google Colab. It contains
**no confirmation results**: they exist only after you run it.

## 1. What you need
- A Google Colab GPU runtime (L4, A100 or T4). Python must be ≥ 3.12, which is the Colab default; the pinned numpy, scipy and jax have no 3.11 wheels.
- Google Drive with enough free space. The preflight prints an estimate for your preset and refuses to start if the estimate exceeds 90% of free space. The table below is computed from the planned snapshot counts and the measured CPU snapshot size; the preflight recomputes it on your runtime.

  | preset | estimated Drive need (max design, 0.61 MB per snapshot) |
  |---|---|
  | Pilot | ~0.2 GB |
  | Standard | ~7 GB |
  | Extended | ~24 GB |
  | High | ~100 GB |

- The file `cor_inv20_colab_three_track_research.zip`. Nothing else is needed: the historical parents and reference checkpoints are vendored inside it, and their hashes are checked.

## 2. First run: pilot (its hours are measured and printed by the preflight)
1. Upload the ZIP to `MyDrive/`.
2. Open `notebooks/00_RUN_COMPLETE_INVESTIGATION_XX.ipynb` in Colab (GPU runtime). The first code cells unpack the package from Drive.
3. In the configuration cell, keep `PRESET = "pilot"` and `USE_DRIVE = True`.
4. Run all cells. If the dependency cell installs anything, it restarts the runtime once; when that happens, run all cells again.
5. The pilot does the following on separate material (parent nP3901, data order 2901):
   - pretrains one fresh parent (3000 steps);
   - screens it on two data orders;
   - trains a few arms to 4000 steps, then scores and reports.

   Every verdict is forced to `TECHNICAL_ONLY`.
6. The preflight cell (step 6) prints:
   - the measured throughput;
   - the GPU-hours, storage and memory of **every** preset on your runtime;
   - PASS or FAIL.

   Scientific presets refuse to run without a PASS.

## 3. Scientific run
1. Set `PRESET = "standard"`, or `"extended"` / `"high"` if the printed hours fit your budget. Keep `DRIVE_DIR` the same: every preset gets its own run folder.
2. Optionally set `BUDGET_HOURS`. The runner then stops launching new jobs after that time; nothing is shortened, and the report says INCOMPLETE.
3. Run all cells.

The order of work is:
1. Track C mathematics;
2. pretraining of fresh parents;
3. screening, with the eligibility decision written once;
4. windows → compositions → anchors → other arms → measurements;
5. Track B;
6. stability (Extended and High only);
7. scoring → RESULTS → export.

## 4. Disconnects and restarts
- Reconnect and run all cells again.
- Interrupted jobs resume from their exact saved state: parameters, AdamW state, data-RNG state and loss history.
- Completed jobs never run again.
- Failed jobs stay failed and listed. With `RETRY_FAILED = True`, each one gets a new attempt, and the earlier attempts are kept.
- If you stop a cell yourself, the runner terminates its jobs and records them as interrupted.
- **Same GPU type:** a run folder records its numerical identity (GPU model, package versions, XLA flags, freeze). If Colab gives you a different GPU type, the run refuses to continue. You can wait for the same type, use a new `DRIVE_DIR`, or set `ALLOW_IDENTITY_CHANGE = True`; the change is recorded and the results are labelled mixed-identity.

## 5. What you get back
- `DRIVE_DIR/<preset>/RESULTS/`, with 11 files:
  - the main synthesis, plus one file per track;
  - frozen verdicts, statistics, and negative/inconclusive results;
  - methods, the compute log, and the review status;
  - the evidence manifest.

  Every verdict is labelled. If anything is incomplete, the files say INCOMPLETE and mark every verdict PROVISIONAL.
- `DRIVE_DIR/<preset>/EXPORT/cor_xx_results_<preset>_<time>.zip` and its `.sha256`, which hold:
  - the results, scores and measurements;
  - the job registry, eligibility, benchmark and preflight;
  - logs, lineage, run configs and composer provenance;
  - the post-run review bundle.

  Checkpoints stay on Drive; their hashes are in the archive.
- **Return the export ZIP and its `.sha256` for independent analysis.** The review bundle inside (`REVIEW_BUNDLE/REVIEWER_INSTRUCTIONS.md`) is meant for post-run agent reviewers. The package does not perform a review itself and does not claim one.

## 6. Per-track notebooks
`01` (Track A), `02` (Track B, which trains the Track A anchors it needs), `03` (Track C, CPU, a few minutes) and `04` (analysis, verification, export) call the same `cor_research.pipeline` functions as `00`.

## 7. Honest limits
- The builder tested the package on CPU only. The GPU path has not been executed by the builder. The preflight's runtime resume/determinism test checks it on your GPU before any scientific job runs.
- GPU results are reproducible run to run on the same GPU model and software stack. They are not bitwise equal to CPU.
- All reviews in the package are AI-agent reviews, not human peer review.
