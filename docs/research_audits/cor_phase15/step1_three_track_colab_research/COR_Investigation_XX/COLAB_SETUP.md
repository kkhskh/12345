# Colab setup

1. **Runtime.** Runtime → Change runtime type → a GPU (L4, A100, T4 all work; the model is small, so throughput is dominated by
   per-step overhead and the measured benchmark decides concurrency). Python must be ≥ 3.12 (the Colab default).
2. **Files.** Put `cor_inv20_colab_three_track_research.zip` in `MyDrive/` (default path in the config cell) or upload it to
   `/content`. Nothing else is needed: the historical parents, the XVIII reference checkpoints and every generator are inside.
3. **Open** `notebooks/00_RUN_COMPLETE_INVESTIGATION_XX.ipynb` (File → Upload notebook, or open it from Drive after extracting).
4. **Configure** the first cell only: `PRESET`, `USE_DRIVE` (recommended: True), `DRIVE_DIR`, optional `BUDGET_HOURS`.
5. **Run all.** The dependency cell may restart the runtime once (pinned jax/jaxlib/optax/numpy/scipy); then run all again.
6. **Storage.** With `USE_DRIVE=True` the run root is `DRIVE_DIR/<preset>/` and holds every checkpoint, the job registry, scores
   and results; it survives disconnects. Expected sizes are printed by the preflight (measured snapshot size × planned snapshots;
   intermediate snapshots of non-anchor arms are deleted after completion with their sha256 kept in the registry).
7. **Interruptions.** Reconnect, run all cells again. Interrupted jobs are marked in the registry and resume from `resume.pkl`
   (parameters, AdamW state, data-RNG state, loss history); completed jobs are never re-run; failed jobs stay failed unless
   `RETRY_FAILED=True` (each attempt is kept). Only one runner may use a run root at a time (heartbeat lock).
8. **Budget.** `BUDGET_HOURS` stops launching new jobs after the given time; nothing is shortened. The report then states INCOMPLETE.
9. **Return** the archive written by the last cell (`EXPORT/…zip` + `.sha256`).

CPU-only runtimes: Track C (notebook 03) and the `smoke` preset run on CPU; scientific presets would take days and are not advised.
The package was tested on CPU only before delivery; the GPU path has not been executed by the builder (see INVESTIGATION_XX_PRE_RUN_VERIFICATION.md).
