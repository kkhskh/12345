# COR Investigation XX — three-track Colab research package

Phase 1.5, Step 1 (≈ 25% provisional foundational exploration). No phase closure, no thesis selection; O/D is not validated.
**This package contains no confirmation results.** It contains the frozen protocol, the code and the vendored historical inputs
needed to produce them on Google Colab, plus development notes (`dev/`, labelled, never used by a scorer).

| track | question | where |
|---|---|---|
| A | Does the XIX training-window pathway-selection effect, and its parameter-state mediation, generalise across independent pretrained parents? | `cor_research/scoring/track_a.py`, notebook 01 |
| B | Do the EARLY and MIXTURE organisations implement different decision functions, the same function through different routes, or the same rule at different precision? | `cor_research/scoring/{functional,track_b}.py`, notebook 02 |
| C | Are the original and repaired O/D operators adequate, distinct from Bayesian inference, and do they yield a distinctive neural prediction? | `cor_research/mathematics/`, notebook 03 (CPU) |

## Quick start (Colab)
1. Upload `cor_inv20_colab_three_track_research.zip` to Google Drive (`MyDrive/`) or to `/content`.
2. Open `notebooks/00_RUN_COMPLETE_INVESTIGATION_XX.ipynb` in Colab with a GPU runtime (see `COLAB_SETUP.md`).
3. Edit only the configuration cell (preset, Drive on/off, optional time budget). Run all cells.
4. After a disconnect: reconnect and run all cells again — every job resumes from its exact saved state.
5. Return `EXPORT/cor_xx_results_<preset>_<time>.zip` and its `.sha256` for independent analysis.

Start with `PRESET = "pilot"` (technical validation on your runtime: it measures throughput and prints the GPU-hours of every
preset), then run `"standard"` (or larger) in the same Drive folder. Presets: `COMPUTE_BUDGET.md`. Protocol: `SCIENTIFIC_PROTOCOL.md`.

## Layout
```
notebooks/            00 (complete run) and 01–04 (per track / analysis); all call cor_research.pipeline
cor_research/         env, protocol (single source of constants), pipeline, generators, models, training, interventions,
                      mathematics, scoring, analysis, reporting, jobs (planner, registry/runner, benchmark, budget), verification
cor_research/legacy/  vendored, unchanged historical COR modules and data (relative layout preserved) + the XX harness and composer
protocols/            DESIGN_XX.md, frozen JSON protocols, FREEZE.json (hashes verified by the runner before any job)
configurations/       presets, parameter-name list
reviews/              pre-run agent design reviews, adjudication, re-reviews
tests/                planted scorer tests, pretraining equivalence/resume, tag rules, Track C regression
scripts/              protocol generation, freeze, notebooks, smoke pipeline, review bundle, package build
manifests/            package manifest (sha256 of every file), vendored checkpoint hashes, smoke report
results_templates/    the structure of RESULTS/ (filled only from completed runs)
dev/                  development observations (NOT evidence)
```
Requirements: Python ≥ 3.12 (Colab default), pinned in `requirements.txt` / `requirements-gpu.txt`; `environment_lock.json` records
the build/test environment. GPU determinism is run-to-run on the same GPU model and software stack only (not bitwise equal to CPU).

All reviews in this package are AI-agent reviews, not human peer review.
