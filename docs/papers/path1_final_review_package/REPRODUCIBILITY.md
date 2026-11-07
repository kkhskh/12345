# Reproducibility

Every command below runs from the package root after the ZIP is extracted. None of them needs the research repository, the model checkpoint, a GPU or network access. `logs/` contains the logs of these commands as they were run for this release, including a run in a freshly extracted copy.

## Requirements

Tested with Python 3.9.13, numpy 1.26.4, scipy 1.13.1, matplotlib 3.8.4 and pytest 8.4.2 on macOS. Analysis, tables and verification should work with any recent numpy and scipy. Byte-identical figures require matplotlib 3.8.4. The figure test stops with an explicit message under another matplotlib version, and the verifier then prints a note and checks the figures' input data instead of their bytes. Rebuilding the PDF requires Tectonic (tested with 0.17.0). Checking the PDF text requires pymupdf, which is optional.

Set `MPLBACKEND=Agg` if your matplotlib defaults to an interactive backend. The figure script sets it itself.

## The three required commands

```
python scripts/verify_paper_claims.py
python scripts/reproduce_figures.py
python -m pytest tests/test_paper_reproducibility.py -q
```

- `scripts/verify_paper_claims.py` recomputes every analysis quantity in memory from `data/raw/` and compares the result with `analysis/`. It regenerates the tables into a temporary directory and compares them byte for byte with `tables/`. It then evaluates every row of `audits/claim_evidence_ledger.csv` against its source and requires every number in the manuscript text to be covered by a ledger row. It also checks the official letters, figure and table numbering, protocol constants, the archival excerpts and their hashes, the bibliography, figure provenance, the build record of the TeX and PDF, and package hygiene. It writes nothing. It prints one line per check and ends with the number of passed and failed checks. The exit code is 1 if any check fails.
- `scripts/reproduce_figures.py` redraws the five figures from `analysis/` and `tables/` and rewrites `figures/figure_provenance.json`. With matplotlib 3.8.4 the files are byte-identical to the shipped ones.
- `tests/test_paper_reproducibility.py` tests the numerical primitives on hand examples. It checks that no package module imports the research repository and that the analysis runs under `python -I` with no repository on the path. It also checks raw-data hashes, recomputation against the shipped analysis and the original runs' intervals, the headline numbers, probe invariance relative to the Editor A reference activation, and byte-identical tables and figures. Finally it checks the manuscript structure and the TeX header, runs the verifier, and confirms that the verifier rejects three deliberately altered manuscripts.

## Full pipeline

```
python scripts/run_all.py
```

This runs every step in order through `scripts/run_logged.py` and stops at the first failure. Each step's command, working directory, interpreter, package versions, selected environment variables, exit code, output, and the SHA-256 of every package file afterwards go to `logs/`. The steps are:

| Step | Script | Layer | Reads | Writes |
|---|---|---|---|---|
| 1 | `scripts/recompute_analysis.py` | A to B | `data/raw/`, `data/archival/` | `analysis/` |
| 2 | `scripts/reproduce_tables.py` | B to C | `analysis/results.json`, `data/archival/` | `tables/`, `audits/recomputation_report.md` |
| 3 | `scripts/reproduce_figures.py` | C | `analysis/`, `tables/` | `figures/` |
| 4 | `scripts/verify_paper_claims.py` | D | everything | nothing |
| 5 | `tests/test_paper_reproducibility.py` | D | everything | nothing |
| 6 | `verification/independent_harness.py` | D | `data/raw/`, `analysis/results.json` | nothing |
| 7 | `scripts/build_release_manifest.py` | release | package | `RELEASE_MANIFEST.json`, `audits/provenance_inventory.md` |
| 8 | `scripts/cross_artifact_check.py` | D | package | nothing |

`python scripts/run_all.py --rebuild-pdf` adds `scripts/build_pdf.py` before the verifier. That step regenerates `manuscript/paper_master.tex` from `manuscript/paper_master.md` and compiles the PDF with Tectonic. It fails on any overfull box. `python scripts/run_all.py --pdf-text` makes the cross-artifact check also extract the PDF text and look for every heading, caption and official letter.

The layers fail loudly. The analysis library raises an error if a raw file is missing or its hash differs from the manifest, if a cell is missing, or if an internal identity fails. The table and figure scripts refuse to write empty outputs. Each exception handler in the pipeline either prints the error and exits with a nonzero code or records a failed check. None of them lets a step continue as if it had passed.

## Independent checks

- `verification/independent_harness.py` derives the headline numbers again without importing the analysis library. It builds its own token list, uses scipy's Jensen–Shannon distance as well as a direct formula, and implements its own clustered bootstrap. It then compares its results with `analysis/results.json`.
- `scripts/integration_check_against_repo.py --repo PATH` compares the package's primitives with the research repository's own functions on every saved cell. This is the only file that imports the repository. It is not part of the package tests, and it needs a repository checkout. Its log from this release is `logs/integration_check_against_repo.txt`.

## Clean-extraction test

The release was checked by extracting the ZIP into an empty temporary directory outside the repository. `PYTHONPATH` was unset, and the research repository's `src` package was confirmed not to be importable from there. From that directory, `scripts/verify_paper_claims.py`, `scripts/reproduce_figures.py`, `python -m pytest tests/test_paper_reproducibility.py -q`, and `verification/independent_harness.py` each exited 0. The logs are `logs/clean_extraction_*.txt`.

## What can and cannot be reproduced from this package

| Evidence level | Examples | Reproducible here |
|---|---|---|
| Recomputed | all SPEC, matched-divergence and NST contrasts, intervals and win counts; probe invariance; JS calibration error; direction cosines; FIBER diagnostics | Yes, from saved logits, probe probabilities, directions, the fitted map and legal sets |
| Recorded | map reconstruction \(R^2\), dose-search outcomes, gradient-baseline coverage | Read from files written by the original runs; recomputing them needs the checkpoint |
| Archival | SEM, FRESH and VNF means and intervals, stage-share subgroups | Read from verbatim console excerpts; arrays not found |
| Protocol | floors, gates, seeds, replicate counts, Editor A settings | Checked against the preregistrations and configurations |

Replaying an intervention through the model, refitting the map, or recalibrating a dose requires the checkpoint and probe described in `data/README.md` and the project repository. That is outside this package.
