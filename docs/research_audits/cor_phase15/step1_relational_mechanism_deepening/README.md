# COR Phase 1.5, Step 1 — Foundational Investigation III: relational mechanism deepening

Continues `../step1_foundational_atlas_continuation/` (start commit and dependencies: `sources/START.txt`; mandate:
`sources/MANDATE_III.txt`). Earlier packages are untouched; all historical verdicts stand. Not opened: Step 2, Phase 2.

**Read first:** `MAIN_RESEARCH_SYNTHESIS.md` (answers to the mandate's 42 questions with evidence grades), then
`PROSPECTIVE_PREDICTIONS.md` (frozen predictions Q3-A…E′ and outcomes), then the workstream documents:
PCOUNT_CAUSAL_TRANSFER · GATE_FULL_CAUSAL_CIRCUIT · TRANS_COMPOSITION_RESEARCH · STRUCTURE_AND_EXTRAPOLATION ·
RELATIONAL_COMPUTATIONAL_CLASSES · ADAPTIVE_SOURCE_THEORY · GENERALIZATION_LANDSCAPE · CAUSAL_IDENTIFICATION_AND_INVARIANCE ·
THEOREMS_AND_COUNTEREXAMPLES · BASELINES_AND_NOVELTY · COR_FAILURE_PROGRESS · OPEN_FOUNDATIONAL_RESEARCH_MAP;
reviews and adjudication in `reviews/`.

**Reviews.** Round R1: four independent adversarial reviews (formal math; causal/architecture/representation; statistics/
dynamics/freeze integrity; composition/novelty) → `reviews/ADJUDICATION_R1.md`. Round R2: counterexample construction and
positive-theory defence on the corrected claims → `reviews/ADJUDICATION_R2.md`. Corrections are marked [R: id] / [R2] in the
documents; superseded statements stay visible in strikethrough. The largest change: the pre-review claim that T1 transfer
models store a non-count variable was an encoder artefact (CA1); both curricula carry causally used count variables, at
different sites and codes. Reviewer scripts: `reviews/checks/`; my re-runs: `logs/review_checks/`.

**Provenance rules.** Frozen prediction blocks in PROSPECTIVE_PREDICTIONS.md are append-only (verified by R1 and R2 against
git history); outcome notes are appended below them. Earlier packages and their verdicts are not modified.

## Reproduction
Dependencies: Python 3.13, numpy 2.5, jax 0.12 (CPU), optax 0.2.8, scikit-learn 1.9, scipy 1.18. Scripts import the sibling
packages' `src/` directories (read-only).
- Training: `scripts/run_job3.py JOB` (job list in its docstring); queues `scripts/lane1.sh`, `lane2.sh`; held-out models via
  the atlas `train_curriculum.py` / `train_run.py`; exploratory/confirmatory ADAPT curricula `scripts/adapt_curriculum*.py`.
- Analyses: `pcount_causal.py`, `pp3_dyn3.py`, `gate_circuit.py`, `gate_redundancy.py`, `gate_headpairs.py`,
  `eval_trans.py`, `trans_identifiability.py`, `trans_probe_controls.py`, `trans_restricted_bayes.py`, `extrap_dev.py`,
  `eval_extrap.py`, `adapt_theory.py`, `eval_adapt.py`, `adapt_state_probe.py`, `od_check.py`.
- Clean extraction: `bash scripts/verify_clean_extraction.sh` (exports the committed tree, checks `MANIFEST.sha256`, reruns
  exact-Bayes scripts and one learned-model evaluation; last result in `logs/clean_extraction_check.txt`).
- Compute: 2 CPUs, 7 GB; budget cuts and an OOM incident are logged in `logs/COMPUTE_LOG.txt` (all made before the affected
  jobs ran, or reruns of killed jobs).
