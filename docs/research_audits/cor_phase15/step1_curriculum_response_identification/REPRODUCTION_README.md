# Reproduction — Investigation V

Base: `ad4c629` on `cor-phase15-step1-cross-implementation-equivalence` (`sources/START.txt`); this package is on
`cor-phase15-step1-curriculum-response-identification`. Mandate: `sources/MANDATE_V.txt`. Python 3.13, numpy 2.5, jax 0.12 (CPU),
scipy 1.18, scikit-learn 1.9; 2 CPUs.

**Companion packages (read-only, all under `docs/research_audits/cor_phase15/`):** `step1_latent_reliability` (T1 trainer `nets.py`,
`bayes`, development T1 parents), `step1_relational_computation_atlas` (generator `gen_family.py`, `train_ext.py` transfer trainer),
`step1_foundational_atlas_continuation` (P-count trainer `train_tasks.py`, development P0), `step1_relational_mechanism_deepening`
(`fwd3.py`, development checkpoints P1–P4, T1, T2, T10, T11), `step1_cross_implementation_causal_equivalence` (`eqv.py`).

| step | command | output | time (2 CPUs) |
|---|---|---|---|
| cohort (16 runs) | `bash scripts/cohort_lane.sh` (two copies) | `checkpoints/{pre,ft}_{T1,P}_s6*.{npz,json,log}` | ≈ 7 h |
| quality variants | `bash scripts/budget_watcher.sh` (`transfer_budget.py`) | `checkpoints/budget/` | ≈ 5 min each |
| development data | `python3 scripts/cri_run.py dev P2 P3 P4 T2 T10 T11 P0 P1 T0 T1` (+ dev budget variants) | `logs/dev/` | ≈ 1.5 min/model |
| freeze | commit db60086 (`FROZEN_PREDICTIONS.md`, `logs/freeze_hashes.txt`) | | |
| confirmation data | `bash scripts/conf_watcher.sh` (runs `cri_run.py conf` per model) | `logs/conf/model_*.{npz,json}`, `generator.npz` | |
| confirmatory statistics | `cri_stats.py conf primary <6 T1> <6 P>`; `... budget_b500`, `budget_b250`, `allruns`; `null_structure.py conf null_primary ...`; `p5b_matched_content.py conf p5b ...` | `logs/conf/*.json` | minutes |
| exploratory (post hoc) | `route_selectivity.py`, `phi_sign_posthoc.py` | `logs/conf/route_selectivity.json`, `logs/phi_sign_posthoc.json` | seconds |
| clean extraction | `bash scripts/verify_clean_extraction.sh` | `logs/clean_extraction_check.txt` | ≈ 5 min |

Streams: development 13_000_000 (eval) / 13_010_000 (fit); confirmation 13_100_000 / 13_110_000; budget transfers use the parent seed.
Saved evaluation episodes and generator arrays: `logs/{dev,conf}/generator.npz`; raw per-model response arrays: `logs/{dev,conf}/model_*.npz`.
The clean-extraction check distinguishes a forward-pass reproduction (step 2) from a replay of archived arrays (step 3).
Orchestration scripts written after the freeze (`conf_watcher.sh`) only invoke frozen code.
