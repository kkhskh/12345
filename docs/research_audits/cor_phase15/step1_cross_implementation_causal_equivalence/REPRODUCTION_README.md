# Reproduction

Base: commit 8c31051 on `cor-phase15-step1-relational-mechanism-deepening` (`sources/START.txt`); this package lives on
`cor-phase15-step1-cross-implementation-equivalence`. Mandate: `sources/MANDATE_IV.txt`. Earlier packages are read, never
written. Python 3.13, numpy 2.5, jax 0.12 (CPU), scikit-learn 1.9, scipy 1.18; 2 CPUs, 7 GB.

| step | command | output | time |
|---|---|---|---|
| exact information structure | `python3 scripts/info_theory.py` | `logs/info_theory.json` | < 1 min |
| site-test training (seeds 5–7) | `bash scripts/site_lane.sh` | `checkpoints/{pre,ft}_pcsrc_s*.{npz,json,log}` | ≈ 30 min/seed |
| calibration | `python3 scripts/equiv_run.py cal; analyze.py cal; info_structure.py cal; necessity.py cal` | `logs/cal/` | ≈ 25 min |
| **freeze** | commit 579a7a2 (`FROZEN_PREDICTIONS.md`, `logs/freeze_hashes.txt`) | | |
| confirmation | the same four scripts with `conf` | `logs/conf/` | ≈ 40 min |
| site evaluation | `python3 scripts/site_eval.py S5 S6 S7 P2 P3 P4` | `logs/site/site_eval.json` | ≈ 8 min |
| post hoc | `python3 scripts/transport_posthoc.py` | `logs/conf/transport_posthoc.json` | ≈ 12 min |
| clean extraction | `bash scripts/verify_clean_extraction.sh` | `logs/clean_extraction_check.txt` | ≈ 2 min |

All random streams are fixed in the scripts (fit 12_100_000 / 12_300_000; evaluation 12_200_000 / 12_400_000; necessity
12_250_000 / 12_450_000). Saved evaluation episodes: `logs/{cal,conf}/episodes.npz`; saved abstract-intervention responses:
`logs/{cal,conf}/responses.npz`. Models: Investigation III transfer checkpoints `ft_pcount_s{1..4}`, `ref_T1p{1,2,10,11}`,
continuation `ft_pcount.npz`, `ref_T1.npz` (P0, T0), new `checkpoints/ft_pcsrc_s{5,6,7}.npz`.

Provenance rules: frozen text in FROZEN_PREDICTIONS.md is above the OUTCOMES line and unchanged since 579a7a2; post-hoc scripts
say so in their docstrings; reviewer reports and their check scripts are in `reviews/`; `MANIFEST.sha256` covers every file.
