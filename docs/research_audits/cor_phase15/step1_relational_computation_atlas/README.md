# COR Phase 1.5 Step 1: relational computation atlas

**Principal document:** `FOUNDATIONAL_RESEARCH_ATLAS.md`. Its §0 map and §7 synthesis are the summary. Reviews revised it substantially, so read `ADJUDICATION.md` alongside it.

## Documents

| file | contents |
|---|---|
| `THEOREMS_BOUNDARIES_AND_IMPOSSIBILITIES.md` | S, M, U, U2, D, D2, I1–I7, with domains |
| `SOURCE_RELATIONS_AND_GENERATIVE_MODELS.md` | relation families and error processes |
| `COMPUTATIONAL_EQUIVALENCE_AND_INVARIANCE.md` | the E1–E6 ladder, the common abstraction and where it stops |
| `CAUSAL_IDENTIFICATION_AND_INTERVENTIONS.md` | intervention taxonomy, natural vs forced, corrections |
| `GENERALIZATION_AND_EMERGENCE.md` | curriculum controls, unknown groups, mechanism selection |
| `BASELINES_AND_NOVELTY_AUDIT.md` | what is known, what is new (small) |
| `PROSPECTIVE_PREDICTIONS_AND_FALSIFIERS.md` | PP1–PP5 frozen first; results appended, failures kept |
| `ORIGINAL_COR_FAILURE_MATRIX.md` | O/D and F1–F7, F25, F29, F30 |
| `OPEN_FOUNDATIONAL_PROBLEMS.md` | OP-1 to OP-9 |
| `reviews/` and `ADJUDICATION.md` | three independent reviews and every disposition |

## Code

**Sources** (`src/`):
- `gen_family.py`: extended generator and exact Bayes, validated against `../step1_latent_reliability/src/bayes.py`;
- `local_bayes.py`;
- `fwd.py`: the forward pass with per-head hooks;
- `train_ext.py`.

**Scripts** (`scripts/`):

| purpose | scripts |
|---|---|
| exact Bayes boundaries | `bayes_boundaries.py` |
| predictions PP1–PP5 | `pp1_sign.py` (`PP1_NB=1500` for the frozen specification), `pp2_shift.py`, `pp5_headsplit.py` |
| post hoc | `pp2b_local_posthoc.py`, `pp5b_exact_split_posthoc.py`, `pp1_reference_rules.py`, `tag_*`, `fixes_r2.py`, `onsupport_matched.py`, `site_specificity.py`, `pp4_own_unknown.py` |
| mechanism analyses | `mechanism_atlas.py` |
| training | `run_training.py` |

## Data

- **Logs:** `logs/`.
- **New checkpoints:** `checkpoints/` (PP3, PP4, PP5).
- **Models analysed:** all in `../step1_latent_reliability/checkpoints/`, unchanged.

## Verification

`scripts/verify_extraction.sh ZIP` unpacks the archive and re-runs the exact-Bayes checks and the fast analyses. It reports any differences.
- Clean extraction (2026-11-09T00:34:11Z): bayes_boundaries B1/B2/B4 and the reference-rule check reproduce identically; 12 atlas checkpoints load.
