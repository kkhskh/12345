# 01 — Authoritative source inventory

Audit revision: git `f95d134`. No experimental verdicts were changed. No obstruction energy was implemented on Othello-GPT. No confirmation split was accessed.

## 1. Authoritative theory sources

| Rank | Source | Role | Status |
|---|---|---|---|
| A | `docs/notes/Causal_Obstruction_Response_theory.pdf` | Original COR manuscript. Definitions 1–4, coboundary (12), `D_c` (14), `O_c` (15), response functional (1)/(26), Theorem 1, Propositions 2–6 (original numbering). | PRESENT. Extracted for this audit. |
| B | `docs/notes/Causal_Obstruction_Response_theory_Update.pdf` | Updated manuscript. Adds latent-state germs `s_y`, `O_y`, representation-before-steering, Propositions 1–7 (renumbered), Proposition 6 local-response remainder, Proposition 7 superposition. | PRESENT. Extracted for this audit. |
| C | `docs/theory_mvp_theorem_set.md` | Only in-repo *proofs* of Theorem 1, Proposition 1 (minimum-energy repair), and the cohomology correction. | PRESENT. |
| D | `docs/math_map_v0.md` | Operational atlas of objects and theorem *targets*, not proofs. | PRESENT. |
| E | `src/model_maps.py` `default_obstruction_spec()` | Machine-readable copy of `O_c`, `Õ_c`, `Frag_c`; notes that `D_c` is conceptual and `O_c` is computed. | PRESENT. |

No `.tex` source of the manuscripts exists in the repository. Propositions in the PDFs are stated; several are not proved in markdown.

## 2. Version-conflict table

| Item | Original PDF | Update PDF | MVP markdown | Empirical code |
|---|---|---|---|---|
| Minimum-energy repair | Proposition 2 | Proposition 1 | Proposition 1 | Local quadratic closed form in `run_08_theory_numerical_checks.py` only |
| Obstruction–margin stability | Proposition 3 | Proposition 2 | Named, **not proved** | Not implemented as a theorem check |
| Non-identifiability | Proposition 4 | Proposition 3 | Named, **not proved** | Admitted in 08i “non-identifiability” prose |
| Local affine boundary | Proposition 5 | Proposition 4 | Named, **not proved** | Local linearization used in 2D scores |
| Relative cohomology | Proposition 6 | Proposition 5 | Proved as “cohomology correction” | Synthetic check only |
| Local Taylor remainder | MISSING | Proposition 6 | Cited as “Proposition 6 of the theory document” | Intervention-radius caps in 0B |
| Superposition non-uniqueness | MISSING | Proposition 7 | Cited, not proved | 0B random-subspace controls |
| Projection notation | `P^c_v` | `P^c_v` | unused | 08i writes `P_v^c` |
| Transport symbol | `ρ` | `ρ` | `ρ` | IOI scalar `r`; factual 2B `R_l` |
| Candidate index | `c=(a,b)` | `c=(a,b)` plus latent `y` | often dropped | 0B uses `y` not `c` |
| `O` vs `D` | Called “equivalent” residual | Same | Theorem 1 is a **spectral sandwich**, not equality | Transformers compute `O` only |

Do not silently prefer the Update numbering. This audit cites both and labels the conflict.

## 3. What is implemented on models

| Quantity | Implemented on LMs / toys? | File |
|---|---|---|
| `O` as weighted edge residual | YES, several variants | `src/obstruction.py`, `src/vector_transports.py`, `src/toy_modular.py`, `src/predictive_response.py` |
| `D` as distance to `ker d` | Synthetic sheaves only | `run_08_theory_numerical_checks.py` |
| Learned `ρ` | YES (ridge / scalar / low-rank) | IOI, factual 2B, modular 0B |
| Candidate-specific `P_v^c` as a true projector | NO. Ridge / PCA / logit-lens maps | 2B `clean_pca`; 0B ridge; Othello 08c probe |
| Sheaf cohomology | NO on models | MVP numerical check only |
| Othello obstruction energy | Explicitly **not computed** | `src/othello_r2.py` line 3; `OBSTRUCTION_READY = "NO"` |

## 4. Official empirical letters (frozen; not revised here)

```
08R2_VALIDATION_COMPLETE — NEGATIVE
TARGET_SPECIFIC_NULL_EFFECT_SUPPORTED
FIXED_JS_TARGET_SPECIFICITY_SUPPORTED
TARGET_SPECIFIC_AT_EQUAL_JS_INCONCLUSIVE
DONOR_FREE_NULL_STATE_TRANSFER_SUPPORTED
LEGAL_MASK_SUFFICES_IN_TESTED_FAMILY
SYMMETRIC_TARGET_SWITCHING_NOT_SUPPORTED
SYMMETRIC_TARGET_CENSUS_INFEASIBLE
TARGET_SPECIFIC_VISIBLE_NULL_INTERACTION_NOT_SUPPORTED
0A = C / PARTIAL
SECTION_5_3 = PARTIAL
OBSTRUCTION_READY = NO
```

Constants: `src/othello_r2.py` lines 24–26.

## 5. Missing material

| Item | Status |
|---|---|
| Proof of Update Proposition 2 (obstruction–margin stability) | MISSING |
| Proof of Update Proposition 3 (non-identifiability) as a uniqueness/identification theorem | MISSING (statement only) |
| Unique identification rule for `P`, `ρ`, `M`, `W` | MISSING from Definitions 2–4 |
| Causal bridge from `D_c` to `F_θ` or `L_c` | MISSING as a theorem |
| Composition law `ρ_{v→w} ∘ ρ_{u→v} = ρ_{u→w}` | NOT REQUIRED in the manuscripts |
| Exact IOI Spearman pair 0.936 vs 0.141 as cited in the audit prompt | NOT FOUND as those floats in tracked markdown; qualitative IOI result is documented (attribution-positive / obstruction-negative) |
| `notebooks/03_compute_obstruction.ipynb` | Present but empty |

Stop-condition 1 does **not** fire: authoritative definitions exist. They are incomplete as a causal theory.
