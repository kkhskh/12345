# Source register

**Repository.** `kkhskh/12345` (GitHub).
**Base.** Branch `cursor/overnight-reconstruction-b68c` at `680c5fd2b2f33f9fb7388bff206b4d892f97aacd` ("Record the Step 1E joint-reachability line certificate").
**Work branch.** `cor-phase15-step1-foundational-bridge`.
**Main.** `main` at `b9516a75851e80345ae0494be0d5e77e3af65be8` holds no `docs/research_audits/`. The audits exist only on the overnight branch and on `08r2-implementation-repair` (`71d2925`, an ancestor).

## Authoritative, read for this step

| Source | Role |
|---|---|
| `docs/research_audits/cor_phase1/02, 04, 06, 09, 10, 15` | original audit: definitions, gauge, causal bridge, claim register, counterexamples, verdict |
| `docs/research_audits/cor_phase15/step1_causal_bridge/18_final_causal_bridge_verdict.md` | Step 1A verdict `PHASE15_STEP1_ONLY_GENERIC_BOUNDS_SURVIVE` |
| `.../step1b_mechanism_discovery/30_step1b_scientific_verdict.md` | `STEP1B_NECESSARY_CONDITIONS_IDENTIFIED` |
| `.../step1c_computational_mechanisms/23_step1c_final_verdict.md` | `STEP1C_NECESSARY_CONDITIONS_ADVANCED` |
| `.../step1c_r_full_block/23_final_step1c_r_verdict.md` | `STEP1C_R_ONLY_GENERIC_COMPUTATIONAL_RESULTS_SURVIVE` |
| `.../step1d_structural_bridge/19_final_scientific_verdict.md` | `STEP1D_STRUCTURAL_CERTIFICATES_ONLY` |
| `.../step1e_joint_reachability/22_final_scientific_verdict.md` | `STEP1E_VERIFIED_RELATIONAL_CERTIFICATE_KNOWN_THEORY` |
| `.../overnight_reconstruction/final/PHASE15_FINAL_VERDICT.md` | `PHASE15_NECESSARY_CONDITIONS_AND_BOUNDARIES_ESTABLISHED` |
| `.../overnight_reconstruction/literature/PRIMARY_NOTES.md` | previously verified citations (Geiger et al. 2025; Sutter et al. 2025; Elhage et al. 2021; Vaswani et al. 2017; Ba et al. 2016; Xiong et al. 2020) |
| `src/model.py` | GPT-2 small loaded through TransformerLens with `fold_ln=False`; architecture facts only |

The other files in those directories were listed by size and consulted where a verdict cited them. They were not all read line by line. Superseded statements in the overnight reconstruction are those that Step 1D `01_overnight_claim_corrections.md` corrects. They are not used here.

## External sources verified for this step (web, November 2026)

- Stolfo, Wu, Gurnee, Belinkov, Song, Sachan, Nanda. *Confidence Regulation Neurons in Language Models.* NeurIPS 2024. arXiv:2406.16254.
- Rushing, Nanda. *Explorations of Self-Repair in Language Models.* ICML 2024, PMLR 235:42836–42855. arXiv:2402.15390.
- McGrath et al. *The Hydra Effect: Emergent Self-repair in Language Model Computations.* arXiv 2023 (cited via Rushing and Nanda).
- Luce, Tukey. *Simultaneous conjoint measurement.* J. Math. Psych. 1:1–27, 1964.
- Scott. *Measurement structures and linear inequalities.* J. Math. Psych. 1:233–247, 1964.
- Krantz. *Conjoint measurement: the Luce–Tukey axiomatization and some extensions.* J. Math. Psych. 1:248–277, 1964.
- Krantz, Luce, Suppes, Tversky. *Foundations of Measurement, Vol. I.* Academic Press, 1971.
- Loftus. *On interpretation of interactions.* Memory & Cognition 6(3):312–319, 1978.
- Wagenmakers, Krypotos, Criss, Iverson. *On the interpretation of removable interactions.* Memory & Cognition 40(2):145–160, 2012.

The novelty reviewer's additions are in `reviews/F_novelty_and_prior_work.md`.

## Not used

Sealed confirmation data; historical evaluation outcomes as fitting data; any pretrained weights. Historical experiment files were not modified (checked in `logs/`).

## Missing

No Phase 1 or Phase 1.5 file was found missing. The "original Phase 1 mathematical audit" failure numbering (#1–7, #25, #29–30) is the mandate's own. The Phase 1 directory does not number its failures that way. The mapping is in `ORIGINAL_FAILURES.md`.
