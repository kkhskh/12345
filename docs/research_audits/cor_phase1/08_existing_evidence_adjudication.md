# 08 — Existing evidence adjudication

Official experimental letters are **not** changed by this audit.

## 1. Positive evidence (what it actually is)

| Finding | Establishes about Transformer causality | Establishes about COR obstruction |
|---|---|---|
| 08c probe rank 124, null 388 | A linear board decoder is incomplete as a 512-D description | Nothing about `O`/`D` |
| `TARGET_SPECIFIC_NULL_EFFECT_SUPPORTED` | Probe-invisible directions can change legal-set mass specifically | Not obstruction |
| `FIXED_JS_TARGET_SPECIFICITY_SUPPORTED` | Specificity survives matched output JS in that family | Not obstruction |
| `TARGET_SPECIFIC_AT_EQUAL_JS_INCONCLUSIVE` | One COST protocol did not settle equal-JS specificity | Not obstruction |
| `DONOR_FREE_NULL_STATE_TRANSFER_SUPPORTED` | A train-only 64-D map in `ker W` transfers target-aligned Y on unused games | Donor-free **steering**, not a germ, not `O` |
| 2B low-rank residual `O` detects factual conflict (AUC ≈ 0.95–0.98, partial corr 0.58–0.79) | A variance-subspace residual chain correlates with conflict at matched margin | Monitoring signal for a fitted `O`. Not shown to be `D`, not shown to be the decision mechanism (`docs/paper_claims_after_2d.md`) |
| 2C Spearman(`ΔO`,`Δm`) ≈ 0.36–0.44 | Sites that repair margin often reduce this `O` | Competitive with attribution, does not dominate (`PASS_WEAK`) |
| 0B cyclic latent decode and some causal edits | Toy model has a known latent; destroying it is not always behaviorally fatal | Representation-before-steering; not LLM obstruction |

Natural-anchor symmetric-target means `A_T=+0.0285`, `A_W=+0.0186` were two-sided on average and still failed the 60% both-positive gate. Positive means are not bidirectional switching and not COR compatibility.

## 2. Negative evidence

| Finding | Consequence |
|---|---|
| IOI: attribution-positive / obstruction-negative (`README.md`, `docs/math_map_v0.md`, `docs/notes/current_project_state_t0.md`) | The closed obstruction-response loop failed on the first LM task. Standardized obstruction-response collapsed toward attribution (`run_02i`). |
| Audit prompt cites Spearman ≈ 0.936 (attribution) vs ≈ 0.141 (obstruction) | Those exact floats were **not found** in tracked markdown. Closest documented 2D numbers: attribution repair AUC **0.979** vs best obstruction-using score **0.955**; obstruction-only Spearman with `Δm` ≈ **0.095** vs attribution ≈ **0.905** (`docs/paper_claims_after_2d.md`, `standardized_decision_summary.md`). Pattern is the same: attribution dominates repair ranking. Status of 0.936/0.141: UNIDENTIFIABLE FROM AVAILABLE INFORMATION as an exact citation. |
| 08R2 validation NEGATIVE | Probe-based repair editor failed preregistered behavioral gates |
| `SYMMETRIC_TARGET_SWITCHING_NOT_SUPPORTED` | 32/32 coverage; both-positive 13/32 = 40.6% < 60% |
| `TARGET_SPECIFIC_VISIBLE_NULL_INTERACTION_NOT_SUPPORTED` | `I_T=+0.000808` CI `[-0.000940,+0.003017]`; `I_W=+0.000972` CI `[-0.000086,+0.002208]`; both below +0.005 floor. Visible `R²=0.9405`, so empty reconstruction is not the excuse. Completed negative factorial. |
| Editor-A `D_T`,`D_W` ≈ +0.016 | Confounded (editor moves both `P_R` and `P_N`). Not a compatibility demonstration |

Do not describe the factorial as inconclusive. Do not claim a new metric reverses it.

## 3. Evidence classification matrix

| Theoretical claim | Experiment | Outcome | Direct test? | Sign | Confounds | Remaining assumptions |
|---|---|---|---|---|---|---|
| Distributed evidence compatibility predicts decisions | IOI obstruction-response | Attribution wins | YES for the operational `O`/`S` | Negative | Scalar/low-rank germs, graph choice | `O` was a residual chain, not `D` |
| `O` detects internal conflict beyond margin | Factual 2A scalar | Failed data/partial-corr gates | YES | Negative | Margin residual | — |
| Vector residual `O` detects conflict | 2B `clean_pca` | PASS_STRONG as **monitor** | For that `O`, not for `D` or germs | Positive as correlation | PCA is variance, not a concept | Confound audit still owed in older notes |
| Reducing `O` repairs `m` better than attribution | 2C/2D | 2D repair AUC lift negative | YES | Negative for steering | `β` selection; linearization | — |
| `pred_ΔO` tracks `ΔO` | 2D | AUC 0.943 | YES | Positive for Taylor map of `O` | Not semantic | — |
| Known latent must be steered first | 0B | Mixed; subspace ablation can spare accuracy | Partial | Mixed | Toy | Prop. 7 superposition |
| Probe-invisible causal information exists | 08R2 SPEC/NST | Supported letters | YES for causality in `ker W` | Positive for **information**, not COR | Site, JS, legal Y | — |
| Matching visible and invisible target identity produces interaction | 08R2 VNF | NOT_SUPPORTED | YES for that operationalization | Negative | Fixed `ρ=ν=3.5`; ridge `r̂` | Not an obstruction test |
| Bidirectional symmetric switching | SYM FRESH | NOT_SUPPORTED | YES | Negative | — | — |
| `s_c=P φ` is a Section 5.3 germ | 08i 0A | `C/PARTIAL`; no `ρ`/`O` | Incomplete | Not tested | Abstraction substituted | `OBSTRUCTION_READY=NO` |

**Supported:** probe-invisible causal information and some donor-free steering in Othello; a fitted residual energy that can **monitor** factual conflict in Pythia-1B.

**Not supported:** COR’s obstruction-specific mechanism, obstruction beating attribution at repair, cohomological local-to-global glue, canonical decision germs, visible/null compatibility interaction.
