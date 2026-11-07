# Comparison with existing theory

Claim-to-prior-work table. The novelty review (`reviews/F_novelty_and_prior_work.md`) has the URLs and the verification notes.

| Result | Closest prior work | Equivalence or limit | Classification |
|---|---|---|---|
| SS | Elhage et al. 2021, residual stream as a sum of writes | a direct corollary | KNOWN |
| NC.1–3, NC.6 | Stolfo et al. 2024 (entropy neurons, effective null space of `W_U`); Gurnee et al. 2024 (the temperature analogy) | the same mechanism. Stolfo et al. report little effect on relative token ranking, consistent with NC: flips need `aκ < 0` | KNOWN_IN_SUBSTANCE |
| NC.4–5 (reachable set, flip iff, `κ` limit) | none found | an elementary corollary | POSSIBLY_NEW (minor) |
| NC-RMS | — | `κ ≡ 0`: pure temperature | KNOWN_IN_SUBSTANCE |
| CI.1–4 | the weighted-mean update; Hampel 1974 (influence function of the mean) | identical | KNOWN |
| CI-5 | Rushing and Nanda 2024, LN self-repair decomposition | algebraically identical | KNOWN |
| CI.3* | — | an algebraic rearrangement of CI.2 | KNOWN |
| RB.1–5 | Luce and Tukey 1964; Scott 1964; Krantz et al. 1971; Loftus 1978; Wagenmakers et al. 2012; Gail and Simon 1985 (qualitative interaction) | textbook additive conjoint measurement | KNOWN |
| RB.6 | Greco, Mousseau, and Słowiński 2008 (necessary preference relation); Dunn and James 2003 (signed difference analysis) | the same LP up to scaling | KNOWN |
| removability as an NN-interpretability criterion | NN interaction literature (NID; Shapley-Taylor; Bordt and von Luxburg 2023) uses fixed-scale interactions | no prior use found (moderate confidence) | NEW_APPLICATION |
| "a head is a weighted-mean aggregator" | Geiger et al. 2025 (causal abstraction) | an exact abstraction instance | KNOWN |
| MIX | Phase 1 Prop. 1 | algebra | KNOWN |
| FIB | elementary; 1C N1 | — | KNOWN (the Geiger et al. attribution is withdrawn) |
| Constructions D, LN, NEG, D′ | attention dilution and sinks (Xiao et al. 2024); self-repair and erasure (McGrath et al. 2023; Rushing and Nanda 2024) | hand-built existence examples | illustrative |

Equal-information comparison (mandate §11). With the same frozen weights and the same forward pass to the head, head patching gives the CI quantity exactly, and the attention-weight gradient gives its sign. With full weights and the full forward pass, direct evaluation gives every response. No surviving result beats these at matched information. None of them is presented as a better per-case predictor.
