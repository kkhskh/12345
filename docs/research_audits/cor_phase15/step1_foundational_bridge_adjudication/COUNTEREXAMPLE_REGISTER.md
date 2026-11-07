# Counterexample register

Each entry: the claim it attacks, whether the witness meets that claim's hypotheses, and where it is checked. Negative results are kept (stopping rule E).

| ID | Attacks | Witness | Hypotheses met? | Check |
|---|---|---|---|---|
| CX-1 | "agreement among same-layer components is a mechanism" | 20 zero-sum edits across 3 heads of layer 0, 2-layer model: the pairwise cosine moves, all logits are fixed to `1e-10` | yes (same cut) | `test_SS_sum_preserving_head_edits_leave_every_logit_unchanged` |
| CX-2 | "equal relation ⇒ equal decision" | both heads scaled ×3: cosine unchanged, margin changed | yes | `test_SS_same_relation_different_sum_changes_decision` |
| CX-3 | "supporting evidence always raises the margin" / "more agreement is better" | Construction D: `N→W` gives `+1.25` (B neutral) and `−0.51` (B strong support) | yes (full pre-norm block, attention-only and with a random MLP) | `test_D_dilution_crossover_in_full_prenorm_block` |
| CX-4 | "the CI direct-effect law fixes the decision" | D′: one MLP neuron re-reads the evidence; direct `< 0 <` total | yes | `test_D_downstream_mlp_reverses_the_direct_prediction` |
| CX-5 | "norm effects only change confidence, never direction of a source's effect" | Construction LN: final-LN-only crossover | yes | `test_LN_final_norm_coupling_creates_crossover_without_attention` |
| CX-6 | "NC flips any pair" | same-sign pairs (seeds 6 and 8) and the disagreeing-but-unreachable pair (seed 12) | yes | `test_NC_flip_iff_content_and_bias_disagree_and_threshold_reachable` |
| CX-7 | "NC constrains full-vocabulary ranking" | `|V| = 20 > d = 8`: `N_C = {0}` | yes | `test_NC_vacuous_for_full_vocabulary_when_V_exceeds_d` |
| CX-8 | "nonremovable ⇔ crossover" in designs larger than 2×2 | `[[3,6,6.5],[5,7,8],[7,7.5,8.5]]` | yes | `test_RB_double_cancellation_violation_without_crossover` |
| CX-9 | "mixed-objective success shows consistency helps" | random `(A,g,J)` with consistency term `< 0 <` total at `β = 2` | yes | `test_MIX_output_term_nonnegative_consistency_term_unsigned` |
| CX-10 | "the agreement indicator predicts sign" | conflict cells of both signs (strong-for/weak-against versus weak-for/strong-against); in Construction D, column `OPP` | yes | FIB.1; the D factorial with `OPP` |
| CX-11 | "consistent ⇒ correct" | relabeled world | yes | CORR |

| CX-12 | "the crossover in D is driven by agreement" | neutral and weak tokens given equal salience: the crossover disappears (review C, probe a); BOS sink weight ≥ 0.8: it disappears (review D) | yes | review C and D scripts; `test_CI_star_relation_enters_only_through_attention_reallocation` |
| CX-13 | "a substituted source's direct sign is `sign(v'−M)`" | neutral salience 1, weak salience 0: `v'_W = 1 < M = 3.21`, head term +0.67 (review B, S2) | yes | review B `cx.py` |
| CX-14 | "CI signs the margin" | random 1-layer 1-head models: 591/4000 substitutions with opposite signs for the head's `a` term and `Δm` (review B, S3); 78/600 for live-LN direct effect (review D) | yes | review B and D scripts |
| CX-15 | "NC: at most one flip along any path" | out-and-back path in `N_C`: two flips | yes | `test_NC_two_flips_on_a_non_monotone_path` |
| CX-16 | "the norm channel can flip pairs" (RMSNorm models) | `κ ≡ 0`: 2000 null edits, ranking unchanged | yes | `test_NC_rmsnorm_analogue_never_flips` |
| CX-17 | "RB is invariant to the decision metric" (beyond one scalar) | 3 candidates: additive margin table, crossover in `p_c` (review B, S1) | yes | review B `cx.py` |
| CX-18 | "relation-sensitivity indicates consistency" | 172/200 random 3×3 tables with random tokens as levels are relation-sensitive (review C, probe d) | yes | review C `probe.py` |
| CX-19 | "raw head writes are invariant" | shifting `W_O` along `1`: logits fixed to 3e−15, head cosine −0.55 → 0.92 (review B, M3) | yes | review B `cx2.py` |

Reviewer scripts lived in `/tmp` of the review sessions and are not part of the package. Their numbers are quoted from the review files.

Inherited and still valid: Phase 1 Attacks A–L; 1A CE-01 to CE-15; 1C-R seed-16 projection pair and seed-0 direction; 1E edit `(1, 1.2)`. None is overturned by this step. All of them attack `O`, `D`, or residual surrogates, and those are not part of the surviving object.
