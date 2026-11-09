# Generalization landscape (Workstream G and cross-cutting generality)

What changed when the setting changed, with the evidence grade. Bayes-level results for unknown, unequal and overlapping
groups, biased errors, copying and mixtures are in the continuation's RELATIONAL_GENERATIVE_LANDSCAPE.md and are not repeated.

## 1. Across seeds (the most important axis tested here)

| phenomenon | seeds | stable? |
|---|---|---|
| P-count frozen-feature transfer (≥ 0.6595) | 5 | **yes**: 0.723, 0.724, 0.708, 0.725, 0.704 |
| P-count readout count code causally used (abstract-variable interchange) | 5 | **yes**: 0.60–0.72 |
| T1 cross-question information at source tokens (full-site interchange) | 3 parents | **yes**: 0.84–0.87; linear count code absent (0.04–0.08) |
| T1-parent transfer | 5 parents | **yes**: 0.738–0.742 |
| P-group transfer | 3 | **no**: 0.669, 0.577, 0.571 |
| P-local failure | 2 | yes: 0.576, 0.576 |
| GATE Bayes-level accuracy | 3 | yes: 0.899–0.903 |
| GATE stage-1 at evidence tokens, decision via readout attention | 3 | yes (structure) |
| GATE layer / redundancy layout | 3 | **no** (three different layouts) |
| ADAPT with T1 curriculum: Bayes-level, positional evidence | 2 parents | yes (second prospectively confirmed) |
| T4 off-support direction (DeepSets global) | 5 earlier + 10 held-out | see STRUCTURE_AND_EXTRAPOLATION |

## 2. Across task families

| family | learned to Bayes level? | relation mediated (RC)? | notes |
|---|---|---|---|
| T4 latent reliability | yes (curriculum or DeepSets) | yes | earlier packages |
| GATE within-question constraint | yes (from scratch) | yes | 3 seeds |
| ADAPT change-point reliability | only with T1 curriculum | input-level evidence | from scratch fails |
| TRANS two-hop composition (36 pairs) | **no** (12k steps; aux supervision 8k) | no | maps represented with aux supervision |
| TRANS restricted to 24 pairs | yes on training pairs (≈ restricted-prior Bayes) | input-level | fails on held-out pairs, as the training-prior Bayes rule does |

Pattern: within-question relations (GATE) are learned from scratch; cross-question relations needing an episode-level
latent (T4, ADAPT) are learned with a relational curriculum; two-hop cross-question composition was not learned under any
regime tried.

## 3. Across architectures

Only Transformer vs DeepSets on T4 (earlier packages and the 10 + 2 held-out DeepSets here): DeepSets solve T4 from
scratch and extrapolate globally; Transformers need a curriculum and extrapolate variably. No architecture variation was
run for GATE, TRANS or ADAPT (compute).

## 4. Not examined (explicit)

- Learned models under **unknown or changing source groups** (T4-perm remains unlearned in the atlas; not retried).
- More than three candidates; larger groups; multiple colluding groups; partial copying regimes for learned models.
- Correlated errors for learned models.
- Longer episodes (the token layout fixes Q = 6).
These remain the largest gaps in the learned-model map (OPEN_FOUNDATIONAL_RESEARCH_MAP).
