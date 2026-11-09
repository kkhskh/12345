# Adjudication of review round R3 (two independent subagent reviewers, run after the R2 corrections and the new results)

**Reviews.**
- `R3_learning_dynamics.md` (LD): PP3, P-A and P-B hygiene.
- `R3_beyond_and_recheck.md` (BR): GATE and TRANS, and a re-check of the R2 corrections.

**Verification.** The authors' re-runs are in `scripts/r3_checks.py` and `logs/r3_checks.json`.

| id | finding | verified by authors? | disposition |
|---|---|---|---|
| BR-F1 (critical) | GATE: last-layer attention is causally unnecessary. Zeroing all layer-2 heads leaves Bayes agreement at 1.000 | **yes**. Layer 2 zeroed: 1.000 (acc 0.900). Layer 1 zeroed: 0.978. Layer 0 zeroed: 0.756. The query-only patch figure (0.003) is reviewer-reported | **Accepted.** The GATE reconstruction is corrected: the relation runs from the evidence tokens to the full readout residual, not through routing. "Same operation as latent reliability" is withdrawn in BEYOND, MAIN and COR_TEN |
| BR-F2 | the flip result is implied by unique-argmax Bayes agreement | — | **Accepted.** Re-labelled as a tie-handling check |
| BR-F3 | the GATE relation is equality after a recoding | — | **Accepted.** The O/D implication is restated as "answer-discrepancy O/D cannot express gating by non-answer sources" |
| BR-F4 | TRANS: the model uses no C reports anywhere, and BC cross-entropy is worse than the one-hop reference. "τ not composed" cannot be told apart from "τ not represented" | reviewer-reported | **Accepted.** Findings reworded |
| BR-F5 | TRANS "neural, not statistical" is overstated. The run was cut mid-transition (cosine schedule truncated to 4000 steps) | yes (training log) | **Accepted.** Claim withdrawn; now "not learned in a truncated run" |
| BR-F6, F7 | the 0.33648 equality is a coincidence; the attention is near the uniform baseline | — | **Accepted** |
| BR-F9 | retracted PP1 readings survive in OPEN_MAP and BASELINES | yes (grep) | **Accepted. Fixed** |
| BR-F10 | the PP1 §5 headline states an identity as an empirical fact, and claims NP coverage that was not computed | — | **Accepted.** Rewritten as (i), (ii), (iii). MAIN and COR_TEN F1 also fixed |
| BR-F11 | OD1′ as stated (ranking equality) fails at D-ties with unequal weights | yes (argument) | **Accepted.** Restated as "no strict reversal"; the "only if" part is marked unverified (THEOREMS, COR_TEN, MAIN) |
| BR-F12–F14 | residual wording; the K2 constant; small numbers (clustered CI ±0.004 for tf s2; \|Δm\| < 0.1 band is not at chance; the ρ = 1 flip rate is not like-for-like) | — | **Accepted.** Fixed in text. The ADJUDICATION_R2 "±0.003" is superseded by this note |
| BR-F15 | MAIN carried the refuted GATE claims | — | **Accepted. Fixed** |
| LD-F1 | P-count supervises the reliability sufficient statistic | yes (definition: the counts are sufficient under (U) and (O)) | **Accepted.** "Without reliability supervision" is withdrawn |
| LD-F2 | P-count's aggregate is at the readout tokens (decodable at 1.000) | **yes**: 1.000, 1.000, 1.000 (T1 readouts: 0.48–0.81) | **Accepted.** "Probes are not the transferred variable" is withdrawn; P-B3 is annotated (wrong site) |
| LD-F3 | the 1000-step dynamics fine-tune is confounded with fine-tune speed | **yes**: fully trained P-count features score 0.654 at 1000 steps (0.723 at 2000) | **Accepted.** "Emerges abruptly" and "nothing transfers before 2000" are withdrawn. P-B4's pass is annotated as uninformative. Its verdict is unchanged, per the frozen-text rule |
| LD-F4–F6 | P-local is the right control for generic pretraining; P-group's transfer exceeds what its target alone supports; n = 1 seed per arm | reviewer-reported | **Accepted.** Added to TRANSFERRED_FEATURE §4 |
| LD-F7, F8 | the P-A and P-B verdicts were applied exactly per the frozen text; the EXTRAPOLATION §5 wording overreaches | yes (git log) | **Accepted.** §5 softened (n = 2 held-out Transformers) |

**Unresolved.**
- Two experiments were recommended but not run, because they exceed the compute budget of this session. Both are listed in TRANSFERRED_FEATURE §4 and OPEN_FOUNDATIONAL_MAP:
  - the step-1000 snapshot and P-group under longer fine-tunes;
  - a τ probe for TRANS.
- No disagreement with a re-run number remains.
