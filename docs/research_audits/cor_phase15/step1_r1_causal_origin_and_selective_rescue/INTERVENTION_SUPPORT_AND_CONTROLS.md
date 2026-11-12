# Intervention support and controls

**Sources.** Development: the 16 Investigation VI parents at step 4000, evaluated on the new development set `logs/eval_dev.npz`
(stream 17_100_000), via `scripts/eval_dev_parents.py` → `logs/dev_parents_eval.json`. Family definitions:
`src/r1lib.py` and R1_INFORMATION_REQUIREMENTS.md §4. Every label (regime, orientation, counts) is generator-derived. None is
inferred from model outputs.

## 1. Source-level families and their support

| family | what changes | reports changed | Bayes response | support |
|---|---|---|---|---|
| F45 / F34 / F23 | one group's leave-q count +1 | 1 | regime-dependent | same world (count < 5 identifies a non-colluder); X_q fixed |
| F0 | the F45 token, set to the third value (pair still disagrees) | 1 | exactly 0 | unchanged world |
| FLOC | a count-4 group's disagreement moves to another question | 2 | exactly 0 | unchanged counts |
| FID | an agreeing non-perfect pair changes its common value | 2 | exactly 0 | unchanged counts and agreement |
| S (nested) | the same count-3 group is repaired twice: 3 → 4 → 5 | 1 + 1 | S34 ≥ 0; S45 regime-dependent | all stages supported; same g, same c_g, same other evidence |
| F54o | one of two agreeing perfect groups loses perfect agreement | 1 | −0.239 | natural two-perfect states |

**Nested sequences are possible under the generator.** A count-3 non-colluder can have its two disagreeing questions repaired
one at a time. Every stage is supported: a non-colluder may agree anywhere, and at stage 5 the colluder still agrees on all six
questions. The nested test therefore holds the group, its vote, X_q and all other reports fixed across 3 → 4 → 5.
Checked over 10,473 sequences (`logs/r1_information.json`).

## 2. Strong F0-type statistics (development; mean over 8 parents per arm)

| control | arm | mean signed | mean \|Δm\| | RMS | median \|Δm\| | 95th pct \|Δm\| | largest per-model mean \|Δm\| |
|---|---|---|---|---|---|---|---|
| F0 | T1 | −0.001 | 0.090 | 0.143 | 0.055 | 0.294 | 0.234 |
| F0 | P | +0.004 | 0.138 | 0.231 | 0.081 | 0.456 | 0.173 |
| FLOC | T1 | −0.001 | 0.158 | 0.224 | 0.107 | 0.482 | 0.511 |
| FLOC | P | +0.004 | 0.318 | 0.508 | 0.191 | 1.097 | 0.484 |
| FID | T1 | −0.000 | 0.076 | 0.114 | 0.051 | 0.231 | 0.194 |
| FID | P | +0.001 | 0.125 | 0.185 | 0.084 | 0.378 | 0.192 |
| *for scale:* F45 (all regimes) | T1 / P | | 1.238 / 1.488 | | | | |

These reproduce the independent audit's point. The T1 F0 mean |Δm| is 0.090 (the audit reported 0.088) and the P-count value
0.138 (the audit: 0.139). The signed means are near zero because large responses of opposite sign cancel.

**Bayes-null edits are not behaviourally null for the models.** The agreement-location control FLOC is the largest. In P-count
models its RMS of 0.51 is about a third of the F45 scale.

**F0 conditional on regime** (mean |Δm|):

| regime | T1 | P |
|---|---|---|
| R1 | 0.062 | 0.124 |
| R2 | 0.099 | 0.129 |
| R3 | 0.105 | 0.148 |

**Consequences.**
1. Small *signed* F0 means cannot certify that a mediator is specifically a count variable. Investigation VI's P6d used F0 in
   that way and was near-vacuous. That criticism is upheld.
2. The R1 result is not threatened. The R1 effect is a sign-level difference of about 0.7 in the *signed* mean, against
   Bayes-null noise that is symmetric and of order 0.1–0.3 per pair.

## 3. Nested 3 → 4 → 5 sequences (development; the strongest boundary test available)

Same group, same candidate c_g, same target evidence and same other reports at every stage.

| regime (stage-4 state) | transition | Bayes Δm | T1 Δm_M (parents < 0) | P Δm_M (parents < 0) |
|---|---|---|---|---|
| R1 | 3 → 4 | +0.455 | +0.415 (0/8) | +0.365 (0/8) |
| R1 | **4 → 5** | **+0.281** | **+0.218 (0/8)** | **−0.462 (8/8)** |
| R2 | 3 → 4 | 0 | −0.089 | −0.140 |
| R2 | 4 → 5 | −2.345 | −1.796 | −1.644 |
| R3 | 3 → 4 | +0.513 | +0.283 | +0.176 |
| R3 | 4 → 5 | −1.947 | −1.640 | −1.876 |

**Within the same sequence, every P-count parent credits g's vote at 3→4 and discounts it at 4→5. Every T1 parent credits it at
both steps.** This removes the population-mismatch objection to Investigation VI's F45-versus-F34 comparison.

Status: development. These are the Investigation VI parents on a new evaluation stream. The nested family is new, but the
parents are not.

## 4. F54o (time-reversed R1, natural two-perfect states)

| | Bayes Δm | mean Δm_M | parents with Δm_M > 0 |
|---|---|---|---|
| T1 | −0.239 | −0.208 | 0/8 |
| P | −0.239 | +0.501 | **8/8** |

For every mixture-form rule, F54o is the reverse of the R1 edit (R1_INFORMATION_REQUIREMENTS.md §4). It is a replication from
naturally sampled states, not an independent prediction.

## 5. Internal interventions: what each one targets

| intervention | variable targeted | held fixed | naturally reachable? | hybrid? | supports |
|---|---|---|---|---|---|
| Input-copy factorial at layer 2 (5 switches) | which layer-2 input copy (before or after the edit) each pathway reads: residual of t, query of t, key at t, keys elsewhere, values | frozen layers 0–1; all of layer 2's computation, recomputed exactly | corners yes (00000 and 11111 are the natural forwards; check < 1e-5) | interior points mix copies across pathways | exact Moebius decomposition of the response over pathway inputs; interaction terms are measured, not assumed away |
| Token-set attention attribution | attention re-weighting vs value change at t, by token group | — | — | linearised at the midpoint (exact for the attention term up to the LN-Jacobian linearisation; linear total = exact attention projection, `attn_lin` = `attn_exact`) | which tokens' attention weights move the margin |
| Per-head Q/K/V switches | one head's query, key or value input copy | everything else at before (add) or after (remove) | no | yes | per-head sensitivity bounds |
| Readout-token replacement (`_rtemb`) | readout-token input to layer 2 replaced by its embedding-level state, during training and test | everything else | the replaced state is a natural layer-0 state | no (a different but natural input distribution) | whether the readout-token route is necessary for the error |
| Feature injection (`_injm`, `_injn`, `_injrand`) | an added, trained linear embedding of a fixed function of X | the frozen model | yes (a deterministic function of the input) | no | whether supplying a specified statistic changes the learned solution |
| Accessibility probes | held-out decodability of H2, agree2, m, W and the posterior | — | — | — | accessibility, not use |

No conclusion here rests on mean ablation. Partial patches are reported as hybrid. Wherever possible, claims rest on the
natural corners, on exact factorial identities, or on training interventions that create no hybrid state.
