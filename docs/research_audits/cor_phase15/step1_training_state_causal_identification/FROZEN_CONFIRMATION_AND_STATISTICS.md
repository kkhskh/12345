# Frozen confirmation and statistics

**Status.** COR Phase 1.5, Step 1 (≈ 25%, unchanged).

## Chronology

| step | when / reference |
|---|---|
| Design frozen | commit `478a038`, 2026-11-19T17:21:39Z |
| Design freeze pushed | 17:21:47Z (verified by post-review E from the GitHub activity log) |
| First confirmation job | 17:21:56Z |
| Stage-6 functional battery frozen | commit `8ca9db7`, pushed 17:36:01Z: **14 minutes after confirmation started**. The battery code and its calibration on development and validation models were produced while the confirmation was running; 16 window states already existed. No confirmation output was read before scoring. That is self-attestation, consistent with file access times (post-review E). The battery is therefore labelled [frozen-addendum], not part of the reviewed freeze |
| Confirmation runs | 2026-11-19T17:21Z to 2026-11-20T02:04Z: 54 step-100 states and 99 continuations; 0 failed jobs |
| Container restart | 23:50:27Z. Two continuations were interrupted: tp_M822_IfromB_L0 at step 165 and tp_M822_BfromI_L0L1 at step 111. Both resumed from their last snapshots. Post-review E replayed both uninterrupted from step 100 to 1211: bit-identical, with the final hashes equal to the measured ones. The first-attempt output logs of these two jobs were overwritten |
| Monitoring before scoring | COMPUTE_LOG and file existence only |
| Frozen-file check at scoring | every file listed in FREEZE_XIX.json unchanged, verified before the scorer ran |
| First read of confirmation outputs | the frozen scorer (`logs/conf/score_c19.json`) |

## Independent units

- **Six matched pairs:** E821, E822, E823, M821, M822, M823.
- **One independent parent (uP160).** The data orders are the units: 3 per base, never pooled across bases, and D801 (development) never pooled.
- **Evaluability:** every pair was evaluable on both bases (3/3 and 3/3). Every model was competent. This is the failure point of XVIII, and it did not recur.

## Control gates (predeclared, step-100 states only)

- **pnz:** displacement ratio 1.003–1.007 on all 6 pairs. All pass.
- **lbl, E:** loss-excess ratio 0.98–1.02. All pass.
- **lbl, M:** the block's excess is 0.008–0.020. All three M pairs exceed 0.005, so the ratio rule applied; ratios were 4.7–12.4, and all pass. On M, lbl is a much stronger loss disturbance than the block.

**Duplicate parameter states.** The scorer reported the expected pairs (fullI, paramI_optB) and (fullB, paramB_optI). The duplicate check hashes only the parameter file, and these pairs differ only in their moments. They are distinct states and were scored as distinct arms.

## Primary results

Format of the "k/n" column: pairs inside the band (and with the right class) out of evaluable pairs, followed by the exact one-sided 95% Clopper–Pearson lower bound on the proportion of matched pairs that behave as predicted.

The last column lists the sensitivity variants (12 frozen variants) whose band differs from the primary.

| hypothesis | prediction | per-pair values | band | k/n (CP lower) | variants that differ |
|---|---|---|---|---|---|
| E_P1_L0_sufficient | `phi['BfromI_L0']` ∈ [0.75, 1.25], class I | E821: 0.98, E822: 0.97, E823: 0.99 | **SUPPORTED** | 3/3 (≥0.37) | none |
| E_P2_L0_necessary | `phi['IfromB_L0']` ∈ [-0.25, 0.25], class B | E821: 0.0, E822: 0.02, E823: 0.0 | **SUPPORTED** | 3/3 (≥0.37) | none |
| E_A4_L0QK_restore_insufficient | `phi['IfromB_L0_QK']` ∈ [0.75, 1.25], class I | E821: 0.98, E822: 0.96, E823: 0.96 | **SUPPORTED** | 3/3 (≥0.37) | none |
| E_A4_L0OV_restore_insufficient | `phi['IfromB_L0_OV']` ∈ [0.75, 1.25], class I | E821: 0.97, E822: 0.97, E823: 0.96 | **SUPPORTED** | 3/3 (≥0.37) | none |
| E_P3_params_carry | `phi['paramI_optB']` ∈ [0.75, 1.25], class I | E821: 0.98, E822: 0.98, E823: 0.98 | **SUPPORTED** | 3/3 (≥0.37) | none |
| E_P3_moments_alone_nil | `phi['paramB_optI']` ∈ [-0.25, 0.25], class B | E821: 0.01, E822: 0.01, E823: -0.0 | **SUPPORTED** | 3/3 (≥0.37) | none |
| E_A7_normmatched_L0_nil | `phi['randnorm_L0']` ∈ [-0.25, 0.25], class B | E821: 0.0, E822: 0.01, E823: -0.0 | **SUPPORTED** | 3/3 (≥0.37) | none |
| E_P6_pnz_nil | `phi['pnz']` ∈ [-0.25, 0.25], class B | E821: 0.02, E822: 0.0, E823: 0.01 | **SUPPORTED** | 3/3 (≥0.37) | none |
| E_P6_lbl_nil | `phi['lbl']` ∈ [-0.25, 0.25], class B | E821: -0.01, E822: 0.02, E823: 0.01 | **SUPPORTED** | 3/3 (≥0.37) | none |
| E_P4_vsgo_switch | `phi['vsgo']` ∈ [0.75, 1.25], class I | E821: 0.69, E822: 0.56, E823: 0.95 | **PARTIAL** | 1/3 (≥0.02) | delta0→CONTRADICTED |
| E_P4_vsgo_switch_sens070 | `phi['vsgo']` ∈ [0.7, 1.25], class I | E821: 0.69, E822: 0.56, E823: 0.95 | **PARTIAL** | 1/3 (≥0.02) | delta0→CONTRADICTED |
| E_P4_vsg_partial | `phi['vsg']` ∈ [0.25, 0.75] | E821: 0.02, E822: 0.02, E823: 0.68 | **CONTRADICTED** | 1/3 (≥0.02) | none |
| E_P4_vsgo_exceeds_vsg | `phi['vsgo'] - phi['vsg']` ∈ [0.2, 1.5] | E821: 0.67, E822: 0.55, E823: 0.26 | **SUPPORTED** | 3/3 (≥0.37) | lin→PARTIAL |
| E_P4_vrs_switch | `phi['vrs']` ∈ [0.75, 1.25], class I | E821: 1.0, E822: 1.0, E823: 0.99 | **SUPPORTED** | 3/3 (≥0.37) | none |
| E_P5_frz_nil | `phi['frz']` ∈ [-0.25, 0.25], class B | E821: 0.01, E822: 0.01, E823: 0.5 | **PARTIAL** | 2/3 (≥0.14) | none |
| E_P5_frzp_partial | `phi['frzp']` ∈ [0.25, 0.75] | E821: 0.59, E822: 0.54, E823: 0.69 | **SUPPORTED** | 3/3 (≥0.37) | none |
| E_P7_L0_concordant | category BOTH (switch pairs) | E821: INCONCLUSIVE(causal_SWITCH), E822: BOTH, E823: BOTH | **PARTIAL** | 2/3 (≥0.14) | none |
| E_P7_vrs_concordant | category BOTH (switch pairs) | E821: BOTH, E822: BOTH, E823: BOTH | **SUPPORTED** | 3/3 (≥0.37) | none |
| M_P1_L0_alone_insufficient | `phi['BfromI_L0']` ∈ [-0.25, 0.5] | M821: 0.26, M822: 0.33, M823: 0.38 | **SUPPORTED** | 3/3 (≥0.37) | none |
| M_P2_L0_restore_reduces | `phi['IfromB_L0']` ∈ [-0.25, 0.5] | M821: 0.32, M822: 0.43, M823: 0.75 | **PARTIAL** | 2/3 (≥0.14) | none |
| M_P2_L0L2_restore_prevents | `phi['IfromB_L0L2']` ∈ [-0.25, 0.25], class B | M821: 0.06, M822: 0.13, M823: 0.06 | **SUPPORTED** | 3/3 (≥0.37) | none |
| M_P1_L0L1_sufficient | `phi['BfromI_L0L1']` ∈ [0.75, 1.25], class I | M821: 0.99, M822: 0.99, M823: 0.97 | **SUPPORTED** | 3/3 (≥0.37) | none |
| M_P1_L0L2_sufficient | `phi['BfromI_L0L2']` ∈ [0.75, 1.25], class I | M821: 0.99, M822: 0.98, M823: 0.97 | **SUPPORTED** | 3/3 (≥0.37) | none |
| M_P3_params_carry | `phi['paramI_optB']` ∈ [0.75, 1.25], class I | M821: 1.0, M822: 0.98, M823: 0.98 | **SUPPORTED** | 3/3 (≥0.37) | none |
| M_P3_moments_alone_nil | `phi['paramB_optI']` ∈ [-0.25, 0.25], class B | M821: 0.04, M822: 0.19, M823: 0.02 | **SUPPORTED** | 3/3 (≥0.37) | thr_pm05→PARTIAL; thr_strict→PARTIAL |
| M_A7_normmatched_L0L2_nil | `phi['randnorm_L0L2']` ∈ [-0.25, 0.25], class B | M821: -0.08, M822: 0.0, M823: -0.0 | **SUPPORTED** | 3/3 (≥0.37) | none |
| M_P6_pnz_nil | `phi['pnz']` ∈ [-0.25, 0.25], class B | M821: -0.06, M822: -0.02, M823: 0.01 | **SUPPORTED** | 3/3 (≥0.37) | none |
| M_P6_lbl_nil | `phi['lbl']` ∈ [-0.25, 0.25], class B | M821: -0.08, M822: -0.03, M823: 0.11 | **SUPPORTED** | 3/3 (≥0.37) | none |
| M_P4_vsgo_switch | `phi['vsgo']` ∈ [0.75, 1.25], class I | M821: 0.99, M822: 0.98, M823: 0.98 | **SUPPORTED** | 3/3 (≥0.37) | none |
| M_P4_vsg_partial | `phi['vsg']` ∈ [0.25, 0.75] | M821: 0.97, M822: 0.97, M823: 0.99 | **CONTRADICTED** | 0/3 (≥0.00) | none |
| M_P4_vsgo_exceeds_vsg | `phi['vsgo'] - phi['vsg']` ∈ [0.2, 1.5] | M821: 0.01, M822: 0.01, M823: -0.01 | **CONTRADICTED** | 0/3 (≥0.00) | delta20→PARTIAL |
| M_P4_vrs_switch | `phi['vrs']` ∈ [0.75, 1.25], class I | M821: 1.0, M822: 0.99, M823: 1.0 | **SUPPORTED** | 3/3 (≥0.37) | none |
| M_P5_frz_nil | `phi['frz']` ∈ [-0.25, 0.25], class B | M821: -0.05, M822: -0.02, M823: 0.04 | **SUPPORTED** | 3/3 (≥0.37) | none |
| M_P5_frzp_nil | `phi['frzp']` ∈ [-0.25, 0.25], class B | M821: -0.03, M822: 0.06, M823: 0.05 | **SUPPORTED** | 3/3 (≥0.37) | none |
| M_P7_vsgo_concordant | category BOTH (switch pairs) | M821: BOTH, M822: BOTH, M823: BOTH | **SUPPORTED** | 3/3 (≥0.37) | none |

**Totals.** 27 SUPPORTED, 5 PARTIAL, 3 CONTRADICTED, 0 INCONCLUSIVE.

These 35 items are **not 35 independent confirmations**:
- E_P4_vsgo_switch_sens070 duplicates E_P4_vsgo_switch;
- the two M P4 contradictions are one observation;
- E_P4_vsgo_exceeds_vsg passed only because vsg was nil, which is the opposite of the predicted mechanism.

Family-level tally (post-review D8):

| | E | M |
|---|---|---|
| SUPPORTED elimination families | P1, P2, A4-QK, A4-OV, P3, X8a, X8b, vrs: 8 | P1-L0-alone, P2-L0L2, partner sufficiency, P3, X8a, X8b, vrs, P5: 8 |
| P4 joint reading | not established | failed |
| P5 joint reading | not established (frz PARTIAL) | — |
| PARTIAL | — | P2-L0-alone |
| Concordance checks (eliminate nothing) | P7 ×2 | P7 |

**P3.** This is band inclusion with margin 0.25, not a formal equivalence test. On M the descriptive t-interval for the moments-only arm reaches 0.31.

**Initialisation seed.** Within a base the fresh-layer initialisation is fixed (L = W = 21 on E, 23 on M), and the units vary data order only. E-against-M differences are confounded with that seed and with the target layer.

**Advisory (post-review E).** `score_c19.json`'s `duplicate_composed_states` list order depends on Python set order. No verdict or value depends on it.

Descriptive per-base means and t-intervals, and per-pair E/M route components, are in `logs/conf/stats_c19.json`. With n = 3 the intervals are wide and descriptive only. No p-values are computed.

**Measurement floor.** φ moves by at most 0.008 between measurement streams, and necessity below ≈ 0.02 nats counts as "route absent". Differences between arms smaller than ≈ 0.05 are not interpreted.

## Stage-6 functional battery [frozen-addendum]

The rule was frozen in commit `8ca9db7`, after confirmation had started (see the chronology). `scripts/score_func19.py` was written after the frozen results had been read. It implements the frozen rule literally. M_F3 contains 16 models under that rule. Five INCONCLUSIVE(causal_NONE) M models fall outside the rule; they are reported separately as F3x and match too (post-review E2). The statistic is SR = KL on split-reliable-pair cases / KL on consensus cases, on stream 93 300 000.

| verdict | band | matching | opposite | ambiguous |
|---|---|---|---|---|
| E_F1 | SUPPORTED | 6/6 | 0 | 0 |
| M_F1 | SUPPORTED | 6/6 | 0 | 0 |
| E_F2 | SUPPORTED | 16/16 | 0 | 0 |
| M_F2 | SUPPORTED | 18/18 | 0 | 0 |
| E_F3 | SUPPORTED | 19/19 | 0 | 0 |
| M_F3 | SUPPORTED | 21/21 | 0 | 0 |

**Hybrids (F4, descriptive).** 12 of the 13 causally-hybrid models have a mixture-like error structure (SR 1.16–1.47). The exception is M823 I←B L0 (SR 2.40, causally closest to EARLY).

**Interpretation.**
- SR correlates 0.97 with nec_E and 0.94 with KL [expl, post-review D6].
- It is consistent with "EARLY = Bayes plus unstructured logit noise", which gives SR 2.9–3.2.
- [interp] The result is **causal implementation reorganisation with a precision difference**. Claim 2 (a different rule) and Claim 3 (a different algorithm) are not established.

## Positive, negative and failed predictions

**SUPPORTED: elimination families.** All of these are frozen replications on 3 pairs per base; M_A7 is new.
- **Transplants.** The layer-0 state is sufficient and necessary on E. On M, layer 0 alone is insufficient, and L0 combined with either L1 or L2 is sufficient. Restoring L0 and L2 together prevents the switch.
  - The QK-only and OV-only layer-0 restorations do not prevent the switch on E: no single tensor class is necessary.
- **Optimizer history.** The parameters carry the switch, and the moments alone do nothing, on both bases. M822 shows a small moment effect (0.19).
- **Random displacement (X8a).** Neither norm-matched random directions (E: L0; M: L0+L2) nor a matched training-time random walk switches either base.
- **Loss disturbance / slowed learning (X8b).** Label noise matched (E) or exceeding (M) the block's loss disturbance does not switch either base.
- **vrs (content loss).** A full switch on both bases.
- **vsgo.** A full switch on M (0.98–0.99).
- **frz / frzp on M.** Nil on both, so freezing target-layer routing learning does nothing on M.
- **frzp on E.** Partial (0.54–0.69).

**CONTRADICTED.** These are falsifications. Each contradicts a 1-pair development observation.
- **M_P4_vsg_partial.** Development gave 0.43 (hybrid). On all three fresh M pairs, vsg (value-vector gradient through T removed; forward and routing intact) gives a **full switch** (0.97–0.99).
- **M_P4_vsgo_exceeds_vsg.** Contradicted as a consequence: vsg already switches fully, so the contrast is 0.
- **E_P4_vsg_partial.** Development gave 0.35. On E, vsg is **nil on two pairs** (0.02, 0.02) and partial on one (0.68).

**PARTIAL.** These eliminate nothing.
- **E_P4_vsgo_switch** (0.69, 0.56, 0.95) and its 0.70 sensitivity reading. Predeclared to mean "not established". On E, removing V+O learning through T gives a large but mostly incomplete switch.
- **E_P5_frz_nil** (0.01, 0.01, 0.50). On E823, frz is also partial.
- **E_P7_L0_concordant.** E821 has an RPC-ambiguous switched arm.
- **M_P2_L0_restore_reduces** (0.32, 0.43, 0.75). On M823, restoring B's layer 0 alone hardly reduces the switch.

**Between-pair heterogeneity.** E823 is more susceptible than E821 and E822: vsg 0.68, frz 0.50, vsgo 0.95. On M823, layer-0 restoration is weak. With 3 pairs per base, such unit-level variation cannot be modelled. It is reported as is.
