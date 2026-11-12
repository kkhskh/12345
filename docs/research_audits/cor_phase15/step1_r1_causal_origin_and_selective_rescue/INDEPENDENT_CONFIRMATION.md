# Independent confirmation

## 1. Design and integrity

**Protocol.** FROZEN_CONFIRMATION_PROTOCOL.md, committed as be4a0da. Hashes are in `logs/FREEZE7_HASHES.txt`.

**Cohort.** 8 new P-count lineages, seeds 90–97, pretrained under the Stage-0 freeze (a688902) before any hypothesis existed.
They descend from no development parent.

**Execution.**
- 40 runs: 8 parents × {C1, injm, injn, injrand, W_bal}.
- All finished with rc = 0 (`logs/COMPUTE_LOG.txt`, 00:02–04:28 UTC).
- No confirmation outcome was examined before the last run ended.
- One pre-freeze 20-step smoke run on yP90 is disclosed in the protocol. Its outputs were deleted.

**Success rule.** C1 accuracy ≥ 0.6595 in all 8 parents (0.712–0.736). There were no failed runs, so ITT and successful-only
analyses coincide.

**Pretraining (auxiliary count accuracy).**

| parents | count accuracy |
|---|---|
| yP90–92 | 0.762–0.763 |
| yP93–97 | 1.000 |

## 2. Frozen predictions: results

Source: `logs/confirmatory7.json`, `logs/conf_probes.json`, `logs/conf_mech.json`.

| id | prediction | result | verdict |
|---|---|---|---|
| **P1** | E_R1(injm) − E_R1(C1) > 0 | mean **+0.695**, exact sign-flip p = 1/256 = **0.0039**, CI [0.476, 0.880]; positive in **8/8** | **confirmed** |
| **P2** | Δ(injm) − Δ(injrand) > 0 | mean **+0.714**, p = 0.0039, CI [0.493, 0.896] | **confirmed** |
| **P3** | selective success in ≥ 6/8 | **7/8** (yP96: Δ = +0.146 < 0.20; every preservation constraint met) | **confirmed** |
| P4 | Δ(injn) < 0 | mean worsening 0.637, p = 0.0117; worse in 7/8 | **confirmed** |
| P5 | original C1 raw Δm_R1 < 0 in ≥ 7/8 | **8/8**. The Investigation VI reversal replicates on a third cohort | **confirmed** |
| P6 | H2 linear-probe accuracy at q-sources < 0.96 in ≥ 6/8 | **8/8** (0.913–0.932; majority rate 0.912) | **confirmed** |
| P7 | readout-token-only route negative in C1, ≥ 6/8 | **8/8** (−0.33 to −0.89) | **confirmed** |

The primary pair (P1 ∧ P2) holds, and the selective criterion P3 holds. Under the frozen outcome map this is a **selective
behavioural rescue by the specified relational feature, specific against capacity**.

## 3. Per-parent results

| parent | C1 E_R1 | injm E_R1 | Δ injm | Δ injrand | Δ injn | Δ W_bal | injm \|E\| R2∪R3 (C1) | injm KL (C1) | injm acc (C1) | success |
|---|---|---|---|---|---|---|---|---|---|---|
| yP90 | −0.636 | +0.316 | +0.952 | +0.002 | −0.876 | +0.568 | 0.598 (1.623) | 0.054 (0.103) | 0.727 (0.714) | yes |
| yP91 | −0.445 | +0.425 | +0.870 | −0.011 | −1.777 | +0.448 | 0.727 (1.566) | 0.075 (0.127) | 0.723 (0.712) | yes |
| yP92 | −0.494 | +0.353 | +0.846 | +0.008 | −0.455 | +0.443 | 0.573 (1.783) | 0.044 (0.085) | 0.728 (0.723) | yes |
| yP93 | −1.368 | −0.367 | +1.001 | +0.008 | +0.311 | +1.889 | 0.436 (0.951) | 0.023 (0.039) | 0.732 (0.732) | yes |
| yP94 | −0.852 | −0.131 | +0.721 | −0.006 | −0.857 | +1.401 | 0.541 (1.074) | 0.036 (0.067) | 0.732 (0.729) | yes |
| yP95 | −0.491 | −0.215 | +0.276 | +0.000 | −1.026 | +0.508 | 0.470 (1.130) | 0.042 (0.079) | 0.731 (0.721) | yes |
| yP96 | −0.446 | −0.300 | +0.145 | −0.028 | −0.035 | +0.662 | 0.297 (0.371) | 0.012 (0.018) | 0.735 (0.732) | no |
| yP97 | −1.061 | −0.310 | +0.751 | −0.124 | −0.382 | +1.282 | 0.467 (1.325) | 0.023 (0.045) | 0.735 (0.736) | yes |

**Signs after repair.** The raw R1 response Δm_R1 under injm is positive in 5/8 parents (+0.01 to +0.65) and slightly
negative in 3/8 (−0.07 to −0.14). Under C1 all 8 are negative (mean −0.50). The repair therefore removes most of the error
and the sign reversal in most parents, but not in every parent.

**Expected aggregate impact at natural prevalence.** R1-type decisions are about 2.8 % of natural decisions, and the repair
changes the R1 excess by about 0.7 log-ratio units. The aggregate KL improvement (0.070 → 0.039) is larger than the R1 cells
alone could produce. It comes mostly from R2/R3 (|E| 1.23 → 0.51), so the injected relation helps all two-perfect states.

## 4. Generic baseline (W_bal, descriptive, no frozen prediction)

- **W_bal raises R1 by more than injm** (mean +0.90 vs +0.70).
- **It is not selective:** R2 E +1.46, R3 E +1.24, KL 0.149 (C1 0.070), accuracy 0.691 (C1 0.725).
- **Score:** 0/8 parents meet the frozen selective thresholds.

Generic reweighting therefore does not account for the repair's *selectivity*. It does show that R1 alone can be moved by the
loss composition.

## 5. Exploratory (not frozen) observations on the cohort

- The repair gain correlates negatively with the MLP-probe accessibility of H2 at the q-sources, r = −0.80 over 8 parents.
  yP96, the least-improved parent, has the most accessible H2 (MLP 0.977; readout-token linear 0.961). Its original error was
  also among the smallest.
- Accessibility does not predict the original error: r(H2 MLP, C1 E_R1) = −0.14.
- Both observations are post hoc and n = 8.

## 6. Scope

- **Population.** These are new P-count lineages from the same pipeline: architecture, generator, pretraining and transfer
  code. That is replication across independent seeds and lineages, not across implementations.
- **No T1 panel.** No T1 confirmation panel was trained, so the T1–P contrasts in the probes and routes (P6, P7) rest on
  development T1 parents.
- **What the rescue shows.** It supplies a hand-specified, input-derived relational feature. It shows that this feature is
  sufficient for a selective correction under the original learner. It does not show that the original models compute m
  incorrectly, nor that a network would discover m unaided.
