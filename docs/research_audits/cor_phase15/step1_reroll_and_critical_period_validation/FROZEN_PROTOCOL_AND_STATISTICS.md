# Frozen protocol and statistics (Investigation XIII-E)

COR Phase 1.5, Step 1 — this is not a validation of COR.

## Freeze chronology

All commits are on branch `cor-phase15-step1-reroll-and-critical-period-validation`, which starts from XIII-D HEAD `9f1bb83`.

| freeze | commit | time (UTC) | contents | first governed output |
|---|---|---|---|---|
| A-1 | `2063085` | 01:09:17 | Re-roll design, 64-run seed matrix, anchors, continuation set and rule, measures, training queue, harness changes, initialisation audit, benchmark | Training started 01:09:19 |
| A-2 | `a4f66b7` | 01:37:19 | R1-W, R1-L predictions; power simulation; fail-closed `score_A.py` + preflight (P0, P1, N1–N7); `select_continuation.py`; monitor; manifest. R2 deliberately not frozen | First re-roll measurement started ≈ 11:05 (gated on all 64 runs) |
| B | `b710b38` | 01:59:36 | B design (41 runs), calibration, `ft_factorial_b.py`, R3/R4 fail-closed scorer + preflight (P0, N1–N7), queues | B training started 01:59:48 |
| C | `87177ad` | 04:04:25 | R5/R6 for the held-out carrier unit C-C2 (development: C-C1), fail-closed `score_C.py` + preflight (P0, N1–N4) | C-C2 audit gated on a flag written after the commit |

**Monitoring separation.**
- During training, progress was followed only through `scripts/monitor_A.py`, which counts finished runs, and through file counts.
- No metric job for a confirmation run was queued before its freeze.
- Each scorer was run once, after all its inputs existed:
  - A: 14:3x;
  - B: 14:2x;
  - C: 05:31.
- No scorer can produce a partial verdict. Each refuses with exit code 2 and writes nothing when any input is missing, mismatched or malformed (preflights N1–N7 / N4–N7 / N3–N4).
- There were no interim looks.

**Disclosed protocol events (`logs/COMPUTE_LOG.txt`).**
1. **01:38Z.** The B interventions were briefly added to `ft_factorial.py`, then moved to `ft_factorial_b.py`. The frozen hash was restored within about 2 minutes, and no re-roll run started in that window.
2. **03:4xZ.** The C-C2 audit line in `queue_C.txt` was gated in place on the freeze-C flag before it was read. The gate is a scheduling change only.
3. **04:04Z.** The freeze-C commit accidentally included some in-progress training checkpoint files. They had not been inspected, and they were removed from the index in the next commit. *Post-review (Review C):* in-progress checkpoints were in fact committed in three freeze commits (a4f66b7, b710b38, 87177ad), and `resume.pkl` files remain in their history; see COMPUTE_LOG errata. None was read before scoring.
4. **Resources.** Concurrency was set from a measured benchmark: 2–4 training processes, about 0.8 GB each, aggregate throughput about 4 steps/s. Measurement jobs were gated, and analysis was niced. **There were no out-of-memory kills (0 jobs with rc = 137)**, and every lane job ended with rc = 0.

## Experiment inventory

**Training runs.**

| set | runs | length |
|---|---|---|
| Re-roll screen | 64 | to step 1211 |
| Re-roll continuations | 10 | to step 4000 |
| B short | 33 | to step 1211 |
| B late | 8 | to step 4000 |
| B continuations | 4 | to step 4000 |
| **Total** | **119** | |

Each training run started from one of the two pretrained parents (uP160, uP161).

**Analyses.**
- C: carrier audit of 2 existing XIII-D adapted models.
- C1: ideal references, numerical only.
- B4: route measurements on 4 + 2 runs, descriptive.

**Excluded or not completed.**
- One strong-noise B run (KL 0.051) failed the competence check and was excluded by the frozen rule.
- One E161 re-roll run was not competent at step 1211. It was counted as scored, since the R1 statistic uses continuous mix_mass for all runs.
- Nothing was dropped.

## Verdicts

| prediction | statement | verdict | deciding values |
|---|---|---|---|
| R1-W | No transferable component of a re-drawn readout map (re-roll null) | **FALSIFIED** | mean ρ 0.65 (uP160 0.55, uP161 0.76); p = 0.006 |
| R1-L | No transferable component of a re-drawn layer-2 initialisation | **CONFIRMED** | ρ −0.02, p = 0.54; upper bound 0.52 < 0.6 |
| R2 | Joint initialisation statistic | **not frozen** | development found no consistent statistic; exploratory R_route AUC 0.59 |
| R3 | 100-step real masks flip ≥ 2/3 on both bases; matched noise ≤ 1/3; sham bitwise identical | **CONFIRMED** | real 3/3 + 3/3; noise 0/3 + 0/3; sham identical |
| R4 | Specifically early: late matched ≤ 2/8 flips; strong early noise ≤ 1/3; dose | **FALSIFIED** | late 4/8 (all route-specific late blocks → hybrid; all late noise unchanged); strong noise 2/2 competent |
| R5 | k = 1 ≥ 85% of k = 3 SUFF, with k = 1 stable and k = 3 unstable, on held-out C-C2 | **FALSIFIED** | ratio 0.67; C-C2 needs 2 stable directions |
| R6 | Decoder decodes like the encoder but does not transmit; variance-matched controls fail; signal ratio ≥ 5; overlaps | **CONFIRMED** | 1.11; 0.024; 0.75; 0.068; 7.7; 0.58; 0.066 |

**Summary: 6 frozen predictions; 3 confirmed (R1-L, R3, R6), 3 falsified (R1-W, R4, R5), none inconclusive.**

How the verdict types sort:
- **Discriminating confirmations:** R3 and R6 (each compares a real effect with controls).
- **Null-model confirmation:** R1-L shows compatibility with re-roll, with a bound. It does not show equivalence.
- **Non-diagnostic passes:** none were counted as confirmations.

## Statistical summaries

| quantity | estimate | interval / uncertainty | n and dependence |
|---|---|---|---|
| P(early), uP160 re-draws | 15/32 = 0.47 | 95% CP [0.29, 0.65] | 4 cells of 8 sharing a parent; pooled descriptively |
| P(early), uP161 re-draws | 0/32 | [0, 0.11] | as above |
| P(early) per anchor × type | 0–0.62 | each CP width ≈ 0.6–0.7 | n = 8 |
| early-anchor minus mixture-anchor difference | −0.00 to +0.25 | Newcombe up to [−0.30, +0.59] | per parent |
| homogeneity across anchors | rejected | MC p = 0.003 (W), 0.0015 (L) | driven by parent |
| R1-W transfer | ρ 0.65 | p = 0.006; bootstrap upper 0.86 | 8 seeds × 2 parents |
| R1-L transfer | ρ −0.02 | p = 0.54; upper 0.52 | as above |
| step-1211 → step-4000 class stability | 9/10 | the 1 change was a borderline run | preselected 8 + 2 by rule |
| B: real 100-step mask flips | 6/6 | — | 3 data orders × 2 bases, one parent |
| B: matched-noise flips | 0/6 | — | as above |
| B: late route-specific → hybrid | 4/4 | — | 2 data orders × 2 bases |
| C: k = 1/k = 3 SUFF ratio | 0.96 (C-C1), 0.67 (C-C2) | — | 2 models |

**Threshold sensitivity.** Shifting the class thresholds by ±0.05 changes at most one re-roll count per cell (`score_A.json` → `sensitivity`). No R1 verdict depends on class thresholds, because R1 uses continuous mix_mass.

**Parent-level uncertainty.** The re-roll result comes from 2 parents and Workstream B from 1. Statements about "parents in general" cannot be made.

**Non-significance is not equivalence.** R1-L bounds a transferable L effect below roughly a latent correlation of 0.6 at the frozen power; it does not show the effect is zero. The early-versus-mixture anchor differences are bounded only to about ±0.5.

## Post-review errata

- Count slips in frozen records (left unedited because they are hash-listed): FREEZE_A2 "2 of 64 finished" should be 4; FREEZE_B "34 short-window runs" should be 33.
- `logs/c_audit.json` frozen hash fails because C-C2 was appended; C-C1 block unchanged.
- The pre-freeze power simulation script was not preserved; `scripts/a_power_sim.py` reconstructs it (`logs/A_power_reproduced.json`) and adds heterogeneous-transfer scenarios.
- The A/B scorers would silently overwrite a duplicated metric line (none exist); no scorer verifies hashes at run time; `score_A.py` checks the continuation rule only by count (Review C re-derived it independently).
- Intervals treat cells sharing seeds as independent (anti-conservative for pooled and difference quantities).
- One post-review exploratory measurement (late-run trajectories, `logs/unit_metrics_Blate_traj.jsonl`) was added; it changes no frozen verdict.
- Archived checkpoints are limited to the steps any analysis reads plus all optimiser snapshots (REPRODUCTION.md); about 1.1 GB of unmeasured intermediate saves and all `resume.pkl` files are excluded and are regenerable deterministically.
