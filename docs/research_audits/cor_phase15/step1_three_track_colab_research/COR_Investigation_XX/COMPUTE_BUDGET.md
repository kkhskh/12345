# Compute budget

GPU-hours are **not** precomputed here: they depend on the GPU type, its utilisation and the Colab allocation. The preflight
(notebook 00, step 6) measures on your runtime the total fine-tuning steps/s at concurrency 1, 2, 4, 8 (excluding JIT compilation),
the pretraining step rate, the measurement and Track-B battery wall times, the snapshot size and the GPU memory per process; it
chooses the concurrency and prints the hours, storage and memory of every preset (`cor_research/jobs/budget.py`), and it refuses
to start if the estimated storage exceeds 90% of the free space of the run root. Colab compute units are not converted to GPU-hours.

Exact experiment counts, taken from the real planner on a synthetic eligibility file. *max design* = every preregistered parent
valid (except uP162, which fails the frozen pretraining-validity threshold) and both bases eligible; *half units* = the
historical-eligibility scenario. Realised counts are known after screening and are reported by the registry.

| preset | scenario | parents (fresh) | screening runs | units | window runs | compositions | continuations (to 4000) | stability runs (to 12000) | measurements | Track B jobs | training steps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pilot | max design | 1 (1) | 8 | 2 | 8 | 6 | 10 | 0 | 62 | 9 | 52,488 |
| pilot | half units | 1 (1) | 8 | 1 | 4 | 3 | 5 | 0 | 35 | 5 | 32,588 |
| standard | max design | 9 (4) | 128 | 16 | 320 | 576 | 768 | 0 | 4096 | 264 | 3,194,208 |
| standard | half units | 9 (4) | 128 | 8 | 160 | 288 | 384 | 0 | 2112 | 132 | 1,680,608 |
| extended | max design | 13 (8) | 192 | 24 | 720 | 1296 | 1728 | 576 | 10848 | 588 | 11,675,712 |
| extended | half units | 13 (8) | 192 | 12 | 360 | 648 | 864 | 288 | 5520 | 294 | 5,966,112 |
| high | max design | 17 (12) | 256 | 32 | 960 | 2112 | 2688 | 3072 | 23296 | 784 | 35,501,216 |
| high | half units | 17 (12) | 256 | 16 | 480 | 1056 | 1344 | 1536 | 11776 | 392 | 17,923,616 |

Schedules (unchanged historical recipe): pretraining 3000 steps (fresh parents); screening baselines to 1211 on two data orders
(901, 902; 4000-step schedule); windows 100 steps; continuations 100 -> 4000 (snapshots include 150, 300 and 1211); stability
(Extended: anchors; High: all arms) 4000 -> 12000 with a restarted warmup-cosine at peak 1e-4, plus a full-LR (1e-3) positive control
for the anchors.

Storage: (measured snapshot size) x (about 26 snapshots per anchor / screening / stability run, 4 per pruned non-anchor arm, 8 per window).
Memory: one process per job; the model (3 layers, d = 64) needs well under 1 GB of GPU memory; the concurrency is the measured
throughput optimum within 85% of device memory.

## What each preset can and cannot support

- **pilot**: Pilot: end-to-end technical validation on separate material (parent nP3901, data order 2901). NOT confirmation evidence.. Can: the pipeline runs end to end on this runtime; measured throughput; storage. Cannot: any scientific claim (verdicts forced to TECHNICAL_ONLY).
- **standard**: Standard research. Can: parent-level Track A verdicts if >= 4 parents are evaluable per base (fresh-only verdicts need >= 4 fresh parents); Track B indicative (< 6 parents). Cannot: persistence beyond 4000 steps; generalisation to other architectures/tasks.
- **extended**: Extended research. Can: as standard, with 6 data orders per unit, fresh-only verdicts, and the stability continuations (low-LR + full-LR control) for anchors. Cannot: other architectures/tasks.
- **high**: High-compute research. Can: as extended, plus exploratory QK/OV splits and interpolation, stability for every arm. Cannot: other architectures/tasks (no validated alternative architecture exists in COR).

If the budget runs out, the runner stops launching jobs, keeps every checkpoint, and the reports state INCOMPLETE; no schedule is shortened.
Priority order (mandate 8.5): Track C mathematics -> pretraining/screening -> windows -> Track A anchors (fullB/fullI) -> other arms -> Track B -> stability.

**Power caveat (tests/test_verdict_oc.py, review A round 2 N3).** With every pair evaluable, a strong Track A effect gives GENERALIZES with
probability ≥ 0.97 at 4–12 parents. If 20% of pairs are lost (non-native anchors, failed switches), the Standard preset (4 data orders) at 4
parents reaches GENERALIZES only 0.42 of the time (INCONCLUSIVE 0.55); Extended (6 data orders) restores about 0.90. Track B equivalence is
reachable at 4–8 parents only if the between-parent SD of HB_c is ≲ 0.01–0.02 (unknown before the run).
