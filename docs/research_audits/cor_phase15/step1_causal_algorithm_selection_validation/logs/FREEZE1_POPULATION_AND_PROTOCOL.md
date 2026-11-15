# XIII-D FREEZE 1 (Workstream B) -- committed before any F1 run is launched

Cohort, predictions, competence and class rules, and the measurement streams are in `scripts/predictions_13d_f1.json`. Scorer:
`scripts/score_f1.py`. Preflight: `scripts/preflight_f1.py`, whose six checks (P0, N1–N5) are all OK; results in `logs/preflight_f1.json`.
Training queue: `logs/queue_f1_train.txt`. Analysis queue: `logs/queue_f1_analysis.txt`.

The two base reruns (uP160 23/23/23 and 21/21/21 with FX_SAVEOPT=1) are full-length replays of XIII-B seeds 23 and 21. They provide the
branch points for F1e/F1f, are verification runs rather than confirmation units, and were launched at 17:36Z, before this freeze.

No output of any F1 run exists at the time of this commit. A second freeze (Workstreams A, C, D, E) will follow.
