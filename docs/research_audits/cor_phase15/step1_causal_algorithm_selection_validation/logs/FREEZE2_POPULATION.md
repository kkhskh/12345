# XIII-D freeze-2 confirmation population, part 1 (declared BEFORE these runs are launched; predictions are frozen later, before any output is opened)

| id | run | role |
|---|---|---|
| A-C1 | uP161 L21 / **W22** / D21 | W-only change relative to the diagonal uP161 21/21/21 (mixture) |
| A-C2 | uP161 L22 / **W21** / D22 | W-only change relative to the diagonal uP161 22/22/22 (early) |
| A-C3 | uP160 L21 / **W23** / D21 | W-only change relative to uP160 21/21/21 (early) |
| A-C4 | uP160 L23 / **W21** / D23 | W-only change relative to uP160 23/23/23 (mixture) |
| C-C1 | early01 adaptation (β = 0.6, adapt4) of uP160 s23 | carrier confirmation unit (launched 18:08Z, not analysed) |
| C-C2 | early01 adaptation of uP165 s21 | carrier confirmation unit (queued, not analysed) |

The uP161 W-only cells were first queued as development runs and were re-designated as confirmation before any of them started (no output
exists). Development for workstream A uses the completed uP163 L/W/D 2^3 and the readout-map transform screen. Further freeze-2 units
(readout-map transform confirmation, accessibility arms, E-workstream analysis units) will be declared in a second part before they are run.

## Part 2 (declared 2026-11-15 ~19:00Z, with the freeze-2 predictions; none of these analyses has been run)

| prediction | units | analysis |
|---|---|---|
| P6 E1 eligibility | uP164 s21 (early), uP165 s20 (mixture) | d_das2.py, site ALLX2, k = 8, stream 92 700 000 |
| P6 E2 rows | uP161 s20, uP165 s23 | d_rows.py, stream 92 600 000 |
| P7 E4 | uP164 s22, uP163 s20, uP160 s20 | e4_noninv.py, stream 92 800 000 |
| P8 E3 routes | uP165 s20 (mixture), uP163 s22 (early), uP164 s21 (early) | d_routes2.py, stream 92 900 000 |
| P3/P4 carrier | C-C1, C-C2 (part 1) | c_carrier.py, stream 92 200 000, checkpoints/adapt4_carrier |
| P1 W-only | A-C1..A-C4 (part 1) | unit_metrics.py final, stream 92 100 000 |

Class labels used by P8 (mixture/early) are taken from logs/known_outcomes.json (measured in earlier packages). Some of these natural models
were analysed in earlier packages (XIII-B/XIII-C) with other analyses; none appears in any XIII-D development output.
P1b (readout-map property) and P5 (accessibility) are deferred to freeze 3.
