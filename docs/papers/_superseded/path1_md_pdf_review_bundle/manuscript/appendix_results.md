# Technical appendix: additional results and provenance

## B.1 Recorded 48-pair numbers (not locally recomputed)

See `reported_numbers.json` and `05_statistical_recomputation.md`.

SPEC (equal L2): \(\Delta_{\mathrm{specific}}=+0.391627\), CI about \([0.317,0.476]\); random contrast \(+0.708584\); \(Y_{\mathrm{correct}}=+0.143649\).

Fixed-JS: \(\Delta_{\mathrm{wrong}}=+0.02655\) \([0.0179,0.0357]\); \(\Delta_{\mathrm{random}}=+0.02297\) \([0.0103,0.0360]\); L2 5.30 / 6.37 / 12.27.

NST: pooled stdout \(\Delta_{\mathrm{wrong}}=0.020057\) \([0.010791,0.029784]\); \(Y_{\mathrm{learned}}=-0.582110\); \(Y_{\mathrm{editor}}=-0.635823\).

## B.2 FRESH contingency

Both positive 13, T-only 13, W-only 5, neither 1. Gate uses both-positive fraction, not the sum of mean \(A_T\) and \(A_W\).

FRESH ablation means from VNF Stage 1 stdout (same frozen maps, FRESH triplets):

| Map | \(A_T\) | \(A_W\) | both-positive |
|---|---|---|---|
| FULL | 0.028500 | 0.018561 | 0.40625 |
| LEGAL_ONLY | 0.024770 | 0.018623 | 0.375 |
| BOARD_ONLY | 0.024299 | 0.006683 | 0.375 |
| STAGE_ONLY | 0.0 | 0.0 | 0.0 |

STAGE_ONLY both-arms zero on this natural-anchor switching estimand is expected if stage features are shared across T and W when occupancy/player match the design; it is not a new gate.

## B.3 VNF controls

Random-null and label-shuffle did not reproduce a passing interaction (`control_reproduces=0` in the Colab summary). That does not convert the near-zero \(I_T,I_W\) into a subtle positive mechanism.

## B.4 What is not in the appendix

Sealed confirmation outcomes. 08R3. Obstruction energies. New \(\alpha\) or \(k\). Pair-level CSVs (absent locally).
