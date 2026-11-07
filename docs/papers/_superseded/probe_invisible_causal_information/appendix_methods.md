# Technical appendix: methods

This appendix restates implementation details that a reviewer needs and that would clutter the main prose. It does not add experiments.

## A.1 Scoring

Legal sets are the simulator lists stored with each prefix, not a re-derived Othello engine at scoring time. `legality_sets(rec_legal, tgt_legal)` returns became-legal, became-illegal, stayed-legal, stayed-illegal. \(G\) and \(Q\) sum softmax mass on the corresponding move tokens via `total_legal_mass`. \(Y=G-Q\). JS uses `kl_row` with natural log.

## A.2 Probe contrasts

For each square, present classes come from the sklearn pack. Binary squares contribute one contrast; three-class squares contribute two. Constant squares contribute none. SVD uses `full_matrices=False`. Primary rank is \(\#\{s_i > 10^{-8}s_{\max}\}\). Sensitivity at \(10^{-6}\) and \(10^{-10}\) is recorded in FIBER but the paper uses the primary split.

## A.3 Equal-JS search

Grid: `0.003125, 0.00625, 0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 1.5, 2.0, 4.0, 8.0`. First point with \(\mathrm{JS}\ge 0.01\), then 16 bisections against the previous point. Failures remain in the attempt record. Coverage is complete-pair count, not a complete-case mean that drops failures silently.

## A.4 Symbolic features for the 64-D map

- 64×3 player-relative occupancy one-hot
- 64 legal-mask bits
- player-is-black
- occupied/64

PCA is fit on training \(P_N h\) only. Ridge is train-game grouped CV. Frozen \(k=64\), \(\alpha=10\), no refit on evaluation games.

## A.5 Editor A

Full residual, budget 16, 40 steps, lr 0.4, \(\lambda=0.01\), \(\eta=1\), \(\mu=0\). 08R2 validation letter remains NEGATIVE.

## A.6 Seeds

See `04_dataset_split_and_leakage_audit.md`. Confirmation games are rejected in code by `require_no_confirmation`.

## A.7 Hardware

NOT AVAILABLE FROM ARCHIVED ARTIFACTS in this package. Colab was used for real-model stages. Local unit tests do not require a GPU.
