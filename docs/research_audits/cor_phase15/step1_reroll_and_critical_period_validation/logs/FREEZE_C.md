# XIII-E freeze C: carrier geometry and dimensionality, predicted for the held-out carrier unit

Units:
- **Development:** C-C1 (uP160 s23 early01). Its audit is `logs/c_audit.json`, which was complete when this freeze was written.
- **Held-out unit:** C-C2 (uP165 s21 early01). Its XIII-E audit (`scripts/c_audit.py`, stream 96.2 M) is gated on `logs/FREEZE_C.flag`, which is created only after this commit.

XIII-D measured SUFF, SRCF, reverse and the trace on C-C2. None of those quantities is predicted here. The predicted quantities (the k-sweep, decoding from projections, signal and noise variance, principal-angle overlaps, and variance-matched random controls) have never been measured on C-C2.

Predictions (exact clauses in `scripts/score_C.py`, which fails closed):

- **R5 (dimensionality).**
  - Clauses:
    - a single DAS direction (k = 1) carries at least 85% of the k = 3 sufficiency;
    - it is seed-stable (mean principal cosine ≥ 0.9);
    - the k = 3 subspace is not seed-stable (≤ 0.8).
  - Development: the k = 1/k = 3 ratio was 0.96, and stability was 0.995 at k = 1 against 0.51 at k = 3.
- **R6 (decoder geometry).**
  - Clauses:
    - the 3-d decoder-filter span decodes m from the site projection about as well as the encoding span (ratio ≥ 0.9);
    - interchange along the decoder span transfers nothing (≤ 0.1), while the encoding span transfers (≥ 0.5);
    - variance-matched random subspaces fail (≤ 0.2);
    - the encoding span carries at least 5 times the signal variance of the decoder span;
    - DAS overlaps the encoding span (≥ 0.2) and not the decoder span (≤ 0.15).
  - Development values:
    - decoding R²: 0.859 against 0.859;
    - SUFF: −0.0 (decoder), 0.835 (encoding), 0.07–0.16 (variance-matched random);
    - signal variance: 15.7 against 0.95;
    - overlaps: 0.40 and 0.05.

Preflight (`logs/preflight_C.json`): P0 and N1–N4 all OK.
