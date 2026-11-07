# Reviewer response and revisions (internal)

No new experiments were added. The following were applied to the existing evidence.

## Addressed after the independent ZIP audit

- Section 4.4 no longer treats Pearson \(r\approx 0.998\) on raw \(Y\) as a direction correlation. The manuscript now reports mean cosine 0.792 (learned vs oracle) and 0.850 (learned vs mean-state).
- Figure 2 is generated from `verified_data/fixed_js_verified_data.csv` (336 interventions, 48 pairs). Probe max abs \(1.19\times 10^{-14}\); JS error \(1.23\times 10^{-6}\); \(\Delta_{\mathrm{wrong}}=+0.02655064\); \(\Delta_{\mathrm{random}}=+0.02296572\).
- Figure 1 now routes both rowspace and nullspace into the frozen model.
- Related work includes Jacopin (2026) and Lin et al. (2024). Novelty is still not priority-proven.
- Bootstrap counts distinguished: 2000 for SPEC/NST, 5000 for FRESH/VNF.
- FRESH 24 identical-state prefixes are a cohort fact; presence in the selected 32 triplets is NOT VERIFIED.

## Previously addressed from current artifacts

- Title and abstract scoped to Othello-GPT and this probe; COR removed from the manuscript body except the preserved official letter `OBSTRUCTION_READY = NO`.
- COST protocol deviation kept as a separate inconclusive letter.
- Absolute negative \(Y\) stated in abstract and Section 4.4.
- Equal JS ≠ equal L2 stated with recorded L2 means.
- Donor-free defined as evaluation-time.
- Legal mask disclosed as an input.
- FRESH 60% gate failure and VNF near-zero interaction in the main results, not an appendix-only dump.
- Gradient 7/48 excluded from full-sample means.
- STAGE share described as a ratio of means.
- Figure 2 explicitly blocked rather than faked from copied numbers.

## Unresolved (need data, not prose)

- Level 1 recomputation from Drive NPZ (fatal for verification, Reviewer 2).
- Local hash of checkpoint and probe files.
- Pair-level plots of probe-probability invariance (Figure 2).
- Independent confirmation split (forbidden to open here).
- On-manifold checks (no archived density model).
- Second model or task (out of scope).
- Human authors, affiliations, and venue-specific AI disclosure signed by humans.

## What we did not do in response to reviews

We did not retrain a stronger map, search \(\rho,\nu\), or reopen confirmation. We did not convert the switching failure into a “partial success” narrative. We did not claim DAS faithfulness.
