# FREEZE_XVIII

- **Frozen design.** PRE_EXPERIMENT_IDENTIFIABILITY_REVIEW.md §10, comprising design v2 and §10.8–§10.11 (the addenda, declarations, development results and dose reading).
- **Frozen files.** Listed with their SHA-256 in `logs/FREEZE_XVIII_HASHES.txt`. These include the harness, the measurement, the scorers, the preflight (ALL_PASS), the design JSON and queues, the parent rule, the continuation and H3 scripts, the unit checks, and every sibling module imported, as listed in `logs/DEPENDENCIES.json`.
- **Confirmation cohort.**
  - Parent uP160; bases E (21/21, T = own0) and M (23/23, T = own2); fresh data orders D = 811, 812 and 813.
  - Measurement stream 95_400_000, never used before in COR.
  - Arms: baseline, block, vrg, gsg, gsv, sha-hi-rolled, sha-lo-real, gof b = 2.3. That is 48 runs.
- **Parents.** Screening uses uP162–uP165, seeds 20–23 (24 and 25 if needed), D = 811, stream 95_500_000. The arms run on D = 812, with stream 95_600_000.
- **Orchestration.** `scripts/run_all18.sh`.
- **Integrity rule.** Until stage 1 has been scored, the investigator monitors only `logs/COMPUTE_LOG.txt` and the existence of files. Nothing is read from `logs/measure_*.jsonl`, the lane outputs (`logs/run/`), the checkpoints or `window_log`/`slog` files of confirmation runs, and nothing is line-counted. The scorer is the first reader.
