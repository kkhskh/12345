# XIII-E freeze A-1: re-roll design and training commands (committed before any re-roll run starts)

* Design: `logs/A_design.json`. It contains the 4 anchors (selected BY known outcome, disclosed), seed lists W' = 601..608 and L' = 701..708 (same
  lists for every anchor: crossed design), all 64 runs, the stop step (1211), the preselected continuation set (W' = 601 and L' = 701 for
  every anchor: 8 runs) and the rule for 2 extra borderline continuations, and the measurement definitions (stream 95.1 M).
* Training queue: `logs/queue_A.txt` (64 lines, anchors interleaved). Harness: `scripts/ft_factorial.py` (XIII-E change: STOP runs resume
  and write stop_done.json; FX_SAVEOPT_AT saves the optimiser only at listed steps). Concurrency: 2 training lanes (benchmark:
  aggregate 3.3 / 4.1 / 3.9 steps/s for 1 / 2 / 3 concurrent runs; peak RSS 0.77-0.79 GB per run; logs/BENCHMARK.json).
* Initialisation audit: `scripts/a2_init_audit.py` -> `logs/a2_init_audit.json`: W_A uses sub-key 5 and the layer-2 tensors sub-keys 20-25 of
  the same jax split for a given seed; matched and mismatched seed pairs give indistinguishable joint W-L statistics (KS p 0.27-0.99) and no
  tensor re-use; D drives a separate numpy generator. "Matching seeds" therefore carries no statistical coupling.
* Statistical analysis, predictions and the fail-closed scorer are frozen in freeze A-2, which will be committed before ANY measurement of a
  re-roll run is made (no measurement job is queued until then); training progress is monitored only through step counts.
