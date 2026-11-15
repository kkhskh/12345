# XIII-D freeze 2: protocol

* Predictions (full text, units, thresholds, measurements): `scripts/predictions_13d_f2.json` (P1, P3, P4, P6, P7, P8).
* Scoring implementation: `scripts/score_f2.py` (reads only `logs/*_f2*` outputs; verdict rules in its docstring and in the predictions file).
* Preflight: `scripts/preflight_f2.py` -> `logs/preflight_f2.json`: P0 and N1-N10 all OK (synthetic inputs built from development outputs
  relabelled to confirmation unit names; no confirmation output read).
* Population: `logs/FREEZE2_POPULATION.md` parts 1 and 2.
* Training queue: `logs/queue_f2A_train.txt` (A-C1..A-C4). Analysis queue: `logs/queue_f2_analysis.txt`. Gates: `scripts/when_ready.sh`,
  `scripts/when_file.sh`. Carrier checkpoint override `CA_CKDIR` added to `scripts/c_carrier.py` (selects `checkpoints/adapt4_carrier`).
* Recorded before freeze: the C-C1 adaptation training summary line was printed by a monitoring grep (COMPUTE_LOG 18:44:55Z); the A-C2
  claim-key collision was fixed by an inert `FX_CONF=1` prefix (COMPUTE_LOG).
* Hashes: `logs/FREEZE2_HASHES.txt` (sha256 of every file above). This commit precedes every freeze-2 confirmation output.
