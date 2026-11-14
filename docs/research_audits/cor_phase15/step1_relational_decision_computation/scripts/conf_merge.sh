#!/bin/bash
# Merge per-parent confirmation outputs (waits for every parent's outputs), then run the frozen scorer.
cd "$(dirname "$0")/.."
for P in P160 P161 P162 P163 P164 P165; do
  for a in l2_scan l2_mix_causal l2_mediation score_contrast l1abl_mix decoy_disc; do while [ ! -f logs/${a}_conf_$P.json ]; do sleep 30; done; done
  while [ ! -f logs/adapt_sel_conf/u${P}_full_c1_s4000_seed20_eval_conf13.json ]; do sleep 30; done
done
python3 - <<'PY'
import json, glob
for a in ("l2_scan", "l2_mix_causal", "l2_mediation", "score_contrast", "l1abl_mix", "decoy_disc"):
    m = {}
    for f in sorted(glob.glob(f"logs/{a}_conf_P*.json")): m.update({k: v for k, v in json.load(open(f)).items() if k.startswith("u")})
    json.dump(m, open(f"logs/{a}_conf.json", "w"))
PY
python3 scripts/score_xiii.py
