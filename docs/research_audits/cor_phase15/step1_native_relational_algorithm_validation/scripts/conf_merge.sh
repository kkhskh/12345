#!/bin/bash
# Merge per-parent confirmation outputs into logs/<analysis>_conf.json and run the component comparison.
cd "$(dirname "$0")/.."
# Amendment 1: wait until every pretrained parent's per-parent outputs exist
for P in P150 P151 P152 P153 P154 P155 P156; do [ -f checkpoints/z$P/pre.json ] || continue
  for a in factorial alt_ablation circuit_l1 restore disc_ident vs_linear dyn_mech; do while [ ! -f logs/${a}_conf_$P.json ]; do sleep 30; done; done
  while [ ! -f logs/traj_alg_xii_$P.json ]; do sleep 30; done
done
python3 - <<'PY'
import json, glob, os
for a in ("factorial", "alt_ablation", "circuit_l1", "restore", "disc_ident", "vs_linear", "dyn_mech"):
    m = {}
    for f in sorted(glob.glob(f"logs/{a}_conf_P*.json")):
        d = json.load(open(f)); m.update({k: v for k, v in d.items() if k not in ("sanity_dS", "n_pairs")})
        if "sanity_dS" in d: m.setdefault("sanity_dS", {})[os.path.basename(f)] = d["sanity_dS"]
    json.dump(m, open(f"logs/{a}_conf.json", "w"))
t = {}
for f in sorted(glob.glob("logs/traj_alg_xii_P*.json")): t.update(json.load(open(f)))
json.dump(t, open("logs/traj_alg_xii.json", "w"), indent=1)
PY
cp logs/factorial_conf.json logs/factorial_permc.json; python3 scripts/perm_compare.py _permc && mv logs/perm_compare_permc.json logs/perm_compare_conf.json
python3 scripts/score_xii.py
