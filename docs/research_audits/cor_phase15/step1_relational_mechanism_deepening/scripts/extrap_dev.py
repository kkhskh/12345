"""Workstream D development (POST HOC: these 14 models' off-support outcomes are already known): GCD and SAT on the 10
development T4 models and the 4 continuation held-out models, against their recorded p_c = 0 local index.
Writes logs/extrap/dev.json."""
import json, os, sys
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, graded_sig as GS, pp2_sig as S
from scipy.stats import spearmanr
LR = os.path.join(HERE, "..", "..", "step1_latent_reliability", "checkpoints"); CO = os.path.join(HERE, "..", "..", "step1_foundational_atlas_continuation")
dev = json.load(open(os.path.join(CO, "logs", "pp2_dev.json"))); pa = json.load(open(os.path.join(CO, "logs", "pa_outcome.json")))
M = {f"tf_s{s}": ("tf", f"{LR}/posthoc_curriculum/T4_M_K3_s{s}_fromT1.npz", dev[f"tf_s{s}"]["local_index"]) for s in [0, 1, 2]}
M.update({f"tf_s{s}": ("tf", f"{LR}/replication/T4_M_K3_s{s}_fromT1.npz", dev[f"tf_s{s}"]["local_index"]) for s in [10, 11, 12, 13]})
M.update({f"ds_s{s}": ("ds", f"{LR}/main/T4_M_K3_s{s}_deepsets.npz", dev[f"ds_s{s}"]["local_index"]) for s in [0, 1, 2]})
M.update({"tf_s14": ("tf", f"{CO}/checkpoints/heldout/T4_M_K3_s14_fromT1.npz", pa["observed_local_index"]["tf_s14"]), "tf_s15": ("tf", f"{CO}/checkpoints/heldout/T4_M_K3_s15_fromT1.npz", pa["observed_local_index"]["tf_s15"]),
          "ds_s3": ("ds", f"{CO}/checkpoints/heldout/T4_M_K3_s3_deepsets.npz", pa["observed_local_index"]["ds_s3"]), "ds_s4": ("ds", f"{CO}/checkpoints/heldout/T4_M_K3_s4_deepsets.npz", pa["observed_local_index"]["ds_s4"])})
res = {}
for nm, (k, p, li) in M.items():
    r = GS.signature(S.load(k, p)); r["local_index"] = li; r["is_tf"] = int(k == "tf"); res[nm] = r; print(nm, r, flush=True)
n = list(res); tf = [x for x in n if x.startswith("tf")]
for f in ("GCD", "SAT"):
    res[f"spearman_{f}_all"] = float(spearmanr([res[x][f] for x in n], [res[x]["local_index"] for x in n])[0])
    res[f"spearman_{f}_tf"] = float(spearmanr([res[x][f] for x in tf], [res[x]["local_index"] for x in tf])[0])
print({k: v for k, v in res.items() if k.startswith("spearman")}); json.dump(res, open(os.path.join(HERE, "..", "logs", "extrap", "dev.json"), "w"), indent=1)
