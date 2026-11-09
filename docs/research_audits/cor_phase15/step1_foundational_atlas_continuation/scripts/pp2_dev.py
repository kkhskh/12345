"""PP2 DEVELOPMENT (post hoc relative to the frozen PP2; uses the 10 existing T4 models only): signatures, baselines,
p_c = 0 local index split by T4 support, leave-one-out comparison of predictors.  Writes logs/pp2_dev.json."""
import json, os, sys
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, pp2_sig as S
LR = S.LR
M = {f"tf_s{s}": ("tf", f"checkpoints/posthoc_curriculum/T4_M_K3_s{s}_fromT1.npz") for s in [0, 1, 2]}
M.update({f"tf_s{s}": ("tf", f"checkpoints/replication/T4_M_K3_s{s}_fromT1.npz") for s in [10, 11, 12, 13]})
M.update({f"ds_s{s}": ("ds", f"checkpoints/main/T4_M_K3_s{s}_deepsets.npz") for s in [0, 1, 2]})
pc = json.load(open(os.path.join(HERE, "..", "logs", "pp1_calibration.json"))); ec = json.load(open(os.path.join(HERE, "..", "logs", "pp1_learned_ece.json")))
res = {}
for nm, (k, p) in M.items():
    ap = S.load(k, os.path.join(LR, p)); r = S.signatures(ap); r.update(S.pc0_local_index(ap))
    r.update(T4_acc=ec[nm]["acc"], T4_KL=ec[nm]["KL"], T4_ECE=ec[nm]["ECE"], is_tf=int(k == "tf"))
    res[nm] = r; print(nm, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in r.items()}, flush=True)
names = list(M); y = np.array([res[n]["local_index"] for n in names])
def loo(feat):
    x = np.array([res[n][feat] for n in names], float); err = []
    for i in range(len(y)):
        m = np.arange(len(y)) != i; A = np.c_[np.ones(m.sum()), x[m]]; w = np.linalg.lstsq(A, y[m], rcond=None)[0]; err.append(abs(w[0] + w[1] * x[i] - y[i]))
    return float(np.mean(err))
cands = ["CCP_1", "CCP_2", "CCP_3", "LPI_deg", "T4_acc", "T4_KL", "T4_ECE", "is_tf"]
res["LOO_MAE"] = {c: loo(c) for c in cands}; res["LOO_MAE"]["constant"] = float(np.mean([abs(y[i] - np.delete(y, i).mean()) for i in range(len(y))]))
rng = np.random.default_rng(0); rnd = []
for _ in range(200):
    for n in names: res[n]["_rand"] = float(rng.normal())
    rnd.append(loo("_rand"))
res["LOO_MAE"]["random_feature_mean"] = float(np.mean(rnd))
for n in names: res[n].pop("_rand")
from scipy.stats import spearmanr
res["spearman_with_local_index"] = {c: float(spearmanr([res[n][c] for n in names], y)[0]) for c in cands}
tfn = [n for n in names if n.startswith("tf")]; yt = np.array([res[n]["local_index"] for n in tfn])
res["spearman_within_TF"] = {c: float(spearmanr([res[n][c] for n in tfn], yt)[0]) for c in cands if c != "is_tf"}
print(json.dumps({k: res[k] for k in ["LOO_MAE", "spearman_with_local_index", "spearman_within_TF"]}, indent=1))
json.dump(res, open(os.path.join(HERE, "..", "logs", "pp2_dev.json"), "w"), indent=1)
