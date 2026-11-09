"""Q3-D evaluation (frozen in PROSPECTIVE_PREDICTIONS; code src/graded_sig.py and fits logs/extrap/frozen_fits.json verified
by hash) on the 16 held-out models of checkpoints/heldout3, plus Workstream D5 shifts.
 Target: p_c = 0 local index (continuation pp2_sig.pc0_local_index, unchanged).  Predictors: architecture, GCD, SAT, A2
 (LPI_deg via pp2_sig.signatures), constants.  Verdicts Q3-D1..D3.
 D5: for alpha shift (0.7), copy-rate shift (rho 0.9) and prevalence shift (p_c 0.5): fraction of questions where true Bayes
 and the T4 plug-in disagree, and each model's agreement with each on those questions.  Writes logs/extrap/heldout_outcome.json."""
import json, os, sys, hashlib
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, graded_sig as GS, pp2_sig as S, gen_family as G
from scipy.stats import spearmanr
H = os.path.join(HERE, "..", "checkpoints", "heldout3")
res = dict(graded_sig_sha256=hashlib.sha256(open(os.path.join(HERE, "..", "src", "graded_sig.py"), "rb").read()).hexdigest(),
           fits_sha256=hashlib.sha256(open(os.path.join(HERE, "..", "logs", "extrap", "frozen_fits.json"), "rb").read()).hexdigest(), models={})
fits = json.load(open(os.path.join(HERE, "..", "logs", "extrap", "frozen_fits.json")))["fits"]
M = {f"tf_s{s}": ("tf", os.path.join(H, f"T4_M_K3_s{s}_fromT1.npz")) for s in range(40, 46)}
M.update({f"ds_s{s}": ("ds", os.path.join(H, f"T4_M_K3_s{s}_deepsets.npz")) for s in range(6, 16)})
shifts = {"alpha0.7": G.Fam(pc=1.0, rho=1.0, alpha=0.7), "rho0.9": G.Fam(pc=1.0, rho=0.9), "pc0.5": G.Fam(pc=0.5, rho=1.0)}
plug = G.Fam(pc=1.0, rho=0.999); sd = {}
for nm, f in shifts.items():
    d = G.sample(f, 4000, np.random.default_rng(9_990_000 + len(nm))); X = d["X"]
    tb = G.posterior(f, X)["post_y"].argmax(-1); pb = np.nan_to_num(G.posterior(plug, X)["post_y"], nan=0).argmax(-1); sd[nm] = (X, tb, pb, tb != pb)
res["shift_disagreement_rate"] = {nm: float(v[3].mean()) for nm, v in sd.items()}
for nm, (k, p) in M.items():
    ap = S.load(k, p); r = GS.signature(ap); r.update(S.pc0_local_index(ap)); r["LPI_deg"] = S.signatures(ap)["LPI_deg"]; r["is_tf"] = int(k == "tf")
    for sn, (X, tb, pb, dis) in sd.items():
        a = ap(X)[0].argmax(-1); r[f"{sn}_agree_true_on_dis"] = float((a == tb)[dis].mean()) if dis.any() else None; r[f"{sn}_agree_plugin_on_dis"] = float((a == pb)[dis].mean()) if dis.any() else None
    res["models"][nm] = r; print(nm, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in r.items() if a in ("GCD", "SAT", "LPI_deg", "local_index")}, flush=True)
names = list(M); tf = [n for n in names if n.startswith("tf")]; ds = [n for n in names if n.startswith("ds")]
y = {n: res["models"][n]["local_index"] for n in names}
def pred(f, n, within=False):
    w = fits[f + ("_withinTF" if within else "")]; return w["intercept"] + w["slope"] * res["models"][n][f]
mae = lambda ps, ns: float(np.mean([abs(ps[i] - y[n]) for i, n in enumerate(ns)]))
res["Q3D1"] = dict(all_ds_below_030=bool(all(y[n] < 0.30 for n in ds)), mean_tf=float(np.mean([y[n] for n in tf])))
res["Q3D1"]["pass"] = bool(res["Q3D1"]["all_ds_below_030"] and res["Q3D1"]["mean_tf"] > 0.30)
const_tf = fits["constant_TF"]["intercept"]; res["within_TF"] = {"constant_MAE": mae([const_tf] * len(tf), tf)}
for f in ("GCD", "SAT", "LPI_deg"):
    res["within_TF"][f] = dict(spearman=float(spearmanr([res["models"][n][f] for n in tf], [y[n] for n in tf])[0]), MAE=mae([pred(f, n, True) for n in tf], tf))
res["Q3D2_pass"] = bool(all(res["within_TF"][f]["spearman"] <= 0.5 and res["within_TF"][f]["MAE"] >= res["within_TF"]["constant_MAE"] for f in ("GCD", "SAT")))
res["Q3D3_pass"] = bool(res["within_TF"]["LPI_deg"]["spearman"] >= 0.6 and res["within_TF"]["LPI_deg"]["MAE"] < res["within_TF"]["constant_MAE"])
res["all_models_MAE"] = {f: mae([pred(f, n) for n in names], names) for f in ("is_tf", "GCD", "SAT", "LPI_deg")}; res["all_models_MAE"]["constant"] = mae([fits["constant"]["intercept"]] * len(names), names)
print(json.dumps({k: res[k] for k in ("Q3D1", "within_TF", "Q3D2_pass", "Q3D3_pass", "all_models_MAE", "shift_disagreement_rate")}, indent=1))
json.dump(res, open(os.path.join(HERE, "..", "logs", "extrap", "heldout_outcome.json"), "w"), indent=1)
