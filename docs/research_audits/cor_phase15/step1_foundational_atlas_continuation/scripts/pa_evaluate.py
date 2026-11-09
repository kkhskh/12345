"""P-A evaluation (frozen in PROSPECTIVE_PREDICTIONS.md, commit dc1dd3b): held-out models only.  Applies the frozen fits in
logs/pp2_frozen_fits.json; computes signatures with src/pp2_sig.py (unchanged since freeze, sha256 2b91afcd...), the
p_c = 0 local index, and the P-A1..P-A4 verdicts.  Ridge baseline B7 fitted on the 10 development models only.
Writes logs/pa_outcome.json."""
import json, os, sys, hashlib
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, pp2_sig as S
src_hash = hashlib.sha256(open(os.path.join(HERE, "..", "src", "pp2_sig.py"), "rb").read()).hexdigest()
fits = json.load(open(os.path.join(HERE, "..", "logs", "pp2_frozen_fits.json"))); dev = json.load(open(os.path.join(HERE, "..", "logs", "pp2_dev.json")))
H = os.path.join(HERE, "..", "checkpoints", "heldout")
M = {"tf_s14": ("tf", "T4_M_K3_s14_fromT1.npz"), "tf_s15": ("tf", "T4_M_K3_s15_fromT1.npz"), "ds_s3": ("ds", "T4_M_K3_s3_deepsets.npz"), "ds_s4": ("ds", "T4_M_K3_s4_deepsets.npz")}
import jax, jax.numpy as jnp, gen_family as G, nets as N, ckpt as C
f4 = G.Fam(pc=1.0, rho=1.0); Dn = G.sample(f4, 3000, np.random.default_rng(6_150_000)); Pn = G.posterior(f4, Dn["X"])["post_y"]
def ece(Pm, Y, nb=10):
    conf = Pm.max(-1).ravel(); cor = (Pm.argmax(-1) == Y).ravel(); e = 0
    for j in range(nb):
        m = (conf > j / nb) & (conf <= (j + 1) / nb)
        if m.any(): e += m.mean() * abs(conf[m].mean() - cor[m].mean())
    return float(e)
res = dict(pp2_sig_sha256=src_hash, models={})
for nm, (k, f) in M.items():
    ap = S.load(k, os.path.join(H, f)); r = S.signatures(ap); r.update(S.pc0_local_index(ap))
    l = ap(Dn["X"])[0]; l = l - np.log(np.exp(l).sum(-1, keepdims=True)); r.update(T4_acc=float((l.argmax(-1) == Dn["Y"]).mean()), T4_ECE=ece(np.exp(l), Dn["Y"]),
                                                                              T4_KL=float((Pn * (np.log(Pn + 1e-12) - l)).sum(-1).mean()), is_tf=int(k == "tf"))
    res["models"][nm] = r; print(nm, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in r.items()}, flush=True)
names = list(M); y = np.array([res["models"][n]["local_index"] for n in names])
pred = {}
for f_, key in [("A1_CCP_3", "CCP_3"), ("A2_LPI_deg", "LPI_deg"), ("B1_arch", "is_tf"), ("B2_KL", "T4_KL"), ("B3_acc", "T4_acc"), ("B4_ECE", "T4_ECE")]:
    w = fits[key]; pred[f_] = [w["intercept"] + w["slope"] * res["models"][n][key] for n in names]
pred["B5_constant"] = [fits["constant"]["intercept"]] * len(names)
rng = np.random.default_rng(0); devn = [k for k in dev if k[:3] in ("tf_", "ds_")]; ydev = np.array([dev[n]["local_index"] for n in devn])
rr = []
for _ in range(200):
    xd = rng.normal(size=len(devn)); w = np.linalg.lstsq(np.c_[np.ones(len(devn)), xd], ydev, rcond=None)[0]; rr.append(np.mean(np.abs(w[0] + w[1] * rng.normal(size=len(names)) - y)))
feat = ["T4_acc", "T4_KL", "T4_ECE", "is_tf"]; Xd = np.array([[dev[n][f] for f in feat] for n in devn]); mu, sd = Xd.mean(0), Xd.std(0) + 1e-12
A = np.c_[np.ones(len(devn)), (Xd - mu) / sd]; wr = np.linalg.solve(A.T @ A + np.diag([0] + [1.0] * len(feat)), A.T @ ydev)
Xh = np.array([[res["models"][n][f] for f in feat] for n in names]); pred["B7_ridge"] = list(np.c_[np.ones(len(names)), (Xh - mu) / sd] @ wr)
mae = {k: float(np.mean(np.abs(np.array(v) - y))) for k, v in pred.items()}; mae["B6_random_mean"] = float(np.mean(rr))
res["predictions"] = {k: [float(x) for x in v] for k, v in pred.items()}; res["MAE"] = mae; res["observed_local_index"] = dict(zip(names, map(float, y)))
base = [v for k, v in mae.items() if k.startswith("B")]
res["verdict"] = {"P-A1_internal_beats_all_baselines": bool(mae["A1_CCP_3"] < min(base)),
                  "P-A2_behavioural_MAE_le_0.10": bool(mae["A2_LPI_deg"] <= 0.10),
                  "P-A3_support": bool(all(res["models"][n]["local_index_inT4"] < 0.15 and res["models"][n]["agree_global_inT4"] > 0.80 for n in names)),
                  "P-A4_deepsets_global": bool(all(res["models"][n]["local_index"] < 0.30 for n in names if n.startswith("ds")))}
print(json.dumps({k: res[k] for k in ["MAE", "observed_local_index", "verdict"]}, indent=1))
json.dump(res, open(os.path.join(HERE, "..", "logs", "pa_outcome.json"), "w"), indent=1)
