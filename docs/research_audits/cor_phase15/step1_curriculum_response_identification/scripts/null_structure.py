"""Structure of Bayesian-null (family N) responses (mandate §5.3-5.4).  Usage: null_structure.py PHASE OUTNAME MODEL [...]
Exact class structure (proved in BAYES_NULL_RESPONSE_STRUCTURE.md §1): minimal null edits exist only when exactly one
group has count 5 (the colluder candidate h); they shift the other two counts together, keeping d = n_g1 − n_g2 fixed and
changing the level L = min(n_g1, n_g2) by ±1.
Per model and N pair: scalar confidence response s = <Δc, c*> / |c*| (change of the centred logits along the exact Bayes
log-odds direction of the base episode; s > 0 = more confident in the Bayes direction).  Regressed on generator features:
  dir (±1), dir·L, dir·|d|, dir·[A_{q,h} = 1 for both reliable-candidate groups], dir·maxp* (base Bayes confidence),
  and indicators of the colluder group h (location of perfect agreement).
Reported: R², coefficients, and correlations of coefficient vectors between models (within/between arms).
Also (§5.4): per model, mean |Δ| on N pairs, mean |R-error| on R pairs (after per-model scaling), KL to Bayes, and the
across-episode correlation of |N response| with |R error| for base episodes that have both edits.  Writes logs/PHASE/OUTNAME.json."""
import itertools, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
import cri as R
E = R.E
PH, OUTNAME, names = sys.argv[1], sys.argv[2], sys.argv[3:]
D = os.path.join(HERE, "..", "logs", PH); Gd = np.load(os.path.join(D, "generator.npz"))
X, q, cfg = Gd["X"], Gd["qsel"], Gd["cfg"]; NE = len(X); idx = np.arange(NE)
okN, okR = Gd["ok_N"].astype(bool), Gd["ok_R"].astype(bool)
n0 = E.loo(X)[idx, q]; n1 = E.loo(Gd["X_N"])[idx, q]; cB = Gd["cB_base"]
H = np.array([np.where(v == 5)[0][0] if (v == 5).sum() == 1 else -1 for v in n0])
oth = np.array([[g for g in range(3) if g != h] if h >= 0 else [0, 1] for h in H])
dirn = np.sign((n1 - n0)[idx, oth[:, 0]]); L = np.minimum(n0[idx, oth[:, 0]], n0[idx, oth[:, 1]]); dd = np.abs(n0[idx, oth[:, 0]] - n0[idx, oth[:, 1]])
Aq = E.agree(X)[idx, q]; both = (Aq[idx, oth[:, 0]] == 1) & (Aq[idx, oth[:, 1]] == 1)
maxp = np.exp(cB).max(-1) / np.exp(cB).sum(-1)
Xd = np.c_[np.ones(NE), dirn, dirn * L, dirn * dd, dirn * both, dirn * maxp, *(H == h for h in range(3))][okN].astype(float)
cols = ["const", "dir", "dir*L", "dir*|d|", "dir*both_reliable_candidates_agree_at_q", "dir*maxp_bayes", "h=0", "h=1", "h=2"]
u = cB / (np.linalg.norm(cB, axis=-1, keepdims=True) + 1e-12)
out = dict(models=names, columns=cols, per_model={}, n=int(okN.sum()), frac_N_valid_with_single_H=float((H[okN] >= 0).mean()))
coefs = {}
for m in names:
    z = np.load(os.path.join(D, f"model_{m}.npz")); j = json.load(open(os.path.join(D, f"model_{m}.json")))
    dN = z["c_N"] - z["c_base"]; s = (dN * u).sum(-1)[okN]
    b = np.linalg.lstsq(Xd[:, :6], s, rcond=None)[0]; pred = Xd[:, :6] @ b      # h indicators collinear with const; separate fit
    b2 = np.linalg.lstsq(np.c_[Xd[:, 1:6], Xd[:, 6:]], s, rcond=None)[0]; pred2 = np.c_[Xd[:, 1:6], Xd[:, 6:]] @ b2
    r2 = lambda p: float(1 - ((s - p) ** 2).sum() / ((s - s.mean()) ** 2).sum())
    dR = (z["c_R"] - z["c_base"]); dBR = Gd["cB_R"] - Gd["cB_base"]; beta = (dR[okR] * dBR[okR]).sum() / (dBR[okR] ** 2).sum()
    Rerr = np.abs(dR - beta * dBR).sum(-1); both_ok = okN & okR
    out["per_model"][m] = dict(kind=R.kind_of(m), coef=dict(zip(cols[:6], b.tolist())), r2=r2(pred), r2_with_h=r2(pred2),
                              mean_abs_N=float(np.abs(dN[okN]).mean()), mean_abs_s=float(np.abs(s).mean()), mean_s=float(s.mean()),
                              mean_abs_Rerr=float(Rerr[okR].mean() / 3), R_scale_beta=float(beta), KL_bayes=j["quality"]["KL_bayes"],
                              corr_absN_absRerr_across_episodes=float(np.corrcoef(np.abs(dN[both_ok]).sum(-1), Rerr[both_ok])[0, 1]))
    coefs[m] = b[1:6]
cv = {}
for a, b in itertools.combinations(names, 2):
    cv[f"{a}~{b}"] = float(np.corrcoef(coefs[a], coefs[b])[0, 1])
out["coef_vector_correlations"] = cv
kinds = {m: R.kind_of(m) for m in names}
for g in ("TT", "PP", "TP"):
    vals = [v for k_, v in cv.items() if "".join(sorted(kinds[x] for x in k_.split("~"))) == "".join(sorted(g))]
    out[f"mean_coef_corr_{g}"] = float(np.mean(vals)) if vals else None
mm = np.array([[out["per_model"][m][k] for k in ("mean_abs_N", "mean_abs_Rerr", "KL_bayes")] for m in names])
out["model_level_corr"] = dict(absN_vs_Rerr=float(np.corrcoef(mm[:, 0], mm[:, 1])[0, 1]), absN_vs_KL=float(np.corrcoef(mm[:, 0], mm[:, 2])[0, 1]),
                               Rerr_vs_KL=float(np.corrcoef(mm[:, 1], mm[:, 2])[0, 1]))
json.dump(out, open(os.path.join(D, f"{OUTNAME}.json"), "w"), indent=1)
for m in names:
    p = out["per_model"][m]; print(m, p["kind"], "r2", round(p["r2"], 3), round(p["r2_with_h"], 3), {k: round(v, 3) for k, v in p["coef"].items()}, "|N|", round(p["mean_abs_N"], 3), "|Rerr|", round(p["mean_abs_Rerr"], 3), "KL", round(p["KL_bayes"], 3), "corr", round(p["corr_absN_absRerr_across_episodes"], 3))
print({k: v for k, v in out.items() if k.startswith("mean_coef") or k == "model_level_corr"})
