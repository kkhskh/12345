"""Counterexample reviewer, check 1: what the operator-class verdict and the phi-sign statistic can and cannot identify.
(a) Synthetic V-surface models built from the exact generator (conf surface episodes, cri.J_at, exact pi):
    exact Bayes; a DIFFERENT pooling algorithm (log-opinion pool of config-conditional predictives, posterior weights);
    MAP-config selection; pure output ceilings applied to exact Bayes (label smoothing, logit clip, tanh saturation).
    Each is fitted with the package's F_mix and F_vote (cri.fit_family); report R2_within, tau, s = phi_rel(5)-phi_rel(4).
(b) Real cohort models: fit an alternative class F_log (same 11 parameters as F_mix, log pooling instead of linear pooling)
    to the frozen input-level surfaces and compare with the stored F_mix R2_within.
Read-only on package data.  Usage: python rv_cex_1_classes.py"""
import json, os, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); PK = os.path.join(HERE, "..", ".."); sys.path.insert(0, os.path.join(PK, "src"))
import numpy as np
import cri as R
from scipy.optimize import minimize
E = R.E; t0 = time.time()
Gd = np.load(os.path.join(PK, "logs", "conf", "generator.npz")); X, qsel, V = Gd["X"], Gd["qsel"], Gd["surfV"]; NS = V.shape[0]
Jq = np.concatenate([R.J_at(X[i:i + 1], qsel[i]) for i in range(NS)]); Jrep = np.repeat(Jq, 36, 0); Vs = V.reshape(-1, 3)
Xqr = np.repeat(X[np.arange(NS), qsel[:NS]], 36, 0); grp = np.repeat(np.arange(NS), 36)
ex = R.mix_exact(Jrep, Vs)
assert np.abs(ex - Gd["cBsurf"].reshape(-1, 3)).max() < 1e-6, "exact mixture != generator Bayes on the surface"


def cen(L): return L - L.mean(-1, keepdims=True)


def logpool(theta, Jq_, Vs_):
    """F_log: same weights as F_mix (exp(phi_rel+phi_col) times the own-question evidence M(g,h)), but the decision is the
    weighted average of config-conditional LOG predictives (logarithmic opinion pool) instead of the log of the average."""
    pr = np.r_[0, theta[0:5]]; pc = np.r_[theta[5:10], 0]; tau = np.exp(theta[10])
    M = Jq_.sum(-1); w = np.exp(pr[Vs_[:, R.G_OF]] + pc[Vs_[:, R.H_OF]]) * M; w = w / w.sum(-1, keepdims=True)
    lp = np.log(np.where(M[..., None] > 0, Jq_ / np.maximum(M[..., None], 1e-300), 1.0 / 3))
    return cen(tau * (w[..., None] * lp).sum(1))


th_ex = np.r_[np.arange(1, 6) * np.log(E.RHO), np.full(5, -50.0), 0.0]
P_ex = np.exp(ex); P_ex /= P_ex.sum(-1, keepdims=True)
post_cfg = np.exp(np.r_[0, th_ex[:5]][Vs[:, R.G_OF]] + np.r_[th_ex[5:10], 0][Vs[:, R.H_OF]]) * Jq.sum(-1).repeat(36, 0)
post_cfg /= post_cfg.sum(-1, keepdims=True)
amax = post_cfg.argmax(-1); tie = np.isclose(post_cfg, post_cfg.max(-1, keepdims=True))
Pmap = (tie[..., None] * Jrep).sum(1); Pmap /= Pmap.sum(-1, keepdims=True)
models = {"exact_bayes": ex, "logpool_exact_weights": logpool(th_ex, Jrep, Vs), "MAP_config": R.bayes_logodds(Pmap)}
for eps in (0.03, 0.1, 0.2):
    models[f"label_smooth_{eps}"] = R.bayes_logodds((1 - eps) * P_ex + eps / 3)
for C in (1.0, 2.0):
    models[f"logit_clip_{C}"] = cen(np.clip(ex, -C, C)); models[f"tanh_sat_{C}"] = cen(C * np.tanh(ex / C))
out = {"synthetic": {}}
print("synthetic V-surface models (ceiling = 1: noiseless)", flush=True)
for nm, tgt in models.items():
    row = {}
    for fk in ("mix", "vote"):
        th, pred = R.fit_family(fk, tgt, Jrep, Xqr, Vs); row[fk] = R.r2(pred, tgt, grp)["r2_within"]
        if fk == "mix": row["s"] = float(th[4] - th[3]); row["tau"] = float(np.exp(th[10])); row["phi_rel"] = np.round(th[:5], 2).tolist()
    thl = minimize(lambda t: ((logpool(t, Jrep, Vs) - tgt) ** 2).mean(), th_ex.clip(-8, None), method="L-BFGS-B", options=dict(maxiter=3000)).x
    row["log"] = R.r2(logpool(thl, Jrep, Vs), tgt, grp)["r2_within"]
    out["synthetic"][nm] = row
    print(f"  {nm:24s} R2w mix {row['mix']:.3f} log {row['log']:.3f} vote {row['vote']:.3f} | F_mix tau {row['tau']:.2f} s {row['s']:+.2f} phi_rel {row['phi_rel']}  [{time.time()-t0:.0f}s]", flush=True)

print("\nreal cohort models: F_log vs stored F_mix (relative R2_within = R2_within / split-half ceiling)", flush=True)
names = ["cT61", "cT62", "cT63", "cT64", "cT65", "cT67", "cP61", "cP62", "cP63", "cP64", "cP66", "cP67"]; out["real"] = {}
for m in names:
    z = np.load(os.path.join(PK, "logs", "conf", f"model_{m}.npz")); j = json.load(open(os.path.join(PK, "logs", "conf", f"model_{m}.json")))
    tgt = z["surf"].reshape(-1, 3); ceil = j["fits"]["split_half_r2_within"]; th0 = np.array(j["fits"]["mix"]["theta"])
    r = minimize(lambda t: ((logpool(t, Jrep, Vs) - tgt) ** 2).mean(), th0, method="L-BFGS-B", options=dict(maxiter=2000))
    rl = R.r2(logpool(r.x, Jrep, Vs), tgt, grp)["r2_within"]; rm = j["fits"]["mix"]["r2_within"]
    out["real"][m] = dict(mix_rel=rm / ceil, log_rel=rl / ceil, vote_rel=j["fits"]["vote"]["r2_within"] / ceil,
                          s_mix=th0[4] - th0[3], s_log=float(r.x[4] - r.x[3]), tau_log=float(np.exp(r.x[10])))
    o = out["real"][m]
    print(f"  {m}: rel R2 mix {o['mix_rel']:.3f} log {o['log_rel']:.3f} vote {o['vote_rel']:.3f}; s mix {o['s_mix']:+.2f} s log {o['s_log']:+.2f}  [{time.time()-t0:.0f}s]", flush=True)
json.dump(out, open(os.path.join(PK, "logs", "review_checks", "rv_cex_1_classes.json"), "w"), indent=1, default=float)
