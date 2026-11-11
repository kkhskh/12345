"""EXPLORATORY (not preregistered).  Fitted-operator identifiability on observable boundary data.
Usage: fitted_operators.py PHASE MODEL [...]   (reads logs/PHASE/generator.npz and model_<name>.npz from boundary_run.py)
Targets: the model's centred logits at (X, q) for base episodes and their F45, F34, F23 edits (oriented and non-oriented pairs).
Families fitted by least squares on centred logits, using the exact leave-q counts n of each input and the exact J(x_q, y | cfg):
  gain   : tau * Bayes                                   (1 parameter)
  mix    : Bayes mixture, w = exp(phi_rel[n_g] + phi_col[n_h])   (11; Inv V parameterisation)
  logpool: logit = sum_cfg w_cfg log J(cfg) / sum w   with the same w parameterisation (11)
  vote   : count-weighted voting (Inv V vote family) (6)
Then: (i) profile of s = phi_rel[5] - phi_rel[4] in mix (refit everything else on a grid of s) and its 'flat' set (loss within 2 %
of the minimum); (ii) each fitted member's predicted B45, B45-R1, B34-R1 versus the observed values (an s-interpretation is
rejected if its predicted observable is outside the observed value's model-level ±2 SE... here per model: |pred-obs| > 0.1)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
from scipy.optimize import minimize
import boundary as B
R, E, G = B.R, B.E, B.G
PH, names = sys.argv[1], sys.argv[2:]; L = os.path.join(HERE, "..", "logs", PH); g = np.load(os.path.join(L, "generator.npz"), allow_pickle=True)
NF = int(os.environ.get("NF", 2500)); X, q = g["X"][:NF], g["q"][:NF]; idx = np.arange(NF)
fams = ["base", "F45", "F34", "F23"]; Xs = {"base": X} | {f: g[f"X_{f}"][:NF] for f in fams[1:]}
Jq = {f: R.J_at(Xs[f], 0) * 0 for f in fams}
for f in fams:
    J = np.zeros((NF, 6, 3))
    for qq in range(6):
        s = q == qq; J[s] = R.J_at(Xs[f][s], qq)
    Jq[f] = J
Vs = {f: E.loo(Xs[f])[idx, q] for f in fams}; Xq = {f: Xs[f][idx, q] for f in fams}
Jc = np.concatenate([Jq[f] for f in fams]); Vc = np.concatenate([Vs[f] for f in fams]); Xqc = np.concatenate([Xq[f] for f in fams])
G_OF, H_OF = np.array(R.G_OF), np.array(R.H_OF)
def weights(th):
    pr = np.r_[0, th[0:5]]; pc = np.r_[th[5:10], 0]; return np.exp(np.clip(pr[Vc[:, G_OF]] + pc[Vc[:, H_OF]], -200, 200))
def f_mix(th): return R.mix_logits(th, Jc, Vc)
def f_log(th):
    w = weights(th); w = w / w.sum(1, keepdims=True); Lg = np.exp(th[10]) * (w[:, :, None] * np.log(Jc / Jc.sum(2, keepdims=True).clip(1e-300) + 1e-300)).sum(1)
    return Lg - Lg.mean(-1, keepdims=True)
def f_vote(th): return R.vote_logits(th, Xqc, Vc)
Bex = R.mix_exact(Jc, Vc)
def f_gain(th): return np.exp(th[0]) * Bex
th_ex = np.r_[np.arange(1, 6) * np.log(E.RHO), np.full(5, -3.0), 0.0]
def fit(f, target, th0, fixed=None):
    if fixed is None: loss = lambda th: ((f(th) - target) ** 2).mean(); r = minimize(loss, th0, method="L-BFGS-B", options=dict(maxiter=2000)); return r.x, r.fun
    i, v = fixed  # fix th[4] - th[3] = v (s), reparametrised
    def full(t): th = np.insert(t, 4, t[3] + v if i == "s" else v); return th
    loss = lambda t: ((f(full(t)) - target) ** 2).mean(); r = minimize(loss, np.delete(th0, 4), method="L-BFGS-B", options=dict(maxiter=2000)); return full(r.x), r.fun
gen_meta = {f: dict(orient=g[f"orient_{f}"][:NF], vote=g[f"vote_{f}"][:NF], regime=g[f"regime_{f}"][:NF]) for f in fams[1:]}
def observables(pred):
    """B (excess over exact Bayes) predicted by a fitted member: pred and Bex blocks of NF rows per family"""
    blk = {f: pred[i * NF:(i + 1) * NF] for i, f in enumerate(fams)}; bb = {f: Bex[i * NF:(i + 1) * NF] for i, f in enumerate(fams)}; out = {}
    for f in ("F45", "F34"):
        o = gen_meta[f]["orient"]; v = np.maximum(gen_meta[f]["vote"], 0)
        dm = B.oriented_margin(blk[f], v) - B.oriented_margin(blk["base"], v); db = B.oriented_margin(bb[f], v) - B.oriented_margin(bb["base"], v)
        e = (dm - db); out[f"B{f[1:]}"] = float(e[o].mean()); r1 = o & (gen_meta[f]["regime"] == "R1_with_colluder_candidate"); out[f"B{f[1:]}R1"] = float(e[r1].mean())
    return out
res = {}
for m in names:
    z = np.load(os.path.join(L, f"model_{m.replace('@', '_')}.npz")); target = np.concatenate([z["c_base"][:NF]] + [z[f"c_{f}"][:NF] for f in fams[1:]])
    obs = observables(target); var = ((target - target.mean(0)) ** 2).mean() * 3; out = dict(observed=obs, families={})
    for nm, f, th0 in (("gain", f_gain, np.zeros(1)), ("mix", f_mix, th_ex), ("logpool", f_log, th_ex), ("vote", f_vote, np.ones(6))):
        th, lo = fit(f, target, th0); out["families"][nm] = dict(theta=th.tolist(), mse=float(lo), r2=float(1 - lo * 3 / var), predicted=observables(f(th)))
        if nm == "mix": thm = th
    s_hat = float(thm[4] - thm[3]); prof = []
    for s in np.linspace(-2, 4, 25):
        th, lo = fit(f_mix, target, thm, fixed=("s", s)); prof.append(dict(s=float(s), mse=float(lo), predicted=observables(f_mix(th)), gate=th[5:10].tolist()))
    mn = min(p["mse"] for p in prof + [dict(mse=out["families"]["mix"]["mse"])]); flat = [p["s"] for p in prof if p["mse"] <= 1.02 * mn]
    out["mix_s_hat"] = s_hat; out["s_profile"] = prof; out["s_flat_set_2pct"] = [min(flat), max(flat)] if flat else None
    out["s_rejected_by_observables"] = [p["s"] for p in prof if abs(p["predicted"]["B45R1"] - obs["B45R1"]) > 0.1]
    res[m] = out; fm = out["families"]
    print(m, "obs B45R1 %.3f B34R1 %.3f" % (obs["B45R1"], obs["B34R1"]), {k: (round(v["r2"], 3), round(v["predicted"]["B45R1"], 3)) for k, v in fm.items()}, "s_hat %.2f flat %s" % (s_hat, out["s_flat_set_2pct"]), flush=True)
    json.dump(res, open(os.path.join(HERE, "..", "logs", f"fitted_operators_{PH}.json"), "w"), indent=1)
