"""Positive-theory review check 4: what exactly is shared inside the Bayes-mixture class F_mix.
(i) Fitted parameters (archived theta): reliability increments dphi(n) = phi_rel(n+1) - phi_rel(n), n = 0..4; collusion gate
    phi_col(n<5) (relative to phi_col(5) = 0); tau.  (ii) Share of each model's Bayes deviation captured in-class:
    1 - ||g - mix||^2 / ||g - exact_scaled||^2 on the within-episode surface.  (iii) Analytic decomposition of in-class null
    responses: refit-free counterfactual thetas (affine phi_rel on 0..4; hard gate) and their predicted null responses on the
    pi-equivalent surface pairs.  (iv) Split-half stability of s = dphi(4) for the 12 cohort models (warm-started refits)."""
import itertools, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
import cri as R
from scipy.optimize import minimize
D = os.path.join(ROOT, "logs", "conf"); G = np.load(os.path.join(D, "generator.npz"))
X, qsel, V = G["X"], G["qsel"], G["surfV"]; NS = V.shape[0]
Jq = np.concatenate([R.J_at(X[i:i + 1], qsel[i]) for i in range(NS)]); Jrep = np.repeat(Jq, 36, 0); Vs = V.reshape(-1, 3)
grp = np.repeat(np.arange(NS), 36)
def dm(Z):
    Z = Z.reshape(NS, 36, 3); return (Z - Z.mean(1, keepdims=True)).reshape(-1, 3)
pairs = []
for i in range(NS):
    P = R.E.pi_of_n(V[i])
    for a, b in itertools.combinations(range(36), 2):
        if np.abs(P[a] - P[b]).max() < 1e-12: pairs.append((i, a, b))
pairs = np.array(pairs)
def null_of(pred):
    p = pred.reshape(NS, 36, 3); return p[pairs[:, 0], pairs[:, 2]] - p[pairs[:, 0], pairs[:, 1]]
Ts = [61, 62, 63, 64, 65, 67]; Ps = [61, 62, 63, 64, 66, 67]
models = [f"cT{s}" for s in Ts] + [f"cP{s}" for s in Ps] + [f"bcT{s}_b500" for s in Ts]
print("n pi-equivalent surface pairs:", len(pairs))
ex_check = null_of(R.mix_exact(Jrep, Vs)); print("exact Bayes null response on these pairs (max abs):", float(np.abs(ex_check).max()))
print("\nmodel        KL     dphi(0..4)                          curv  s=dphi4  gate(max phi_col<5) tau   inclass_share  null|fit  null|affine  null|affine+hard  obs_null")
rows = {}
for m in models:
    js = json.load(open(os.path.join(D, f"model_{m}.json"))); z = np.load(os.path.join(D, f"model_{m}.npz"))
    th = np.array(js["fits"]["mix"]["theta"]); pr = np.r_[0, th[:5]]; d = np.diff(pr); tau = float(np.exp(th[10]))
    curv = float(np.std(d[:4]) / abs(np.mean(d[:4]))); gate = float(th[5:10].max())
    g = dm(z["surf"]); mix = dm(z["surf_pred_mix"]); exs = dm(z["surf_pred_exact"])
    share = float(1 - ((g - mix) ** 2).sum() / ((g - exs) ** 2).sum())
    # counterfactual thetas
    n = np.arange(5); A = np.c_[np.ones(5), n]; cfit = np.linalg.lstsq(A, pr[:5], rcond=None)[0]
    pra = np.r_[A @ cfit, pr[5]]; pra = pra - pra[0]
    th_aff = th.copy(); th_aff[:5] = pra[1:]
    th_hard = th_aff.copy(); th_hard[5:10] = -50.0
    nf = null_of(R.mix_logits(th, Jrep, Vs)); na = null_of(R.mix_logits(th_aff, Jrep, Vs)); nh = null_of(R.mix_logits(th_hard, Jrep, Vs))
    no = null_of(z["surf"])
    rows[m] = dict(KL=js["quality"]["KL_bayes"], d=d, curv=curv, s=float(d[4]), gate=gate, tau=tau, share=share,
                   nf=float(np.abs(nf).mean()), na=float(np.abs(na).mean()), nh=float(np.abs(nh).mean()), no=float(np.abs(no).mean()))
    r = rows[m]
    print(f"{m:12s} {r['KL']:.3f}  {np.round(d, 2).tolist()!s:34s} {curv:.2f}  {r['s']:+.2f}   {gate:+.2f}              {tau:.2f}  {share:.2f}           "
          f"{r['nf']:.3f}     {r['na']:.3f}       {r['nh']:.4f}          {r['no']:.3f}")
print("\nUniversal features over the 12 cohort models:")
coh = [f"cT{s}" for s in Ts] + [f"cP{s}" for s in Ps]
print("  phi_rel strictly increasing on 0..4 (dphi(0..3) > 0):", sum(all(rows[m]["d"][:4] > 0) for m in coh), "/ 12")
print("  phi_rel increasing incl. 4->5:", sum(all(rows[m]["d"] > 0) for m in coh), "/ 12")
print("  collusion gate: all phi_col(n<5) < -2:", sum(rows[m]["gate"] < -2 for m in coh), "/ 12;  gate values:", [round(rows[m]["gate"], 1) for m in coh])
print("  mean dphi(0..3) T1 / P:", round(np.mean([rows[m]["d"][:4].mean() for m in coh[:6]]), 2), "/", round(np.mean([rows[m]["d"][:4].mean() for m in coh[6:]]), 2), " (exact log rho = %.2f)" % np.log(R.E.RHO))
print("  in-class share of Bayes deviation T1 / P:", [round(rows[m]["share"], 2) for m in coh[:6]], [round(rows[m]["share"], 2) for m in coh[6:]])
# (iv) split-half stability of s
print("\nSplit-half refits (warm start, maxiter 400): s_a, s_b, s_full")
loss_rows = []
for m in coh:
    z = np.load(os.path.join(D, f"model_{m}.npz")); th0 = np.array(json.load(open(os.path.join(D, f"model_{m}.json")))["fits"]["mix"]["theta"])
    ss = []
    for h in ("surf_half_a", "surf_half_b"):
        tgt = z[h].reshape(-1, 3)
        r = minimize(lambda t: ((R.mix_logits(t, Jrep, Vs) - tgt) ** 2).mean(), th0, method="L-BFGS-B", options=dict(maxiter=400))
        ss.append(float(r.x[4] - r.x[3]))
    print(f"  {m:6s} {ss[0]:+.2f} {ss[1]:+.2f} {rows[m]['s']:+.2f}")
