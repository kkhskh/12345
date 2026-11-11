"""Robustness of the post hoc sign finding s = phi_rel(5) - phi_rel(4) in the Bayes-mixture fit (arrays only; surfaces from
logs/conf/model_*.npz).  Per model: (0) reproduce frozen theta; (1) multistart L-BFGS (random inits): best loss and s over
near-optimal solutions; (2) episode bootstrap of s (warm start); (3) profile: loss / R2_within with s fixed at 0 and at log rho;
(4) s refitted using only cells with no non-colluder count at 5 vs only cells with one (identification source);
(5) corrected noise ceiling for the 4-realisation target vs the split-half R2 used in the package."""
import json, os, sys, time, numpy as np
from scipy.optimize import minimize
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "..", "..", "logs", "conf")
sys.path.insert(0, os.path.join(HERE, "..", "..", "src")); import cri as R; E = R.E
G = np.load(os.path.join(D, "generator.npz")); X, qsel, surfV = G["X"], G["qsel"], G["surfV"]; NS = surfV.shape[0]; Si = np.arange(NS)
Jq = np.concatenate([R.J_at(X[i:i + 1], qsel[i]) for i in Si]); Jrep = np.repeat(Jq, 36, 0); Vs = surfV.reshape(-1, 3)
grp = np.repeat(Si, 36); h = np.repeat(R.true_h(G["cfg"][Si]), 36)
oth5 = ((Vs == 5).sum(1) >= 2)                                   # a non-colluder also at 5
names = sys.argv[1:] or ["cT61", "cT62", "cT63", "cT64", "cT65", "cT67", "cP61", "cP62", "cP63", "cP64", "cP66", "cP67"]
rng = np.random.default_rng(0); LR = np.log(E.RHO)

def fit(tgt, J, V, th0, fix_s=None):
    if fix_s is None: f = lambda th: R.mix_logits(th, J, V)
    else:
        f = lambda t: R.mix_logits(np.r_[t[:4], t[3] + fix_s, t[4:]], J, V); th0 = np.r_[th0[:4], th0[5:]]
    r = minimize(lambda th: ((f(th) - tgt) ** 2).mean(), th0, method="L-BFGS-B", options=dict(maxiter=3000))
    th = r.x if fix_s is None else np.r_[r.x[:4], r.x[3] + fix_s, r.x[4:]]
    return th, float(r.fun), R.mix_logits(th, J, V)

def dm(Z, g):
    Z = Z.copy()
    for k in np.unique(g): s = g == k; Z[s] -= Z[s].mean(0)
    return Z
out = {}
for m in names:
    t0 = time.time(); z = np.load(os.path.join(D, f"model_{m}.npz")); tgt = z["surf"].reshape(-1, 3)
    th_f = np.array(json.load(open(os.path.join(D, f"model_{m}.json")))["fits"]["mix"]["theta"]); s_f = th_f[4] - th_f[3]
    TH_DEF = np.r_[np.arange(1, 6) * LR, np.full(5, -3.0), 0.0]   # package default init
    th0, L0, p0 = fit(tgt, Jrep, Vs, TH_DEF); s0 = th0[4] - th0[3]
    tw = dm(tgt, grp); r2w = lambda p: float(1 - ((tw - dm(p, grp)) ** 2).sum() / (tw ** 2).sum())
    ms = []
    for k in range(6):
        init = np.r_[np.cumsum(rng.uniform(0, 2.5, 5)), rng.uniform(-15, 0, 5), rng.normal(0, .2)]
        th, L, _ = fit(tgt, Jrep, Vs, init); ms.append((L, th[4] - th[3]))
    ms = np.array(ms); near = ms[ms[:, 0] <= L0 * 1.001 + 1e-9]
    bs = []
    for b in range(15):
        e = rng.integers(0, NS, NS); rows = (e[:, None] * 36 + np.arange(36)).ravel()
        th, _, _ = fit(tgt[rows], Jrep[rows], Vs[rows], th0); bs.append(th[4] - th[3])
    bs = np.array(bs)
    prof = {}
    for nm, sv in (("s=0", 0.0), ("s=logrho", LR), ("s=-s_hat", -s0)):
        _, L, p = fit(tgt, Jrep, Vs, th0, sv); prof[nm] = dict(dloss_rel=(L - L0) / L0, r2w=r2w(p))
    sub = {}
    for nm, sel in (("cells_no_other5", ~oth5), ("cells_other5", oth5)):
        th, _, _ = fit(tgt[sel], Jrep[sel], Vs[sel], th0); sub[nm] = th[4] - th[3]
    ha, hb = z["surf_half_a"].reshape(-1, 3), z["surf_half_b"].reshape(-1, 3); ta, tb = dm(ha, grp), dm(hb, grp)
    sig2 = ((ta - tb) ** 2).sum() / 2                     # = sum over entries of per-realisation-pair noise for a 2-mean, times... see report
    # noise var of a 2-mean = v2 ; diff var = 2 v2 ; 4-mean noise var = v2/2
    v4 = ((ta - tb) ** 2).sum() / 4                        # total noise SS in the 4-mean target
    ceil4 = float(1 - v4 / (tw ** 2).sum()); sh = float(1 - ((ta - tb) ** 2).sum() / (tb ** 2).sum())
    out[m] = dict(s_frozen=s_f, s_refit=s0, r2w_mix=r2w(p0), multistart_s=ms[:, 1].tolist(), multistart_relloss=(ms[:, 0] / L0).tolist(),
                  s_near_opt_range=[float(near[:, 1].min()), float(near[:, 1].max())] if len(near) else None,
                  boot_s_mean=float(bs.mean()), boot_s_sd=float(bs.std()), boot_frac_pos=float((bs > 0).mean()), profile=prof, s_by_cells=sub,
                  split_half_r2=sh, ceiling_4mean=ceil4, rel_r2_pkg=r2w(p0) / sh, rel_r2_corrected=r2w(p0) / ceil4)
    o = out[m]
    print(f"{m}: s frozen {s_f:+.2f} refit {s0:+.2f} | multistart s {np.round(ms[:,1],2)} relloss {np.round(ms[:,0]/L0,3)} | boot s {bs.mean():+.2f}±{bs.std():.2f} pos {o['boot_frac_pos']:.2f} | "
          f"profile s=0 dloss {prof['s=0']['dloss_rel']:+.3f} r2w {prof['s=0']['r2w']:.3f} vs free {o['r2w_mix']:.3f}; s=logrho {prof['s=logrho']['dloss_rel']:+.3f} ; s=-s {prof['s=-s_hat']['r2w']:.3f} | "
          f"s(no-other5 cells) {sub['cells_no_other5']:+.2f} s(other5 cells) {sub['cells_other5']:+.2f} | ceiling split-half {sh:.3f} 4-mean {ceil4:.3f} rel {o['rel_r2_pkg']:.2f}->{o['rel_r2_corrected']:.2f} [{time.time()-t0:.0f}s]", flush=True)
json.dump(out, open(os.path.join(HERE, "..", "..", "logs", "review_checks", "rv_mech_phi.json"), "w"), indent=1)
