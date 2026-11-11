"""Counterexample reviewer, check 3: null-response coherence produced by generic function approximators that share only the
target (exact Bayes), the training distribution and the input features, with no common internal algorithm.
Approximators are fitted by (ridge) least squares to exact centred Bayes log-odds on natural episodes (conf encoder-fit stream
Xfit, all six decisions), and evaluated on the package's exact conf N pairs (Bayes response exactly 0) and R pairs.
  RF-w : random-feature ReLU network of width w (random first layer = 'implementation', seed-specific; LS readout)
  POLY : quadratic polynomial in the leave-q counts, interacted with the own-question report one-hots (deterministic)
  KNN  : k-nearest-neighbour average in (one-hot x_q, v) space over the fit set
For each width, 6 seeds -> within-group coherence (package statistic), KL to Bayes, |N|; cross-family coherences.
Also (B2) unbiased latent noise: config log-weights perturbed by iid noise with count-dependent sd, exact mixture readout.
Usage: python rv_cex_3_approx.py"""
import itertools, json, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
HERE = os.path.dirname(os.path.abspath(__file__)); PK = os.path.join(HERE, "..", ".."); sys.path.insert(0, os.path.join(PK, "src"))
import numpy as np
import cri as R
E = R.E
Gd = np.load(os.path.join(PK, "logs", "conf", "generator.npz")); q = Gd["qsel"]; NE = len(q); idx = np.arange(NE)
okN, okR = Gd["ok_N"].astype(bool), Gd["ok_R"].astype(bool)
soft = lambda c: np.exp(c - c.max(-1, keepdims=True)) / np.exp(c - c.max(-1, keepdims=True)).sum(-1, keepdims=True)


def feats(X, qq):
    """(one-hot own-question reports (18), one-hot leave-q counts (18), own-question agreement (3))"""
    i = np.arange(len(X)); xq = X[i, qq]; v = E.loo(X)[i, qq]; aq = E.agree(X)[i, qq]
    return np.c_[np.eye(3)[xq].reshape(len(X), -1), np.eye(6)[v].reshape(len(X), -1), aq].astype(float), v, xq


Xf = Gd["Xfit"]; Xfit = np.repeat(Xf, 6, 0); qfit = np.tile(np.arange(6), len(Xf))
Pf = E.bayes_from(Xfit, E.loo(Xfit))[np.arange(len(Xfit)), qfit]; ok = np.isfinite(Pf).all(-1)
Ff, vf, xqf = feats(Xfit[ok], qfit[ok]); Tf = R.bayes_logodds(Pf[ok])
ev = {f: feats(Gd[x], q) for f, x in (("base", "X"), ("N", "X_N"), ("R", "X_R"))}
cB = {"base": Gd["cB_base"], "N": Gd["cB_N"], "R": Gd["cB_R"]}; Pb = soft(cB["base"])
KL = lambda c: float((Pb * (np.log(Pb + 1e-12) - np.log(soft(c) + 1e-12))).sum(-1).mean())


def ridge(H, T, lam=1e-3):
    return np.linalg.solve(H.T @ H + lam * len(H) * np.eye(H.shape[1]), H.T @ T)


def rf_model(w, seed):
    r = np.random.default_rng(seed); W = r.standard_normal((Ff.shape[1], w)) / np.sqrt(4.0); b = r.standard_normal(w) * 0.5
    phi = lambda F: np.c_[np.maximum(F @ W + b, 0), np.ones(len(F))]
    B = ridge(phi(Ff), Tf, 1e-5); return {f: (lambda c: c - c.mean(-1, keepdims=True))(phi(ev[f][0]) @ B) for f in ev}


def poly_model():
    def phi(F, v, xq):
        vv = v / 5.0; quad = np.c_[vv, vv ** 2, vv[:, [0]] * vv[:, [1]], vv[:, [0]] * vv[:, [2]], vv[:, [1]] * vv[:, [2]], np.ones(len(v))]
        oh = np.c_[F[:, :18], F[:, 36:39], np.ones(len(F))]
        return (oh[:, :, None] * quad[:, None, :]).reshape(len(F), -1)
    B = ridge(phi(Ff, vf, xqf), Tf, 1e-6); return {f: (lambda c: c - c.mean(-1, keepdims=True))(phi(*ev[f]) @ B) for f in ev}


def knn_model(k=25, sub=None, seed=0):
    r = np.random.default_rng(seed); s = r.choice(len(Ff), sub, replace=False) if sub else np.arange(len(Ff))
    A, T = Ff[s], Tf[s]; out = {}
    for f in ev:
        F = ev[f][0]; d = (F ** 2).sum(1)[:, None] - 2 * F @ A.T + (A ** 2).sum(1)[None]
        nn = np.argpartition(d, k, 1)[:, :k]; c = T[nn].mean(1); out[f] = c - c.mean(-1, keepdims=True)
    return out


def summarize(models):
    rN = [m["N"] - m["base"] for m in models]; rR = [m["R"] - m["base"] for m in models]
    dBR = (cB["R"] - cB["base"])[okR]
    rRe = [r[okR] - ((r[okR] * dBR).sum() / (dBR ** 2).sum()) * dBR for r in rR]
    pr = lambda L, mk=None: float(np.mean([np.corrcoef(a.ravel() if mk is None else a[mk].ravel(), b.ravel() if mk is None else b[mk].ravel())[0, 1]
                                          for a, b in itertools.combinations(L, 2)])) if len(L) > 1 else np.nan
    return dict(coh_N=pr(rN, okN), coh_N_prob=pr([soft(m["N"]) - soft(m["base"]) for m in models], okN), coh_Rerr=pr(rRe),
                KL=float(np.median([KL(m["base"]) for m in models])), mean_abs_N=float(np.mean([np.abs(r[okN]).mean() for r in rN]))), rN


out = {}; groups = {}
print("approximators fitted to exact Bayes on natural episodes; evaluated on the conf N/R pairs (6 seeds per width)")
for w in (64, 256, 1024, 2048):
    ms = [rf_model(w, 1000 * w + s) for s in range(6)]; sm, rN = summarize(ms); out[f"RF-{w}"] = sm; groups[f"RF-{w}"] = rN
    print(f"  RF-{w:<5d} coh_N {sm['coh_N']:+.3f} (prob {sm['coh_N_prob']:+.3f}) coh_Rerr {sm['coh_Rerr']:+.3f} KL {sm['KL']:.4f} |N| {sm['mean_abs_N']:.3f}", flush=True)
pm = poly_model(); km = [knn_model(25, 6000, s) for s in range(3)]
for nm, ms in (("POLY", [pm]), ("KNN", km)):
    sm, rN = summarize(ms); out[nm] = sm; groups[nm] = rN
    print(f"  {nm:8s} coh_N {sm['coh_N']:+.3f} KL {sm['KL']:.4f} |N| {sm['mean_abs_N']:.3f}", flush=True)
cr = {}
for a, b in itertools.combinations(groups, 2):
    cr[f"{a}~{b}"] = float(np.mean([np.corrcoef(x[okN].ravel(), y[okN].ravel())[0, 1] for x in groups[a] for y in groups[b]]))
out["cross_family_coherence"] = cr
print("  cross-family null-response coherence:", {k: round(v, 3) for k, v in cr.items()})

# (B2) unbiased latent noise in configuration log-weights, exact mixture readout, count-dependent noise sd
def JV(X):
    J = np.zeros((len(X), 6, 3)); v = E.loo(X)[idx, q]
    for k in range(6):
        s = q == k; J[s] = R.J_at(X[s], k)
    return J, v
D = {f: JV(Gd[x]) for f, x in (("base", "X"), ("N", "X_N"), ("R", "X_R"))}
lpi = {f: np.log(np.maximum(E.pi_of_n(D[f][1]), 1e-30)) for f in D}
B2 = {}
for s0 in (0.5, 1.0, 2.0):
    ms = []
    for m in range(6):
        r = np.random.default_rng(5000 + m + int(10 * s0)); o = {}
        for f in D:
            sd = s0 * np.exp(0.6 * (10 - D[f][1].sum(-1)) / 5.0)[:, None]
            w = soft(lpi[f] + sd * r.standard_normal(lpi[f].shape)); P = (w[..., None] * D[f][0]).sum(1); o[f] = R.bayes_logodds(P / P.sum(-1, keepdims=True))
        ms.append(o)
    sm, _ = summarize(ms); B2[s0] = sm
    print(f"  (B2) latent noise sd0 {s0}: coh_N {sm['coh_N']:+.3f} (prob {sm['coh_N_prob']:+.3f}) KL {sm['KL']:.4f} |N| {sm['mean_abs_N']:.3f}")
out["B2_latent_noise"] = B2
json.dump(out, open(os.path.join(PK, "logs", "review_checks", "rv_cex_3_approx.json"), "w"), indent=1, default=float)
