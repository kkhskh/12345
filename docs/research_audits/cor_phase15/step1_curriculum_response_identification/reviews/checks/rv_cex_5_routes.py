"""Counterexample reviewer, check 5: can the package's route-selectivity statistic rem = <D_block, D_full>/|D_full|^2 report
'blocking the readout route removes most of the null response and keeps the relevant one' in a model where blocking makes the
null response LARGER and the model LESS Bayesian?
(a) Toy two-route model on the exact conf N/R pairs.  Layer-2 input is additive over routes: source route S carries the exact
    Bayes response plus  +a*dSum*w + n_S ; readout route Rq carries  -a*dSum*w + n_R  (a large count-level signal w that
    cancels between routes, plus small independent route errors n_S, n_R).  Natural response = Bayes + n_S + n_R.
(b) The same quantities for the real cohort T1 / P models from the frozen path-patching arrays: rem, and the NORM ratio
    |D_block| / |D_full| on N, and the R-error norm natural vs Rq-blocked.
Usage: python rv_cex_5_routes.py"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); PK = os.path.join(HERE, "..", "..")
import numpy as np
Gd = np.load(os.path.join(PK, "logs", "conf", "generator.npz")); okN, okR = Gd["ok_N"].astype(bool), Gd["ok_R"].astype(bool)
rng = np.random.default_rng(3); NE = len(okN)
cen = lambda c: c - c.mean(-1, keepdims=True)
sys.path.insert(0, os.path.join(PK, "src"))
import cri as R
E = R.E; q = Gd["qsel"]; idx = np.arange(NE)
dSum = {f: E.loo(Gd[x])[idx, q].sum(-1) - E.loo(Gd["X"])[idx, q].sum(-1) for f, x in (("N", "X_N"), ("R", "X_R"))}
dB = {f: Gd[f"cB_{f}"] - Gd["cB_base"] for f in ("N", "R")}
w = cen(rng.standard_normal((NE, 3)))                               # per-episode direction of the cancelling level signal
rem = lambda d, full, mk: float((d[mk] * full[mk]).sum() / (full[mk] ** 2).sum())
nrm = lambda d, mk: float(np.linalg.norm(d[mk], axis=-1).mean())
out = {"toy": {}}
for a in (0.15, 0.3):
    r = {}
    for f in ("N", "R"):
        nS, nR = 0.05 * cen(rng.standard_normal((NE, 3))), 0.05 * cen(rng.standard_normal((NE, 3)))
        S = dB[f] + a * dSum[f][:, None] * w + nS; Rr = -a * dSum[f][:, None] * w + nR
        full, block = S + Rr, S; mk = okN if f == "N" else okR
        err_full, err_block = full - dB[f], block - dB[f]
        r[f] = dict(rem_block_Rq=rem(block, full, mk), norm_full=nrm(full, mk), norm_block=nrm(block, mk),
                    err_norm_full=nrm(err_full, mk), err_norm_block=nrm(err_block, mk))
    out["toy"][a] = r
    print(f"(a) toy a={a}: N rem(block Rq) {r['N']['rem_block_Rq']:.2f} ('removes {1-r['N']['rem_block_Rq']:.0%}') but |N| {r['N']['norm_full']:.3f} -> {r['N']['norm_block']:.3f}; "
          f"R rem {r['R']['rem_block_Rq']:.2f} ('keeps {r['R']['rem_block_Rq']:.0%}'), R-error norm {r['R']['err_norm_full']:.3f} -> {r['R']['err_norm_block']:.3f}")
print("(b) real cohort models: rem (package) vs norm ratio of the blocked response (mean per-pair L2 norm)")
out["real"] = {}
for m in ["cT61", "cT62", "cT63", "cT64", "cT65", "cT67", "cP61", "cP62", "cP63", "cP64", "cP66", "cP67"]:
    z = np.load(os.path.join(PK, "logs", "conf", f"model_{m}.npz")); r = {}
    for f, mk in (("N", okN), ("R", okR)):
        full = z[f"c_{f}"] - z["c_base"]; blk = z[f"c_{f}_block_Rq"] - z["c_base"]
        r[f] = dict(rem=rem(blk, full, mk), norm_ratio=nrm(blk, mk) / nrm(full, mk))
        if f == "R":
            beta = (full[mk] * dB["R"][mk]).sum() / (dB["R"][mk] ** 2).sum()
            r[f].update(err_full=nrm(full - beta * dB["R"], mk), err_block=nrm(blk - beta * dB["R"], mk),
                        err_block_refit=nrm(blk - ((blk[mk] * dB["R"][mk]).sum() / (dB["R"][mk] ** 2).sum()) * dB["R"], mk))
    out["real"][m] = r
    print(f"  {m}: N rem {r['N']['rem']:.2f} |blocked|/|natural| {r['N']['norm_ratio']:.2f} | R rem {r['R']['rem']:.2f} ratio {r['R']['norm_ratio']:.2f}; "
          f"R-error norm natural {r['R']['err_full']:.3f} -> blocked {r['R']['err_block']:.3f} (refit scale {r['R']['err_block_refit']:.3f})")
json.dump(out, open(os.path.join(PK, "logs", "review_checks", "rv_cex_5_routes.json"), "w"), indent=1, default=float)
