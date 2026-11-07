import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, jax.numpy as jnp
PKG = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(PKG / "src"))
import learned as Lm, learned_analysis as LA
from sklearn.linear_model import LogisticRegression
for task, arch in [("T3", "TF2"), ("T3", "TF1"), ("T1", "TF2")]:
    P = LA.load(task, arch, 0); L = len(P["layers"]); pv = 1.0 if task == "T3" else 0.0
    agp, cfp = LA.matched_pairs()
    bi = LA.inp_bin(agp[:, 0], agp[:, 1], np.full(len(agp), pv)); si = LA.inp_bin(cfp[:, 0], cfp[:, 1], np.full(len(cfp), pv))
    mb = LA.fwd_margin(P, task, bi); ms = LA.fwd_margin(P, task, si); diff = np.sign(mb) != np.sign(ms)
    bt = LA.sample_binary(task, 4000, np.random.default_rng(3000)); _, st = LA.capture(P, task, LA.inp_bin(bt.a, bt.b, bt.p))
    _, ss = LA.capture(P, task, si)
    for site in [f"resid_mid{l}" for l in range(L)] + [f"resid_post{l}" for l in range(L)]:
        X = st[site][:2000, -1]; res = {}
        W = {nm: LogisticRegression(C=10., max_iter=3000).fit(X, lab[:2000]).coef_[0] for nm, lab in
             [("sab", bt.a * bt.b > 0), ("sa", bt.a > 0), ("sb", bt.b > 0)]}
        for nm, dirs in [("sign(ab)", ["sab"]), ("sign(a)", ["sa"]), ("sign(b)", ["sb"]), ("span{sa,sb}", ["sa", "sb"]), ("span{sa,sb,sab}", ["sa", "sb", "sab"])]:
            Q, _ = np.linalg.qr(np.column_stack([W[d] for d in dirs]))
            def hk(name, arr, Q=Q):
                if name == site:
                    a_ = np.asarray(arr).copy(); a_[:, -1] = a_[:, -1] + (ss[site][:, -1] - a_[:, -1]) @ Q @ Q.T; return jnp.asarray(a_)
                return arr
            mp = LA.fwd_margin(P, task, bi, hk); res[nm] = round(float(np.mean(np.sign(mp[diff]) == np.sign(ms[diff]))), 3)
        print(task, arch, site, "n_diff", int(diff.sum()), res)
