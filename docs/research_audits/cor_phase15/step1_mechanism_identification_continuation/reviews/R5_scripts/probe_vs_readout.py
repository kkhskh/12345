"""R5: is the T1 'agreement probe' at the last residual site just the readout (margin) direction?
Also: does interchange along the readout direction itself transfer the same decisions?"""
import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, jax.numpy as jnp
SRC = Path(__file__).resolve().parents[2] / "src"; sys.path.insert(0, str(SRC))
import learned as Lm, learned_analysis as LA
from sklearn.linear_model import LogisticRegression
for task in ("T1","T3"):
  for arch in ("TF1","TF2","TF2AO"):
    for seed in (0,1,2,3,4):
        P = LA.load(task, arch, seed); L = len(P["layers"]); site = f"resid_post{L-1}"
        bt = LA.sample_binary(task, 4000, np.random.default_rng(3000))
        _, st = LA.capture(P, task, LA.inp_bin(bt.a, bt.b, bt.p))
        X = st[site][:2000, -1]; lab = (bt.a*bt.b > 0).astype(int)[:2000]
        w = LogisticRegression(C=10.0, max_iter=3000).fit(X, lab).coef_[0]; w /= np.linalg.norm(w)
        # readout direction through final LN (linearised: centring + gain), margin = (WU0-WU1).gamma.(centred x)/sigma
        r = (P["WU"][:,0]-P["WU"][:,1]) * P["gf"]; r = r - r.mean(); r /= np.linalg.norm(r)
        # interchange along readout direction on matched pairs with differing decisions
        agp, cfp = LA.matched_pairs(); pv = 1.0
        bi = LA.inp_bin(agp[:,0], agp[:,1], np.full(len(agp), pv)); si = LA.inp_bin(cfp[:,0], cfp[:,1], np.full(len(cfp), pv))
        mb = LA.fwd_margin(P, task, bi); ms = LA.fwd_margin(P, task, si); diff = np.sign(mb) != np.sign(ms)
        _, ss = LA.capture(P, task, si)
        def tr(u):
            sc = ss[site][:, -1] @ u
            def hk(name, arr):
                if name == site:
                    a_ = np.asarray(arr).copy(); a_[:, -1] += np.outer(sc - a_[:, -1] @ u, u); return jnp.asarray(a_)
                return arr
            mp = LA.fwd_margin(P, task, bi, hk); return float(np.mean(np.sign(mp[diff]) == np.sign(ms[diff]))) if diff.any() else float("nan")
        print(f"{task} {arch} s{seed}: |cos(agree-probe, readout)|={abs(w@r):.3f}  transfer probe={tr(w):.3f}  readout={tr(r):.3f}  n_diff={int(diff.sum())}", flush=True)
