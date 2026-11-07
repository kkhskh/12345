"""R2: specificity controls for the probe-direction interchange (learned_posthoc.interchange_differing)."""
import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, jax.numpy as jnp
PKG = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(PKG / "src"))
import learned as Lm, learned_analysis as LA
from sklearn.linear_model import LogisticRegression, Ridge

def unit(v): v = np.asarray(v, float); v = v - v.mean(); return v / np.linalg.norm(v)
def run(task, arch, seed, pv=1.0):
    P = LA.load(task, arch, seed); L = len(P["layers"]); site = f"resid_post{L-1}"
    agp, cfp = LA.matched_pairs()
    bi = LA.inp_bin(agp[:, 0], agp[:, 1], np.full(len(agp), pv)); si = LA.inp_bin(cfp[:, 0], cfp[:, 1], np.full(len(cfp), pv))
    mb = LA.fwd_margin(P, task, bi); ms = LA.fwd_margin(P, task, si); diff = np.sign(mb) != np.sign(ms)
    bt = LA.sample_binary(task, 4000, np.random.default_rng(3000))
    _, st = LA.capture(P, task, LA.inp_bin(bt.a, bt.b, bt.p)); X = st[site][:2000, -1]
    _, ss = LA.capture(P, task, si); _, sb = LA.capture(P, task, bi)
    dirs = {}
    dirs["probe sign(ab)"] = LogisticRegression(C=10., max_iter=3000).fit(X, (bt.a * bt.b > 0)[:2000]).coef_[0]
    dirs["probe sign(a)"] = LogisticRegression(C=10., max_iter=3000).fit(X, (bt.a > 0)[:2000]).coef_[0]
    dirs["probe sign(b)"] = LogisticRegression(C=10., max_iter=3000).fit(X, (bt.b > 0)[:2000]).coef_[0]
    dirs["ridge a+b"] = Ridge(1e-3).fit(X, (bt.a + bt.b)[:2000]).coef_
    dirs["readout gf*dWU"] = np.asarray(P["gf"] * (P["WU"][:, 0] - P["WU"][:, 1]))
    rng = np.random.default_rng(999); dirs["random"] = rng.normal(size=Lm.D_MODEL)
    # random direction rescaled: random dir patched with magnitude matched to base-source difference (patch full diff projected on random 1D)
    ro = unit(dirs["readout gf*dWU"])
    out = {"n_diff": int(diff.sum())}
    for nm, w in dirs.items():
        u = np.asarray(w, float); u = u / np.linalg.norm(u)
        sc = ss[site][:, -1] @ u
        def hk(name, arr, u=u, sc=sc):
            if name == site:
                a_ = np.asarray(arr).copy(); a_[:, -1] = a_[:, -1] + np.outer(sc - a_[:, -1] @ u, u); return jnp.asarray(a_)
            return arr
        mp = LA.fwd_margin(P, task, bi, hk)
        out[nm] = (round(float(np.mean(np.sign(mp[diff]) == np.sign(ms[diff]))), 3) if diff.sum() else None, round(float(abs(unit(w) @ ro)), 3))
    return out
for task in ["T1", "T3", "T4"]:
    for arch in ["TF1", "TF2", "TF2AO"]:
        for seed in [0, 3]:
            print(task, arch, seed, run(task, arch, seed))
