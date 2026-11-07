"""(g) Is IIA trivially 1? Readout alignment of U at resid_mid1@R, role of final MLP, unmatched-pair baseline."""
import numpy as np
import jax.numpy as jnp
from common import *  # noqa

T = "REL-XNOR"
for where, s in XNOR_TF2:
    P = load("T1", "TF2", s, where)
    site = MD.Site(P, T, "resid_mid1", "R")
    A, C = G.matched_pairs(T, 800, np.random.default_rng(5001))
    Ae, Ce = G.matched_pairs(T, 500, np.random.default_rng(6002))
    U, k, _ = MD.s_rel(site, A, C)
    w = readout_dir(P)
    cos_w = float(np.linalg.svd(U.T @ w, compute_uv=False).max())
    # final MLP knock-out: resid_post1 := resid_mid1 (at all positions; only R is read)
    store = {}
    def hk(name, arr):
        if name == "resid_mid1":
            store["m"] = arr
        if name == "resid_post1":
            return store["m"]
        return arr
    ev = G.sample(T, 3000, np.random.default_rng(6004))
    lg_full = np.asarray(M.forward(P, "T1", MD.to_inp(T, ev)))
    lg_nomlp = np.asarray(M.forward(P, "T1", MD.to_inp(T, ev), hk))
    acc_full = float((MD.decision(T, lg_full) == ev["y"]).mean())
    acc_nomlp = float((MD.decision(T, lg_nomlp) == ev["y"]).mean())
    # projection of mid1 state on readout dir: sign accuracy (after LN centring; LN does not change sign of centred projection)
    _, h = site.get(ev)
    hc = h - h.mean(1, keepdims=True)
    proj = (hc @ w)[:, 0]
    bias = float(np.asarray(P["bU"])[0] - np.asarray(P["bU"])[1])
    # unmatched baseline: diff of means agree vs conflict on train-distribution samples
    bank = G.sample(T, 4000, np.random.default_rng(77))
    _, hb = site.get(bank)
    dm = hb[bank["s"] == 1].mean(0) - hb[bank["s"] == 0].mean(0)
    dm = (dm / np.linalg.norm(dm))[:, None]
    cos_dm = float(np.abs(U[:, :1].T @ dm).max())
    out = dict(seed=s, k=k, cos_U_readout=round(cos_w, 3), cos_U_unmatchedDiffMeans=round(cos_dm, 3),
               acc_full=acc_full, acc_noFinalMLP=acc_nomlp,
               S_rel=causal(site, Ae, Ce, U), readout_dir=causal(site, Ae, Ce, w), unmatched_dm=causal(site, Ae, Ce, dm))
    print(out, flush=True)
