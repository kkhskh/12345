"""(a) Sensitivity of S_rel to number of fit pairs, pair seed, and forced k; plus content R2 for the margin itself."""
import numpy as np
from common import *  # noqa

T = "REL-XNOR"
for where, s in XNOR_TF2:
    P = load("T1", "TF2", s, where)
    site = MD.Site(P, T, "resid_mid1", "R")
    Ae, Ce = G.matched_pairs(T, 500, np.random.default_rng(6002))
    A0, C0 = G.matched_pairs(T, 800, np.random.default_rng(5001))
    U0, k0, sv = MD.s_rel(site, A0, C0)
    rows = []
    for n in (10, 25, 100, 800, 3200):
        for ps in (11, 12, 13):
            A, C = G.matched_pairs(T, n, np.random.default_rng(ps * 1000 + n))
            U, k, _ = MD.s_rel(site, A, C)
            cos = float(np.linalg.svd(U0[:, :1].T @ U, compute_uv=False).max())
            r = causal(site, Ae, Ce, U)
            rows.append((n, ps, k, round(cos, 3), r["iia"], r["nec"], r["suf"]))
    worst = dict(k_set=sorted({x[2] for x in rows}), min_cos=min(x[3] for x in rows), min_iia=min(x[4] for x in rows),
                 max_nec=max(x[5] for x in rows), min_suf=min(x[6] for x in rows),
                 n10=[x for x in rows if x[0] == 10])
    forced = {}
    _, hA = site.get(A0); _, hC = site.get(C0)
    _, _, Vt = np.linalg.svd(hA - hC, full_matrices=False)
    for kk in (1, 2, 3, 4):
        U = Vt[:kk].T
        r = causal(site, Ae, Ce, U)
        rr = [MD.iia(site, Ae, Ce, MD.s_rand(16, kk, q))[0] for q in range(20)]
        forced[kk] = (r["iia"], r["nec"], r["suf"], round(float(np.mean(rr)), 3), round(float(np.max(rr)), 3))
    # margin (logit difference) content on the grid: interaction part R2 by sign(ab)
    g = np.linspace(-3, 3, 61); Aa, Bb = np.meshgrid(g, g, indexing="ij")
    d = pack(T, Aa.ravel(), Bb.ravel())
    m = MD.margin(np.asarray(M.forward(P, "T1", MD.to_inp(T, d)))).reshape(61, 61)
    Rz = m - (m.mean(1, keepdims=True) + m.mean(0, keepdims=True) - m.mean())
    sab = np.sign(Aa * Bb).ravel(); x = Rz.ravel()
    r2m = 1 - np.var(x - np.polyval(np.polyfit(sab, x, 1), sab)) / np.var(x)
    addr2m = 1 - (Rz ** 2).sum() / ((m - m.mean()) ** 2).sum()
    print(dict(seed=s, k_protocol=k0, sv_energy=np.round(sv[:4], 3).tolist(), fit_sweep=worst,
               forced_k=forced, margin_interaction_R2_sign_ab=round(float(r2m), 4), margin_additive_R2=round(float(addr2m), 3)), flush=True)
