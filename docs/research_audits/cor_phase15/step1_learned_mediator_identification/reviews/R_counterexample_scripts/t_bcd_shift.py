"""(b) |z| ranges and agreement boundary; (c) extrapolation outside [-3,3]; (d) Gaussian matched-pair generator."""
import numpy as np
from common import *  # noqa

T = "REL-XNOR"


def gauss_pairs(n, rng, sd=1.2, lim=3.0):
    """Agreement member (a,b) ~ N(0,sd^2)^2 | ab>0; conflict partner with same z, d' ~ N(0, 2 sd^2) | |d'|>|z|."""
    A_, C_ = [], []
    while len(A_) < n:
        a, b = rng.normal(0, sd, 2)
        if a * b <= 0 or abs(a + b) < 0.05:
            continue
        z = a + b
        for _ in range(200):
            d = rng.normal(0, sd * np.sqrt(2))
            if abs(d) > abs(z) + 1e-3 and abs((z + d) / 2) <= lim and abs((z - d) / 2) <= lim:
                A_.append((a, b)); C_.append(((z + d) / 2, (z - d) / 2)); break
    A_, C_ = np.array(A_), np.array(C_)
    return pack(T, A_[:, 0], A_[:, 1]), pack(T, C_[:, 0], C_[:, 1])


for where, s in XNOR_TF2:
    P = load("T1", "TF2", s, where)
    site = MD.Site(P, T, "resid_mid1", "R")
    A0, C0 = G.matched_pairs(T, 800, np.random.default_rng(5001))
    U, k, _ = MD.s_rel(site, A0, C0)
    rng = np.random.default_rng(900 + s)
    res = {}
    for nm, kw in [("z~0.2", dict(zlo=0.15, zhi=0.25)), ("z~2.5", dict(zlo=2.4, zhi=2.6)),
                   ("boundary(u.9-.98,e.02-.1)", dict(zlo=0.2, zhi=2.5, da_frac=(0.9, 0.98), dc_extra=(0.02, 0.1))),
                   ("boundary(u.99-.999,e.001-.01)", dict(zlo=0.2, zhi=2.5, da_frac=(0.99, 0.999), dc_extra=(0.001, 0.01))),
                   ("tiny z 0.01-0.05", dict(zlo=0.01, zhi=0.05)),
                   ("lim5 z0.2-4.5", dict(zlo=0.2, zhi=4.5, lim=5.0)),
                   ("lim5 z3-4.8", dict(zlo=3.0, zhi=4.8, lim=5.0))]:
        A, C = custom_pairs(T, 600, rng, **kw)
        accA = float((MD.model_dec(site, A) == A["y"]).mean()); accC = float((MD.model_dec(site, C) == C["y"]).mean())
        r = causal(site, A, C, U); r.update(accA=round(accA, 3), accC=round(accC, 3))
        res[nm] = r
    # pairs where some coordinate exceeds 3 (lim 5) only
    A, C = custom_pairs(T, 3000, rng, zlo=0.2, zhi=4.5, lim=5.0)
    out = (np.maximum.reduce([np.abs(A["a"]), np.abs(A["b"]), np.abs(C["a"]), np.abs(C["b"])]) > 3.0)
    A = {kk: v[out] for kk, v in A.items()}; C = {kk: v[out] for kk, v in C.items()}
    r = causal(site, A, C, U)
    r.update(accA=round(float((MD.model_dec(site, A) == A["y"]).mean()), 3), accC=round(float((MD.model_dec(site, C) == C["y"]).mean()), 3))
    res["lim5 any|x|>3"] = r
    # accuracy on uniform [-5,5]^2 outside [-3,3]^2
    a = rng.uniform(-5, 5, 20000); b = rng.uniform(-5, 5, 20000); o = (np.abs(a) > 3) | (np.abs(b) > 3)
    d = pack(T, a[o], b[o]); res["acc_outside3_in5"] = round(float((MD.model_dec(site, d) == d["y"]).mean()), 3)
    a = rng.uniform(-8, 8, 20000); b = rng.uniform(-8, 8, 20000); o = (np.abs(a) > 5) | (np.abs(b) > 5)
    d = pack(T, a[o], b[o]); res["acc_outside5_in8"] = round(float((MD.model_dec(site, d) == d["y"]).mean()), 3)
    # Gaussian generator
    Ag, Cg = gauss_pairs(600, rng)
    res["gauss_pairs"] = causal(site, Ag, Cg, U)
    Ug, kg, _ = MD.s_rel(site, *gauss_pairs(800, np.random.default_rng(31)))
    res["gauss_fit_k"] = kg; res["cos(U_train,U_gauss)"] = round(float(np.linalg.svd(U.T @ Ug, compute_uv=False).max()), 3)
    res["U_gauss on train eval"] = causal(site, *G.matched_pairs(T, 500, np.random.default_rng(6002)), Ug)
    print(f"seed {s} k={k}")
    for kk, v in res.items():
        print("   ", kk, v)
