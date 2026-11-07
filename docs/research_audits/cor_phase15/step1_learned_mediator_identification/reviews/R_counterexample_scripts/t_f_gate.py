"""(f) REL-GATE: is the source-additive verdict robust to grid resolution, grid range, p-subsets, and eval pair ranges?"""
import numpy as np
from common import *  # noqa
T = "REL-GATE"
def sub(X, m): return {k: v[m] for k, v in X.items()}
for arch, hook in (("TF2", "resid_mid1"), ("TF1", "resid_mid0")):
    for s in (2, 3, 4, 7, 8, 9):
        P = load("T3", arch, s); site = MD.Site(P, T, hook, "R")
        U, k, _ = MD.s_rel(site, *G.matched_pairs(T, 800, np.random.default_rng(5001)))
        Ae, Ce = G.matched_pairs(T, 500, np.random.default_rng(6002))
        ev = G.sample(T, 3000, np.random.default_rng(6004))
        res = {}
        for n, lim in ((61, 3.0), (121, 3.0), (241, 3.0), (121, 2.0)):
            gc = MD.GridCoords(site, U, n=n, lim=lim)
            ex = MD.surrogate_test(site, U, gc, "exact", ev, Ae, Ce); ad = MD.surrogate_test(site, U, gc, "additive", ev, Ae, Ce)
            res[f"grid{n}/lim{lim}"] = (round(gc.additive_r2(), 3), round(ex["acc"] - ad["acc"], 3), round(ex["pair_flip"] - ad["pair_flip"], 3))
        gc = MD.GridCoords(site, U, n=121)
        for pv in (-1.0, 1.0):
            m = Ae["p"] == pv; mev = ev["p"] == pv
            ex = MD.surrogate_test(site, U, gc, "exact", sub(ev, mev), sub(Ae, m), sub(Ce, m))
            ad = MD.surrogate_test(site, U, gc, "additive", sub(ev, mev), sub(Ae, m), sub(Ce, m))
            res[f"p={pv:+.0f}"] = (round(ex["acc"] - ad["acc"], 3), round(ex["pair_flip"] - ad["pair_flip"], 3))
        for nm, kw in (("z~0.2", dict(zlo=0.15, zhi=0.25)), ("boundary", dict(zlo=0.2, zhi=2.5, da_frac=(0.9, 0.98), dc_extra=(0.02, 0.1)))):
            rng = np.random.default_rng(55)
            pp = rng.choice([-1.0, 1.0], 600)
            A, C = custom_pairs(T, 600, rng, p=pp, **kw)
            ex = MD.surrogate_test(site, U, gc, "exact", ev, A, C); ad = MD.surrogate_test(site, U, gc, "additive", ev, A, C)
            res[nm] = (int((MD.model_dec(site, A) != MD.model_dec(site, C)).sum()), round(ex["pair_flip"] - ad["pair_flip"], 3), causal(site, A, C, U))
        print(arch, s, "k", k, res, flush=True)
