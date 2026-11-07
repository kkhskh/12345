"""(g2) Content test applied to the weight-defined readout direction w (no data, no relation labels)."""
import numpy as np
from common import *  # noqa
T = "REL-XNOR"
for where, s in XNOR_TF2:
    P = load("T1", "TF2", s, where); site = MD.Site(P, T, "resid_mid1", "R")
    w = readout_dir(P)
    Ae, Ce = G.matched_pairs(T, 500, np.random.default_rng(6002)); ev = G.sample(T, 3000, np.random.default_rng(6004))
    gc = MD.GridCoords(site, w)
    sur = {k: MD.surrogate_test(site, w, gc, k, ev, Ae, Ce) for k in ("exact", "additive")}
    g = gc.g; Aa, Bb = np.meshgrid(g, g, indexing="ij"); sab = np.sign(Aa * Bb).ravel()
    Z = gc.C[0.0][..., 0]; Rz = Z - (Z.mean(1, keepdims=True) + Z.mean(0, keepdims=True) - Z.mean()); x = Rz.ravel()
    r2 = 1 - np.var(x - np.polyval(np.polyfit(sab, x, 1), sab)) / np.var(x)
    print(s, "addR2", round(gc.additive_r2(), 3), "intR2_sign_ab", round(float(r2), 4),
          {k: (round(v["acc"], 3), round(v["pair_flip"], 3)) for k, v in sur.items()})
