"""s4 necessity across eval-pair seeds (protocol generator) and fit seeds."""
import numpy as np
from common import *  # noqa
T = "REL-XNOR"
for s in (2, 4, 7, 8, 9):
    P = load("T1", "TF2", s); site = MD.Site(P, T, "resid_mid1", "R")
    U, k, _ = MD.s_rel(site, *G.matched_pairs(T, 800, np.random.default_rng(5001)))
    nec = [causal(site, *G.matched_pairs(T, 500, np.random.default_rng(es)), U)["nec"] for es in range(7000, 7020)]
    necf = []
    for fs in range(8000, 8010):
        Uf, _, _ = MD.s_rel(site, *G.matched_pairs(T, 800, np.random.default_rng(fs)))
        necf.append(causal(site, *G.matched_pairs(T, 500, np.random.default_rng(6002)), Uf)["nec"])
    print(s, "eval-seed nec min/med/max", min(nec), np.median(nec), max(nec), "n>0.2:", sum(x > 0.2 for x in nec),
          "| fit-seed nec min/max", min(necf), max(necf))
