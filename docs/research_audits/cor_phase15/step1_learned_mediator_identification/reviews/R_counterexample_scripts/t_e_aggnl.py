"""(e) AGG-NL: do matched pairs ever get different decisions, especially near the |z|=1.5 boundary?"""
import numpy as np
from common import *  # noqa
T = "AGG-NL"
for arch in ("TF1", "TF2", "TF2AO"):
    for s in range(5):
        P = load("T6", arch, s, "new")
        rng = np.random.default_rng(40 + s)
        out = {}
        for nm, (lo, hi, lim) in {"z1.45-1.55": (1.45, 1.55, 3), "z1.49-1.51": (1.49, 1.51, 3),
                                  "z1.499-1.501": (1.499, 1.501, 3), "z0.2-2.5": (0.2, 2.5, 3), "lim5 z0.2-4.5": (0.2, 4.5, 5)}.items():
            A, C = custom_pairs(T, 2000, rng, lo, hi, lim=lim)
            lA = np.asarray(M.forward(P, "T6", MD.to_inp(T, A))); lC = np.asarray(M.forward(P, "T6", MD.to_inp(T, C)))
            dA, dC = MD.decision(T, lA), MD.decision(T, lC)
            mA, mC = MD.margin(lA), MD.margin(lC)
            out[nm] = (int((dA != dC).sum()), len(dA), round(float(np.abs(mA - mC).max()), 4), round(float(np.median(np.abs(mA))), 4))
        # locate model's own boundary: margin zero crossing along z with a=b
        zz = np.linspace(0, 3, 3001); d = pack(T, zz / 2, zz / 2)
        m = MD.margin(np.asarray(M.forward(P, "T6", MD.to_inp(T, d))))
        zc = zz[np.where(np.diff(np.sign(m)) != 0)[0]]
        # matched pairs exactly at model's own boundary z*, |z| in z*+-0.002
        if len(zc):
            A, C = custom_pairs(T, 4000, rng, zc[0] - 0.002, zc[0] + 0.002)
            lA = np.asarray(M.forward(P, "T6", MD.to_inp(T, A))); lC = np.asarray(M.forward(P, "T6", MD.to_inp(T, C)))
            out["at_model_boundary"] = (round(float(zc[0]), 4), int((MD.decision(T, lA) != MD.decision(T, lC)).sum()), len(lA),
                                        round(float(np.abs(MD.margin(lA) - MD.margin(lC)).max()), 4))
        print(arch, s, out, flush=True)
