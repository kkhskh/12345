"""(e3) AGG-NL: model's decision boundary in z for agreeing (d=0) vs conflicting (|d|=|z|+1) inputs."""
import numpy as np
from common import *  # noqa
T = "AGG-NL"
zz = np.linspace(1.2, 1.8, 6001)
for arch in ("TF1", "TF2", "TF2AO"):
    row = []
    for s in range(5):
        P = load("T6", arch, s, "new")
        zs = {}
        for nm, d in (("agree d=0", 0 * zz), ("agree d=.9z", 0.9 * zz), ("conf d=z+.1", zz + 0.1), ("conf d=z+1", zz + 1.0), ("conf d=z+2", zz + 2.0)):
            x = pack(T, (zz + d) / 2, (zz - d) / 2)
            m = MD.margin(np.asarray(M.forward(P, "T6", MD.to_inp(T, x))))
            i = np.where(np.diff(np.sign(m)) != 0)[0]
            zs[nm] = round(float(zz[i[0]]), 3) if len(i) else None
        row.append((s, zs))
    for s, zs in row:
        print(arch, s, zs)
