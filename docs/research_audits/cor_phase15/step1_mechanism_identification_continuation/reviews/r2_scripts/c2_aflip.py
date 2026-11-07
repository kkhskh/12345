import sys
from pathlib import Path
import numpy as np
PKG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG / "src")); sys.path.insert(0, str(PKG.parent / "step1_causal_bridge_deep_investigation" / "src"))
import cbmath
C, D, E = cbmath.rereading(); R = 3
rng = np.random.default_rng(51); rows = []
for a, b in rng.uniform(0.3, 3.0, (200, 2)):
    base = cbmath.forward(C, E(a, b), R); cf = cbmath.forward(C, E(-a, b), R)
    mb, mc = cbmath.margin(base), cbmath.margin(cf)
    pv = cbmath.margin(cbmath.forward(C, E(a, b), R, cbmath.replace_site(1, "V0", cf.V[1][0])))
    ps = cbmath.margin(cbmath.forward(C, E(a, b), R, cbmath.replace_site(1, "scores0", cf.scores[1][0])))
    v = base.mlp_out[0].copy(); v[3] = cf.mlp_out[0][3]
    pq = cbmath.margin(cbmath.forward(C, E(a, b), R, cbmath.replace_site(0, "mlp_out", v)))
    rows.append([(x - mb) / (mc - mb) for x in (pq, ps, pv)] + [np.sign(pv) == np.sign(mc), base.pattern[1][0][R][1]])
r = np.array(rows, float)
print("(a,b)->(-a,b): median share q_R %.3f scores %.3f values %.3f ; values decision->cf %.3f; median alpha_A %.3f" % tuple(list(np.median(r[:, :3], 0)) + [r[:, 3].mean(), np.median(r[:, 4])]))
print("values share pct10/50/90:", np.percentile(r[:, 2], [10, 50, 90]).round(3))
