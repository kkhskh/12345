import sys, copy
from pathlib import Path
import numpy as np
PKG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG / "src")); sys.path.insert(0, str(PKG.parent / "step1_causal_bridge_deep_investigation" / "src"))
import cbmath
from x4_twin import additive_twin
C, D, E = cbmath.rereading(); R = 3; fU = D["fU"]
def info(m, a, b, hk=None):
    t = cbmath.forward(m, E(a, b), R, hk)
    A = t.pattern[1][0][R]; v = (t.VO[1][0] @ (m.WU[:, 0] - m.WU[:, 1]))
    return cbmath.margin(t), A, v
for ab in [(1, 1), (2, 0.5), (1, -1)]:
    m, A, v = info(C, *ab); print(ab, "m", round(m, 3), "alpha", A.round(3), "v", v.round(3))
# value interchange
rng = np.random.default_rng(51); rows = []
for a, b in rng.uniform(0.3, 3.0, (200, 2)):
    base = cbmath.forward(C, E(a, b), R); cf = cbmath.forward(C, E(a, -b), R)
    pv = cbmath.margin(cbmath.forward(C, E(a, b), R, cbmath.replace_site(1, "V0", cf.V[1][0])))
    ps = cbmath.margin(cbmath.forward(C, E(a, b), R, cbmath.replace_site(1, "scores0", cf.scores[1][0])))
    # reverse direction: values from agree into conflict run
    pv2 = cbmath.margin(cbmath.forward(C, E(a, -b), R, cbmath.replace_site(1, "V0", base.V[1][0])))
    mb, mc = cbmath.margin(base), cbmath.margin(cf)
    rows.append(((pv - mb) / (mc - mb), (ps - mb) / (mc - mb), (pv2 - mc) / (mb - mc), (pv - mb), (ps-mb)+(pv-mb) - (mc-mb)))
r = np.array(rows)
print("value-share agree<-conf median %.3f  IQR %s" % (np.median(r[:, 0]), np.percentile(r[:, 0], [10, 25, 75, 90]).round(3)))
print("value-share conf<-agree (reverse) median %.3f  pct %s" % (np.median(r[:, 2]), np.percentile(r[:, 2], [10, 25, 75, 90]).round(3)))
print("raw value effect agree-run median %.3f; interaction (score+value-TE) median %.3f" % (np.median(r[:, 3]), np.median(r[:, 4])))
# factorial on q_R across all 4 quadrants, C' vs additive twin; and routing to P
T = additive_twin()
for name, m in [("C'", C), ("twin", T)]:
    out = []
    for ab in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        t = cbmath.forward(m, E(*ab), R)
        out.append((ab, round(float(t.mlp_out[0][R] @ D["e3"]), 3), round(float(t.pattern[1][0][R][0]), 3), round(cbmath.margin(t), 3)))
    print(name, "(ab, q_R, alpha_P, m):", out)
