"""R2 check: does the X4 matched relation-blind contrast and the C2 q_R interchange
distinguish C' from a same-architecture twin whose block-1 MLP writes a PER-SOURCE (additive)
gate feature onto e3 (no unit reads two sources)?"""
import copy, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
PKG = HERE.parents[1]
sys.path.insert(0, str(PKG / "src")); sys.path.insert(0, str(PKG.parent / "step1_causal_bridge_deep_investigation" / "src"))
import cbmath
from relation_free_equivalent import relation_free

C, D, E = cbmath.rereading(); Cb, _, _ = cbmath.rereading(block1_mlp=False)
e1, e2, e3 = D["e1"], D["e2"], D["e3"]
R = 3
mg = lambda m, a, b, hk=None: cbmath.margin(cbmath.forward(m, E(a, b), R, hk))

# e3 magnitude of C' q at R for reference
tr = cbmath.forward(C, E(1, 1), R); print("C' q_R(1,1) e3 =", tr.mlp_out[0][R] @ e3, " q_R(1,-1)=", cbmath.forward(C, E(1, -1), R).mlp_out[0][R] @ e3)

def additive_twin(k=20.0, lam=None):
    m = copy.deepcopy(C); L1 = m.layers[0]
    q11 = cbmath.forward(C, E(1, 1), R).mlp_out[0][R] @ e3
    lam = lam or q11 / 1.0
    # phi(t)=clamp(k t,-1,1) = relu(kt+1)-relu(kt-1)-1 ; units read e1 only or e2 only
    L1.Win = np.column_stack([k * e1, k * e1, k * e2, k * e2]); L1.bin = np.array([1., -1., 1., -1.])
    L1.Wout = np.stack([lam * e3, -lam * e3, lam * e3, -lam * e3]) * 0.5
    L1.bout = -2.0 * lam * e3 * 0.5 - lam * 0.5 * e3   # q = lam/2 (phi_u+phi_v) - lam/2  -> +lam/2 iff both positive
    L1.act = "relu"
    return m

T = additive_twin()
reads_both = [abs(T.layers[0].Win[:, j] @ e1) > 1e-12 and abs(T.layers[0].Win[:, j] @ e2) > 1e-12 for j in range(4)]
print("twin: any unit reads both sources:", any(reads_both))
# blind pathway identical by construction (ablating block-1 MLP gives the same model as C' ablated)
Tb = copy.deepcopy(T); Tb.layers[0].mlp_on = False
g = np.concatenate([-np.linspace(3, 0.1, 30), np.linspace(0.1, 3, 30)])
print("blind margins identical C' vs twin:", max(abs(mg(Cb, a, b) - mg(Tb, a, b)) for a in g[::5] for b in g[::5]))
agree = same = 0; n = 0
for a in g:
    for b in g:
        n += 1; same += np.sign(mg(C, a, b)) == np.sign(mg(T, a, b))
print(f"twin vs C' decision agreement on 60x60 grid |a|,|b|>=0.1: {same/n:.4f}")
print("X4 pairs on twin:")
for (a, b), (a2, b2) in [((1, 1), (3, -0.2381)), ((1.5, .5), (3, -0.2002)), ((.6, .6), (2, -0.6343))]:
    print(f"  ({a},{b}) blind {mg(Tb,a,b):+.4f} m_twin {mg(T,a,b):+.4f} | ({a2},{b2}) blind {mg(Tb,a2,b2):+.4f} m_twin {mg(T,a2,b2):+.4f}  m_C' {mg(C,a,b):+.3f}/{mg(C,a2,b2):+.3f}")

# C2 on twin: q_R interchange share
rng = np.random.default_rng(51); sh = []; fl = []
for a, b in rng.uniform(0.3, 3.0, (200, 2)):
    base = cbmath.forward(T, E(a, b), R); cf = cbmath.forward(T, E(a, -b), R)
    v = base.mlp_out[0].copy(); v[3] = cf.mlp_out[0][3]
    p = mg(T, a, b, cbmath.replace_site(0, "mlp_out", v))
    TE = cbmath.margin(cf) - cbmath.margin(base); sh.append((p - cbmath.margin(base)) / TE); fl.append(np.sign(p) == np.sign(cbmath.margin(cf)))
print(f"twin C2: q_R interchange median share {np.median(sh):.3f}, decision->cf {np.mean(fl):.3f}")

# Relation-free twin of relation_free_equivalent.py under the X4 criterion
mR, embR = relation_free(); mRb = copy.deepcopy(mR); mRb.layers[0].mlp_on = False
f = lambda m, a, b: cbmath.margin(cbmath.forward(m, embR(a, b), 3))
print("RF twin blind margin (block-1 MLP removed) at (1,1),(3,-0.238),(-2,1):", [round(f(mRb, *p), 4) for p in [(1, 1), (3, -0.238), (-2, 1)]])
print("RF twin full margins on X4 pair:", f(mR, 1, 1), f(mR, 3, -0.2381))
