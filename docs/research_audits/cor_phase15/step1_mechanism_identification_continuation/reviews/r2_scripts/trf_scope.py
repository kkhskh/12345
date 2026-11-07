import sys, itertools
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
PKG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG / "src")); sys.path.insert(0, str(PKG.parent / "step1_causal_bridge_deep_investigation" / "src"))
import cbmath
from cprime import closed_form
from relation_free_equivalent import relation_free
mR, emb = relation_free()
rf = lambda a, b: cbmath.margin(cbmath.forward(mR, emb(a, b), 3))
print("C' margins  (1,1) %.3f (-1,1) %.3f (1,-1) %.3f (-1,-1) %.3f" % tuple(closed_form(*p)["m"] for p in [(1,1),(-1,1),(1,-1),(-1,-1)]))
print("RF twin     (1,1) %.3f (-1,1) %.3f (1,-1) %.3f (-1,-1) %.3f" % tuple(rf(*p) for p in [(1,1),(-1,1),(1,-1),(-1,-1)]))
# single-cancellation (independence) violation on a grid: count pairs a,a' and b,b' where order of m(.,b) reverses
g = np.concatenate([-np.linspace(3, .1, 15), np.linspace(.1, 3, 15)])
M = np.array([[closed_form(a, b)["m"] for b in g] for a in g])
viol = 0; tot = 0
for i, j in itertools.combinations(range(len(g)), 2):
    s = np.sign(M[i] - M[j]); s = s[np.abs(M[i] - M[j]) > 1e-3]
    tot += 1; viol += (s > 0).any() and (s < 0).any()
print(f"C' margin: row pairs (a,a') whose order reverses across b: {viol}/{tot}  -> no monotone transform of f(a)+g(b) fits")
MR = np.array([[rf(a, b) for b in g] for a in g]); viol = 0
for i, j in itertools.combinations(range(len(g)), 2):
    s = np.sign(MR[i] - MR[j]); s = s[np.abs(MR[i] - MR[j]) > 1e-3]; viol += (s > 0).any() and (s < 0).any()
print(f"RF twin margin: reversals {viol}/{tot}")
# K=3 candidate-directed rule: agree on k -> k; disagree -> the candidate neither picked. Additive scores feasible?
K = 3; rule = lambda kA, kB: kA if kA == kB else ({0, 1, 2} - {kA, kB}).pop()
# variables f[k][kA], g[k][kB]  (score_k = f[k][kA] + g[k][kB])
nv = 2 * K * K; idx = lambda which, k, s: which * K * K + k * K + s
A, bvec = [], []
for kA in range(K):
    for kB in range(K):
        w = rule(kA, kB)
        for k in range(K):
            if k == w: continue
            row = np.zeros(nv); row[idx(0, k, kA)] += 1; row[idx(1, k, kB)] += 1; row[idx(0, w, kA)] -= 1; row[idx(1, w, kB)] -= 1
            A.append(row); bvec.append(-1.0)
res = linprog(np.zeros(nv), A_ub=np.array(A), b_ub=np.array(bvec), bounds=[(-50, 50)] * nv)
print("K=3 'agree->k, else the unpicked candidate' additive-representable:", res.status == 0)
