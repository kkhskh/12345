"""Independent checks of Theorems A, C2, D4, D5 (formal reviewer).  Uses only gen_family (generator) + own enumeration."""
import itertools, sys, os
from fractions import Fraction as Fr
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "step1_relational_computation_atlas", "src"))
import gen_family as G
f = G.Fam(); K = 3; GR = G.GROUPS; CFG = f.configs
al = Fr(4, 5); e = (1 - al) / (K - 1)
# ---- A1 exact: P(X_k | R) by enumeration over Y and all 729 X_k
def lik_pair(xa, xb, y, role):
    if role == "rel": return (al if xa == y else e) * (al if xb == y else e)
    if role == "noise": return Fr(1, K * K)
    return Fr(1, K) if xa == xb else Fr(0)
f1 = (al**2 + (K-1)*e**2) / K; f0 = (2*al*e + (K-2)*e**2) / K; rho = f1 / f0
bad = 0
for (g, h) in CFG:
    roles = G.roles_of((g, h))
    for x in itertools.product(range(K), repeat=6):
        p = sum(Fr(1, K) * np.prod([lik_pair(x[a], x[b], y, r) for (a, b), r in zip(GR, roles)]) for y in range(K))
        A = [x[a] == x[b] for a, b in GR]
        pred = Fr(1, K**3) * (f1 if A[g] else f0) * (1 if A[h] else 0)
        bad += (p != pred)
print("A1 exact factorization mismatches over 6x729:", bad, " f1,f0,rho =", f1, f0, rho, float(rho))
# ---- A2/A4 exact pi classes with rationals
supp = [v for v in itertools.product(range(6), repeat=3) if 5 in v]
def pi(v):
    w = [rho**v[g] * (1 if v[h] == 5 else 0) for g, h in CFG]; s = sum(w); return tuple(x / s for x in w)
cls = {}
for v in supp: cls.setdefault(pi(v), []).append(v)
print("support vectors", len(supp), "exact distinct pi", len(cls))
def char(v, w):   # A4 characterisation
    H, H2 = {h for h in range(3) if v[h] == 5}, {h for h in range(3) if w[h] == 5}
    if H != H2: return False
    pos = [(g, h) for g, h in CFG if h in H]
    gs = {g for g, _ in pos}
    return all(v[a] - v[b] == w[a] - w[b] for a in gs for b in gs)
mism = sum((pi(v) == pi(w)) != char(v, w) for v in supp for w in supp)
print("A4 characterisation vs exact equality, mismatching ordered pairs:", mism)
sizes = sorted(len(c) for c in cls.values()); print("class size histogram", {s: sizes.count(s) for s in set(sizes)})
# natural distribution of V = n^{(-q)}
ap = dict(rel=al**2 + (K-1)*e**2, noise=Fr(1, 3))
from math import comb
def pn_R(v, g, h):
    p = Fr(1)
    for k in range(3):
        if k == h: p *= (1 if v[k] == 5 else 0)
        else:
            a = ap["rel"] if k == g else ap["noise"]; p *= comb(5, v[k]) * a**v[k] * (1-a)**(5-v[k])
    return p
Pn = {v: sum(Fr(1, 6) * pn_R(v, g, h) for g, h in CFG) for v in supp}
print("sum P(n) =", float(sum(Pn.values())))
# ---- D5 / lossy code: best per-component accuracy of decoding n from pi alone (MAP within class)
accs = []
for c in range(3):
    tot = Fr(0)
    for members in cls.values():
        m = {}
        for v in members: m[v[c]] = m.get(v[c], 0) + Pn[v]
        tot += max(m.values())
    accs.append(float(tot))
print("per-component accuracy of decoding n from pi only (Bayes-optimal):", [round(a, 4) for a in accs], "mean", round(np.mean(accs), 4))
print("joint accuracy of decoding full n from pi only:", round(float(sum(max(Pn[v] for v in m) for m in cls.values())), 4))
# ---- D5: additive one-hot functionals constant on pi classes
Fe = np.array([np.concatenate([[1.0], np.eye(6)[v[0]], np.eye(6)[v[1]], np.eye(6)[v[2]]]) for v in supp])
rows = []
for m in cls.values():
    for v in m[1:]:
        rows.append(Fe[supp.index(v)] - Fe[supp.index(m[0])])
Cn = np.array(rows); u, s, vt = np.linalg.svd(Cn); rank = (s > 1e-9).sum()
null = vt[rank:]                              # functionals w with w.f(n) constant on classes
vals = Fe @ null.T                            # 91 x dim
Hkey = [tuple(int(x == 5) for x in v) for v in supp]
# is every such functional a function of H only?
dev = 0.0
for hk in set(Hkey):
    idx = [i for i, k in enumerate(Hkey) if k == hk]; dev = max(dev, np.abs(vals[idx] - vals[idx].mean(0)).max())
print(f"additive functionals constant on pi-classes: dim {null.shape[0]} (of 19); max within-H variation {dev:.2e}  -> function of H only")
# additive R^2 of pi-only representations under natural P(n)
w = np.array([float(Pn[v]) for v in supp]); W = np.sqrt(w)[:, None]
def r2(Yt):
    B = np.linalg.lstsq(W * Fe, W * Yt, rcond=None)[0]; res = Yt - Fe @ B
    mu = (w[:, None] * Yt).sum(0) / w.sum(); return 1 - (w[:, None] * res**2).sum() / (w[:, None] * (Yt - mu)**2).sum()
PI = np.array([[float(x) for x in pi(v)] for v in supp])
print("weighted additive R^2: pi (6-vec) %.4f ; sqrt(pi) %.4f ; log(pi+1e-3) %.4f" % (r2(PI), r2(np.sqrt(PI)), r2(np.log(PI + 1e-3))))
# ---- A3 formula as written vs exact posterior (sampled episodes)
rng = np.random.default_rng(7); D = G.sample(f, 3000, rng); X = D["X"]; post = G.posterior(f, X)
J = G.joint(f, X)                                  # P(X_q, Y_q=y | R)
with np.errstate(invalid="ignore"): cond = np.nan_to_num(J / J.sum(-1, keepdims=True))   # P(Y_q|X_q,R); 0 where P(X_q|R)=0
wrong = (post["pi_minus"][..., None] * cond).sum(2); wrong = wrong / wrong.sum(-1, keepdims=True)
print("A3 as written  sum_R pi(R) P(Y|X_q,R) vs exact: max abs diff %.3f, argmax disagreement %.4f"
      % (np.abs(wrong - post["post_y"]).max(), (wrong.argmax(-1) != post["post_y"].argmax(-1)).mean()))
right = (post["pi_minus"][..., None] * J).sum(2); right /= right.sum(-1, keepdims=True)
print("A3 corrected   sum_R pi(R) P(X_q,Y|R) normalised vs exact: max abs diff %.1e" % np.abs(right - post["post_y"]).max())
# ---- decision-level coarsening: for each X_q, number of distinct P(Y_q | X_q, pi) over the 43 classes
reps = [m[0] for m in cls.values()]
def postY(xq, v):
    p = pi(v); out = []
    for y in range(K):
        out.append(sum(p[c] * Fr(1, K) * np.prod([lik_pair(xq[a], xq[b], y, r) for (a, b), r in zip(GR, G.roles_of(CFG[c]))]) for c in range(6)))
    s = sum(out); return tuple(o / s for o in out) if s > 0 else None
nd = []
for xq in itertools.product(range(K), repeat=6):
    ps = {postY(xq, v) for v in reps}; ps.discard(None)
    if ps: nd.append(len(ps))
print("distinct decision posteriors across the 43 pi-classes, over the %d possible X_q: min %d, median %d, max %d" % (len(nd), min(nd), int(np.median(nd)), max(nd)))
# ---- C2 converse counterexample: donor decoder returns class representative; recipient = exact Bayes-from-counts
rep = {v: m[0] for m in cls.values() for v in m}
okc = all(postY(xq, rep[v]) == postY(xq, v) for xq in [(0,0,1,1,2,2), (0,1,0,0,2,2), (1,1,0,2,0,0)] for v in supp)
print("C2 counterexample: Bayes recipient response identical under representative-decoded transport:", okc)
# ---- D4 reverse direction: y = sign(V1*V2), V in {-1,+1}: ablate V1:=0 vs interchange V1:=v'
vs = [-1, 1]; ii = np.mean([np.sign(vp * v) == np.sign(vp * vp) for v in vs for vp in vs if vp != v])
ab = np.mean([np.sign(0 * v) != np.sign(v * v) for v in vs])
print(f"D4 reverse: interchange IIA on informative cases {ii:.2f}; ablation changes output in {ab:.2f} of cases")
