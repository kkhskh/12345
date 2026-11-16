"""XIV Stage III [proof/num]: exact per-question values of candidate GATE decision rules by enumeration of all 3^6 inputs with their exact
generator probabilities (K=3, beta=0.9).  Rules:
  bayes      : exact posterior (uses the gate relation g = [x0+x1 = 0 mod 3])
  gatefree   : best rule without the gate relation = Bayes after marginalising g (mixture over 'pair 23 reliable' / 'pair 45 reliable');
               it uses within-pair agreement implicitly (the T4-like trust cue)
  sourcewise : naive Bayes with the true single-source marginals P(x_s | y) (no relation between any sources)
  pair23 / pair45 : always trust one pair;  prior : uniform.
Reports expected NLL, accuracy (expected prob. of correct argmax with ties split), KL(Bayes||rule), and the mutual information
I(g; x_s) for every single source and I(g; (x0,x1)).  Writes logs/g_rules_exact.json."""
import itertools, json, os
import numpy as np
K, b = 3, 0.9; e = (1 - b) / (K - 1)
def p_src(x, y, rel): return (b if x == y else e) if rel else 1 / K
rows = []
for x in itertools.product(range(K), repeat=6):
    for y in range(K):
        g = int((x[0] + x[1]) % K == 0)
        pg = 1 / K if g else 2 / K        # P(x1 | x0, g): open -> unique x1 among... (open: x1 determined, prob 1; closed: 2 values, 1/2 each)
        pgate = 0.5 * (1 / K) * (1.0 if g else 0.5)          # P(g) P(x0) P(x1 | x0, g)
        pe = np.prod([p_src(x[s], y, (s in (2, 3)) == bool(g)) for s in (2, 3, 4, 5)])
        rows.append((x, y, g, pgate * pe / K))
P = {}
for x, y, g, p in rows: P.setdefault(x, np.zeros(K))[y] += p
X = list(P); W = np.array([P[x].sum() for x in X]); PY = np.array([P[x] / P[x].sum() for x in X])
assert abs(W.sum() - 1) < 1e-12
def lik(x, S):
    return np.array([np.prod([b if x[s] == y else e for s in S]) for y in range(K)])
def rule(name, x):
    if name == "bayes":
        L = lik(x, (2, 3)) if (x[0] + x[1]) % K == 0 else lik(x, (4, 5)); return L / L.sum()
    if name == "gatefree": L = 0.5 * lik(x, (2, 3)) * (1 / K) ** 2 + 0.5 * lik(x, (4, 5)) * (1 / K) ** 2; return L / L.sum()
    if name == "sourcewise":
        m = 0.5 * b + 0.5 / K; mo = (1 - m) / (K - 1); L = np.array([np.prod([m if x[s] == y else mo for s in (2, 3, 4, 5)]) for y in range(K)]); return L / L.sum()
    if name == "pair23": L = lik(x, (2, 3)); return L / L.sum()
    if name == "pair45": L = lik(x, (4, 5)); return L / L.sum()
    return np.ones(K) / K
out = {}
for name in ("bayes", "gatefree", "sourcewise", "pair23", "prior"):
    R = np.array([rule(name, x) for x in X])
    nll = float(-(W[:, None] * PY * np.log(R)).sum())
    acc = float(sum(W[i] * PY[i][np.flatnonzero(np.isclose(R[i], R[i].max()))].mean() for i in range(len(X))))
    kl = float((W[:, None] * PY * (np.log(PY) - np.log(R))).sum())
    out[name] = dict(NLL=nll, acc=acc, KL_from_bayes=kl)
def H(p): p = p[p > 0]; return float(-(p * np.log(p)).sum())
# mutual information of g with single sources and with the gate pair (exact, from the joint)
J = {}
for x, y, g, p in rows: J[(x, g)] = J.get((x, g), 0) + p
def MI(f):
    j = {}
    for (x, g), p in J.items(): j[(f(x), g)] = j.get((f(x), g), 0) + p
    pf, pg = {}, {}
    for (a, g), p in j.items(): pf[a] = pf.get(a, 0) + p; pg[g] = pg.get(g, 0) + p
    return float(sum(p * np.log(p / (pf[a] * pg[g])) for (a, g), p in j.items() if p > 0))
out["MI_nats"] = {f"x{s}": MI(lambda x, s=s: x[s]) for s in range(6)}
out["MI_nats"]["(x0,x1)"] = MI(lambda x: (x[0], x[1])); out["MI_nats"]["agree23"] = MI(lambda x: x[2] == x[3]); out["MI_nats"]["(agree23,agree45)"] = MI(lambda x: (x[2] == x[3], x[4] == x[5]))
out["H_g"] = float(np.log(2))
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "logs", "g_rules_exact.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
