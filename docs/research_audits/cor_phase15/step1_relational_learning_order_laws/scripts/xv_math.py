"""XV [proof-check / num]: exact references for the task family (src/xvtask.py), by enumeration of all 3^6 inputs.
For each (a, beta): exact Bayes NLL / accuracy; exact gate-free mixture (g marginalised); expected-NLL-optimal log-linear rules of
interaction order k = 1 .. a+1 (logits = sum over source subsets |A| <= k of tables w_A(x_A, y)); for each: NLL, accuracy, KL from
Bayes, gate-flip slope G (exact, all on-support flips), swap(4,5) response slopes; mutual information of g with every single source
and every pair of gate sources; the minimal decision order (smallest k whose optimum reaches Bayes, KL < 1e-6) and the check that the
order-a optimum has G = 0.  Usage: xv_math.py OUT.json a:beta[:nu] ..."""
import itertools, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, xvtask as T
from scipy.optimize import minimize
XA = T.XALL; IDX = {tuple(x): i for i, x in enumerate(XA)}; K = 3
def design(k):
    cols = []
    for A in itertools.chain.from_iterable(itertools.combinations(range(6), r) for r in range(1, k + 1)):
        code = sum(XA[:, s] * K ** i for i, s in enumerate(A)); M = np.zeros((len(XA), K ** len(A))); M[np.arange(len(XA)), code] = 1; cols.append(M)
    return np.concatenate(cols, 1)
def fit(k, px, PB):
    F = design(k); d = F.shape[1]
    def f(w):
        Z = F @ w.reshape(d, K); Z -= Z.max(1, keepdims=True); L = Z - np.log(np.exp(Z).sum(1, keepdims=True))
        return -(px[:, None] * PB * L).sum(), (F.T @ (px[:, None] * (np.exp(L) - PB))).ravel()
    r = minimize(f, np.zeros(d * K), jac=True, method="L-BFGS-B", options=dict(maxiter=20000, gtol=1e-12, ftol=1e-15))
    Z = F @ r.x.reshape(d, K); P = np.exp(Z - Z.max(1, keepdims=True)); return P / P.sum(1, keepdims=True), bool(r.success)
def cen(P): L = np.log(P + 1e-15); return L - L.mean(-1, keepdims=True)
def flips(a):
    """all on-support gate flips: (i, j, weight) pairs, the last gate source redrawn to switch g (uniform over valid redraws)."""
    out = []
    for i, x in enumerate(XA):
        part = x[:a - 1].sum() % K; g = (x[:a].sum() % K) == 0
        news = [(-part + 1) % K, (-part + 2) % K] if g else [(-part) % K]
        for v in news: y = x.copy(); y[a - 1] = v; out.append((i, IDX[tuple(y)], 1 / len(news)))
    return out
def slope(P, PB, px, pairs):
    num = den = 0.0
    for i, j, w in pairs:
        dB = cen(PB[j]) - cen(PB[i]); dm = cen(P[j]) - cen(P[i]); num += px[i] * w * (dm * dB).sum(); den += px[i] * w * (dB * dB).sum()
    return num / den if den > 0 else float("nan")
def MI(px_g, f):
    j = {}
    for (i, g), p in px_g.items(): k = (f(XA[i]), g); j[k] = j.get(k, 0) + p
    pf, pg = {}, {}
    for (u, g), p in j.items(): pf[u] = pf.get(u, 0) + p; pg[g] = pg.get(g, 0) + p
    return float(sum(p * np.log(p / (pf[u] * pg[g])) for (u, g), p in j.items() if p > 0))
out = {}
for spec in sys.argv[2:]:
    sp = spec.split(":"); a, beta = int(sp[0]), float(sp[1]); nu = float(sp[2]) if len(sp) > 2 else 1 / 3; px, PB = T.joint(a, beta, nu); r = dict(a=a, beta=beta, nu=nu)
    nll = lambda P: float(-(px[:, None] * PB * np.log(P + 1e-15)).sum()); acc = lambda P: float((px * PB[np.arange(len(P)), P.argmax(1)]).sum())
    GF = T.gatefree(XA, a, beta, nu); fl = flips(a)
    sw = [(i, IDX[tuple(np.array([x[0], x[1], x[2], x[3], x[5], x[4]]))], 1.0) for i, x in enumerate(XA)]
    r["bayes"] = dict(NLL=nll(PB), acc=acc(PB)); r["gatefree"] = dict(NLL=nll(GF), acc=acc(GF), G=slope(GF, PB, px, fl))
    r["prior"] = dict(NLL=float(np.log(K)))
    for k in range(1, a + 2):
        P, ok = fit(k, px, PB); r[f"order{k}"] = dict(NLL=nll(P), acc=acc(P), KL=nll(P) - nll(PB), G=slope(P, PB, px, fl), converged=ok)
    r["min_decision_order"] = next((k for k in range(1, a + 2) if r[f"order{k}"]["KL"] < 1e-6), None)
    r["order_a_has_zero_G"] = bool(abs(r[f"order{a}"]["G"]) < 1e-6)
    pg = {}
    for i, x in enumerate(XA):
        g = (x[:a].sum() % K) == 0; pg[(i, g)] = pg.get((i, g), 0) + px[i]
    r["MI_g_single"] = {f"x{s}": MI(pg, lambda x, s=s: x[s]) for s in range(6)}
    r["MI_g_gatepairs"] = {f"x{s}x{t}": MI(pg, lambda x, s=s, t=t: (x[s], x[t])) for s, t in itertools.combinations(range(a), 2)} if a == 3 else {}
    r["MI_g_all_gate_sources"] = MI(pg, lambda x: tuple(x[:a])); r["MI_g_agree34"] = MI(pg, lambda x: x[3] == x[4])
    # usefulness: gate gain = best rule without the gate (order a optimum; equals the gate-free optimum) minus Bayes
    r["gain_gate"] = r[f"order{a}"]["NLL"] - r["bayes"]["NLL"]; r["gain_intermediate"] = r["order1"]["NLL"] - r[f"order{a}"]["NLL"]
    r["gain_gate_vs_gatefree_mixture"] = r["gatefree"]["NLL"] - r["bayes"]["NLL"]; r["acc_gain_gate"] = r["bayes"]["acc"] - r[f"order{a}"]["acc"]
    out[spec] = r; print(spec, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if not isinstance(v, dict)}), {k: {kk: round(vv, 4) for kk, vv in v.items() if isinstance(vv, float)} for k, v in r.items() if isinstance(v, dict) and (k.startswith("order") or k in ("bayes", "gatefree"))}, flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
