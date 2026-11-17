"""XV Workstream F (mathematics only): a noisy-gate variant.  As condition B (a = 2, beta = 0.9, nu = 1/3), except that the evidence
regime follows the gate only with probability gamma: regime r = g with prob gamma, 1 - g otherwise (the gate is still exactly computable,
but trust becomes uncertain given it).  Exact enumeration: Bayes, gate-free mixture, order-k optima k = 1..5, gate-flip slope, and the
minimal decision order.  Writes logs/xv_noisy_math.json."""
import itertools, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, HERE)
import numpy as np, xvtask as T, xv_math_core as MC
XA = T.XALL; K = 3; b = 0.9; e = (1 - b) / 2; nu = 1 / 3; IDX = {tuple(x): i for i, x in enumerate(XA)}
def joint(gamma):
    pxy = np.zeros((len(XA), K))
    for i, x in enumerate(XA):
        g = (x[0] + x[1]) % K == 0; pgate = (1 / K) * 0.5 * (1.0 if g else 0.5) * (1 / K)
        for y in range(K):
            pr1 = (b if x[3] == y else e) * (b if x[4] == y else e) / K                    # regime 1: pair reliable, source 5 noise
            pr0 = (b if x[5] == y else e) * ((nu if x[3] == x[4] else (1 - nu) / 2) / K)   # regime 0: source 5 reliable, pair noise
            w1 = gamma if g else 1 - gamma; pxy[i, y] = pgate * (w1 * pr1 + (1 - w1) * pr0) / K
    px = pxy.sum(1); return px, pxy / px[:, None]
def slope(P, PB, px):
    num = den = 0.0
    for i, x in enumerate(XA):
        g = (x[0] + x[1]) % K == 0; news = [(-x[0] + 1) % K, (-x[0] + 2) % K] if g else [(-x[0]) % K]
        for v in news:
            y = x.copy(); y[1] = v; j = IDX[tuple(y)]; dB = MC_cen(PB[j]) - MC_cen(PB[i]); dm = MC_cen(P[j]) - MC_cen(P[i]); w = px[i] / len(news)
            num += w * (dm * dB).sum(); den += w * (dB * dB).sum()
    return num / den
def MC_cen(P): L = np.log(P + 1e-15); return L - L.mean(-1, keepdims=True)
out = {}
for gamma in (1.0, 0.95, 0.85, 0.75):
    px, PB = joint(gamma); nll = lambda P: float(-(px[:, None] * PB * np.log(P + 1e-15)).sum()); r = dict(bayes=nll(PB))
    for k in (1, 2, 3, 4):
        P, _ = MC.fit(k, px, PB); r[f"order{k}"] = dict(NLL=nll(P), KL=nll(P) - nll(PB), G=slope(P, PB, px))
    r["gain_gate"] = r["order2"]["NLL"] - r["bayes"]; r["min_order_KL_lt_1e-6"] = next((k for k in (1, 2, 3, 4) if r[f"order{k}"]["KL"] < 1e-6), ">4")
    out[str(gamma)] = r; print(gamma, json.dumps(r), flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "xv_noisy_math.json"), "w"), indent=1)
