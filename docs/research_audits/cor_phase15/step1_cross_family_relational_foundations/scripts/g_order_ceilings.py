"""XIV Stage III [num]: interaction-order ceilings of the GATE decision (exact, by enumeration of all 3^6 inputs).
The order-k rule class: log-linear rules logit_y(x) = sum over source subsets A with |A| <= k of w_A(x_A, y) (ANOVA-style interaction
order in the log domain).  For k = 1, 2, 3 the expected-NLL-optimal rule is fitted (convex; L-BFGS on the exact input distribution);
reported: NLL, accuracy, KL from Bayes, and its responses to the gate flip and the (x3,x5) swap.  Also verifies that Bayes is exactly
order 3 (log-odds = g * l23 + (1-g) * l45 with g a function of (x0, x1)).  Writes logs/g_order_ceilings.json."""
import itertools, json, os
import numpy as np
from scipy.optimize import minimize
K, b = 3, 0.9; e = (1 - b) / 2
Xall = np.array(list(itertools.product(range(K), repeat=6)))
def lik(x, S): return np.array([np.prod([b if x[s] == y else e for s in S]) for y in range(K)])
g = (Xall[:, 0] + Xall[:, 1]) % K == 0
PB = np.array([(lik(x, (2, 3)) if gg else lik(x, (4, 5))) for x, gg in zip(Xall, g)]); PB /= PB.sum(-1, keepdims=True)
# exact input probabilities P(x) = P(x0) P(x1|x0) * prod-evidence marginal (x | g), Y marginalised
px = np.array([(1 / K) * (1.0 if gg else 0.5) * 0.5 * sum((1 / K) * np.prod([(b if x[s] == y else e) if ((s in (2, 3)) == gg) else 1 / K for s in (2, 3, 4, 5)]) for y in range(K)) for x, gg in zip(Xall, g)])
assert abs(px.sum() - 1) < 1e-12
def design(k):
    cols = []
    for A in itertools.chain.from_iterable(itertools.combinations(range(6), r) for r in range(1, k + 1)):
        code = sum(Xall[:, a] * K ** i for i, a in enumerate(A)); n = K ** len(A); M = np.zeros((len(Xall), n)); M[np.arange(len(Xall)), code] = 1; cols.append(M)
    return np.concatenate(cols, 1)
def fit(k):
    F = design(k); d = F.shape[1]
    def f(w):
        W = w.reshape(d, K); Z = F @ W; Z -= Z.max(1, keepdims=True); L = Z - np.log(np.exp(Z).sum(1, keepdims=True)); P = np.exp(L)
        return -(px[:, None] * PB * L).sum(), (F.T @ (px[:, None] * (P - PB))).ravel()
    r = minimize(f, np.zeros(d * K), jac=True, method="L-BFGS-B", options=dict(maxiter=5000, gtol=1e-10))
    W = r.x.reshape(d, K); Z = F @ W; P = np.exp(Z - Z.max(1, keepdims=True)); P /= P.sum(1, keepdims=True); return P, r
def respond(P):
    idx = {tuple(x): i for i, x in enumerate(Xall)}
    def cen(Q): L = np.log(Q + 1e-15); return L - L.mean(-1, keepdims=True)
    # gate flip: average over all on-support x1 redraws; swap x3<->x5
    flip_num = flip_den = 0.0; sw_num = sw_den = 0.0
    sw_ref = []
    for i, x in enumerate(Xall):
        x2 = x.copy(); x2[3], x2[5] = x[5], x[3]; j = idx[tuple(x2)]
        dB = cen(PB[j]) - cen(PB[i]); dm = cen(P[j]) - cen(P[i]); sw_num += px[i] * (dm * dB).sum(); sw_den += px[i] * (dB * dB).sum()
        gg = (x[0] + x[1]) % K == 0
        for x1n in ([(-x[0] + 1) % K, (-x[0] + 2) % K] if gg else [(-x[0]) % K]):
            x3 = x.copy(); x3[1] = x1n; j = idx[tuple(x3)]; w = px[i] / (2 if gg else 1)
            dB = cen(PB[j]) - cen(PB[i]); dm = cen(P[j]) - cen(P[i]); flip_num += w * (dm * dB).sum(); flip_den += w * (dB * dB).sum()
    return dict(gateflip_slope=flip_num / flip_den, swap_slope_vs_bayes=sw_num / sw_den)
out = {}
for k in (1, 2, 3):
    P, r = fit(k); kl = float((px[:, None] * PB * (np.log(PB) - np.log(P + 1e-15))).sum()); nll = float(-(px[:, None] * PB * np.log(P + 1e-15)).sum())
    acc = float((px * PB[np.arange(len(P)), P.argmax(1)]).sum()); out[f"order{k}"] = dict(NLL=nll, acc=acc, KL_from_bayes=kl, converged=bool(r.success), **respond(P))
    print(k, out[f"order{k}"], flush=True)
out["bayes"] = dict(NLL=float(-(px[:, None] * PB * np.log(PB)).sum()), **respond(PB))
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "logs", "g_order_ceilings.json"), "w"), indent=1); print(out["bayes"])
