"""Positive-theory review check 5: is the sign/ordering of s = phi_rel(5) - phi_rel(4) an artefact of the trade-off with the
leaky collusion gate?  Refit F_mix with the gate fixed hard (phi_col(n<5) = -50), free phi_rel(1..5) and tau, warm start.
Also: exact permutation p for the ordering of s between arms at full budget and at the frozen b* = 500 (post hoc statistic)."""
import itertools, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
import cri as R
from scipy.optimize import minimize
D = os.path.join(ROOT, "logs", "conf"); G = np.load(os.path.join(D, "generator.npz"))
X, qsel, V = G["X"], G["qsel"], G["surfV"]; NS = V.shape[0]
Jrep = np.repeat(np.concatenate([R.J_at(X[i:i + 1], qsel[i]) for i in range(NS)]), 36, 0); Vs = V.reshape(-1, 3)
Ts = [61, 62, 63, 64, 65, 67]; Ps = [61, 62, 63, 64, 66, 67]
def th_of(m): return np.array(json.load(open(os.path.join(D, f"model_{m}.json")))["fits"]["mix"]["theta"])
print("model        s(free gate)  s(hard gate)  dphi(0..3) hard-gate               loss ratio hard/free")
S = {}
for m in [f"cT{s}" for s in Ts] + [f"cP{s}" for s in Ps] + [f"bcT{s}_b500" for s in Ts]:
    tgt = np.load(os.path.join(D, f"model_{m}.npz"))["surf"].reshape(-1, 3); th = th_of(m)
    full = lambda t6: np.r_[t6[:5], np.full(5, -50.0), t6[5]]
    lf = ((R.mix_logits(th, Jrep, Vs) - tgt) ** 2).mean()
    r = minimize(lambda t6: ((R.mix_logits(full(t6), Jrep, Vs) - tgt) ** 2).mean(), np.r_[th[:5], th[10]], method="L-BFGS-B", options=dict(maxiter=400))
    pr = np.r_[0, r.x[:5]]; d = np.diff(pr); S[m] = (th[4] - th[3], d[4])
    print(f"{m:12s} {th[4] - th[3]:+.2f}         {d[4]:+.2f}          {np.round(d[:4], 2).tolist()!s:34s} {r.fun / lf:.2f}")
def perm(a, b):
    v = np.r_[a, b]; obs = np.mean(a) - np.mean(b)
    null = np.array([v[list(c)].mean() - np.delete(v, list(c)).mean() for c in itertools.combinations(range(12), 6)])
    return obs, float((np.abs(null) >= abs(obs) - 1e-12).mean())
for k, lab in ((0, "free gate"), (1, "hard gate")):
    for tset, nm in (([f"cT{s}" for s in Ts], "T2000"), ([f"bcT{s}_b500" for s in Ts], "T500 (b*)")):
        a = [S[m][k] for m in tset]; b = [S[f"cP{s}"][k] for s in Ps]
        o, p = perm(a, b); print(f"{lab:9s} {nm:9s} vs P-count: min T = {min(a):+.2f}, max P = {max(b):+.2f}, mean diff {o:+.2f}, exact two-sided p = {p:.4f}")
