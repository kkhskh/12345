"""(a) Matched-content control: token sets touched by N vs Cm (generator only).  (b) Where count information sits at the
layer-2 input: ridge decodability (held-out R2) of total count S = sum_g n^(-q)_g, level L = min of non-colluder counts and
difference d, from the readout token 36+q vs from question q's six source tokens, per model (fit stream)."""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "..", "..", "logs", "conf")
sys.path.insert(0, os.path.join(HERE, "..", "..", "src")); import cri as R; E = R.E
G = np.load(os.path.join(D, "generator.npz")); X = G["X"]; mk = G["ok_N"].astype(bool)
tN, tC = (G["X_N"] != X)[mk], (G["X_Cm"] != X)[mk]
print("(a) tokens changed per pair: N", np.bincount(tN.sum((1, 2))), " Cm", np.bincount(tC.sum((1, 2))),
      "| Cm superset of N:", float((tC >= tN).all((1, 2)).mean()), "| Cm == N token set:", float((tC == tN).all((1, 2)).mean()))
Xf = G["Xfit"]; n = E.loo(Xf); cfg = E.sample(len(Xf), 13_110_000)["cfg"]; assert (E.sample(len(Xf), 13_110_000)["X"] == Xf).all()
hh = R.true_h(cfg)
def r2_ridge(F, y, lam=10.0):
    pm = np.random.default_rng(1).permutation(len(F)); F, y = F[pm], y[pm]; k = len(F) // 2; mu, sd = F[:k].mean(0), F[:k].std(0) + 1e-6; A = (F - mu) / sd; A = np.c_[np.ones(len(A)), A]
    w = np.linalg.solve(A[:k].T @ A[:k] + lam * np.eye(A.shape[1]), A[:k].T @ y[:k]); p = A[k:] @ w
    return float(1 - ((y[k:] - p) ** 2).sum() / ((y[k:] - y[k:].mean()) ** 2).sum())
for m in sys.argv[1:]:
    M = R.Model(m); A = M.acts(Xf)[0]; res = {}
    rows_R, rows_Q, S, L, Dd = [], [], [], [], []
    for q in range(R.Q):
        nq = n[:, q]; o = np.array([[g for g in range(3) if g != hh[i]] for i in range(len(Xf))])
        a, b = nq[np.arange(len(Xf)), o[:, 0]], nq[np.arange(len(Xf)), o[:, 1]]
        rows_R.append(A[:, 36 + q]); rows_Q.append(A[:, 6 * q:6 * q + 6].reshape(len(Xf), -1)); S.append(nq.sum(1)); L.append(np.minimum(a, b) + 0.0); Dd.append(np.abs(a - b) + 0.0)
    FR, FQ = np.concatenate(rows_R), np.concatenate(rows_Q); S, L, Dd = map(np.concatenate, (S, L, Dd))
    print(m, " ".join(f"{nm}: readout {r2_ridge(FR, y):.2f} sources {r2_ridge(FQ, y):.2f} |" for nm, y in (("S", S), ("L", L), ("|d|", Dd))), flush=True)
