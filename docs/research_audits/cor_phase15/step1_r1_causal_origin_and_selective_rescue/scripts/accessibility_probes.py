"""Stage II: accessibility of the R1-relevant statistics in the frozen layer-2 input Z = resid_pre2 (development models).
Natural decisions (stream 17_300_000 fit, 17_400_000 test; 6000 episodes each, all 6 questions).  Targets (generator-derived):
  H2      : two or more groups perfect and agreeing at q (vs exactly one)
  agree2  : among two-perfect decisions, the two perfect groups vote the same candidate at q (R1-type) vs different (R2/R3-type)
  m       : eligibility m_g of each group (0/1/2; 3 x 3-class)
  W       : normalised exact Bayes mixture weights W_g / sum W (3-vector; R^2)
  post    : exact Bayes posterior over candidates at q (3-vector; R^2)
Feature sites: readout token 36+q (64-d), the six q-source tokens (384-d), the readout token + q-source tokens (448-d).
Linear (logistic / ridge) and MLP (1 hidden layer of 256) probes, held-out.  A failed probe is not proof of absence."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
import nnlib as NL, r1lib as L
E, B = L.E, L.B
def data(stream, ne=6000):
    D = E.sample(ne, stream); X = D["X"]; n = E.loo(X); V = np.stack([L.votes(X, qq) for qq in range(6)], 1); perf = (n == 5) & (V >= 0)
    H = perf.sum(-1); m = perf.sum(-1, keepdims=True) - perf; W = L.RHO ** n * m; W = W / np.maximum(W.sum(-1, keepdims=True), 1e-12)
    post = E.bayes_from(X, n); ag2 = np.zeros(H.shape, int)
    for a, b in ((0, 1), (0, 2), (1, 2)): ag2 |= (perf[..., a] & perf[..., b] & (V[..., a] == V[..., b])).astype(int)
    return X, dict(H2=(H >= 2).astype(int), agree2=ag2, m=m, W=W, post=post)
Xf, Tf = data(17_300_000); Xt, Tt = data(17_400_000)
def sites(p, X):
    z = np.asarray(jax.jit(lambda X: NL.frozen_z(p, X))(jnp.asarray(X)))      # N,43,64
    out = {}
    rd = z[:, 36:42]                                                           # N,6,64 (per question)
    src = z[:, :36].reshape(len(X), 6, 6 * 64)                                 # N,6,384: the q-source tokens
    out["readout"] = rd.reshape(-1, 64); out["qsources"] = src.reshape(-1, 384); out["readout+qsources"] = np.concatenate([out["readout"], out["qsources"]], 1)
    return out
def flat(T): return {k: v.reshape(-1, *v.shape[2:]) if v.ndim > 2 else v.reshape(-1) for k, v in T.items()}
tf, tt = flat(Tf), flat(Tt)
res = {}
for mname in sys.argv[1:]:
    p = NL.load(NL.dev_ckpt(mname)); Sf, St = sites(p, Xf), sites(p, Xt); r = {}
    for sn in Sf:
        sc = StandardScaler().fit(Sf[sn]); A, Bt = sc.transform(Sf[sn]), sc.transform(St[sn]); rr = {}
        sel_f = tf["H2"] == 1; sel_t = tt["H2"] == 1
        for nm, y, yt, sf, st in (("H2", tf["H2"], tt["H2"], None, None), ("agree2", tf["agree2"], tt["agree2"], sel_f, sel_t)):
            a_, b_ = (A, Bt) if sf is None else (A[sf], Bt[st]); y_, yt_ = (y, yt) if sf is None else (y[sf], yt[st])
            lin = LogisticRegression(max_iter=2000, C=1.0).fit(a_, y_); mlp = MLPClassifier(hidden_layer_sizes=(256,), max_iter=300, random_state=0).fit(a_, y_)
            base = max(yt_.mean(), 1 - yt_.mean())
            rr[nm] = dict(linear=float(lin.score(b_, yt_)), mlp=float(mlp.score(b_, yt_)), majority=float(base))
        accs = []
        for g in range(3):
            lin = LogisticRegression(max_iter=2000).fit(A, tf["m"][:, g]); accs.append(float(lin.score(Bt, tt["m"][:, g])))
        rr["m_linear_acc"] = accs
        for nm in ("W", "post"):
            lin = Ridge(1.0).fit(A, tf[nm]); pr = lin.predict(Bt); r2 = 1 - ((pr - tt[nm]) ** 2).sum() / ((tt[nm] - tt[nm].mean(0)) ** 2).sum()
            mlp = MLPRegressor(hidden_layer_sizes=(256,), max_iter=200, random_state=0).fit(A, tf[nm]); pm = mlp.predict(Bt); r2m = 1 - ((pm - tt[nm]) ** 2).sum() / ((tt[nm] - tt[nm].mean(0)) ** 2).sum()
            rr[nm] = dict(linear_r2=float(r2), mlp_r2=float(r2m))
        r[sn] = rr
    res[mname] = r
    print(mname, {sn: dict(agree2=(round(v["agree2"]["linear"], 3), round(v["agree2"]["mlp"], 3)), H2=round(v["H2"]["linear"], 3), m=[round(x, 3) for x in v["m_linear_acc"]],
                          W=(round(v["W"]["linear_r2"], 3), round(v["W"]["mlp_r2"], 3)), post=(round(v["post"]["linear_r2"], 3), round(v["post"]["mlp_r2"], 3))) for sn, v in r.items()}, flush=True)
    json.dump(res, open(os.path.join(HERE, "..", "logs", f"accessibility_probes_{sys.argv[1]}.json"), "w"), indent=1)
