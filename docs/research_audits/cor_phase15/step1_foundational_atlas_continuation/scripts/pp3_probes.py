"""PP3 feature identification (mandate §4): which variables do the TRANSFERRED layers (0-1 + embeddings) make linearly
available at the input of the last layer (resid_pre2), for each pretraining arm?  Probe data: natural T4 episodes
(stream 9_100_000 fit / 9_100_001 test, 1500 + 1000).  Per source token of group g in question q:
  agree_q      : 1[g's pair agrees in q]                          (local, within-question)
  count_loo    : n_g^{-q} (0..5)                                   (cross-question aggregate)
  perfect_loo  : 1[n_g^{-q} = 5]                                   (colluder signature under T4)
  P_rel        : T4 Bayes P(g reliable | X_{-q})                  (reliability, regression R^2)
  P_coll       : T4 Bayes P(g colluder | X_{-q})
Ridge (alpha 1) for continuous targets (R^2), logistic for binary/categorical (accuracy) vs majority baseline.
Arms: pre_plocal, pre_pcount, pre_pgroup (this continuation), T1 s0 (atlas; P-reliability), random init (P-control).
Writes logs/pp3_probes.json."""
import json, os, sys
HERE = os.path.dirname(__file__); LR = os.path.join(HERE, "..", "..", "step1_latent_reliability"); AT = os.path.join(HERE, "..", "..", "step1_relational_computation_atlas")
sys.path.insert(0, os.path.join(AT, "src")); sys.path.insert(0, os.path.join(LR, "src"))
import numpy as np, jax, jax.numpy as jnp, gen_family as G, fwd as F, ckpt as C, nets as N, bayes as B
from sklearn.linear_model import Ridge, LogisticRegression
f4 = G.Fam(); Q = 6
def feats(X):
    A = np.stack([(X[:, :, a] == X[:, :, b]) for a, b in G.GROUPS], -1).astype(int)          # N,Q,3
    loo = A.sum(1, keepdims=True) - A
    lM = np.zeros(1)
    out = {"agree_q": A, "count_loo": loo, "perfect_loo": (loo == 5).astype(int)}
    # leave-q configuration posterior under T4 (direct leave-one-out)
    J = G.joint(f4, X) if hasattr(G, "joint") else None
    Mq = J.sum(-1); lMq = np.log(Mq); pr = np.log(f4.prior)
    loo_l = np.stack([lMq[:, [k for k in range(Q) if k != q]].sum(1) for q in range(Q)], 1) + pr
    m = loo_l.max(-1, keepdims=True); w = np.exp(loo_l - m); w = np.nan_to_num(w / w.sum(-1, keepdims=True))
    cf = f4.configs
    out["P_rel"] = np.stack([w[..., [i for i, c in enumerate(cf) if c[0] == g]].sum(-1) for g in range(3)], -1)
    out["P_coll"] = np.stack([w[..., [i for i, c in enumerate(cf) if c[1] == g]].sum(-1) for g in range(3)], -1)
    return out
def tokens(p, X):
    @jax.jit
    def f(Xb):
        rec = {}
        def hk(n, a):
            if n == "resid_pre2": rec["r"] = a
            return a
        F.forward(p, Xb, hk); return rec["r"]
    return np.concatenate([np.asarray(f(jnp.asarray(X[i:i + 500]))) for i in range(0, len(X), 500)])[:, :36]   # N,36,d
def table(X, R):
    fe = feats(X); rows = {k: [] for k in fe}; Z = []
    for q in range(Q):
        for s in range(6):
            g = s // 2; Z.append(R[:, 6 * q + s])
            for k in fe: rows[k].append(fe[k][:, q, g])
    return np.concatenate(Z), {k: np.concatenate(v) for k, v in rows.items()}
Dtr = G.sample(f4, 1500, np.random.default_rng(9_100_000))["X"]; Dte = G.sample(f4, 1000, np.random.default_rng(9_100_001))["X"]
arms = {"P-local": os.path.join(HERE, "..", "checkpoints", "pre_plocal.npz"), "P-count": os.path.join(HERE, "..", "checkpoints", "pre_pcount.npz"),
        "P-group": os.path.join(HERE, "..", "checkpoints", "pre_pgroup.npz"), "P-reliability(T1)": os.path.join(LR, "checkpoints", "main", "T1_M_K3_s0.npz"), "P-control(random)": None}
res = {}
for nm, path in arms.items():
    if path is not None and not os.path.exists(path): print("missing", nm); continue
    if path is None: p = N.init("M", B.Env("T4", K=3, Q=6), 0)
    else: p = C.load(path); p.pop("WX", None)
    p = jax.tree_util.tree_map(jnp.asarray, p)
    Ztr, Ttr = table(Dtr, tokens(p, Dtr)); Zte, Tte = table(Dte, tokens(p, Dte)); mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-6; Ztr = (Ztr - mu) / sd; Zte = (Zte - mu) / sd
    r = {}
    for k in Ttr:
        if k in ("P_rel", "P_coll"):
            m = Ridge(alpha=1.0).fit(Ztr, Ttr[k]); pr_ = m.predict(Zte); r[k + "_R2"] = float(1 - ((pr_ - Tte[k]) ** 2).mean() / Tte[k].var())
        else:
            m = LogisticRegression(max_iter=1000).fit(Ztr[::2], Ttr[k][::2]); acc = float(m.score(Zte, Tte[k]))
            base = float(np.bincount(Tte[k]).max() / len(Tte[k])); r[k + "_acc"] = acc; r[k + "_majority"] = base
    res[nm] = r; print(nm, {a: round(b, 3) for a, b in r.items()}, flush=True)
json.dump(res, open(os.path.join(HERE, "..", "logs", "pp3_probes.json"), "w"), indent=1)
