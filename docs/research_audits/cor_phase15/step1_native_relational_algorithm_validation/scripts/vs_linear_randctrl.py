"""Workstream C (neural test of a mathematical prediction): the pooled-S intervention VS ('set S := S_src, keep e_base') is NOT realizable by
any projection of an I2-like (per-group m) representation, but IS realizable by the fixed linear map 1/2 11^T (Investigation XI, X1').
If the network's 6-dim E-subspace stores eligibility linearly per group and layer 2 reads it like I2, a NON-projective linear map inside the
E-subspace should realise VS while every projection fails.  If layer 2 cannot use off-image states (S != sum e), both fail.
Interventions on resid2 at the q7 tokens, restricted to the E-subspace Q (6 dims):
   z' = z_b + Q M Q^T (z_s - z_b)
   PROJ_k : M = U U^T, U (6 x k) orthonormal, learned (k = 1, 2, 3)          [the class DAS searches]
   LIN    : M an unconstrained 6 x 6 matrix, learned                          [R_lin]
   ID     : M = I (Investigation XI E-interchange; reference)
Targets: VS (p ∝ S_src A_b - B_b) and VE (p ∝ sum_g W_g L_g (S_src - e_src,g)); W, L from the base.  600 Adam steps, batch 128, training pairs
stream 66_100_000, evaluation 66_600_000 (pairs with S_src != S_base for VS; e_src != e_base for VE).
Usage: vs_linear.py RSUF SPEC... -> logs/vs_linear<TAG>.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp, optax
import ftlib as F, native as NV, genlib as GL
np.seterr(all="ignore")
RSUF = sys.argv[1] if sys.argv[1] != "-" else ""; X_DAS = os.environ.get("XII_RDIR", os.path.join(NV.ROOT, "step1_causal_equivalence", "logs", "das"))
env = GL.Env(3, 1); TR, EV = int(os.environ.get("VS_TRAIN", 66_100_000)), int(os.environ.get("VS_EVAL", 66_600_000))
def hl(X, q):
    rel, col, noi = GL._group_lik(env, X); i = np.arange(len(X)); relm = rel.mean(-1); lr = np.log(relm / noi)
    W = np.exp(lr.sum(1) - lr[i, q]); Lg = rel[i, q] / noi; a_ = X.reshape(len(X), 6, 3, 2)[i, q]
    return W, Lg, ((GL.counts(X, 3)[i, q] == 5) & (a_[..., 0] == a_[..., 1])).astype(float)
def pairs(n, stream, var):
    rng = np.random.default_rng(stream); d = NV.Bm.sample_fast(NV.ENV, 4 * n, rng); X = d["X"]; q = rng.integers(0, 6, len(X)); W, L, e = hl(X, q); S = e.sum(1)
    two, one = np.where(S >= 2)[0], np.where(S == 1)[0]; bi = np.concatenate([rng.choice(two, n // 2), rng.choice(one, n - n // 2)]); rng.shuffle(bi)
    si = np.where(rng.random(n) < 0.5, rng.choice(two, n), rng.choice(one, n))
    Ss, Sb = S[si][:, None], S[bi][:, None]; f = (Ss - e[bi]) if var == "VS" else (Ss - e[si])
    P = np.maximum((W[bi][..., None] * f[..., None] * L[bi]).sum(1), 1e-30); P /= P.sum(-1, keepdims=True)
    keep = (S[si] != S[bi]) if var == "VS" else (np.abs(e[si] - e[bi]).sum(1) > 0)
    return X[bi], X[si], q[bi], q[si], P, keep
tokj = lambda q_: jnp.concatenate([(36 + q_)[:, None], q_[:, None] * 6 + jnp.arange(6)[None]], 1)
out = {}
for sp in sys.argv[2:]:
    parent, spec, seed, suf = (sp.split(":") + ["", "", "0", ""])[:4]; M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf)
    Q = jnp.asarray(np.linalg.qr(np.load(os.path.join(X_DAS, f"{M['tag']}_E_L2_q7_k6{RSUF}_R.npy")))[0], jnp.float32)
    if os.environ.get("VS_RANDOM") == "1": Q = jnp.asarray(np.linalg.qr(np.random.default_rng(11).standard_normal((448, 6)))[0], jnp.float32)   # control: random 6-dim subspace
    def logits(Mm, Xb, Xs, qb, qs):
        rs = {}; F.forward(M, Xs, rec=rs); xs = rs["resid2"]; tb, ts = tokj(qb), tokj(qs); i = jnp.arange(Xb.shape[0])[:, None]
        def fn(x):
            vb = x[i, tb].reshape(Xb.shape[0], -1); vs = xs[i, ts].reshape(Xb.shape[0], -1); vp = vb + (((vs - vb) @ Q) @ Mm.T) @ Q.T
            return x.at[i, tb].set(vp.reshape(Xb.shape[0], 7, 64))
        return F.forward(M, Xb, resid_patch={2: fn})[jnp.arange(Xb.shape[0]), qb]
    lj = jax.jit(logits); res = {}
    for var in ("VS", "VE"):
        Xb, Xs, qb, qs, P, _ = pairs(12000, TR, var); Eb, Es, eqb, eqs, EP, keep = pairs(4000, EV, var)
        Eb, Es, eqb, eqs, EP = Eb[keep], Es[keep], eqb[keep], eqs[keep], EP[keep]
        def ev(Mm):
            L = np.concatenate([np.asarray(lj(Mm, jnp.asarray(Eb[i:i + 500]), jnp.asarray(Es[i:i + 500]), jnp.asarray(eqb[i:i + 500]), jnp.asarray(eqs[i:i + 500]))) for i in range(0, len(Eb), 500)])
            Pm = np.exp(L - L.max(-1, keepdims=True)); Pm /= Pm.sum(-1, keepdims=True)
            return dict(IIA=float((L.argmax(-1) == EP.argmax(-1)).mean()), KL=float((EP * (np.log(EP) - np.log(Pm + 1e-12))).sum(-1).mean()))
        r = {"none": ev(jnp.zeros((6, 6))), "ID": ev(jnp.eye(6)), "n_eval": int(len(Eb))}
        for kind in ("PROJ_1", "PROJ_2", "PROJ_3", "LIN"):
            k = int(kind[-1]) if kind != "LIN" else 6
            par = jax.random.normal(jax.random.PRNGKey(k), (6, k)) * 0.3 if kind != "LIN" else jnp.eye(6) * 0.5
            def Mof(p_):
                if kind == "LIN": return p_
                U, _ = jnp.linalg.qr(p_); return U @ U.T
            def loss(p_, a, b, c, d_, Pt): return -(Pt * jax.nn.log_softmax(logits(Mof(p_), a, b, c, d_), -1)).sum(-1).mean()
            g = jax.jit(jax.value_and_grad(loss)); opt = optax.adam(1e-2); st = opt.init(par); br = np.random.default_rng(1)
            for it in range(600):
                j = br.integers(0, len(Xb), 128); l, gr = g(par, jnp.asarray(Xb[j]), jnp.asarray(Xs[j]), jnp.asarray(qb[j]), jnp.asarray(qs[j]), jnp.asarray(P[j])); up, st = opt.update(gr, st); par = optax.apply_updates(par, up)
            r[kind] = ev(Mof(par)); r[kind]["final_loss"] = float(l)
            if kind == "LIN": r["LIN_eigs"] = sorted(np.linalg.eigvals(np.asarray(par)).real.round(3).tolist())
        res[var] = r
    out[sp] = res; print(sp, {v: {k: (round(x["IIA"], 3), round(x["KL"], 3)) for k, x in r.items() if isinstance(x, dict)} for v, r in res.items()}, "LIN eig VS", res["VS"]["LIN_eigs"], flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "vs_linear" + os.environ.get("VS_TAG", "") + ".json"), "w"), indent=1)
