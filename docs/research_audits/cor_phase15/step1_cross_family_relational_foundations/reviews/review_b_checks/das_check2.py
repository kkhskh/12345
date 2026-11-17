"""Reviewer check (not part of package): GATE DAS k=1 restarts (shared) and per-token 1-d DAS (4 separate directions)."""
import sys, os, json
PKG = "/home/claude/12345/docs/research_audits/cor_phase15/step1_cross_family_relational_foundations"
sys.path.insert(0, os.path.join(PKG, "src"))
import numpy as np, jax, jax.numpy as jnp, optax, gatelib as GL
name, site = sys.argv[1], sys.argv[2]; ck = GL.DEV_FINALS[name]
p = GL.load(ck); rng = np.random.default_rng(98_300_000 + int(name[1])); n = 2000
X = GL.TX.sample_gate(n, rng)["X"]; q = rng.integers(0, 6, n); I = np.arange(n)
def edit_q(X, fn): Xe = X.copy(); Xe[I, q] = fn(X[I, q][:, None])[:, 0]; return Xe
Xf = edit_q(X, lambda Z: GL.gate_flip(Z, rng))
B0, Bf = GL.rule("bayes", X)[I, q], GL.rule("bayes", Xf)[I, q]
dec = GL.unique(B0) & GL.unique(Bf) & (B0.argmax(-1) != Bf.argmax(-1)); tr = I < n // 2; te = ~tr
def rec_site(Xin):
    rec = {}
    def hk(nm, a):
        if nm == site: rec[nm] = a
        return a
    GL.F3.forward(p, jnp.asarray(Xin), hk); return rec[site]
Hf = rec_site(Xf); Xj = jnp.asarray(X); qj = jnp.asarray(q)
# token-role masks: role j in 0..3 -> token 6q+2+j
Mr = np.zeros((4, n, 43), bool)
for j in range(4): Mr[j, I, 6 * q + 2 + j] = True
Mr = jnp.asarray(Mr)
def fwd(Us, idx, per_token, D=None):
    Mi, Hi, Xi, qi = D if D is not None else (Mr[:, idx], Hf[idx], Xj[idx], qj[idx])
    def hk(nm, a):
        if nm == site:
            for j in range(4):
                U = Us[j] if per_token else Us[0]
                a = a + Mi[j][..., None] * (((Hi - a) @ U) @ U.T)
        return a
    return jax.nn.softmax(GL.F3.forward(p, Xi, hk)["answer"], -1)[jnp.arange(Xi.shape[0]), qi]
idx_tr = jnp.asarray(np.flatnonzero(tr & dec)); tgt = jnp.asarray(Bf[tr & dec]); idx_te = jnp.asarray(np.flatnonzero(te & dec))
def train(k, per_token, seed, iters):
    nU = 4 if per_token else 1
    W = jnp.asarray(np.random.default_rng(seed).normal(size=(nU, 64, k)) * 0.1); opt = optax.adam(1e-2); st = opt.init(W)
    def orth(W): return [jnp.linalg.qr(W[i])[0] for i in range(nU)]
    Dtr = (Mr[:, idx_tr], Hf[idx_tr], Xj[idx_tr], qj[idx_tr])
    @jax.jit
    def stepf(W, st, Dtr):
        def L(W): P = fwd(orth(W), None, per_token, Dtr); return -(tgt * jnp.log(P + 1e-9)).sum(-1).mean()
        l, g = jax.value_and_grad(L)(W); up, st = opt.update(g, st); return optax.apply_updates(W, up), st, l
    for it in range(iters): W, st, l = stepf(W, st, Dtr)
    P = np.asarray(fwd(orth(W), idx_te, per_token)); return float(l), float((P.argmax(-1) == Bf[np.asarray(idx_te)].argmax(-1)).mean())
out = dict(name=name, site=site, n_test_dec=int(len(idx_te)))
out["shared_k1_restarts"] = []; out["pertoken_k1_restarts"] = []; out["shared_k2"] = []
for s_ in (11, 12, 13):
    out["shared_k1_restarts"].append(train(1, False, s_, 400)); print("shared k1", out["shared_k1_restarts"][-1], flush=True)
for s_ in (21, 22):
    out["pertoken_k1_restarts"].append(train(1, True, s_, 400)); print("pertoken k1", out["pertoken_k1_restarts"][-1], flush=True)
out["shared_k2"].append(train(2, False, 31, 300)); print("shared k2", out["shared_k2"][-1], flush=True)
out["pertoken_k1_restarts"] = out["pertoken_k1_restarts"]
print(json.dumps(out)); json.dump(out, open(f"/tmp/review_b/das_check_{name}.json", "w"))
