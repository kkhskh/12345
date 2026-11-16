"""XIV Stage III: distributed-alignment (DAS) interchange for the GATE gate variable at the evidence tokens.  Usage: g_das.py NAME CKPT SITE STREAM OUT
A k-dim subspace U (shared by the 4 evidence tokens of question q) at residual SITE is trained so that interchanging it from the gate-flipped
donor makes the model output the donor's exact posterior (training half); evaluated on the held-out half as flip transfer on Bayes-decisive
cases.  Controls: random k-dim subspaces (5; variance ratio reported), relation-preserving null donor (decision change; ideal 0), and the
same U applied at the G tokens and at the R token (should not transfer).  Ideal mechanism (g stored in one direction): k=1 -> 1, null 0."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp, optax, gatelib as GL
name, ck, site, stream, outp = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
p = GL.load(ck); rng = np.random.default_rng(stream); n = 2000
X = GL.TX.sample_gate(n, rng)["X"]; q = rng.integers(0, 6, n); I = np.arange(n)
def edit_q(X, fn): Xe = X.copy(); Xe[I, q] = fn(X[I, q][:, None])[:, 0]; return Xe
Xf = edit_q(X, lambda Z: GL.gate_flip(Z, rng)); Xs = edit_q(X, lambda Z: GL.gate_keep_shift(Z, rng))
B0, Bf = GL.rule("bayes", X)[I, q], GL.rule("bayes", Xf)[I, q]
dec = GL.unique(B0) & GL.unique(Bf) & (B0.argmax(-1) != Bf.argmax(-1)); tr = (I < n // 2); te = ~tr
def masks(kind):
    M = np.zeros((n, 43), bool)
    for i in range(n):
        if kind == "E": M[i, 6 * q[i] + 2: 6 * q[i] + 6] = True
        if kind == "G": M[i, 6 * q[i]: 6 * q[i] + 2] = True
        if kind == "R": M[i, 36 + q[i]] = True
    return jnp.asarray(M)
def rec_site(Xin):
    rec = {}
    def hk(nm, a):
        if nm == site: rec[nm] = a
        return a
    GL.F3.forward(p, jnp.asarray(Xin), hk); return rec[site]
Hf, Hs = rec_site(Xf), rec_site(Xs); Xj = jnp.asarray(X); qj = jnp.asarray(q); Ij = jnp.asarray(I)
def fwd_patch(U, src, M, idx):
    def hk(nm, a):
        if nm == site: a = a + M[idx][..., None] * (((src[idx] - a) @ U) @ U.T)
        return a
    return jax.nn.softmax(GL.F3.forward(p, Xj[idx], hk)["answer"], -1)[jnp.arange(len(idx)), qj[idx]]
ME = masks("E"); res = dict(name=name, site=site, n_decisive_test=int((dec & te).sum()), das={})
def evalU(U, src=Hf, M=ME):
    idx = jnp.asarray(np.flatnonzero(te)); P = np.asarray(fwd_patch(U, src, M, idx)); return P
def transfer(P): d = dec[te]; return float((P.argmax(-1) == Bf[te].argmax(-1))[d].mean())
P0 = GL.probs(p, X)[I, q]
for k in (1, 2, 4, 8):
    W = jnp.asarray(np.random.default_rng(k).normal(size=(64, k)) * 0.1); opt = optax.adam(1e-2); st = opt.init(W)
    idx = jnp.asarray(np.flatnonzero(tr & dec)); tgt = jnp.asarray(Bf[tr & dec])
    @jax.jit
    def stepf(W, st):
        def L(W):
            U, _ = jnp.linalg.qr(W); P = fwd_patch(U, Hf, ME, idx); return -(tgt * jnp.log(P + 1e-9)).sum(-1).mean()
        l, g = jax.value_and_grad(L)(W); up, st = opt.update(g, st); return optax.apply_updates(W, up), st, l
    for it in range(300): W, st, l = stepf(W, st)
    U, _ = jnp.linalg.qr(W); r = dict(train_loss=float(l), transfer=transfer(evalU(U)))
    r["at_G_tokens"] = transfer(evalU(U, M=masks("G"))); r["at_R_token"] = transfer(evalU(U, M=masks("R")))
    Pn = evalU(U, src=Hs); r["null_donor_decision_change"] = float((Pn.argmax(-1) != P0[te].argmax(-1)).mean())
    H0 = rec_site(X); hE = np.asarray(H0)[np.asarray(ME)]; C = np.cov(hE.T); vU = float(np.trace(np.asarray(U).T @ C @ np.asarray(U)))
    rt, vr = [], []
    for t in range(5):
        R, _ = np.linalg.qr(np.random.default_rng(100 + t).normal(size=(64, k))); rt.append(transfer(evalU(jnp.asarray(R)))); vr.append(float(np.trace(R.T @ C @ R) / vU))
    r.update(random_transfer_mean=float(np.mean(rt)), random_transfer_max=float(np.max(rt)), random_var_ratio=float(np.mean(vr)))
    res["das"][str(k)] = r; print(name, k, {a: round(b, 3) for a, b in r.items()}, flush=True)
res["ideal_reference"] = dict(transfer_k1=1.0, null_donor=0.0, random=0.0, at_G=0.0, at_R=0.0)
json.dump(res, open(outp, "w"), indent=1)
