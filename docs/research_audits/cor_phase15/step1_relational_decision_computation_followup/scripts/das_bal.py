"""Workstream B2 (XIII-B): BALANCED E-versus-A1 identification by DAS.
XIII's training pairs almost never separated E from A1 (2.65% of decisions), and the E-, A1- and PERM-trained subspaces largely overlapped.
Here the training distribution is BALANCED: 50% of (base, source) pairs are DISCRIMINATING (the E-counterfactual and the A1-counterfactual
DECISIONS differ; this needs a source episode with a group that is perfect elsewhere but disagrees at its q), 50% are ordinary pairs (as XIII).
Everything else is matched across rules: site (resid2, the 7 q tokens of base/source), k = 6, 1200 Adam(3e-3) steps, batch 128, 24,000 training
pairs, init key, objective (cross-entropy to the rule's counterfactual distribution).  Rules: E, A1, and RANDE (target = e of an unrelated
episode: a learnable-nothing decoy).  Held-out evaluation (stream DB_EVAL, default 77_500_000): discriminating pairs only.
Per trained subspace and per control patch (NOOP R = 0; RANDOM 6-dim; SOURCE-FREE = the subspace filled with the MEAN source projection):
  fidelity of the observed centred-log-prob change to the E-cf change and to the A1-cf change (R^2, slope), decision agreement with each cf.
Also principal angles between the E- and A1-trained subspaces.  -> logs/das_bal<TAG>/<model>.json (+ _R_<rule>.npy)
Usage: das_bal.py SPEC"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp, optax
import ftlib as F, native as NV, genlib as GL, relpairs as RP
np.seterr(all="ignore"); t0 = time.time()
parent, spec, seed, suf = (sys.argv[1].split(":") + ["", "", "0", ""])[:4]; M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf)
OUT = os.path.join(HERE, "..", "logs", "das_bal" + os.environ.get("DB_TAG", "")); os.makedirs(OUT, exist_ok=True)
def cfp(W, L, x):
    P = np.maximum((W[..., None] * (x.sum(1, keepdims=True) - x)[..., None] * L).sum(1), 1e-30); return P / P.sum(-1, keepdims=True)
def sample(n, stream, frac_disc):
    """accumulate chunks until n*frac_disc discriminating and n*(1-frac_disc) ordinary pairs are collected (ordinary = admissible, not discriminating)"""
    rng = np.random.default_rng(stream); need_d = int(frac_disc * n); need_o = n - need_d; acc = {k: [] for k in ("Xb", "Xs", "qb", "qs", "cfE", "cfA1", "cfRANDE", "disc")}; nd = no = 0
    while nd < need_d or no < need_o:
        d = NV.Bm.sample_fast(NV.ENV, 20000, rng); X = d["X"]; q = rng.integers(0, 6, len(X)); W, L, pf, e, S = RP.parts(X, q)
        ok_b = np.where(e.sum(1) >= 1)[0]; bi = rng.choice(ok_b, 200000); si = rng.integers(0, len(X), 200000); ri = rng.integers(0, len(X), 200000)
        okp = (e[si].sum(1) >= 1) & (pf[si].sum(1) >= 1); cE = cfp(W[bi], L[bi], e[si]); cA = cfp(W[bi], L[bi], pf[si]); disc = okp & (cE.argmax(-1) != cA.argmax(-1))
        D_ = np.where(disc)[0][:max(need_d - nd, 0)]; O_ = np.where(okp & ~disc)[0][:max(need_o - no, 0)]; sel = np.concatenate([D_, O_]); nd += len(D_); no += len(O_)
        for k, v in (("Xb", X[bi[sel]]), ("Xs", X[si[sel]]), ("qb", q[bi[sel]]), ("qs", q[si[sel]]), ("cfE", cE[sel]), ("cfA1", cA[sel]), ("cfRANDE", cfp(W[bi[sel]], L[bi[sel]], e[ri[sel]])), ("disc", disc[sel])): acc[k].append(v)
    out = {k: np.concatenate(v) for k, v in acc.items()}; perm = rng.permutation(len(out["Xb"])); out = {k: v[perm] for k, v in out.items()}; out["n_disc"] = int(nd); return out
tok = lambda q_, xp=np: xp.concatenate([(36 + q_)[:, None], q_[:, None] * 6 + xp.arange(6)[None]], 1)
def patched(R, Xb, Xs, qb, qs, src_override=None):
    rs = {}; F.forward(M, Xs, rec=rs); xs = rs["resid2"]; tb, ts = tok(qb, jnp), tok(qs, jnp); i = jnp.arange(Xb.shape[0])[:, None]
    def fn(x):
        vb = x[i, tb].reshape(Xb.shape[0], -1); vs = xs[i, ts].reshape(Xb.shape[0], -1) if src_override is None else src_override
        Q, _ = jnp.linalg.qr(R); vp = vb + ((vs - vb) @ Q) @ Q.T; return x.at[i, tb].set(vp.reshape(Xb.shape[0], 7, 64))
    return F.forward(M, Xb, resid_patch={2: fn})[jnp.arange(Xb.shape[0]), qb]
pj = jax.jit(patched)
tr = sample(24_000, int(os.environ.get("DB_TRAIN", 77_000_000)), 0.5); ev = sample(3000, int(os.environ.get("DB_EVAL", 77_500_000)), 1.0)
J = jnp.asarray; Rs = {}
for rule in ("E", "A1", "RANDE"):
    def loss(R, Xb, Xs, qb, qs, P): return -(P * jax.nn.log_softmax(patched(R, Xb, Xs, qb, qs), -1)).sum(-1).mean()
    g = jax.jit(jax.value_and_grad(loss)); R = jax.random.normal(jax.random.PRNGKey(6), (448, 6)) * 0.1; opt = optax.adam(3e-3); st = opt.init(R); br = np.random.default_rng(3)
    for it in range(1200):
        j = br.integers(0, len(tr["Xb"]), 128); l, gr = g(R, J(tr["Xb"][j]), J(tr["Xs"][j]), J(tr["qb"][j]), J(tr["qs"][j]), J(tr[f"cf{rule}"][j])); up, st = opt.update(gr, st); R = optax.apply_updates(R, up)
    Rs[rule] = np.asarray(R); np.save(os.path.join(OUT, f"{M['tag']}_R_{rule}.npy"), Rs[rule])
L0 = np.asarray(F.batched(jax.jit(lambda X_: F.forward(M, X_)), ev["Xb"]))[np.arange(len(ev["Xb"])), ev["qb"]]; P0 = RP.sm(L0)
dE = RP.cen(ev["cfE"]) - RP.cen(P0); dA = RP.cen(ev["cfA1"]) - RP.cen(P0)
# mean source projection for the source-free control
rr = {}; F.forward(M, J(ev["Xs"][:1500]), rec=rr); zbar = np.asarray(rr["resid2"])[np.arange(1500)[:, None], tok(ev["qs"][:1500])].reshape(1500, -1).mean(0)
def evaluate(R, src_free=False):
    Ls = []
    for i in range(0, len(ev["Xb"]), 500):
        sl = slice(i, i + 500); nb = len(ev["Xb"][sl]); so = J(np.repeat(zbar[None], nb, 0)) if src_free else None
        Ls.append(np.asarray(pj(J(R), J(ev["Xb"][sl]), J(ev["Xs"][sl]), J(ev["qb"][sl]), J(ev["qs"][sl]), so)) if not src_free else np.asarray(patched(J(R), J(ev["Xb"][sl]), J(ev["Xs"][sl]), J(ev["qb"][sl]), J(ev["qs"][sl]), so)))
    P = RP.sm(np.concatenate(Ls)); o = RP.cen(P) - RP.cen(P0)
    return dict(fid_E=RP.fid(o, dE), fid_A1=RP.fid(o, dA), agree_cfE=float((P.argmax(-1) == ev["cfE"].argmax(-1)).mean()), agree_cfA1=float((P.argmax(-1) == ev["cfA1"].argmax(-1)).mean()), mag=float(np.abs(o).mean()))
res = dict(model=M["tag"], n_train_disc=tr["n_disc"], n_eval=len(ev["Xb"]), eval_all_discriminating=bool(ev["disc"].all()), base_agree_cfE=float((P0.argmax(-1) == ev["cfE"].argmax(-1)).mean()))
for rule, R in Rs.items(): res[f"trained_{rule}"] = evaluate(R); res[f"sourcefree_{rule}"] = evaluate(R, True)
res["RANDOM"] = evaluate(np.random.default_rng(9).standard_normal((448, 6))); res["NOOP"] = evaluate(np.zeros((448, 6)) + 1e-9)
QE, _ = np.linalg.qr(Rs["E"]); QA, _ = np.linalg.qr(Rs["A1"]); res["principal_cosines_E_A1"] = np.linalg.svd(QE.T @ QA)[1].tolist()
res["seconds"] = time.time() - t0; json.dump(res, open(os.path.join(OUT, f"{M['tag']}.json"), "w"), indent=1)
print(M["tag"][:6], "disc train", tr["n_disc"], {k: (round(v["fid_E"]["R2"], 2), round(v["fid_A1"]["R2"], 2), round(v["agree_cfE"], 2), round(v["agree_cfA1"], 2)) for k, v in res.items() if isinstance(v, dict)}, "cos", np.round(res["principal_cosines_E_A1"], 2), round(res["seconds"]), "s", flush=True)
