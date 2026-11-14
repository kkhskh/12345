"""Workstream C2: false positives and false negatives of interchange-based identification (DAS), and which diagnostics separate them.
Train a DAS subspace (same site/procedure as Investigation X: L2 / q7 tokens / k=6, 1200 Adam steps, batch 128) against the counterfactuals
of a high-level RULE, then evaluate several diagnostics.  Rules (W_g, L_g always from the base episode; x = per-group variable):
  E     : true Bayes eligibility, x = e_src                    p ∝ sum_g W L (S_x - x_g)
  PERM  : wrong variable assignment: x = roll(e_src, 1)  (group g receives source group g-1's eligibility)   [decoy]
  A1    : perfect-only rule, x = pf_src (n_g = 5, agreement ignored) p ∝ sum_g W L (S_x - x_g)              [rule followed by C1 models]
  RANDE : target noise: x = e of an unrelated random episode (not the source)                                [decoy]
Diagnostics (held-out stream 64_500_000; pairs whose rule value differs, unless stated):
  IIA, KL           : patched argmax == rule-counterfactual argmax; KL(cf || patched)
  IIA_disc          : on pairs where the rule cf and the separable-A3 cf decisions differ
  IIA_trueE         : agreement of the patched decision with the TRUE-E counterfactual (cross-fidelity)
  preserve_KL/flip  : pairs whose rule value is EQUAL but whose episodes differ: KL(base || patched) and argmax-flip rate
                      (a valid variable predicts no change)
  sel_D2c/D2, D2/CNT: displacement inside the subspace for natural edits (Investigation X F6 selectivity)
Usage: das_decoy.py SPEC RULE -> logs/decoy/<tag>_<RULE>.json"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp, optax
import ftlib as F, native as NV, genlib as GL
np.seterr(all="ignore"); t0 = time.time()
parent, spec, seed, suf = (sys.argv[1].split(":") + ["", "", "0", ""])[:4]; RULE = sys.argv[2]
M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf); OUT = os.path.join(HERE, "..", "logs", os.environ.get("XI_OUT", "decoy")); os.makedirs(OUT, exist_ok=True)
env = GL.Env(G=3, C=1)
def hl(X, q):
    rel, col, noi = GL._group_lik(env, X); i = np.arange(len(X)); relm = rel.mean(-1); lr = np.log(relm / noi)
    W = np.exp(lr.sum(1) - lr[i, q]); Lg = rel[i, q] / noi; cnt = GL.counts(X, 3)[i, q]; a_ = X.reshape(len(X), 6, 3, 2)[i, q]
    pf = (cnt == 5).astype(float); return W, Lg, pf * (a_[..., 0] == a_[..., 1]), pf
def cfp(W, L, x):
    P = np.maximum((W[..., None] * (x.sum(1, keepdims=True) - x)[..., None] * L).sum(1), 1e-30); return P / P.sum(-1, keepdims=True)
def sample(n, stream):
    rng = np.random.default_rng(stream); d = NV.Bm.sample_fast(NV.ENV, 4 * n, rng); X = d["X"]; q = rng.integers(0, 6, len(X))
    W, L, e, pf = hl(X, q); S = e.sum(1); P_ = pf.sum(1); key = P_ if RULE == "A1" else S
    two = np.where(key >= 2)[0]; one = np.where(key == 1)[0]
    bi = np.concatenate([rng.choice(two, n // 2), rng.choice(one, n - n // 2)]); rng.shuffle(bi)
    si = np.where(rng.random(n) < 0.6, rng.choice(two, n), rng.integers(0, len(X), n)); ri = rng.integers(0, len(X), n)
    xs = {"E": e[si], "PERM": np.roll(e[si], 1, axis=1), "A1": pf[si], "RANDE": e[ri]}[RULE]
    xb = pf[bi] if RULE == "A1" else e[bi]
    return dict(Xb=X[bi], Xs=X[si], qb=q[bi], qs=q[si], cf=cfp(W[bi], L[bi], xs), cfE=cfp(W[bi], L[bi], e[si]), cfA3=(lambda P: P / P.sum(-1, keepdims=True))(np.maximum((W[bi][..., None] * (1 - xs)[..., None] * L[bi]).sum(1), 1e-30)),
                diff=np.abs(xs - xb).sum(1) > 0, same_ep=(si == bi))
tok = lambda q, xp=np: xp.concatenate([(36 + q)[:, None], q[:, None] * 6 + xp.arange(6)[None]], 1)
def patched_logits(R, Xb, Xs, qb, qs):
    rs = {}; F.forward(M, Xs, rec=rs); xs = rs["resid2"]; tb, ts = tok(qb, jnp), tok(qs, jnp); i = jnp.arange(Xb.shape[0])[:, None]
    def fn(x):
        vb = x[i, tb].reshape(Xb.shape[0], -1); vs = xs[i, ts].reshape(Xb.shape[0], -1); Q, _ = jnp.linalg.qr(R); vp = vb + ((vs - vb) @ Q) @ Q.T
        return x.at[i, tb].set(vp.reshape(Xb.shape[0], 7, 64))
    return F.forward(M, Xb, resid_patch={2: fn})[jnp.arange(Xb.shape[0]), qb]
pj = jax.jit(patched_logits)
def loss(R, Xb, Xs, qb, qs, P): return -(P * jax.nn.log_softmax(patched_logits(R, Xb, Xs, qb, qs), -1)).sum(-1).mean()
g = jax.jit(jax.value_and_grad(loss))
R = jax.random.normal(jax.random.PRNGKey(6), (448, 6)) * 0.1; opt = optax.adam(3e-3); st = opt.init(R)
tr = sample(24_000, 64_000_000); brng = np.random.default_rng(3); J = lambda a: jnp.asarray(a)
for it in range(1200):
    j = brng.integers(0, len(tr["Xb"]), 128); l, gr = g(R, J(tr["Xb"][j]), J(tr["Xs"][j]), J(tr["qb"][j]), J(tr["qs"][j]), J(tr["cf"][j])); up, st = opt.update(gr, st); R = optax.apply_updates(R, up)
ev = sample(6000, 64_500_000)
def run(Rm, d, sel):
    Lp = np.concatenate([np.asarray(pj(Rm, J(d["Xb"][sel][i:i + 500]), J(d["Xs"][sel][i:i + 500]), J(d["qb"][sel][i:i + 500]), J(d["qs"][sel][i:i + 500]))) for i in range(0, sel.sum(), 500)])
    return Lp
L0 = F.batched(jax.jit(lambda X_: F.forward(M, X_)), ev["Xb"])[np.arange(len(ev["Xb"])), ev["qb"]]
sm = lambda L: (lambda E_: E_ / E_.sum(-1, keepdims=True))(np.exp(L - L.max(-1, keepdims=True)))
def diag(Rm):
    s1 = ev["diff"]; Lp = run(Rm, ev, s1); cf = ev["cf"][s1]; P = sm(Lp); kl = (cf * (np.log(cf) - np.log(P + 1e-12))).sum(-1)
    disc = cf.argmax(-1) != ev["cfA3"][s1].argmax(-1)
    s2 = (~ev["diff"]) & (~ev["same_ep"]); Lq = run(Rm, ev, s2); Pb = sm(L0[s2]); Pq = sm(Lq)
    return dict(n=int(s1.sum()), IIA=float((Lp.argmax(-1) == cf.argmax(-1)).mean()), KL=float(kl.mean()), n_disc=int(disc.sum()),
                IIA_disc=float((Lp[disc].argmax(-1) == cf[disc].argmax(-1)).mean()), IIA_trueE=float((Lp.argmax(-1) == ev["cfE"][s1].argmax(-1)).mean()),
                base_IIA=float((L0[s1].argmax(-1) == cf.argmax(-1)).mean()),
                n_pres=int(s2.sum()), preserve_KL=float((Pb * (np.log(Pb + 1e-12) - np.log(Pq + 1e-12))).sum(-1).mean()), preserve_flip=float((Lq.argmax(-1) != L0[s2].argmax(-1)).mean()))
# selectivity on natural edits (dev eval set)
evd = NV.load_eval() if os.environ.get("DD_EVAL", "dev") == "dev" else NV.load_eval(os.environ["DD_EVAL"]); X = evd["X"]; q = evd["q"]; n = len(X); ix = np.arange(n); cnt = GL.counts(X, 3)[ix, q]; a_ = X.reshape(n, 6, 3, 2)[ix, q]; ag = a_[..., 0] == a_[..., 1]
rng = np.random.default_rng(1102)
def edits(kind, nmax=400):
    b, Xe = [], []
    for i in range(n):
        cand = [h for h in range(3) if not ag[i, h] and ((cnt[i, h] == 5) if kind == "D2" else (cnt[i, h] <= 3))]
        if not cand: continue
        h = cand[rng.integers(len(cand))]; x = X[i].copy(); x[q[i], 2 * h + 1] = x[q[i], 2 * h]; b.append(i); Xe.append(x)
        if len(b) >= nmax: break
    b = np.array(b); return X[b], np.array(Xe), q[b]
def from_eval(mask, key, nmax=400): s = np.where(mask)[0][:nmax]; return X[s], evd[key][s], q[s]
fams = {"D2": edits("D2"), "D2c": edits("D2c"), "CNT": from_eval(np.isin(evd["reg_F45"], ["R2", "R3"]), "X_F45"), "CNTc": from_eval(evd["reg_F34"] == "R1", "X_F34")}
st_f = jax.jit(lambda X_: (lambda r: (F.forward(M, X_, rec=r), r["resid2"])[1])({}))
def disp(Rm):
    Q, _ = np.linalg.qr(np.asarray(Rm)); o = {}
    for k, (Xb, Xa, qq) in fams.items():
        i = np.arange(len(qq)); t = tok(qq); zb = np.asarray(st_f(J(Xb)))[i[:, None], t].reshape(len(qq), -1); za = np.asarray(st_f(J(Xa)))[i[:, None], t].reshape(len(qq), -1)
        o[k] = float(np.linalg.norm((za - zb) @ Q, axis=1).mean())
    return dict(disp=o, sel_D2c_over_D2=o["D2c"] / o["D2"], sel_CNTc_over_CNT=o["CNTc"] / o["CNT"], D2_over_CNT=o["D2"] / o["CNT"])
Rr = J(np.linalg.qr(np.random.default_rng(9).standard_normal((448, 6)))[0])
res = dict(model=M["tag"], rule=RULE, final_loss=float(l), trained=dict(**diag(R), **disp(R)), rand=dict(**diag(Rr), **disp(Rr)))
np.save(os.path.join(OUT, f"{M['tag']}_{RULE}_R.npy"), np.asarray(R)); res["seconds"] = time.time() - t0
json.dump(res, open(os.path.join(OUT, f"{M['tag']}_{RULE}.json"), "w"), indent=1)
t = res["trained"]; print(M["tag"][:24], RULE, {k: round(t[k], 3) for k in ("IIA", "KL", "IIA_disc", "IIA_trueE", "preserve_KL", "preserve_flip", "sel_D2c_over_D2", "D2_over_CNT")}, f"{res['seconds']:.0f}s", flush=True)
