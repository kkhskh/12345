"""Workstream C1: mapped cross-model interchange.
Target model B (with its own Investigation X E-subspace Q_B, L2/q7/k6).  Source model A.  A linear map M_AB: q7 states of A -> q7 states of B
(448 -> 448, ridge lambda=1e-2 relative), fit by least squares on NATURAL episodes only (stream 63_000_000; no counterfactuals, no outcomes).
Mapped interchange in B: base episode x_b run in B; source episode x_s run in A, mapped, then  z_B' = z_b^B + Q_B Q_B^T (M_AB z_s^A - z_b^B).
Compared against:
  within    : source state from B itself (Investigation X procedure)                         [upper reference]
  mapped    : as above
  shuffled  : map fit on episodes paired with WRONG partners (A's state of episode i -> B's state of episode pi(i))   [alignment control]
  rand_sub  : mapped source but a random 6-dim subspace in B instead of Q_B                     [subspace control]
  mean      : B's mean natural q7 state patched into Q_B (source-free control)
  base      : no intervention
Evaluation: held-out pairs (stream 63_500_000) with e_src != e_base; counterfactual of the high-level E intervention (Bayes, W/L from base).
Metrics: IIA (argmax), KL(cf || patched), and on the discriminating subset (Bayes vs separable counterfactual argmax differ).
Usage: cross_interchange.py A_SPEC B_SPEC RSUF_B -> logs/cross/<A>__<B>.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import ftlib as F, native as NV, genlib as GL
np.seterr(all="ignore")
def load(sp):
    parent, spec, seed, suf = (sp.split(":") + ["", "", "0", ""])[:4]; return F.load_model(parent, spec, seed=int(seed or 0), suffix=suf)
MA, MB = load(sys.argv[1]), load(sys.argv[2]); RSUF = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "-" else ""
X_DAS = os.environ.get("XII_RDIR", os.path.join(NV.ROOT, "step1_causal_equivalence", "logs", "das")); QB, _ = np.linalg.qr(np.load(os.path.join(X_DAS, f"{MB['tag']}_E_L2_q7_k6{RSUF}_R.npy")))
OUT = os.path.join(HERE, "..", "logs", os.environ.get("XI_OUT", "cross")); os.makedirs(OUT, exist_ok=True)
env = GL.Env(G=3, C=1)
def hl(X, q):
    rel, col, noi = GL._group_lik(env, X); i = np.arange(len(X)); relm = rel.mean(-1); lr = np.log(relm / noi)
    W = np.exp(lr.sum(1) - lr[i, q]); Lg = rel[i, q] / noi; c = GL.counts(X, 3)[i, q]; a_ = X.reshape(len(X), 6, 3, 2)[i, q]
    return W, Lg, ((c == 5) & (a_[..., 0] == a_[..., 1])).astype(float)
tok = lambda q: np.concatenate([(36 + q)[:, None], q[:, None] * 6 + np.arange(6)[None]], 1)
def states(M, X):
    f = jax.jit(lambda X_: (lambda r: (F.forward(M, X_, rec=r), r["resid2"])[1])({}))
    return np.concatenate([np.asarray(f(jnp.asarray(X[i:i + 1000]))) for i in range(0, len(X), 1000)])
def q7(Z, q): i = np.arange(len(q)); return Z[i[:, None], tok(q)].reshape(len(q), -1)
# fit map on natural episodes
rng = np.random.default_rng(63_000_000); Xf = NV.Bm.sample_fast(NV.ENV, 8000, rng)["X"]; qf = rng.integers(0, 6, len(Xf))
ZA, ZB = q7(states(MA, Xf), qf), q7(states(MB, Xf), qf)
def ridge(Za, Zb, lam=1e-2):
    mu_a, mu_b = Za.mean(0), Zb.mean(0); A0, B0 = Za - mu_a, Zb - mu_b; G = A0.T @ A0; G += lam * np.trace(G) / len(G) * np.eye(len(G))
    Wm = np.linalg.solve(G, A0.T @ B0); return lambda z: (z - mu_a) @ Wm + mu_b
Mmap = ridge(ZA[:6000], ZB[:6000]); perm = rng.permutation(6000); Mshuf = ridge(ZA[:6000], ZB[:6000][perm])
# XII correction: Investigation XI fitted the map on all 8000 states and reported R^2 on states 6000-8000, i.e. IN-SAMPLE.  Here the map is fit
# on states 0-5999 only; R^2 is reported in-sample (0-5999), on the held-out tail of the fit stream (6000-7999), and on the separate
# evaluation stream (the base states of the evaluation pairs, computed below).
r2 = lambda Za, Zb: float(1 - ((Mmap(Za) - Zb) ** 2).sum() / ((Zb - Zb.mean(0)) ** 2).sum())
map_r2_in = r2(ZA[:6000], ZB[:6000]); map_r2 = r2(ZA[6000:], ZB[6000:])
# held-out pairs
rng = np.random.default_rng(63_500_000); d = NV.Bm.sample_fast(NV.ENV, 24000, rng); X = d["X"]; q = rng.integers(0, 6, len(X))
W, Lg, e = hl(X, q); S = e.sum(1); two = np.where(S >= 2)[0]; npair = 3000
bi = np.concatenate([rng.choice(two, npair // 2), rng.choice(np.where(S == 1)[0], npair - npair // 2)]); si = np.where(rng.random(npair) < 0.6, rng.choice(two, npair), rng.integers(0, len(X), npair))
keep = np.abs(e[si] - e[bi]).sum(1) > 0; bi, si = bi[keep], si[keep]
norm = lambda P: (np.maximum(P, 0) + 1e-12) / (np.maximum(P, 0) + 1e-12).sum(-1, keepdims=True)
cfB = norm((W[bi][..., None] * (e[si].sum(1, keepdims=True) - e[si])[..., None] * Lg[bi]).sum(1)); cfA = norm((W[bi][..., None] * (1 - e[si])[..., None] * Lg[bi]).sum(1))
disc = cfB.argmax(-1) != cfA.argmax(-1)
Xb, Xs, qb, qs = X[bi], X[si], q[bi], q[si]
zsA = q7(states(MA, Xs), qs); zsB = q7(states(MB, Xs), qs); map_r2_evalstream = r2(zsA, zsB)
def patched_logits(src_q7, Qm):
    def run(Xb_, qb_, src_):
        i = jnp.arange(Xb_.shape[0])[:, None]; tb = jnp.concatenate([(36 + qb_)[:, None], qb_[:, None] * 6 + jnp.arange(6)[None]], 1)
        def fn(x):
            vb = x[i, tb].reshape(Xb_.shape[0], -1); vp = vb + ((src_ - vb) @ Qm) @ Qm.T; return x.at[i, tb].set(vp.reshape(Xb_.shape[0], 7, 64))
        return F.forward(MB, Xb_, resid_patch={2: fn})[jnp.arange(Xb_.shape[0]), qb_]
    rj = jax.jit(run)
    return np.concatenate([np.asarray(rj(jnp.asarray(Xb[i:i + 500]), jnp.asarray(qb[i:i + 500]), jnp.asarray(src_q7[i:i + 500]))) for i in range(0, len(Xb), 500)])
def score(L):
    P = np.exp(L - L.max(-1, keepdims=True)); P /= P.sum(-1, keepdims=True); kl = (cfB * (np.log(cfB) - np.log(P + 1e-12))).sum(-1)
    return dict(IIA=float((L.argmax(-1) == cfB.argmax(-1)).mean()), KL=float(kl.mean()), IIA_disc=float((L[disc].argmax(-1) == cfB[disc].argmax(-1)).mean()), KL_disc=float(kl[disc].mean()),
                per_pair_KL=kl.astype(np.float32).tolist())
Qr, _ = np.linalg.qr(np.random.default_rng(5).standard_normal(QB.shape)); Qbj, Qrj = jnp.asarray(QB), jnp.asarray(Qr)
L0 = F.batched(jax.jit(lambda X_: F.forward(MB, X_)), Xb)[np.arange(len(Xb)), qb]
res = dict(A=MA["tag"], B=MB["tag"], map_heldout_R2=map_r2, map_insample_R2=map_r2_in, map_evalstream_R2=map_r2_evalstream, n=int(len(bi)), n_disc=int(disc.sum()),
           within=score(patched_logits(zsB, Qbj)), mapped=score(patched_logits(Mmap(zsA), Qbj)), shuffled=score(patched_logits(Mshuf(zsA), Qbj)),
           rand_sub=score(patched_logits(Mmap(zsA), Qrj)), mean=score(patched_logits(np.repeat(ZB.mean(0)[None], len(Xb), 0), Qbj)), base=score(L0))
for k in ("mapped", "shuffled", "mean", "rand_sub"): res[k]["gain_KL"] = (res["base"]["KL"] - res[k]["KL"]) / (res["base"]["KL"] - res["within"]["KL"])
json.dump(res, open(os.path.join(OUT, f"{MA['tag']}__{MB['tag']}.json"), "w"), indent=1)
print(MA["tag"][:22], "->", MB["tag"][:22], "R2 in/held/eval", round(map_r2_in, 3), round(map_r2, 3), round(map_r2_evalstream, 3), {k: (round(v["IIA"], 3), round(v["IIA_disc"], 3), round(v["KL"], 3), round(v.get("gain_KL", 1.0), 3)) for k, v in res.items() if isinstance(v, dict)}, flush=True)
