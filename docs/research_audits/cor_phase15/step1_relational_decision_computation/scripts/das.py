"""Workstream A3: distributed interchange intervention (DAS-style causal abstraction) inside one fine-tuned native model.
High-level model H (Theorem 1, C=1):  inputs -> (W_g, L_g, e_g) -> S = sum_g e_g -> p(y) ∝ sum_g W_g L_g(y) (S - e_g).
Variables tested (interchange from a SOURCE episode into a BASE episode; base keeps everything else):
   S    : only the eligible count.   counterfactual p ∝ S_src * A_base - B_base
   E    : the whole eligibility vector (e_0,e_1,e_2).  counterfactual p ∝ sum_g W_g L_g (S_src - e_src,g)  [W, L from base]
   NULL : negative control -- counterfactual = base output (no high-level change), trained with the same sources
Site: residual stream before layer LAYER (1: after fine-tuned layer 0; 2: after layers 0-1 = downstream input; 3: final), token set TOK:
   ro (readout token q) | q7 (readout + 6 source tokens of q) | all (43 tokens).  Subspace: k orthonormal directions in the concatenated
   (|TOK|*64)-dim space; patched = x_b + R R^T (x_s - x_b) (in the concatenated coordinates).
Training: 1200 Adam steps on pairs from fresh generator episodes (stream 61_000_000+seed); 50% of pairs have V_src != V_base (S=2 states
oversampled).  Evaluation on held-out pairs drawn from a separate stream (61_500_000+...), V_src != V_base only:
   IIA_argmax : patched model argmax at q == counterfactual argmax ;  KLcf : mean KL(counterfactual || patched model)
   KLbase     : KL(counterfactual || unpatched model)  (how much the intervention must move the output)
   RAND       : same k, random orthonormal subspace (untrained)
Usage: das.py PARENT SPEC SEED SUFFIX VAR LAYER TOK K -> logs/das/<tag>.json"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp, optax
import ftlib as F, native as NV, genlib as GL
np.seterr(all="ignore")
parent, spec, seed, suf, VAR, LAYER, TOK, K = sys.argv[1], sys.argv[2], int(sys.argv[3]), (sys.argv[4] if sys.argv[4] != "-" else ""), sys.argv[5], int(sys.argv[6]), sys.argv[7], int(sys.argv[8])
EVSTREAM = int(os.environ.get("DAS_EVAL_STREAM", 61_500_000))
M = F.load_model(parent, spec, seed=seed, suffix=suf); tag = f"{M['tag']}_{VAR}_L{LAYER}_{TOK}_k{K}" + ("" if EVSTREAM == 61_500_000 else f"_ev{EVSTREAM}")
OUT = os.path.join(HERE, "..", "logs", "das"); os.makedirs(OUT, exist_ok=True); t0 = time.time()
env = GL.Env(G=3, C=1)
def hl(X, q):
    rel, col, noi = GL._group_lik(env, X); i = np.arange(len(X)); relm = rel.mean(-1); lr = np.log(relm / noi)
    W = np.exp(lr.sum(1) - lr[i, q]); Lg = rel[i, q] / noi; cnt = GL.counts(X, 3)[i, q]; a_ = X.reshape(len(X), 6, 3, 2)[i, q]
    e = ((cnt == 5) & (a_[..., 0] == a_[..., 1])).astype(float); return W, Lg, e
def counterfactual(Xb, Xs, qb, qs):
    Wb, Lb, eb = hl(Xb, qb); _, _, es = hl(Xs, qs); Ss = es.sum(1, keepdims=True); Sb = eb.sum(1, keepdims=True)
    if VAR == "S": f = Ss - eb
    elif VAR == "E": f = Ss - es
    else: f = Sb - eb
    P = np.maximum((Wb[..., None] * f[..., None] * Lb).sum(1), 1e-30); return P / P.sum(-1, keepdims=True), (Ss[:, 0] if VAR != "NULL" else Sb[:, 0]), Sb[:, 0], es, eb
def tokens(q, xp=np):
    if TOK == "ro": return xp.stack([36 + q], 1)
    if TOK == "q7": return xp.concatenate([(36 + q)[:, None], q[:, None] * 6 + xp.arange(6)[None]], 1)
    if TOK == "src6": return q[:, None] * 6 + xp.arange(6)[None]
    return xp.broadcast_to(xp.arange(43)[None], (q.shape[0], 43))
def sample_pairs(n, stream, balanced=True):
    rng = np.random.default_rng(stream); d = NV.Bm.sample_fast(NV.ENV, 4 * n, rng); X = d["X"]; q = rng.integers(0, 6, len(X))
    _, _, e = hl(X, q); S = e.sum(1); two = np.where(S >= 2)[0]; one = np.where(S == 1)[0]
    bi = np.concatenate([rng.choice(two, n // 2), rng.choice(one, n - n // 2)]); rng.shuffle(bi)
    si = np.where(rng.random(n) < 0.5, rng.choice(two, n), rng.choice(one, n))      # sources: half S=2, half S=1
    if VAR == "E":                                                                      # for E, also vary WHICH groups are eligible
        si = rng.integers(0, len(X), n); si = np.where(rng.random(n) < 0.6, rng.choice(two, n), si)
    return X[bi], X[si], q[bi], q[si]
Dfull = len(tokens(np.zeros(1, int))[0]) * 64
def patched_logits(R, Xb, Xs, qb, qs):
    rs = {}; F.forward(M, Xs, rec=rs); xs = rs[f"resid{LAYER}"] if LAYER < 3 else rs["resid3"]
    tb, ts = tokens(qb, jnp), tokens(qs, jnp); i = jnp.arange(Xb.shape[0])[:, None]
    def fn(x):
        vb = x[i, tb].reshape(len(Xb), -1); vs = xs[i, ts].reshape(len(Xb), -1)
        Q, _ = jnp.linalg.qr(R); vp = vb + ((vs - vb) @ Q) @ Q.T
        return x.at[i, tb].set(vp.reshape(len(Xb), tb.shape[1], 64))
    L = F.forward(M, Xb, resid_patch={LAYER: fn})
    return L[jnp.arange(len(Xb)), qb]
pj = jax.jit(patched_logits)
def loss(R, Xb, Xs, qb, qs, P):
    lp = jax.nn.log_softmax(patched_logits(R, Xb, Xs, qb, qs), -1); return -(P * lp).sum(-1).mean()
g = jax.jit(jax.value_and_grad(loss))
R = jax.random.normal(jax.random.PRNGKey(seed + K), (Dfull, K)) * 0.1; opt = optax.adam(3e-3); st = opt.init(R)
Xb, Xs, qb, qs = sample_pairs(24_000, 61_000_000 + seed); P, _, _, _, _ = counterfactual(Xb, Xs, qb, qs); brng = np.random.default_rng(3)
for it in range(1200):
    j = brng.integers(0, len(Xb), 128)
    l, gr = g(R, jnp.asarray(Xb[j]), jnp.asarray(Xs[j]), jnp.asarray(qb[j]), jnp.asarray(qs[j]), jnp.asarray(P[j])); up, st = opt.update(gr, st); R = optax.apply_updates(R, up)
def evaluate(Rm, n=3000):
    Xb, Xs, qb, qs = sample_pairs(n, EVSTREAM + (1 if VAR != "E" else 2)); P, Ss, Sb, es, eb = counterfactual(Xb, Xs, qb, qs)
    keep = (Ss != Sb) if VAR == "S" else (np.abs(es - eb).sum(1) > 0) if VAR == "E" else (Ss == Ss)
    Xb, Xs, qb, qs, P = Xb[keep], Xs[keep], qb[keep], qs[keep], P[keep]
    Lp = np.concatenate([np.asarray(pj(Rm, jnp.asarray(Xb[i:i + 500]), jnp.asarray(Xs[i:i + 500]), jnp.asarray(qb[i:i + 500]), jnp.asarray(qs[i:i + 500]))) for i in range(0, len(Xb), 500)])
    L0 = F.batched(jax.jit(lambda X_: F.forward(M, X_)), Xb)[np.arange(len(Xb)), qb]
    sm = lambda L: np.exp(L - L.max(-1, keepdims=True)) / np.exp(L - L.max(-1, keepdims=True)).sum(-1, keepdims=True)
    kl = lambda Pm: float((P * (np.log(P + 1e-12) - np.log(Pm + 1e-12))).sum(-1).mean())
    return dict(n=int(len(Xb)), IIA_argmax=float((Lp.argmax(-1) == P.argmax(-1)).mean()), KLcf=kl(sm(Lp)), KLbase=kl(sm(L0)),
                base_argmax_agree=float((L0.argmax(-1) == P.argmax(-1)).mean()))
res = dict(tag=tag, var=VAR, layer=LAYER, tok=TOK, k=K, trained=evaluate(R),
           rand=evaluate(jnp.asarray(np.linalg.qr(np.random.default_rng(9).standard_normal((Dfull, K)))[0])), seconds=None, final_loss=float(l))
res["seconds"] = time.time() - t0; json.dump(res, open(os.path.join(OUT, tag + ".json"), "w"), indent=1)
np.save(os.path.join(OUT, tag + "_R.npy"), np.asarray(R))
print(tag, json.dumps({k: res[k] for k in ("trained", "rand")}), f"{res['seconds']:.0f}s", flush=True)
