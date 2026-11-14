"""Workstream C2/C3: WHAT must change to solve the shared-copying two-colluder task, and does the repaired model compute the joint relation J_H?
Layers 0-1 frozen (states resid2 precomputed on a pool of 40k C=2 episodes, stream AD_STREAM).  Adaptation conditions (1500 AdamW steps,
warmup-cosine 1e-3, batch 256, generator labels), all initialised from the native layer 2:
  readout : final LayerNorm + readout matrix WA only
  attn    : layer-2 attention parameters only (g1, b1, WQ, WK, WV, WO)
  mlp     : layer-2 MLP parameters only (g2, b2, W1, c1, W2, c2)
  full    : the whole layer-2 block + final LN + WA        (Investigation XII condition)
  fresh   : a freshly initialised layer 2 + final LN + WA  (capacity baseline)
Evaluation (held-out C=2 episodes):
  accuracy, KL(exact || model), where 'exact' = xii_bayes.posterior_shared (the actual sampler);
  J_H-DEPENDENT SUBSET: episodes with S = 3 at q (all three groups perfect and agreeing; 7.8% of C=2 episodes).  There every per-group
  eligibility rule is uninformative about which group is reliable, while the exact posterior uses the joint relation J_H (which pair is
  identical everywhere).  Accuracy and KL on S = 3 vs S <= 2 separate 'uses J_H' from 'uses per-group e'.
  (A pair-contrast design that breaks the colluders' identity at one question was attempted first and is OFF-SUPPORT under beta = 1: the
  exact posterior is undefined (NaN) -- recorded in logs/COMPUTE_LOG.txt; it is not used.)
Usage: adapt_sel.py RSUF SPEC -> logs/adapt_sel<OUT>/<tag>.json"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp, optax
import ftlib as F, native as NV, genlib as GL, xii_bayes as XB
np.seterr(all="ignore"); t0 = time.time()
RSUF = sys.argv[1] if sys.argv[1] != "-" else ""; parent, spec, seed, suf = (sys.argv[2].split(":") + ["", "", "0", ""])[:4]; STEPS = int(os.environ.get("AD_STEPS", 1000))
M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf); OUT = os.path.join(HERE, "..", "logs", os.environ.get("AD_OUT", "adapt_sel_dev")); os.makedirs(OUT, exist_ok=True)
E2 = GL.Env(3, 2); E1 = GL.Env(3, 1); ST0 = int(os.environ.get("AD_STREAM", 73_100_000))
def resid2(X):
    f = jax.jit(lambda X_: (lambda r: (F.forward(M, X_, rec=r), r["resid2"])[1])({}))
    return np.concatenate([np.asarray(f(jnp.asarray(X[i:i + 1000]))).astype(np.float16) for i in range(0, len(X), 1000)])
rng = np.random.default_rng(ST0); dp = GL.sample(E2, 40000, rng); Zp = resid2(dp["X"]); Yp = dp["Y"]
rng_e = np.random.default_rng(ST0 + 500_000); de = GL.sample(E2, 3000, rng_e); Ze = resid2(de["X"]); Ye = de["Y"]
# J_H contrast pairs
rng_j = np.random.default_rng(ST0 + 700_000); dj = GL.sample(E2, 20000, rng_j); Xj = dj["X"]; qj = rng_j.integers(0, 6, len(Xj)); ridx = dj["ridx"]
roles = E2.roles; base, edit, qs_ = [], [], []
for i in range(len(Xj)):
    g, hs = roles[ridx[i]]; h = hs[rng_j.integers(2)]; qp = [x for x in range(6) if x != qj[i]][rng_j.integers(5)]
    x = Xj[i].copy(); old = x[qp, 2 * h]; new = (old + rng_j.integers(1, 3)) % 3; x[qp, 2 * h] = new; x[qp, 2 * h + 1] = new
    base.append(Xj[i]); edit.append(x); qs_.append(qj[i])
    if len(base) >= 1500: break
Xb, Xe2, qJ = np.array(base), np.array(edit), np.array(qs_); iJ = np.arange(len(qJ))
eb = ((GL.counts(Xb, 3)[iJ, qJ] == 5) & (Xb.reshape(-1, 6, 3, 2)[iJ, qJ][..., 0] == Xb.reshape(-1, 6, 3, 2)[iJ, qJ][..., 1])); ee = ((GL.counts(Xe2, 3)[iJ, qJ] == 5) & (Xe2.reshape(-1, 6, 3, 2)[iJ, qJ][..., 0] == Xe2.reshape(-1, 6, 3, 2)[iJ, qJ][..., 1]))
same_e = bool((eb == ee).all())
cen = lambda P: np.log(P + 1e-12) - np.log(P + 1e-12).mean(-1, keepdims=True)
dB = np.zeros((len(qJ), 3)); dInd = dB   # J_H pair contrast disabled (off-support, see docstring)

def head(t, x):
    x = F.block_heads(t["l2"], x, None, 2, None, None); return F.N.ln(x, t["gf"], t["bf"])[:, 36:42] @ t["WA"]
ATT = ("g1", "b1", "WQ", "WK", "WV", "WO"); MLP = ("g2", "b2", "W1", "c1", "W2", "c2")
def mask_for(cond, t):
    m = jax.tree_util.tree_map(lambda a: False, t)
    if cond in ("full", "fresh"): return jax.tree_util.tree_map(lambda a: True, t)
    m["gf"] = m["bf"] = m["WA"] = True if cond == "readout" else False
    for k in t["l2"]: m["l2"][k] = (cond == "attn" and k in ATT) or (cond == "mlp" and k in MLP)
    if cond in ("attn", "mlp"): m["gf"] = m["bf"] = m["WA"] = False
    return m
def train(t, cond):
    # NOTE (pre-freeze fix): optax.masked passes the RAW gradient through for masked-out leaves, which apply_updates then applies (lr = 1) -> the
    # 'frozen' parameters diverged (readout/attn/mlp conditions, dev run 1).  multi_transform with set_to_zero truly freezes them.
    msk = mask_for(cond, t); labels = jax.tree_util.tree_map(lambda b: "train" if b else "freeze", msk)
    opt = optax.multi_transform({"train": optax.adamw(optax.warmup_cosine_decay_schedule(0.0, 1e-3, STEPS // 10, STEPS), weight_decay=1e-4), "freeze": optax.set_to_zero()}, labels); st = opt.init(t)
    def lossf(tt, x, Y): lp = jax.nn.log_softmax(head(tt, x), -1); return (-jnp.take_along_axis(lp, Y[..., None], -1)[..., 0]).mean()
    @jax.jit
    def step(tt, st, x, Y): l, g = jax.value_and_grad(lossf)(tt, x, Y); up, st = opt.update(g, st, tt); return optax.apply_updates(tt, up), st, l
    r = np.random.default_rng(5)
    for i in range(STEPS): j = r.integers(0, len(Zp), 256); t, st, l = step(t, st, jnp.asarray(Zp[j], jnp.float32), jnp.asarray(Yp[j]))
    return t, float(l)
P_true = np.stack([XB.posterior_shared(E2, de["X"], np.full(len(de["X"]), qq)) for qq in range(6)], 1)
A6 = de["X"].reshape(len(de["X"]), 6, 3, 2); S_e = np.stack([((GL.counts(de["X"], 3)[:, qq] == 5) & (A6[:, qq, :, 0] == A6[:, qq, :, 1])).sum(-1) for qq in range(6)], 1)
kl = lambda P, Qm: (P * (np.log(P + 1e-12) - np.log(Qm + 1e-12))).sum(-1)
def fid(o, p):
    o, p = o.ravel(), p.ravel(); b = float((o * p).sum() / max((p * p).sum(), 1e-12)); return dict(slope=b, R2=float(((1 - ((o - b * p) ** 2).sum() / ((o - o.mean()) ** 2).sum()) if ((o - o.mean()) ** 2).sum() > 1e-10 else 0.0)))
def evaluate(t):
    f = jax.jit(lambda x: head(t, x))
    L = np.concatenate([np.asarray(f(jnp.asarray(Ze[i:i + 1000], jnp.float32))) for i in range(0, len(Ze), 1000)]); P = np.exp(L - L.max(-1, keepdims=True)); P /= P.sum(-1, keepdims=True)
    hit = P.argmax(-1) == Ye; k_ = kl(P_true, P)
    return dict(acc=float(hit.mean()), KL_true=float(k_.mean()), acc_S3=float(hit[S_e == 3].mean()), acc_S12=float(hit[S_e < 3].mean()), KL_S3=float(k_[S_e == 3].mean()), KL_S12=float(k_[S_e < 3].mean()),
                per_ep_hit_S3=hit[S_e == 3].astype(int).tolist(), per_ep_hit=hit.astype(int).ravel()[:6000].tolist())
base_t = dict(l2=M["layers"][2], gf=M["gf"], bf=M["bf"], WA=M["WA"])
hitB = P_true.argmax(-1) == Ye
res = dict(model=M["tag"], steps=STEPS, bayes_acc=float(hitB.mean()), bayes_acc_S3=float(hitB[S_e == 3].mean()), bayes_acc_S12=float(hitB[S_e < 3].mean()), frac_S3=float((S_e == 3).mean()), native=evaluate(base_t))
for cond in os.environ.get("AD_CONDS", "readout,attn,mlp,full,fresh").split(","):
    init = dict(base_t)
    if cond == "fresh": init["l2"] = F.N.init("M", F.NL.ENV, 9191)["layers"][2]
    t, fl = train(init, cond); res[cond] = evaluate(t); res[cond]["final_loss"] = fl
    res[cond]["frozen_params_unchanged"] = bool(all(bool(jnp.all(a == b)) for a, b, m in zip(jax.tree_util.tree_leaves(init), jax.tree_util.tree_leaves(t), jax.tree_util.tree_leaves(mask_for(cond, t))) if not m))
res["seconds"] = time.time() - t0; json.dump(res, open(os.path.join(OUT, f"{M['tag']}.json"), "w"))
print(M["tag"][:28], "bayes_acc", round(res["bayes_acc"], 3), "S3", round(res["bayes_acc_S3"], 3), "S12", round(res["bayes_acc_S12"], 3), flush=True)
for k in ("native",) + tuple(os.environ.get("AD_CONDS", "readout,attn,mlp,full,fresh").split(",")):
    v = res[k]; print(f"   {k:8s} acc {v['acc']:.3f} KL {v['KL_true']:.3f} | S3 acc {v['acc_S3']:.3f} KL {v['KL_S3']:.3f} | S12 acc {v['acc_S12']:.3f}", flush=True)
print("   seconds", round(res["seconds"]), flush=True)
