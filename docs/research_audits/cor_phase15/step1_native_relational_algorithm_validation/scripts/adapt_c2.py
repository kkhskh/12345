"""Workstream D: is the C=2 generalization failure in the eligibility VARIABLE or in the downstream evidence-COMBINATION rule?
Mathematics (exact, G=3, beta=1):  C=1 rule  p ∝ sum_g W_g L_g (S - e_g)          (sum over the OTHER groups' eligibility)
                                     C=2 rule  p ∝ sum_g W_g L_g prod_{h!=g} e_h   (BOTH other groups must be eligible: AND, not SUM)
  plus a correction the e-statistic cannot express: under C=2 the two colluders copy the SAME value, so when all three groups are
  eligible (S=3) the exact posterior also uses cross-group equality.  'C2prod' below is the product rule on e (sufficiency gap reported).
Selective adaptation (layers 0-1 frozen = the learned relational representation is kept):
  native      : the C=1-trained model, no adaptation
  adapt_l2    : layer 2 + final LN + readout re-trained on C=2 data, initialised from the native layer 2         [selective adaptation]
  adapt_l2_noE: same, but the E-subspace coordinates (Investigation X DAS, per question's q7 block) are replaced by their pool mean at the
                layer-2 input during training and evaluation (can the correction be learned WITHOUT the eligibility representation?)
  fresh_l2    : a freshly initialised layer 2 trained the same way (capacity baseline)
TRUE = exact posterior of the actual sampler (src/xii_bayes.posterior_shared; colluders share one copied value); 'indep' = genlib's
posterior_brute (independent copying; used as 'true' in Investigation XI -- corrected here).  C2prod == indep exactly (checked).
Outcomes on held-out C=2 episodes (all 6 questions): accuracy, KL(true C2 posterior || model), KL(C2prod || model), KL(C1 rule || model);
  E-dependence of each model: KL(model(x) || model(x with E coordinates mean-patched)).
Usage: adapt_c2.py RSUF SPEC [STEPS] -> logs/adapt_c2/<tag>.json"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp, optax
import ftlib as F, native as NV, genlib as GL, xii_bayes as XB
np.seterr(all="ignore"); t0 = time.time()
RSUF = sys.argv[1] if sys.argv[1] != "-" else ""; parent, spec, seed, suf = (sys.argv[2].split(":") + ["", "", "0", ""])[:4]; STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf); OUT = os.path.join(HERE, "..", "logs", os.environ.get("AD_OUT", "adapt_c2")); os.makedirs(OUT, exist_ok=True)
RDIR = os.environ.get("XII_RDIR", os.path.join(NV.ROOT, "step1_causal_equivalence", "logs", "das"))
Q, _ = np.linalg.qr(np.load(os.path.join(RDIR, f"{M['tag']}_E_L2_q7_k6{RSUF}_R.npy"))); Qj = jnp.asarray(Q, jnp.float32)
E2 = GL.Env(3, 2); E1 = GL.Env(3, 1)
def resid2(X):
    f = jax.jit(lambda X_: (lambda r: (F.forward(M, X_, rec=r), r["resid2"])[1])({}))
    return np.concatenate([np.asarray(f(jnp.asarray(X[i:i + 1000]))).astype(np.float16) for i in range(0, len(X), 1000)])
rng = np.random.default_rng(int(os.environ.get("AD_STREAM", 71_000_000))); dp = GL.sample(E2, 40000, rng); Zp = resid2(dp["X"]); Yp = dp["Y"]
rng_e = np.random.default_rng(int(os.environ.get("AD_STREAM", 71_000_000)) + 500_000); de = GL.sample(E2, 4000, rng_e); Ze = resid2(de["X"]); Ye = de["Y"]
TOK = np.stack([np.concatenate([[36 + qq], qq * 6 + np.arange(6)]) for qq in range(6)])      # (6,7) q7 token blocks
zbar = jnp.asarray(Zp[:5000].astype(np.float32)[:, TOK].reshape(5000, 6, -1).mean(0))         # (6,448) pool mean per block
def noE(x):
    blk = x[:, TOK].reshape(x.shape[0], 6, -1); blk = blk + ((zbar[None] - blk) @ Qj) @ Qj.T; return x.at[:, TOK].set(blk.reshape(x.shape[0], 6, 7, 64))
def head(t, x, ablate=False):
    if ablate: x = noE(x)
    x = F.block_heads(t["l2"], x, None, 2, None, None); return F.N.ln(x, t["gf"], t["bf"])[:, 36:42] @ t["WA"]
def train(t, ablate):
    opt = optax.adamw(optax.warmup_cosine_decay_schedule(0.0, 1e-3, STEPS // 10, STEPS), weight_decay=1e-4); st = opt.init(t)
    def lossf(tt, x, Y): lp = jax.nn.log_softmax(head(tt, x, ablate), -1); return (-jnp.take_along_axis(lp, Y[..., None], -1)[..., 0]).mean()
    @jax.jit
    def step(tt, st, x, Y): l, g = jax.value_and_grad(lossf)(tt, x, Y); up, st = opt.update(g, st, tt); return optax.apply_updates(tt, up), st, l
    r = np.random.default_rng(5)
    for i in range(STEPS):
        j = r.integers(0, len(Zp), 256); t, st, l = step(t, st, jnp.asarray(Zp[j], jnp.float32), jnp.asarray(Yp[j]))
    return t, float(l)
# references on the evaluation set
qs = np.arange(6)
def posts(env, f):
    return np.stack([f(env, de["X"], np.full(len(de["X"]), qq)) for qq in qs], 1)         # (n,6,K)
P_true = posts(E2, XB.posterior_shared)          # exact posterior of the actual sampler (shared copied value)
P_indep = posts(E2, GL.posterior_brute)          # genlib / Investigation XI reference (independent copying)
def rule(kind):
    out = []
    rel, col, noi = GL._group_lik(E1, de["X"]); relm = rel.mean(-1); lr = np.log(relm / noi); n_ = len(de["X"]); ii = np.arange(n_)
    for qq in qs:
        qv = np.full(n_, qq); W = np.exp(lr.sum(1) - lr[ii, qv]); Lg = rel[ii, qv] / noi; A_ = de["X"].reshape(n_, 6, 3, 2)[ii, qv]
        e = ((GL.counts(de["X"], 3)[ii, qv] == 5) & (A_[..., 0] == A_[..., 1])).astype(float)
        m = (e.sum(1, keepdims=True) - e) if kind == "C1" else np.stack([e[:, [1, 2]].prod(1), e[:, [0, 2]].prod(1), e[:, [0, 1]].prod(1)], 1)
        P = np.maximum((W[..., None] * m[..., None] * Lg).sum(1), 1e-30); out.append(P / P.sum(-1, keepdims=True))
    return np.stack(out, 1)
P_c1, P_c2p = rule("C1"), rule("C2prod")
S_all = np.stack([((GL.counts(de["X"], 3)[:, qq] == 5) & (de["X"].reshape(len(de["X"]), 6, 3, 2)[:, qq, :, 0] == de["X"].reshape(len(de["X"]), 6, 3, 2)[:, qq, :, 1])).sum(-1) for qq in qs], 1)
kl = lambda P, Qm: (P * (np.log(P + 1e-12) - np.log(Qm + 1e-12))).sum(-1)
def evaluate(t, ablate):
    f = jax.jit(lambda x: head(t, x, ablate)); fa = jax.jit(lambda x: head(t, x, True))
    L = np.concatenate([np.asarray(f(jnp.asarray(Ze[i:i + 1000], jnp.float32))) for i in range(0, len(Ze), 1000)]); P = np.exp(L - L.max(-1, keepdims=True)); P /= P.sum(-1, keepdims=True)
    La = np.concatenate([np.asarray(fa(jnp.asarray(Ze[i:i + 1000], jnp.float32))) for i in range(0, len(Ze), 1000)]); Pa = np.exp(La - La.max(-1, keepdims=True)); Pa /= Pa.sum(-1, keepdims=True)
    r = dict(acc=float((P.argmax(-1) == Ye).mean()), KL_true=float(kl(P_true, P).mean()), KL_C2prod=float(kl(P_c2p, P).mean()), KL_C1rule=float(kl(P_c1, P).mean()), KL_indep=float(kl(P_indep, P).mean()),
             E_dependence=float(kl(P, Pa).mean()), acc_S2=float((P.argmax(-1) == Ye)[S_all == 2].mean()), acc_S3=float((P.argmax(-1) == Ye)[S_all == 3].mean()) if (S_all == 3).any() else None)
    return r
base = dict(l2=M["layers"][2], gf=M["gf"], bf=M["bf"], WA=M["WA"])
fresh = dict(base); fresh["l2"] = jax.tree_util.tree_map(lambda a: a, F.N.init("M", F.NL.ENV, 9191)["layers"][2])
res = dict(model=M["tag"], steps=STEPS, refs=dict(bayes_acc=float((P_true.argmax(-1) == Ye).mean()), indep_acc=float((P_indep.argmax(-1) == Ye).mean()),
                                    KL_true_indep=float(kl(P_true, P_indep).mean()), KL_indep_C2prod=float(kl(P_indep, P_c2p).mean()), C2prod_acc=float((P_c2p.argmax(-1) == Ye).mean()), C1rule_acc=float((P_c1.argmax(-1) == Ye).mean()),
                                    KL_true_C2prod=float(kl(P_true, P_c2p).mean()), KL_true_C1rule=float(kl(P_true, P_c1).mean()), frac_S3=float((S_all == 3).mean()),
                                    KL_true_C2prod_S3=float(kl(P_true, P_c2p)[S_all == 3].mean()) if (S_all == 3).any() else None))
res["native"] = evaluate(base, False)
for nm, init, abl in (("adapt_l2", base, False), ("adapt_l2_noE", base, True), ("fresh_l2", fresh, False)):
    t, fl = train(init, abl); res[nm] = evaluate(t, abl); res[nm]["final_loss"] = fl
res["seconds"] = time.time() - t0; json.dump(res, open(os.path.join(OUT, f"{M['tag']}.json"), "w"), indent=1)
print(M["tag"][:28], "refs", {k: (round(v, 3) if v is not None else None) for k, v in res["refs"].items()}, flush=True)
for k in ("native", "adapt_l2", "adapt_l2_noE", "fresh_l2"): print("  ", k, {kk: (round(v, 3) if v is not None else None) for kk, v in res[k].items()}, flush=True)
print("   seconds", round(res["seconds"]), flush=True)
