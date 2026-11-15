"""Workstream C (XIII-B): two-colluder (C=2, shared copying) adaptation, extended.
Conditions (matched budget: AD_STEPS steps, batch 256, AdamW warmup-cosine 1e-3, wd 1e-4, data stream AD_STREAM):
  layer-2-side (layers 0-1 frozen; trained on frozen resid2 as in XIII adapt_sel.py):  readout | attn | mlp | full | fresh
  early01   : layers 0 AND 1 trainable; embeddings, layer 2 and the readout FROZEN (necessity arm, C4)
  early01ro : layers 0-1 + readout trainable, layer 2 frozen (control: is the frozen layer-2 the bottleneck?)
Generator: C = 2, beta = AD_BETA (1.0 = XIII; 0.6 = partial copying, where per-group statistics are insufficient almost everywhere).
Evaluation (3000 fresh episodes x 6 questions): acc, KL(exact || model) with the EXACT shared-copy posterior (xii_bayes.posterior_shared);
  beta = 1: slices S3_notall (S = 3, not all groups agree at q: the J_H-dependent slice), S3_all, S12;
  J-PAIRS (valid at any beta < 1; at beta = 1 they are off-support and skipped): x' = x with ONE group's two sources relabelled by a non-identity
  permutation of the 3 values at every question except q.  Every per-group statistic (self-agreement pattern -> e_g, c_g, W_g; values at q -> L_g)
  is unchanged, the cross-group identity relation changes.  Exact change vs observed change of centred log-probs at q -> fidelity; the
  independent-copying product rule predicts exactly 0 (checked).
Adapted parameters are saved (checkpoints/adapt2/<tag>_<cond>_b<beta>.npz) for dissection (C3).  -> logs/adapt2<OUT>/<tag>_b<beta>.json"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp, optax
import ftlib as F, native as NV, genlib as GL, xii_bayes as XB, relpairs as RP
np.seterr(all="ignore"); t0 = time.time()
sp = sys.argv[1]; parent, spec, seed, suf = (sp.split(":") + ["", "", "0", ""])[:4]; M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf)
STEPS = int(os.environ.get("AD_STEPS", 1000)); BETA = float(os.environ.get("AD_BETA", 1.0)); ST0 = int(os.environ.get("AD_STREAM", 76_000_000)) + int(100 * BETA)
CONDS = os.environ.get("AD_CONDS", "readout,attn,mlp,full,fresh,early01,early01ro").split(",")
OUT = os.path.join(HERE, "..", "logs", "adapt2" + os.environ.get("AD_OUT", "")); os.makedirs(OUT, exist_ok=True)
CK = os.path.join(HERE, "..", "checkpoints", "adapt2"); os.makedirs(CK, exist_ok=True)
E2 = GL.Env(3, 2, beta=BETA)
def resid2(X, Mx=M):
    f = jax.jit(lambda X_: (lambda r: (F.forward(Mx, X_, rec=r), r["resid2"])[1])({}))
    return np.concatenate([np.asarray(f(jnp.asarray(X[i:i + 1000]))).astype(np.float16) for i in range(0, len(X), 1000)])
rng = np.random.default_rng(ST0); dp = GL.sample(E2, 40000, rng); Xp, Yp = dp["X"], dp["Y"]; Zp = resid2(Xp)
rng_e = np.random.default_rng(ST0 + 500_000); de = GL.sample(E2, 3000, rng_e); Xe, Ye = de["X"], de["Y"]; Ze = resid2(Xe)
P_true = np.stack([XB.posterior_shared(E2, Xe, np.full(len(Xe), qq)) for qq in range(6)], 1)
A6 = Xe.reshape(len(Xe), 6, 3, 2); S_e = np.stack([((GL.counts(Xe, 3)[:, qq] == 5) & (A6[:, qq, :, 0] == A6[:, qq, :, 1])).sum(-1) for qq in range(6)], 1)
allq = np.stack([(A6[:, qq, 0, 0] == A6[:, qq, 1, 0]) & (A6[:, qq, 1, 0] == A6[:, qq, 2, 0]) for qq in range(6)], 1)
# J-pairs
JP = None
if BETA < 1:
    rj = np.random.default_rng(ST0 + 700_000); dj = GL.sample(E2, 1500, rj); Xj = dj["X"]; qj = rj.integers(0, 6, len(Xj)); hj = rj.integers(0, 3, len(Xj))
    perms = [p for p in __import__("itertools").permutations(range(3)) if p != (0, 1, 2)]
    Xj2 = Xj.copy()
    for i in range(len(Xj)):
        pm = np.array(perms[rj.integers(len(perms))]); cols = [2 * hj[i], 2 * hj[i] + 1]; others = [t for t in range(6) if t != qj[i]]
        Xj2[i][np.ix_(others, cols)] = pm[Xj[i][np.ix_(others, cols)]]
    ij = np.arange(len(Xj)); dB = RP.cen(XB.posterior_shared(E2, Xj2, qj)) - RP.cen(XB.posterior_shared(E2, Xj, qj))
    dInd = RP.cen(GL.posterior_brute(E2, Xj2, qj)) - RP.cen(GL.posterior_brute(E2, Xj, qj))
    ok = np.isfinite(dB).all(1)
    JP = dict(Xj=Xj[ok], Xj2=Xj2[ok], qj=qj[ok], dB=dB[ok], ind_max_abs=float(np.abs(dInd[ok]).max()), n=int(ok.sum()), exact_mean_abs=float(np.abs(dB[ok]).mean()),
              frac_decision_change=float((XB.posterior_shared(E2, Xj2[ok], qj[ok]).argmax(-1) != XB.posterior_shared(E2, Xj[ok], qj[ok]).argmax(-1)).mean()))
def head(t, x):
    x = F.block_heads(t["l2"], x, None, 2, None, None); return F.N.ln(x, t["gf"], t["bf"])[:, 36:42] @ t["WA"]
ATT = ("g1", "b1", "WQ", "WK", "WV", "WO"); MLP = ("g2", "b2", "W1", "c1", "W2", "c2")
def full_logits(t, X_):
    Mx = dict(M); Mx["layers"] = [t["l0"], t["l1"], t["l2"]]; Mx["gf"], Mx["bf"], Mx["WA"] = t["gf"], t["bf"], t["WA"]; return F.forward(Mx, X_)
def labels_for(cond, t):
    lab = jax.tree_util.tree_map(lambda a: "freeze", t)
    def setall(k): lab[k] = jax.tree_util.tree_map(lambda a: "train", t[k])
    if cond in ("full", "fresh"): [setall(k) for k in t]
    elif cond == "readout": [setall(k) for k in ("gf", "bf", "WA")]
    elif cond in ("attn", "mlp"): lab["l2"] = {k: ("train" if (k in ATT if cond == "attn" else k in MLP) else "freeze") for k in t["l2"]}
    elif cond == "early01": setall("l0"); setall("l1")
    elif cond == "early01ro": setall("l0"); setall("l1"); [setall(k) for k in ("gf", "bf", "WA")]
    return lab
def evaluate(t, early):
    if early:
        f = jax.jit(lambda X_: full_logits(t, X_)); L = np.concatenate([np.asarray(f(jnp.asarray(Xe[i:i + 500]))) for i in range(0, len(Xe), 500)])
    else:
        f = jax.jit(lambda z: head(t, z)); L = np.concatenate([np.asarray(f(jnp.asarray(Ze[i:i + 1000], jnp.float32))) for i in range(0, len(Ze), 1000)])
    P = RP.sm(L); hit = P.argmax(-1) == Ye; kl = (P_true * (np.log(P_true + 1e-12) - np.log(P + 1e-12))).sum(-1)
    r = dict(acc=float(hit.mean()), KL=float(np.nanmean(kl)))
    if BETA == 1.0:
        for nm, m in (("S3_notall", (S_e == 3) & ~allq), ("S3_all", (S_e == 3) & allq), ("S12", S_e < 3)): r[f"acc_{nm}"] = float(hit[m].mean()); r[f"KL_{nm}"] = float(np.nanmean(kl[m]))
    if JP is not None:
        if early:
            Lb = np.asarray(jax.jit(lambda X_: full_logits(t, X_))(jnp.asarray(JP["Xj"]))); La = np.asarray(jax.jit(lambda X_: full_logits(t, X_))(jnp.asarray(JP["Xj2"])))
        else:
            fz = jax.jit(lambda z: head(t, z)); Lb = np.asarray(fz(jnp.asarray(resid2(JP["Xj"]), jnp.float32))); La = np.asarray(fz(jnp.asarray(resid2(JP["Xj2"]), jnp.float32)))
        i = np.arange(JP["n"]); dO = RP.cen(RP.sm(La[i, JP["qj"]])) - RP.cen(RP.sm(Lb[i, JP["qj"]])); r["Jpair"] = RP.fid(dO, JP["dB"])
        r["Jpair"]["decision_follows_exact"] = float((RP.sm(La[i, JP["qj"]]).argmax(-1) == XB.posterior_shared(E2, JP["Xj2"], JP["qj"]).argmax(-1)).mean())
    return r
def train(t, cond):
    lab = labels_for(cond, t); early = cond.startswith("early")
    opt = optax.multi_transform({"train": optax.adamw(optax.warmup_cosine_decay_schedule(0.0, 1e-3, STEPS // 10, STEPS), weight_decay=1e-4), "freeze": optax.set_to_zero()}, lab); st = opt.init(t)
    if early:
        def lossf(tt, X_, Y): lp = jax.nn.log_softmax(full_logits(tt, X_), -1); return (-jnp.take_along_axis(lp, Y[..., None], -1)[..., 0]).mean()
    else:
        def lossf(tt, z, Y): lp = jax.nn.log_softmax(head(tt, z), -1); return (-jnp.take_along_axis(lp, Y[..., None], -1)[..., 0]).mean()
    @jax.jit
    def step(tt, st, a, Y): l, g = jax.value_and_grad(lossf)(tt, a, Y); up, st = opt.update(g, st, tt); return optax.apply_updates(tt, up), st, l
    r = np.random.default_rng(5); l = None
    for i in range(STEPS):
        j = r.integers(0, len(Xp), 256)
        a = jnp.asarray(Xp[j]) if early else jnp.asarray(Zp[j], jnp.float32); t, st, l = step(t, st, a, jnp.asarray(Yp[j]))
    unchanged = all(bool(jnp.all(x == y)) for x, y, m in zip(jax.tree_util.tree_leaves(t0_[cond]), jax.tree_util.tree_leaves(t), jax.tree_util.tree_leaves(lab)) if m == "freeze")
    return t, float(l), unchanged
base_t = dict(l0=M["layers"][0], l1=M["layers"][1], l2=M["layers"][2], gf=M["gf"], bf=M["bf"], WA=M["WA"])
res = dict(model=M["tag"], beta=BETA, steps=STEPS, native=evaluate(base_t, False), bayes_acc=float((P_true.argmax(-1) == Ye).mean()), Jpairs=None if JP is None else {k: v for k, v in JP.items() if not isinstance(v, np.ndarray)})
if BETA == 1.0:
    hB = P_true.argmax(-1) == Ye; res["bayes_acc_S3_notall"] = float(hB[(S_e == 3) & ~allq].mean())
print(M["tag"][:30], "beta", BETA, "bayes", round(res["bayes_acc"], 3), "native", {k: (round(v, 3) if isinstance(v, float) else {kk: round(vv, 3) for kk, vv in v.items()}) for k, v in res["native"].items()}, "Jpairs", res["Jpairs"], flush=True)
t0_ = {}
for cond in CONDS:
    init = dict(base_t)
    if cond == "fresh": init["l2"] = F.N.init("M", F.NL.ENV, 9191)["layers"][2]
    t0_[cond] = init; t, fl, unch = train(init, cond); r = evaluate(t, cond.startswith("early")); r.update(final_loss=fl, frozen_params_unchanged=unch); res[cond] = r
    np.savez(os.path.join(CK, f"{M['tag']}_{cond}_b{BETA}.npz"), **{"/".join(str(getattr(k, 'key', k)) for k in path): np.asarray(v) for path, v in jax.tree_util.tree_flatten_with_path(t)[0]})
    print(f"   {cond:9s}", {k: (round(v, 3) if isinstance(v, float) else ({kk: round(vv, 3) for kk, vv in v.items()} if isinstance(v, dict) else v)) for k, v in r.items()}, round(time.time() - t0), "s", flush=True)
res["seconds"] = time.time() - t0; json.dump(res, open(os.path.join(OUT, f"{M['tag']}_b{BETA}.json"), "w"), indent=1)
