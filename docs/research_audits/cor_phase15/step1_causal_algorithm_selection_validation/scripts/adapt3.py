"""XIII-C Workstream C5/C6: capacity-controlled and information-access-controlled two-colluder adaptation (C = 2, shared copying, beta = AD_BETA).
All arms: AD_STEPS steps (default 1000), batch 256, AdamW warmup-cosine to AD_LR (default 1e-3), wd 1e-4, identical training episodes (stream
AD_STREAM + 100 beta, as XIII-B adapt2) and identical evaluation episodes.  Trainable-parameter counts are recorded for every arm.
  L2       : layer 2 + readout (gf, bf, WA) trainable on frozen resid2                                  (= XIII-B 'full')
  early01  : layers 0 and 1 trainable; layer 2 and readout frozen                                        (= XIII-B 'early01')
  L1only   : layer 1 only trainable (ONE block, same block size as L2); layers 0, 2 and readout frozen   [capacity-matched to L2]
  L0only   : layer 0 only trainable (one block)                                                          [capacity-matched to L2]
  L2plus   : layer 2 + readout + one extra block after layer 2 (zero-init output maps -> starts as identity) [capacity-matched to early01]
  preattn  : ONE fresh attention-only sublayer inserted before frozen layer 2 (zero-init WO); layer 2 and readout frozen   [cross-token adapter]
  pretok   : ONE token-wise MLP adapter (64-128-64, zero-init output) before frozen layer 2; layer 2 + readout frozen [no cross-token mixing]
  supL2    : the sufficient joint statistic m = (m_01, m_02, m_12) is SUPPLIED: m_H(q) = #{q' != q : all four sources of the two groups in H
             equal}; a trainable linear map E (3 -> 64, zero-init) adds E m/5 to resid2 at the readout token and the 6 source tokens of each q;
             layer 2 + readout trainable                                                                 [information-access test]
  supFrozen: same supplied m; ONLY E trainable (layer 2, readout frozen)                                 [can the frozen mixture use it?]
  supG     : the same information in PER-GROUP format (added after development showed supL2 fails): at the six q-source tokens of group g,
             features (m_{H(-g)}, m_{g,h_a}, m_{g,h_b})/5 where H(-g) is the pair NOT containing g (the exact posterior's log-weight of
             "g reliable" is linear in m_{H(-g)}); linear map E (3 -> 64, zero-init); layer 2 + readout trainable
  supGFrozen: per-group supplied features; ONLY E trainable (layer 2 and readout frozen)
  supDecoy : supplies a DECOY statistic of the same shape (raw first-source cross-equality counts at q' != q); layer 2 + readout trainable
Evaluation (3000 fresh episodes x 6 q): acc, KL(exact || model) with xii_bayes.posterior_shared; J-pairs (XIII-B construction) split by whether the
relabelling changes m (DM) or not (D0; exact change 0 by sufficiency); DECOY pairs: one source of a NON-self-agreeing group at q' != q moved to
another value that keeps the pair non-agreeing and changes raw cross-equality, never an all-four-equal set (exact change 0, checked).
Usage: AD_ARMS=L2,early01,... python3 scripts/adapt3.py SPEC   (SPEC per src/xc.py) -> logs/adapt3<OUT>/<unit>_b<beta>_lr<lr>.json ; params in checkpoints/adapt3/"""
import json, os, sys, time, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp, optax
import xc, ftlib as F, native as NV, genlib as GL, xii_bayes as XB, relpairs as RP
np.seterr(all="ignore"); t0 = time.time()
sp = sys.argv[1]; M = xc.load(sp); UN = sp.replace(":", "_")
STEPS = int(os.environ.get("AD_STEPS", 1000)); BETA = float(os.environ.get("AD_BETA", 0.6)); LR = float(os.environ.get("AD_LR", 1e-3))
ST0 = int(os.environ.get("AD_STREAM", 76_000_000)) + int(100 * BETA)
ARMS = os.environ.get("AD_ARMS", "L2,early01,L1only,L0only,L2plus,preattn,pretok,supL2,supFrozen,supDecoy").split(",")
OUT = os.path.join(HERE, "..", "logs", "adapt3" + os.environ.get("AD_OUT", "")); os.makedirs(OUT, exist_ok=True)
CK = os.path.join(HERE, "..", "checkpoints", "adapt3"); os.makedirs(CK, exist_ok=True)
E2 = GL.Env(3, 2, beta=BETA); PAIRS = [(0, 1), (0, 2), (1, 2)]
def mstat(X, decoy=False):
    """(n,6,3): for each target q and pair H, number of q' != q where all four sources equal (decoy: first sources equal)."""
    a = X.reshape(len(X), 6, 3, 2); out = np.zeros((len(X), 6, 3))
    for j, (g, h) in enumerate(PAIRS):
        eq = (a[:, :, g, 0] == a[:, :, h, 0]) if decoy else ((a[:, :, g, 0] == a[:, :, g, 1]) & (a[:, :, h, 0] == a[:, :, h, 1]) & (a[:, :, g, 0] == a[:, :, h, 0]))
        out[:, :, j] = eq.sum(1, keepdims=True) - eq
    return out.astype(np.float32)
def resid2(X):
    f = jax.jit(lambda X_: (lambda r: (F.forward(M, X_, rec=r), r["resid2"])[1])({}))
    return np.concatenate([np.asarray(f(jnp.asarray(X[i:i + 1000]))).astype(np.float16) for i in range(0, len(X), 1000)])
rng = np.random.default_rng(ST0); dp = GL.sample(E2, 40000, rng); Xp, Yp = dp["X"], dp["Y"]; Zp = resid2(Xp); Mp, Mdp = mstat(Xp), mstat(Xp, True)
rng_e = np.random.default_rng(ST0 + 500_000); de = GL.sample(E2, 3000, rng_e); Xe, Ye = de["X"], de["Y"]
P_true = np.stack([XB.posterior_shared(E2, Xe, np.full(len(Xe), qq)) for qq in range(6)], 1)
# ---- J-pairs (XIII-B) and decoy pairs
rj = np.random.default_rng(ST0 + 700_000); dj = GL.sample(E2, 2000, rj); Xj = dj["X"]; qj = rj.integers(0, 6, len(Xj)); hj = rj.integers(0, 3, len(Xj))
perms = [p for p in itertools.permutations(range(3)) if p != (0, 1, 2)]; Xj2 = Xj.copy()
for i in range(len(Xj)):
    pm = np.array(perms[rj.integers(len(perms))]); cols = [2 * hj[i], 2 * hj[i] + 1]; others = [t for t in range(6) if t != qj[i]]; Xj2[i][np.ix_(others, cols)] = pm[Xj[i][np.ix_(others, cols)]]
dBj = RP.cen(XB.posterior_shared(E2, Xj2, qj)) - RP.cen(XB.posterior_shared(E2, Xj, qj)); okj = np.isfinite(dBj).all(1)
ij_ = np.arange(len(Xj)); dm = np.abs(mstat(Xj2)[ij_, qj] - mstat(Xj)[ij_, qj]).sum(-1) > 0
rd = np.random.default_rng(ST0 + 800_000); dd = GL.sample(E2, 4000, rd); Xd = dd["X"]; qd = rd.integers(0, 6, len(Xd)); Xd2 = Xd.copy(); keepd = np.zeros(len(Xd), bool)
for i in range(len(Xd)):
    a = Xd[i].reshape(6, 3, 2); cands = [(p, g) for p in range(6) if p != qd[i] for g in range(3) if a[p, g, 0] != a[p, g, 1]]
    if not cands: continue
    p, g = cands[rd.integers(len(cands))]; s_ = rd.integers(2); v_old = a[p, g, s_]; other = a[p, g, 1 - s_]
    v_new = [v for v in range(3) if v != v_old and v != other][0]; x2 = Xd[i].copy(); x2[p, 2 * g + s_] = v_new
    if (mstat(x2[None])[0] != mstat(Xd[i][None])[0]).any(): continue
    if (mstat(x2[None], True)[0, qd[i]] == mstat(Xd[i][None], True)[0, qd[i]]).all(): continue
    Xd2[i] = x2; keepd[i] = True
Xd, Xd2, qd = Xd[keepd][:1200], Xd2[keepd][:1200], qd[keepd][:1200]
dBd = RP.cen(XB.posterior_shared(E2, Xd2, qd)) - RP.cen(XB.posterior_shared(E2, Xd, qd))
PAIRSETS = dict(J_DM=(Xj[okj & dm], Xj2[okj & dm], qj[okj & dm], dBj[okj & dm]), J_D0=(Xj[okj & ~dm], Xj2[okj & ~dm], qj[okj & ~dm], dBj[okj & ~dm]), DECOY=(Xd, Xd2, qd, dBd))
info = {k: dict(n=len(v[2]), exact_mean_abs=float(np.abs(v[3]).mean())) for k, v in PAIRSETS.items()}
# ---- model pieces
ATT = ("g1", "b1", "WQ", "WK", "WV", "WO"); MLPK = ("g2", "b2", "W1", "c1", "W2", "c2")
def attn_only(lay, x):
    z = F.N.ln(x, lay["g1"], lay["b1"]); qh = jnp.einsum("ntd,hde->nhte", z, lay["WQ"]); kh = jnp.einsum("ntd,hde->nhte", z, lay["WK"]); vh = jnp.einsum("ntd,hde->nhte", z, lay["WV"])
    a = jax.nn.softmax(jnp.einsum("nhte,nhse->nhts", qh, kh) / jnp.sqrt(qh.shape[-1]), -1); return x + jnp.einsum("nhte,hed->ntd", jnp.einsum("nhts,nhse->nhte", a, vh), lay["WO"])
COMP = np.array([2, 1, 0]); WITH = np.array([[0, 1], [0, 2], [1, 2]])       # pair index NOT containing g; pairs containing g
def group_feats(m):
    """m (n,6,3) pair statistics -> per-group relative features (n,6,3 groups,3)."""
    return np.stack([np.stack([m[..., COMP[g]], m[..., WITH[g][0]], m[..., WITH[g][1]]], -1) for g in range(3)], 2).astype(np.float32)
def supply_add_g(t, x, mg):
    v = (mg / 5.0) @ t["E"]                                                   # n,6(q),3(g),64
    src = jnp.repeat(v, 2, axis=2).reshape(x.shape[0], 36, 64)               # token q*6 + s, s//2 = g
    return x + jnp.concatenate([src, jnp.zeros((x.shape[0], 7, 64))], 1)
def supply_add(t, x, m):
    """m (n,6,3) -> add (m/5) @ E at readout token 36+q and at the six source tokens of q."""
    v = (m / 5.0) @ t["E"]                                                    # n,6,64
    add = jnp.concatenate([jnp.repeat(v, 6, axis=1), v, jnp.zeros((x.shape[0], 1, 64))], 1)   # tokens: 36 source (q*6+s) + 6 readout + episode
    return x + add
def head(t, z, arm, m=None):
    x = z
    if arm == "preattn": x = attn_only(t["pre"], x)
    if arm == "pretok": x = x + (jax.nn.gelu(F.N.ln(x, t["pre"]["g"], t["pre"]["b"]) @ t["pre"]["W1"] + t["pre"]["c1"], approximate=True) @ t["pre"]["W2"])
    if arm in ("supG", "supGFrozen"): x = supply_add_g(t, x, m)
    elif arm.startswith("sup"): x = supply_add(t, x, m)
    x = F.block_heads(t["l2"], x, None, 2, None, None)
    if arm == "L2plus": x = F.block_heads(t["l3"], x, None, 3, None, None)
    return F.N.ln(x, t["gf"], t["bf"])[:, 36:42] @ t["WA"]
def full_logits(t, X_):
    Mx = dict(M); Mx["layers"] = [t["l0"], t["l1"], t["l2"]]; Mx["gf"], Mx["bf"], Mx["WA"] = t["gf"], t["bf"], t["WA"]; return F.forward(Mx, X_)
EARLY = ("early01", "L1only", "L0only")
def init_for(arm):
    t = dict(l0=M["layers"][0], l1=M["layers"][1], l2=M["layers"][2], gf=M["gf"], bf=M["bf"], WA=M["WA"])
    k = jax.random.PRNGKey(4040)
    if arm == "L2plus":
        b = F.N.init("M", F.NL.ENV, 5151)["layers"][2]; b = dict(b); b["WO"] = jnp.zeros_like(b["WO"]); b["W2"] = jnp.zeros_like(b["W2"]); b["c2"] = jnp.zeros_like(b["c2"]); t["l3"] = b
    if arm == "preattn":
        b = dict(F.N.init("M", F.NL.ENV, 5252)["layers"][2]); t["pre"] = {kk: b[kk] for kk in ATT}; t["pre"]["WO"] = jnp.zeros_like(b["WO"])
    if arm == "pretok": t["pre"] = dict(g=jnp.ones(64), b=jnp.zeros(64), W1=jax.random.normal(k, (64, 128)) / 8.0, c1=jnp.zeros(128), W2=jnp.zeros((128, 64)))
    if arm.startswith("sup"): t["E"] = jnp.zeros((3, 64))
    return t
def labels_for(arm, t):
    lab = jax.tree_util.tree_map(lambda a: "freeze", t)
    def setall(k): lab[k] = jax.tree_util.tree_map(lambda a: "train", t[k])
    if arm in ("L2", "supL2", "supDecoy"): [setall(k) for k in ("l2", "gf", "bf", "WA")]
    if arm in ("supL2", "supDecoy", "supFrozen", "supG", "supGFrozen"): setall("E")
    if arm == "supG": [setall(k) for k in ("l2", "gf", "bf", "WA")]
    if arm == "early01": setall("l0"); setall("l1")
    if arm == "L1only": setall("l1")
    if arm == "L0only": setall("l0")
    if arm == "L2plus": [setall(k) for k in ("l2", "l3", "gf", "bf", "WA")]
    if arm in ("preattn", "pretok"): setall("pre")
    return lab
def ntrain(t, lab): return int(sum(np.prod(x.shape) for x, m in zip(jax.tree_util.tree_leaves(t), jax.tree_util.tree_leaves(lab)) if m == "train"))
def logits_fn(t, arm):
    if arm in EARLY: f = jax.jit(lambda X_: full_logits(t, X_)); return lambda X_: np.concatenate([np.asarray(f(jnp.asarray(X_[i:i + 500]))) for i in range(0, len(X_), 500)])
    f = jax.jit(lambda z, m: head(t, z, arm, m)); dec = arm == "supDecoy"; gf_ = arm in ("supG", "supGFrozen")
    return lambda X_: np.concatenate([np.asarray(f(jnp.asarray(Z_[i:i + 500], jnp.float32), jnp.asarray(mm[i:i + 500]))) for Z_, mm in [(resid2(X_), group_feats(mstat(X_)) if gf_ else mstat(X_, dec))] for i in range(0, len(X_), 500)])
def evaluate(t, arm):
    lf = logits_fn(t, arm); P = RP.sm(lf(Xe)); hit = P.argmax(-1) == Ye; kl = (P_true * (np.log(P_true + 1e-12) - np.log(P + 1e-12))).sum(-1)
    r = dict(acc=float(hit.mean()), KL=float(np.nanmean(kl)))
    for nm, (Xa, Xb, qq, dB) in PAIRSETS.items():
        i = np.arange(len(qq)); dO = RP.cen(RP.sm(lf(Xb)[i, qq])) - RP.cen(RP.sm(lf(Xa)[i, qq]))
        r[nm] = RP.fid(dO, dB) if nm == "J_DM" else dict(mean_abs_obs=float(np.abs(dO).mean()), mean_abs_exact=float(np.abs(dB).mean()))
    r["J_D0_over_DM_response"] = r["J_D0"]["mean_abs_obs"] / max(r["J_DM"]["mean_abs_obs"], 1e-9); r["DECOY_over_DM_response"] = r["DECOY"]["mean_abs_obs"] / max(r["J_DM"]["mean_abs_obs"], 1e-9)
    return r
def train(arm):
    t = init_for(arm); lab = labels_for(arm, t); t_init = t
    opt = optax.multi_transform({"train": optax.adamw(optax.warmup_cosine_decay_schedule(0.0, LR, STEPS // 10, STEPS), weight_decay=1e-4), "freeze": optax.set_to_zero()}, lab); st = opt.init(t)
    if arm in EARLY:
        def lossf(tt, a, m, Y): lp = jax.nn.log_softmax(full_logits(tt, a), -1); return (-jnp.take_along_axis(lp, Y[..., None], -1)[..., 0]).mean()
    else:
        def lossf(tt, a, m, Y): lp = jax.nn.log_softmax(head(tt, a, arm, m), -1); return (-jnp.take_along_axis(lp, Y[..., None], -1)[..., 0]).mean()
    @jax.jit
    def step(tt, st, a, m, Y): l, g = jax.value_and_grad(lossf)(tt, a, m, Y); up, st = opt.update(g, st, tt); return optax.apply_updates(tt, up), st, l
    r = np.random.default_rng(5); l = None; Ms = Mdp if arm == "supDecoy" else (group_feats(Mp) if arm in ("supG", "supGFrozen") else Mp)
    for i in range(STEPS):
        j = r.integers(0, len(Xp), 256); a = jnp.asarray(Xp[j]) if arm in EARLY else jnp.asarray(Zp[j], jnp.float32)
        t, st, l = step(t, st, a, jnp.asarray(Ms[j]), jnp.asarray(Yp[j]))
    unchanged = all(bool(jnp.all(x == y)) for x, y, m in zip(jax.tree_util.tree_leaves(t_init), jax.tree_util.tree_leaves(t), jax.tree_util.tree_leaves(lab)) if m == "freeze")
    return t, float(l), unchanged, ntrain(t, lab)
outp = os.path.join(OUT, f"{UN}_b{BETA}_lr{LR:g}.json"); res = json.load(open(outp)) if os.path.exists(outp) else {}
if "native" not in res:
    base = init_for("L2"); res.update(model=M["tag"], beta=BETA, steps=STEPS, lr=LR, pairs=info, bayes_acc=float((P_true.argmax(-1) == Ye).mean()), native=evaluate(base, "L2"))
    json.dump(res, open(outp, "w"), indent=1)
print(M["tag"][:30], "beta", BETA, "lr", LR, "bayes", round(res["bayes_acc"], 3), "native acc", round(res["native"]["acc"], 3), "pairs", info, flush=True)
for arm in ARMS:
    if arm in res: continue
    t, fl, unch, nt = train(arm); r = evaluate(t, arm); r.update(final_loss=fl, frozen_params_unchanged=unch, n_trainable=nt, sec=time.time() - t0); res[arm] = r
    np.savez(os.path.join(CK, f"{UN}_{arm}_b{BETA}_lr{LR:g}.npz"), **{"/".join(str(getattr(k, 'key', getattr(k, 'idx', k))) for k in path): np.asarray(v) for path, v in jax.tree_util.tree_flatten_with_path(t)[0]})
    json.dump(res, open(outp, "w"), indent=1)
    print(f"  {arm:9s} acc {r['acc']:.3f} KL {r['KL']:.4f} J_DM slope {r['J_DM']['slope']:+.3f} R2 {r['J_DM']['R2']:+.2f} D0/DM {r['J_D0_over_DM_response']:.2f} DEC/DM {r['DECOY_over_DM_response']:.2f} n_train {nt} {round(time.time() - t0)}s", flush=True)
