"""Frozen-feature (Condition Z) transfer variants and learned repairs.  Usage: repair_train.py PHASE PARENT SPEC [STEPS]
PHASE dev  : PARENT = xP80 / xT70 ... (Investigation VI checkpoints), eval = Investigation VII eval_dev.npz
PHASE conf : PARENT = zP100 / zT1110 ... (Stage-0 lineages, pre.npz), eval = logs/eval_conf8.npz
Everything matches Investigation VII's original transfer (fresh layer 2 from nets.init(seed+7777), inherited W_A and final LN, data stream
1_000_003*(seed+1)+7919*4+93, AdamW warmup-cosine, 4000 steps, generator-label CE) except the SPEC:
  orig      original transfer (C1)
  injm      + Investigation VII m-injection (hand-computed eligibility; oracle REFERENCE)
  lca       + learned conjunction adapter: MLP (45 -> 128 -> 128 -> 64, GELU, last layer zero-init) on the raw relational ingredients of
              question q (one-hot counts, one-hot q-reports, group id, for the three groups), added to the readout-token input of layer 2.
              No m, no combined cross-group flag.  (Repair family A)
  radapt    + learned relational adapter: the group-attention reader (readers.attn, d=40, 1 block) on the same ingredients, projected
              (zero-init) to 64 and added at the readout token.  (Repair family A, attention-structured)
  gadapt    + generic adapter control: same MLP as lca, input = 45 i.i.d. N(0,1) features fixed per (episode, q) by hashing X
              (matched parameters, no task information)
  deep2     downstream = two fresh Transformer blocks (more depth; family B)
  unfreeze1 layer 1 also trainable (lr as layer 2), fresh layer 2 (family B: selective unfreezing)
  lora1     rank-8 LoRA on all layer-1 weight matrices (zero-init B) + fresh layer 2 (family B)
  auxm      original architecture + auxiliary head (linear, from the final-LN readout state) predicting each group's m_g (3x3-class CE,
              weight 0.5) during TRAINING only; inference uses no m (family C: auxiliary relational supervision)
  bayes     original architecture with exact-Bayes soft targets (family C)
Writes checkpoints/PHASE/<parent>_<spec>[_sN].npz/.json (evaluations at 1000 and STEPS)."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp, optax
import v8lib as V, readers as RD
sys.path.insert(0, V.V7); import nnlib as NL, evalrun as ER, ckpt as C, r1lib as L
N = NL.N
PHASE, parent, spec = sys.argv[1], sys.argv[2], sys.argv[3]; STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 4000
seed = int(parent.lstrip("xzPT")[-3:] if parent.startswith("z") else parent[2:])
if parent.startswith("zT1"): seed = int(parent[3:])
OUT = os.path.join(HERE, "..", "checkpoints", PHASE); os.makedirs(OUT, exist_ok=True); tag = f"{parent}_{spec}" + (f"_s{STEPS}" if STEPS != 4000 else "")
if PHASE == "dev":
    ev = dict(np.load(os.path.join(V.V7, "..", "logs", "eval_dev.npz"), allow_pickle=True)); pf = NL.load(NL.dev_ckpt(parent, 0))
else:
    ev = dict(np.load(os.path.join(HERE, "..", "logs", "eval_conf8.npz"), allow_pickle=True)); pf = NL.load(os.path.join(HERE, "..", "checkpoints", parent, "pre.npz"))
fresh = N.init("M", NL.ENV, seed + 7777); stream = 1_000_003 * (seed + 1) + 7919 * 4 + 31 * 3
def ingr(X):
    f = RD.features(X)                                   # N,6,3,15
    return f.reshape(*f.shape[:2], -1)                   # N,6,45
def ingr_rand(X):
    h = (np.asarray(X).reshape(len(X), -1) * np.arange(1, 37)).sum(1) * 2654435761 % (2 ** 31)
    return np.stack([np.random.default_rng(int(s)).standard_normal((6, 45)) for s in h]).astype(np.float32)
def inj_m(X):
    m = V.eligibility_inv7(np.asarray(X)); return np.eye(3)[np.minimum(m, 2)].reshape(*m.shape[:2], 9).astype(np.float32)
XF = {"injm": inj_m, "lca": ingr, "gadapt": ingr_rand, "radapt": lambda X: RD.features(X)}.get(spec)
t = dict(l2=fresh["layers"][2], gf=pf["gf"], bf=pf["bf"], WA=pf["WA"])
if spec == "injm": t["Wi"] = jnp.zeros((9, 64))
if spec in ("lca", "gadapt"):
    t["ad"] = RD.mlp_init(jax.random.PRNGKey(seed + 31), [45, 128, 128, 64]); t["ad"][-1]["W"] = jnp.zeros((128, 64))
if spec == "radapt":
    t["ra"] = RD.init("attn", 15, seed + 32, d=40); t["ra"]["out"] = dict(W=jnp.zeros((40, 64)), b=jnp.zeros(64))
if spec == "deep2": t["l3"] = N.init("M", NL.ENV, seed + 8888)["layers"][2]
if spec == "unfreeze1": t["l1"] = pf["layers"][1]
if spec == "lora1":
    k = jax.random.split(jax.random.PRNGKey(seed + 41), 16); lay = pf["layers"][1]; t["lora"] = {}
    for i, (nm, W) in enumerate(sorted((n_, w) for n_, w in lay.items() if w.ndim >= 2)):
        shp = W.shape; din, dout = int(np.prod(shp[:-1])), shp[-1]
        t["lora"][nm] = dict(A=jax.random.normal(k[i], (din, 8)) * din ** -0.5, B=jnp.zeros((8, dout)))
if spec == "auxm": t["aux"] = dict(W=jnp.zeros((64, 9)), b=jnp.zeros(9))
def layer1_params(tt):
    if spec == "unfreeze1": return tt["l1"]
    if spec == "lora1":
        lay = dict(pf["layers"][1])
        for nm, ab in tt["lora"].items():
            W = lay[nm]; lay[nm] = W + (ab["A"] @ ab["B"]).reshape(W.shape)
        return lay
    return pf["layers"][1]
def zfun(tt, X):
    if spec in ("unfreeze1", "lora1"):
        pz = jax.lax.stop_gradient(pf); x = N.embed(pz, X); x = NL.block(pz["layers"][0], x, l=0); return NL.block(layer1_params(tt), x, l=1)
    return NL.frozen_z(pf, X)
def head(tt, X, Fx):
    z = zfun(tt, X)
    if spec == "injm": z = z.at[:, 36:42].add(Fx @ tt["Wi"])
    if spec in ("lca", "gadapt"): z = z.at[:, 36:42].add(RD.mlp_apply(tt["ad"], Fx))
    if spec == "radapt":
        ra = tt["ra"]; x = Fx @ ra["emb"]["W"] + ra["emb"]["b"]
        for b in ra["blocks"]:
            zz = RD.lnorm(x); a = jax.nn.softmax(jnp.einsum("...gd,...hd->...gh", zz @ b["Wq"], zz @ b["Wk"]) / jnp.sqrt(40.), -1)
            x = x + jnp.einsum("...gh,...hd->...gd", a, zz @ b["Wv"]) @ b["Wo"]; x = x + RD.mlp_apply(b["ff"], RD.lnorm(x))
        z = z.at[:, 36:42].add(x.mean(-2) @ ra["out"]["W"] + ra["out"]["b"])
    x = NL.block(tt["l2"], z, l=2)
    if spec == "deep2": x = NL.block(tt["l3"], x, l=3)
    zf = N.ln(x, tt["gf"], tt["bf"])[:, 36:42]; return zf @ tt["WA"], zf
target = "bayes" if spec == "bayes" else "label"
opt = optax.adamw(optax.warmup_cosine_decay_schedule(0.0, 1e-3, min(200, STEPS // 10), STEPS), weight_decay=1e-4); st = opt.init(t)
@jax.jit
def step(tt, st, X, Fx, Y, P, M):
    def loss(tt):
        lg, zf = head(tt, X, Fx); lp = jax.nn.log_softmax(lg, -1)
        l = (-jnp.take_along_axis(lp, Y[..., None], -1)[..., 0]).mean() if target == "label" else -(P * lp).sum(-1).mean()
        if spec == "auxm":
            la = jax.nn.log_softmax((zf @ tt["aux"]["W"] + tt["aux"]["b"]).reshape(*zf.shape[:2], 3, 3), -1)
            l = l + 0.5 * (-jnp.take_along_axis(la, M[..., None], -1)[..., 0]).mean()
        return l
    l, g = jax.value_and_grad(loss)(tt); up, st = opt.update(g, st, tt); return optax.apply_updates(tt, up), st, l
fwd = jax.jit(lambda tt, X, Fx: head(tt, X, Fx)[0])
def fn_of(tt):
    def f(X):
        out = []
        for i in range(0, len(X), 2000):
            Xi = X[i:i + 2000]; Fx = jnp.asarray(XF(Xi)) if XF else jnp.zeros((len(Xi), 1)); out.append(np.asarray(fwd(tt, jnp.asarray(Xi), Fx)))
        return np.concatenate(out)
    return f
rng = np.random.default_rng(stream); t0 = time.time(); hist = []; evals = {}
for i in range(STEPS):
    d = NL.Bm.sample_fast(NL.ENV, 256, rng); T = NL.targets(d); X = d["X"]
    Fx = jnp.asarray(XF(X)) if XF else jnp.zeros((256, 1)); M = jnp.asarray(np.minimum(V.eligibility_inv7(X), 2))
    t, st, l = step(t, st, jnp.asarray(X), Fx, T["Y"], T["P"], M)
    if (i + 1) % max(STEPS // 20, 1) == 0: hist.append((i + 1, float(l)))
    if (i + 1) in (1000, STEPS):
        r = ER.run(fn_of(t), ev); evals[i + 1] = r; h = ER.headline(r)
        print(f"[{tag}] step {i + 1} " + " ".join(f"{k} {v:+.3f}" for k, v in h.items()), flush=True)
npar = int(sum(np.prod(x.shape) for k_, v_ in t.items() if k_ not in ("l2", "gf", "bf", "WA") for x in jax.tree_util.tree_leaves(v_)))
V.save_tree(os.path.join(OUT, tag + ".npz"), jax.tree_util.tree_map(np.asarray, t))
json.dump(dict(phase=PHASE, parent=parent, spec=spec, steps=STEPS, extra_params=npar, seconds=time.time() - t0, hist=hist, evals=evals), open(os.path.join(OUT, tag + ".json"), "w"), indent=1, default=float)
