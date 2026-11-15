"""Workstream B / C3: native learning inside the parent Transformer (input = raw tokens X only; no side channels at train or inference).
Usage: native_train.py PARENT SPEC [STEPS=4000] [SEED=0] [TARGET=label|bayes] [EVAL=dev|<path>]
SPEC (all start from the pretrained parent; the readout map WA is fresh unless stated):
  c1         original transfer: layers 0-1 frozen, fresh layer 2 (Investigation VI-VIII protocol)          [pool-based: frozen prefix]
  keep3      all 3 pretrained layers frozen (layer 2 KEPT), fresh layer 3 on top                             [pool-based]
  keep3x2    all 3 pretrained layers frozen, two fresh layers 3-4                                             [pool-based]
  deep2      layers 0-1 frozen, two fresh layers 2-3 (Investigation VIII deep2)                               [pool-based]
  full_c1    all layers trainable: pretrained 0-1, fresh layer 2 (full fine-tuning, original architecture)    [streaming]
  full_keep3 all layers trainable: pretrained 0-2 + fresh layer 3                                             [streaming]
  unf1       layer 1 trainable + fresh layer 2 (layer 0, embeddings frozen)                                  [streaming]
  unf0       layer 0 trainable + fresh layer 2 (layer 1 frozen)                                              [streaming]
  lora_attn1 / lora_mlp1 / lora_both1 : rank-8 LoRA on layer-1 attention (WQ,WK,WV,WO) / MLP (W1,W2) / both, + fresh layer 2   [streaming]
  scratch3   3-layer Transformer from random init, all trainable (raw-token from-scratch control)            [streaming]
Optimiser: AdamW, warmup-cosine to 1e-3 (matched across specs), batch 256, generator labels (or exact-Bayes soft targets).
Checkpoints + compact evaluation (half of each eval family) at log-spaced steps for learning dynamics; full eval at the end.
Writes checkpoints/native_train/<tag>/{step<k>.npz, curve.json, final.json}."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp, optax
import native as NV
NL, N = NV.NL, NV.N
parent, spec = sys.argv[1], sys.argv[2]; STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 4000; seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
target = sys.argv[5] if len(sys.argv) > 5 else "label"; evname = sys.argv[6] if len(sys.argv) > 6 else "dev"
tag = f"{parent}_{spec}_s{STEPS}_seed{seed}" + ("_bayes" if target == "bayes" else "") + ("" if evname == "dev" else "_" + os.path.basename(evname).split(".")[0])
OUT = os.path.join(HERE, "..", "checkpoints", "native_train", tag); os.makedirs(OUT, exist_ok=True)
pf = NV.load_parent(parent) if spec != "scratch3" else None
fresh = lambda s: N.init("M", NL.ENV, s)
POOL = spec in ("c1", "keep3", "keep3x2", "deep2"); UPTO = 3 if spec.startswith("keep3") else 2
base = pf if pf is not None else fresh(seed + 4242)
t = dict(gf=base["gf"], bf=base["bf"], WA=fresh(seed + 7777)["WA"])     # final LN inherited (as in the original protocol); fresh readout map
if spec in ("c1", "deep2", "full_c1", "unf1", "unf0", "lora_attn1", "lora_mlp1", "lora_both1"): t["new"] = [fresh(seed + 7777)["layers"][2]]
if spec == "deep2": t["new"].append(fresh(seed + 8888)["layers"][2])
if spec in ("keep3", "full_keep3"): t["new"] = [fresh(seed + 7777)["layers"][2]]
if spec == "keep3x2": t["new"] = [fresh(seed + 7777)["layers"][2], fresh(seed + 8888)["layers"][2]]
if spec in ("full_c1", "full_keep3", "scratch3"):
    t["emb"] = {k: base[k] for k in NV.NL.FROZEN_KEYS}; t["pre"] = [base["layers"][l] for l in range(3 if spec != "full_c1" else 2)]
    if spec == "scratch3": t["new"] = []
if spec == "unf1": t["l1"] = pf["layers"][1]
if spec == "unf0": t["l0"] = pf["layers"][0]
if spec.startswith("lora"):
    names = {"lora_attn1": ("WQ", "WK", "WV", "WO"), "lora_mlp1": ("W1", "W2"), "lora_both1": ("WQ", "WK", "WV", "WO", "W1", "W2")}[spec]
    ks = jax.random.split(jax.random.PRNGKey(seed + 41), 8); t["lora"] = {}
    for i, nm in enumerate(names):
        W = pf["layers"][1][nm]; din, dout = int(np.prod(W.shape[:-1])), W.shape[-1]
        t["lora"][nm] = dict(A=jax.random.normal(ks[i], (din, 8)) * din ** -0.5, B=jnp.zeros((8, dout)))
def prefix(tt, X):
    """Native computation up to the fresh layers (everything from the raw tokens)."""
    if spec in ("full_c1", "full_keep3", "scratch3"):
        p_ = dict(tt["emb"]); x = N.embed(p_, X)
        for l, lay in enumerate(tt["pre"]): x = NL.block(lay, x, l=l)
        return x
    ps = jax.lax.stop_gradient(pf); x = N.embed(ps, X)
    l0 = tt["l0"] if spec == "unf0" else ps["layers"][0]; x = NL.block(l0, x, l=0)
    if spec == "unf1": l1 = tt["l1"]
    elif spec.startswith("lora"):
        l1 = dict(ps["layers"][1])
        for nm, ab in tt["lora"].items(): W = l1[nm]; l1[nm] = W + (ab["A"] @ ab["B"]).reshape(W.shape)
    else: l1 = ps["layers"][1]
    return NL.block(l1, x, l=1)
def head(tt, x):
    for i, lay in enumerate(tt["new"]): x = NL.block(lay, x, l=10 + i)
    return N.ln(x, tt["gf"], tt["bf"])[:, 36:42] @ tt["WA"]
def logits_X(tt, X):
    if POOL: return head(tt, NV.states(jax.lax.stop_gradient(pf), X, UPTO))
    return head(tt, prefix(tt, X))
opt = optax.adamw(optax.warmup_cosine_decay_schedule(0.0, 1e-3, min(200, STEPS // 10), STEPS), weight_decay=1e-4); st = opt.init(t)
def lossf(tt, inp, Y, P):
    lg = head(tt, inp) if POOL else logits_X(tt, inp); lp = jax.nn.log_softmax(lg, -1)
    return (-jnp.take_along_axis(lp, Y[..., None], -1)[..., 0]).mean() if target == "label" else -(P * lp).sum(-1).mean()
@jax.jit
def step(tt, st, inp, Y, P):
    l, g = jax.value_and_grad(lossf)(tt, inp, Y, P); up, st = opt.update(g, st, tt); return optax.apply_updates(tt, up), st, l
fwd = jax.jit(logits_X)
def fn_of(tt):
    return lambda X: np.concatenate([np.asarray(fwd(tt, jnp.asarray(X[i:i + 1000]))) for i in range(0, len(X), 1000)])
ev = NV.load_eval() if evname == "dev" else NV.load_eval(evname)
n6 = len(ev["X"]); n7 = len(ev["X_O"]); evc = {k: (v[: n6 // 3] if len(v) == n6 else v[: n7 // 2] if len(v) == n7 else v) for k, v in ev.items()}
rng = np.random.default_rng(1_000_003 * (seed + 1) + 7919 * 4 + 31 * 3 + (0 if spec == "c1" else 17))
t0 = time.time(); curve = []; hist = []
if POOL:                                                       # frozen prefix: precompute a large pool of native states once
    NPOOL = 150_000; dpool = NV.Bm.sample_fast(NV.ENV, NPOOL, rng); Zpool = NV.states_np(pf, dpool["X"], UPTO, dtype=np.float16)
    Tpool = NL.targets(dpool) if target == "bayes" else None; print(f"[{tag}] pool {Zpool.shape} {time.time() - t0:.0f}s", flush=True)
snaps = sorted(set([int(round(x)) for x in np.geomspace(50, STEPS, 12)] + [STEPS]))
for i in range(STEPS):
    if POOL:
        e = rng.integers(0, len(Zpool), 256); inp = jnp.asarray(Zpool[e], jnp.float32); Y = jnp.asarray(dpool["Y"][e])
        P = Tpool["P"][e] if Tpool is not None else jnp.zeros((256, 6, 3))
    else:
        d = NL.Bm.sample_fast(NL.ENV, 256, rng); inp = jnp.asarray(d["X"]); Y = jnp.asarray(d["Y"])
        P = NL.targets(d)["P"] if target == "bayes" else jnp.zeros((256, 6, 3))
    t, st, l = step(t, st, inp, Y, P)
    if (i + 1) % max(STEPS // 40, 1) == 0: hist.append((i + 1, float(l)))
    if (i + 1) in snaps:
        r = NV.ER.run(fn_of(t), evc if i + 1 < STEPS else ev); s = NV.summarize(r); s["step"] = i + 1; s["seconds"] = time.time() - t0; curve.append(s)
        leaves = jax.tree_util.tree_leaves(t); np.savez(os.path.join(OUT, f"step{i + 1}.npz"), *[np.asarray(x) for x in leaves])
        print(f"[{tag}] step {i + 1} " + " ".join(f"{k} {v:+.3f}" for k, v in s.items() if k not in ("step", "seconds")), flush=True)
        if i + 1 == STEPS: json.dump(dict(summary=s, families=r["families"], quality=r["quality"]), open(os.path.join(OUT, "final.json"), "w"), indent=1, default=float)
json.dump(dict(parent=parent, spec=spec, steps=STEPS, seed=seed, target=target, seconds=time.time() - t0, curve=curve, hist=hist, snaps=snaps,
               trainable=int(sum(np.prod(x.shape) for k_, v_ in t.items() for x in jax.tree_util.tree_leaves(v_)))), open(os.path.join(OUT, "curve.json"), "w"), indent=1, default=float)
