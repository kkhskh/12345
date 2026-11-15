"""XIII-C Workstream A1/A2: full_c1 fine-tuning with the randomness sources SEPARATED.

Audit of XIII-B scripts/native_train.py (spec full_c1, target label) -- every use of the fine-tuning seed `s`:
  (1) fresh layer-2 block      : N.init("M", ENV, s + 7777)["layers"][2]
  (2) fresh readout map WA      : N.init("M", ENV, s + 7777)["WA"]           (same init call as (1); separable by calling init with two seeds)
  (3) training data stream      : np.random.default_rng(1_000_003*(s+1) + 7919*4 + 31*3 + 17) -> NL.Bm.sample_fast(ENV, 256, rng) per step
                                  (one rng draws episodes AND their order; there is no separate shuffling -- each step is a fresh sample)
  other randomness: none in the training loop (no dropout, no data augmentation; AdamW deterministic; final LN gf/bf inherited from parent;
  embeddings and layers 0-1 inherited from parent).  Evaluation (ER.run) uses fixed files and no rng.  Floating-point summation order
  is fixed by XLA_FLAGS (single-threaded Eigen, intra_op 1, as in XIII-B lane.sh) -- verified by bitwise reproduction (repro check below).

Usage: ft_factorial.py PARENT LSEED WSEED DSEED [STEPS=4000]
  LSEED: seed of the fresh layer-2 init; WSEED: seed of the fresh WA init; DSEED: seed of the data stream.
  L = W = D = s reproduces XIII-B native_train.py full_c1 seed s exactly (checked against its saved checkpoints).
Optional training-time intervention (Workstream A4/R2):  FX_INTERV="<name>:<T>"  applies an attention mask for steps < T:
  ro0  : layer 0, readout tokens (36..41) may not attend to source tokens (0..35)   [suppresses early candidate readout]
  ro01 : same mask in layers 0 and 1                                               [suppresses early readout + layer-1 readout pooling]
  rq1  : layer 1, readout tokens may not attend to any source token                 [suppresses layer-1 relational pooling into readout]
  l2q  : layer 2 (fresh), readout token 36+q may not attend to the 6 source tokens of its own q   [suppresses mixture-head formation]
Writes checkpoints/ft/<tag>/{step<k>.npz, curve.json, final.json}; dense log-spaced checkpoints for trajectories.
RESUMABLE (added after the 07:23Z container restart killed the first three runs): at every checkpoint the AdamW state and the numpy
bit-generator state are written to resume.pkl; a restarted run continues from the last checkpoint and is bitwise identical to an
uninterrupted run (checked in logs/repro_resume.json).  A run whose final.json exists exits immediately (idempotent)."""
import json, os, sys, time, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp, optax
import native as NV
NL, N = NV.NL, NV.N
parent = sys.argv[1]; LS, WS, DS = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]); STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 4000
INTERV = os.environ.get("FX_INTERV", ""); iname, iT = (INTERV.split(":") + ["0"])[:2] if INTERV else ("", "0"); iT = int(iT)
STOP = int(os.environ.get("FX_STOP", 0))                                     # repro check: stop after STOP steps (LR schedule unchanged)
tag = f"{parent}_fx_L{LS}_W{WS}_D{DS}" + (f"_s{STEPS}" if STEPS != 4000 else "") + (f"_{iname}T{iT}" if INTERV else "") + (f"_stop{STOP}" if STOP else "")
OUT = os.path.join(HERE, "..", "checkpoints", "ft", tag); os.makedirs(OUT, exist_ok=True)
pf = NV.load_parent(parent); fresh = lambda s: N.init("M", NL.ENV, s)
t = dict(gf=pf["gf"], bf=pf["bf"], WA=fresh(WS + 7777)["WA"], new=[fresh(LS + 7777)["layers"][2]])
t["emb"] = {k: pf[k] for k in NL.FROZEN_KEYS}; t["pre"] = [pf["layers"][l] for l in range(2)]
NEG = -1e9
def mask_for(layer):
    m = np.zeros((43, 43), np.float32)
    if iname in ("ro0", "ro01") and (layer == 0 or (iname == "ro01" and layer == 1)): m[36:42, 0:36] = NEG
    if iname == "rq1" and layer == 1: m[36:42, 0:36] = NEG
    if iname == "l2q" and layer == 2:
        for qq in range(6): m[36 + qq, qq * 6:(qq + 1) * 6] = NEG
    return jnp.asarray(m) if m.any() else None
def block_m(lay, x, bias):
    z = N.ln(x, lay["g1"], lay["b1"])
    q = jnp.einsum("ntd,hde->nhte", z, lay["WQ"]); k = jnp.einsum("ntd,hde->nhte", z, lay["WK"]); v = jnp.einsum("ntd,hde->nhte", z, lay["WV"])
    s = jnp.einsum("nhte,nhse->nhts", q, k) / jnp.sqrt(q.shape[-1])
    if bias is not None: s = s + bias
    a = jax.nn.softmax(s, -1); x = x + jnp.einsum("nhte,hed->ntd", jnp.einsum("nhts,nhse->nhte", a, v), lay["WO"])
    h = jax.nn.gelu(N.ln(x, lay["g2"], lay["b2"]) @ lay["W1"] + lay["c1"], approximate=True); return x + h @ lay["W2"] + lay["c2"]
def logits_X(tt, X, masked=False):
    x = N.embed(dict(tt["emb"]), X)
    for l, lay in enumerate(tt["pre"]):
        bias = mask_for(l) if masked else None
        x = NL.block(lay, x, l=l) if bias is None else block_m(lay, x, bias)
    for i, lay in enumerate(tt["new"]):
        bias = mask_for(2) if masked else None
        x = NL.block(lay, x, l=10 + i) if bias is None else block_m(lay, x, bias)
    return N.ln(x, tt["gf"], tt["bf"])[:, 36:42] @ tt["WA"]
opt = optax.adamw(optax.warmup_cosine_decay_schedule(0.0, 1e-3, min(200, STEPS // 10), STEPS), weight_decay=1e-4); st = opt.init(t)
def lossf(tt, X, Y, masked):
    lp = jax.nn.log_softmax(logits_X(tt, X, masked), -1); return (-jnp.take_along_axis(lp, Y[..., None], -1)[..., 0]).mean()
def mkstep(masked):
    @jax.jit
    def step(tt, st, X, Y):
        l, g = jax.value_and_grad(lambda a: lossf(a, X, Y, masked))(tt); up, st = opt.update(g, st, tt); return optax.apply_updates(tt, up), st, l
    return step
step_plain, step_mask = mkstep(False), (mkstep(True) if INTERV else None)
fwd = jax.jit(logits_X)
def fn_of(tt): return lambda X: np.concatenate([np.asarray(fwd(tt, jnp.asarray(X[i:i + 1000]))) for i in range(0, len(X), 1000)])
rng = np.random.default_rng(1_000_003 * (DS + 1) + 7919 * 4 + 31 * 3 + 17)                  # identical to native_train.py full_c1 with seed DS
snaps = sorted(set([int(round(x)) for x in np.geomspace(50, STEPS, 12)] + [25, 100, 150, 200, 300, 400, 600, 800, 1000, 1500, 2000, 3000, STEPS]))
snaps = [s for s in snaps if s <= STEPS]; t0 = time.time(); hist = []
if os.path.exists(os.path.join(OUT, "final.json")) and not STOP: print(f"[{tag}] already complete", flush=True); sys.exit(0)
start = 0; RP_ = os.path.join(OUT, "resume.pkl")
if os.path.exists(RP_) and not STOP:
    R = pickle.load(open(RP_, "rb")); start = R["step"]; rng.bit_generator.state = R["rng"]; hist = R["hist"]
    z = np.load(os.path.join(OUT, f"step{start}.npz")); t = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(t), [jnp.asarray(z[f"arr_{k}"]) for k in range(len(z.files))])
    st = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(st), [jnp.asarray(a) for a in R["opt"]]); print(f"[{tag}] resumed at step {start}", flush=True)
for i in range(start, STEPS):
    d = NL.Bm.sample_fast(NL.ENV, 256, rng); X = jnp.asarray(d["X"]); Y = jnp.asarray(d["Y"])
    t, st, l = (step_mask if (INTERV and i < iT) else step_plain)(t, st, X, Y)
    if (i + 1) % max(STEPS // 40, 1) == 0: hist.append((i + 1, float(l)))
    if (i + 1) in snaps:
        np.savez(os.path.join(OUT, f"step{i + 1}.npz"), *[np.asarray(x) for x in jax.tree_util.tree_leaves(t)])
        pickle.dump(dict(step=i + 1, rng=rng.bit_generator.state, hist=hist, opt=[np.asarray(a) for a in jax.tree_util.tree_leaves(st)]), open(RP_ + ".tmp", "wb")); os.replace(RP_ + ".tmp", RP_)
        print(f"[{tag}] step {i + 1} loss {float(l):.4f} {time.time() - t0:.0f}s", flush=True)
    if STOP and i + 1 >= STOP: sys.exit(0)
    if (i + 1) in snaps and int(os.environ.get("FX_DIE_AFTER", 0)) == i + 1: sys.exit(3)       # resume test only: simulate a crash
ev = NV.load_eval(os.path.join(NV.X13B, "logs", "eval_conf13.npz")); r = NV.ER.run(fn_of(t), ev); s = NV.summarize(r)
json.dump(dict(summary=s, families=r["families"], quality=r["quality"]), open(os.path.join(OUT, "final.json"), "w"), indent=1, default=float)
json.dump(dict(parent=parent, LSEED=LS, WSEED=WS, DSEED=DS, steps=STEPS, interv=INTERV, seconds=time.time() - t0, hist=hist, snaps=snaps, final=s),
          open(os.path.join(OUT, "curve.json"), "w"), indent=1, default=float)
print(f"[{tag}] done acc {s['acc']:.4f} KL {s['KL']:.4f} {time.time() - t0:.0f}s", flush=True)
