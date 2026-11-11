"""Development downstream experiments on the frozen layers 0-1 of an Investigation VI parent.
Usage: downstream.py PARENT SPEC REP [STEPS]      PARENT e.g. xP80 / xT70; REP = replicate index (0 = Investigation VI's own stream).
Factorial (labels, generator CE, exactly the original transfer objective; matched fresh draw and data stream within a REP):
  C1 inherited WA, inherited final LN (gf, bf), fresh layer 2          (original setup)
  C2 fresh WA, inherited LN, fresh layer 2                              (= mandate conditions 2 and 4: identical in this code)
  C3 inherited WA, fresh LN (g = 1, b = 0), fresh layer 2
  C5 fresh WA, fresh LN, fresh layer 2                                   (fully fresh downstream stack)
Sufficiency readers (exact-Bayes soft targets; training-time supervision only; test-time input is the episode only):
  Asoft (= C5 architecture), Bsoft (two fresh blocks), Csoft (MLP on full frozen state), Ereadsoft (readout token only),
  Esrcsoft (source tokens only)
Readout-site variant (suffix _rtemb, any spec): the six readout-token inputs to layer 2 are replaced by their embedding-level states
  (Eread + Eq, before the frozen layers), so the downstream stack can obtain episode information only by attending to source tokens.
Learning-dynamics (labels; per-decision loss weights from generator-derived decision classes, training stream only):
  W_r1up   decisions in the two-perfect-agreeing class x4 (normalised to mean weight 1 per batch)
  W_bal    classes {one perfect group, two perfect agreeing, two perfect disagreeing} equally weighted in expectation
  (C1 is the matched natural-sampling control; equal steps)
Fresh draws: REP 0 -> nets.init(seed + 7777) (Investigation VI); REP r -> nets.init(seed + 7777 + 1000 r).  Data stream: REP 0 ->
Investigation VI's transfer stream; REP r -> + 104729 r.  Evaluations at 500/1000/2000/4000 on logs/eval_dev.npz."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import nnlib as NL, r1lib as L, evalrun as ER, ckpt as C
N = NL.N
parent, spec, rep = sys.argv[1], sys.argv[2], int(sys.argv[3]); STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 4000
seed = int(parent[2:]); OUT = os.path.join(HERE, "..", "checkpoints", "dev"); os.makedirs(OUT, exist_ok=True); tag = f"{parent}_{spec}_r{rep}" + (f"_s{STEPS}" if STEPS != 4000 else "")
ev = dict(np.load(os.path.join(HERE, "..", "logs", "eval_dev.npz"), allow_pickle=True))
pf = NL.load(NL.dev_ckpt(parent, 0)); fresh = N.init("M", NL.ENV, seed + 7777 + 1000 * rep)
stream = 1_000_003 * (seed + 1) + 7919 * 4 + 31 * 3 + 104729 * rep
snaps = tuple(sorted(set([s for s in (250, 500, 1000, 2000, 4000, 8000, 12000) if s <= STEPS] + [STEPS]))); evals = {}
def decision_class(d):
    X = d["X"]; n = L.E.loo(X); V = np.stack([L.votes(X, qq) for qq in range(6)], 1); perf = (n == 5) & (V >= 0); H = perf.sum(-1)
    agree2 = np.zeros(H.shape, bool)
    for a, b in ((0, 1), (0, 2), (1, 2)): agree2 |= perf[..., a] & perf[..., b] & (V[..., a] == V[..., b])
    return np.where(H >= 2, np.where(agree2, 1, 2), 0)          # 0: one perfect group, 1: two perfect agreeing, 2: two perfect disagreeing
PREV = np.array([0.916, 0.028, 0.056])                           # natural prevalence (logs/r1_information.json)
def weight_fn(d):
    c = decision_class(d)
    if spec == "W_r1up": w = np.where(c == 1, 4.0, 1.0)
    else: w = (1 / 3) / PREV[c]
    return w / w.mean()
RTEMB = spec.endswith("_rtemb")          # readout-token input to layer 2 replaced by its embedding-level state (Eread + Eq)
INJ = spec.split("_inj")[1] if "_inj" in spec else None   # feature injection at the readout tokens: 'm' | 'n' | 'rand'
base_spec = spec[:-6] if RTEMB else (spec.split("_inj")[0] if INJ else spec)
def inj_feats(X):
    """per-question features added (through a trainable linear map, zero-initialised) to the readout-token input of layer 2.
    m    : eligibility m_g of each group (one-hot 3x3) -- the cross-group relational statistic of Proposition 1 (computed from X only)
    n    : leave-q counts of each group (one-hot 3x6) -- information already present in the frozen state (control)
    rand : 9 fixed pseudo-random binary features of the episode (hash of X; capacity-matched irrelevant control)"""
    X = np.asarray(X); n = L.E.loo(X)
    if INJ == "n": return np.eye(6)[n].reshape(*n.shape[:2], 18).astype(np.float32)
    V = np.stack([L.votes(X, qq) for qq in range(6)], 1); perf = (n == 5) & (V >= 0); m = perf.sum(-1, keepdims=True) - perf
    if INJ == "m": return np.eye(3)[np.minimum(m, 2)].reshape(*n.shape[:2], 9).astype(np.float32)
    h = (X.reshape(len(X), -1) * np.arange(1, 37)).sum(1)[:, None, None] * np.arange(1, 7)[None, :, None] * np.arange(1, 10)[None, None, :]
    return ((h * 2654435761) % 7 < 3).astype(np.float32)
def zfun(X, Fx=None, Wi=None):
    z = NL.frozen_z(pf, X)
    if RTEMB: z = z.at[:, 36:42].set(jax.lax.stop_gradient(N.embed(pf, X)[:, 36:42]))
    if INJ: z = z.at[:, 36:42].add(Fx @ Wi)
    return z
if base_spec in ("C1", "C2", "C3", "C5", "W_r1up", "W_bal"):
    p = dict(pf); p["layers"] = list(pf["layers"]); p["layers"][2] = fresh["layers"][2]; p["WR"] = fresh["WR"]
    if base_spec in ("C2", "C5"): p["WA"] = fresh["WA"]
    if base_spec in ("C3", "C5"): p["gf"], p["bf"] = jnp.ones(64), jnp.zeros(64)
    train_keys = ["layers2", "gf", "bf", "WA"]
    def split(p): return dict(l2=p["layers"][2], gf=p["gf"], bf=p["bf"], WA=p["WA"])
    def merge(t):
        q = dict(pf); q["layers"] = [pf["layers"][0], pf["layers"][1], t["l2"]]; q["gf"], q["bf"], q["WA"] = t["gf"], t["bf"], t["WA"]; return q
    def logit_fn(t, X):
        Xa, Fx = (X if INJ else (X, None))
        z = zfun(Xa, Fx, t.get("Wi")); x = NL.block(t["l2"], z); zf = N.ln(x, t["gf"], t["bf"]); return zf[:, 36:42] @ t["WA"]
    params = split(p); target = "label"; wf = weight_fn if base_spec.startswith("W_") else None
    if INJ: params["Wi"] = jnp.zeros(({"m": 9, "n": 18, "rand": 9}[INJ], 64))
    def fnmaker(t):
        if INJ:
            f = jax.jit(lambda X, F: logit_fn(t, (X, F)))
            return lambda X: np.concatenate([np.asarray(f(jnp.asarray(X[i:i + 2000]), jnp.asarray(inj_feats(X[i:i + 2000])))) for i in range(0, len(X), 2000)])
        f = jax.jit(lambda X: logit_fn(t, X)); return lambda X: np.concatenate([np.asarray(f(jnp.asarray(X[i:i + 2000]))) for i in range(0, len(X), 2000)])
else:
    kind = {"Asoft": "A", "Bsoft": "B", "Csoft": "C", "Ereadsoft": "Eread", "Esrcsoft": "Esrc"}[base_spec]
    params = NL.reader_init(kind, seed + 7777 + 1000 * rep); target = "bayes"; wf = None
    if kind in ("A",): params["blocks"] = [fresh["layers"][2]]; params["WA"] = fresh["WA"]
    logit_fn = lambda t, X: NL.reader_logits(kind, t, zfun(X))
    def fnmaker(t):
        f = jax.jit(lambda X: logit_fn(t, X)); return lambda X: np.concatenate([np.asarray(f(jnp.asarray(X[i:i + 2000]))) for i in range(0, len(X), 2000)])
def on_snap(i, t):
    r = ER.run(fnmaker(t), ev); evals[i] = r; h = ER.headline(r)
    print(f"[{tag}] step {i} " + " ".join(f"{k} {v:+.3f}" for k, v in h.items()), flush=True)
t0 = time.time()
if INJ:
    lf0 = logit_fn
    def logit_fn(t, X): return lf0(t, X)
params, hist = NL.train(logit_fn, params, STEPS, stream, xfeat=inj_feats if INJ else None, target=target, weight_fn=wf, snaps=snaps if base_spec[0] in "CW" else (STEPS,), on_snap=on_snap)
C.save(os.path.join(OUT, tag + ".npz"), jax.tree_util.tree_map(np.asarray, params))
json.dump(dict(parent=parent, spec=spec, rep=rep, steps=STEPS, seconds=time.time() - t0, hist=hist, evals=evals), open(os.path.join(OUT, tag + ".json"), "w"), indent=1, default=float)
