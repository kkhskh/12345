"""XIII-C Workstream A4/A5: initialisation-time and very-early features of a fine-tuning run (exploratory predictor search, DEVELOPMENT).
For a (parent, L2-init seed, WA-init seed) the step-0 model is fully determined (layers 0-1 pretrained, fresh layer 2 = N.init(L+7777), fresh WA =
N.init(W+7777)).  Features at step 0, on natural episodes (stream 86_500_000, 500 episodes, random q):
  m0_max   : max over the 4 fresh layer-2 heads of the readout-token (36+q) attention mass on the 6 q source tokens
  m0_all   : the 4 per-head masses (sorted)
  g_ratio  : at step 0, |grad| of the label loss w.r.t. layer-2 (WQ,WK) relative to |grad| w.r.t. layer-0 (WQ,WK)  (which route learns faster)
  g_mass   : first-order change of m0_max under one SGD step (directional derivative of the mass along -grad, on the head with max mass)
Usage: init_features.py PARENT:L:W ...  -> appends logs/init_features<TAG>.jsonl"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import native as NV, ftlib as F
NL, N = NV.NL, NV.N
rng = np.random.default_rng(86_500_000); d = NV.Bm.sample_fast(NV.ENV, 500, rng); X = jnp.asarray(d["X"]); Y = jnp.asarray(d["Y"]); q = rng.integers(0, 6, 500)
i = np.arange(500); qs = q[:, None] * 6 + np.arange(6)[None]
out = os.path.join(HERE, "..", "logs", "init_features" + os.environ.get("IF_TAG", "") + ".jsonl")
for sp in sys.argv[1:]:
    parent, L, W = sp.split(":"); L, W = int(L), int(W); pf = NV.load_parent(parent); fresh = lambda s: N.init("M", NL.ENV, s)
    M = dict(emb={k: pf[k] for k in NL.FROZEN_KEYS}, layers=[pf["layers"][0], pf["layers"][1], fresh(L + 7777)["layers"][2]], gf=pf["gf"], bf=pf["bf"], WA=fresh(W + 7777)["WA"])
    rec = {}; F.forward(M, X, rec=rec); A = np.asarray(rec["pattern2"]); mass = np.array([np.take_along_axis(A[i, h, 36 + q], qs, 1).sum(1).mean() for h in range(4)])
    def loss(layers):
        Mx = dict(M); Mx["layers"] = layers; lp = jax.nn.log_softmax(F.forward(Mx, X), -1); return (-jnp.take_along_axis(lp, Y[..., None], -1)[..., 0]).mean()
    g = jax.grad(loss)(M["layers"])
    nrm = lambda l: float(np.sqrt(sum(float((g[l][k] ** 2).sum()) for k in ("WQ", "WK"))))
    hm = int(np.argmax(mass))
    def mass_of(layers):
        Mx = dict(M); Mx["layers"] = layers; r = {}; F.forward(Mx, X, rec=r); A_ = r["pattern2"]; return jnp.take_along_axis(A_[i, hm, 36 + q], jnp.asarray(qs), 1).sum(1).mean()
    _, dm = jax.jvp(mass_of, (M["layers"],), (jax.tree_util.tree_map(lambda a: -a, g),))
    r = dict(unit=sp, parent=parent, L=L, W=W, m0_max=float(mass.max()), m0_all=sorted(map(float, mass)), g_ratio=nrm(2) / max(nrm(0), 1e-12), g_mass=float(dm))
    with open(out, "a") as fh: fh.write(json.dumps(r) + "\n")
    print(r, flush=True)
