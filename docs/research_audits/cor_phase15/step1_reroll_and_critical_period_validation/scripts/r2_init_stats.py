"""XIII-E A7/R2 development: theory-guided JOINT (parent, L, W) statistics measured at step 0, before any training.
Rationale: the fresh layer 2 + readout start untrained; which route the optimiser exploits first should depend on how much each route can
already move the readout logits at initialisation, and on how strongly the initial gradient points along each route.
  I_mix   = mean KL(f0 || f0 with the mixture-route mask l2q)      (initial influence of layer-2 readout->own-q-source attention)
  I_early = mean KL(f0 || f0 with the early-route mask ro0)        (initial influence of layer-0 readout->source attention)
  R_route = log(I_early / I_mix)
  G_ratio = ||grad_WA L(masked l2q)|| / ||grad_WA L(masked ro0)||   (initial readout-gradient magnitude available without each route)
Episodes: 1024 from stream 95 900 000 (development stream, never used for outcomes).  Usage: r2_init_stats.py OUT.jsonl CONFIG...
CONFIG = parent:L:W  (D is irrelevant at step 0)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import native as NV
NL, N = NV.NL, NV.N; NEG = -1e9
def mask(name, layer):
    m = np.zeros((43, 43), np.float32)
    if name == "ro0" and layer == 0: m[36:42, 0:36] = NEG
    if name == "l2q" and layer == 2:
        for qq in range(6): m[36 + qq, qq * 6:(qq + 1) * 6] = NEG
    return jnp.asarray(m) if m.any() else None
def block_m(lay, x, bias):
    z = N.ln(x, lay["g1"], lay["b1"])
    q = jnp.einsum("ntd,hde->nhte", z, lay["WQ"]); k = jnp.einsum("ntd,hde->nhte", z, lay["WK"]); v = jnp.einsum("ntd,hde->nhte", z, lay["WV"])
    s = jnp.einsum("nhte,nhse->nhts", q, k) / jnp.sqrt(q.shape[-1]) + bias
    a = jax.nn.softmax(s, -1); x = x + jnp.einsum("nhte,hed->ntd", jnp.einsum("nhts,nhse->nhte", a, v), lay["WO"])
    h = jax.nn.gelu(N.ln(x, lay["g2"], lay["b2"]) @ lay["W1"] + lay["c1"], approximate=True); return x + h @ lay["W2"] + lay["c2"]
def logits(t, X, name=None):
    x = N.embed(dict(t["emb"]), X)
    for l, lay in enumerate(t["pre"] + t["new"]):
        b = mask(name, l) if name else None
        x = (NL.block(lay, x, l=l if l < 2 else 10) if b is None else block_m(lay, x, b))
    return N.ln(x, t["gf"], t["bf"])[:, 36:42] @ t["WA"]
d = NL.Bm.sample_fast(NL.ENV, 1024, np.random.default_rng(95_900_000)); X, Y = jnp.asarray(d["X"]), jnp.asarray(d["Y"])
def kl(a, b): pa = jax.nn.log_softmax(a, -1); pb = jax.nn.log_softmax(b, -1); return float((jnp.exp(pa) * (pa - pb)).sum(-1).mean())
out = sys.argv[1]; done = set()
if os.path.exists(out): done = {json.loads(l)["config"] for l in open(out)}
for cf in sys.argv[2:]:
    if cf in done: continue
    p, L, W = cf.split(":"); pf = NV.load_parent(p); fr = lambda s: N.init("M", NL.ENV, s)
    t = dict(gf=pf["gf"], bf=pf["bf"], WA=fr(int(W) + 7777)["WA"], new=[fr(int(L) + 7777)["layers"][2]], emb={k: pf[k] for k in NL.FROZEN_KEYS}, pre=[pf["layers"][l] for l in range(2)])
    f0 = logits(t, X); fm = logits(t, X, "l2q"); fe = logits(t, X, "ro0")
    def gWA(name):
        g = jax.grad(lambda wa: (-jnp.take_along_axis(jax.nn.log_softmax(logits(dict(t, WA=wa), X, name), -1), Y[..., None], -1)).mean())(t["WA"]); return float(jnp.linalg.norm(g))
    r = dict(config=cf, I_mix=kl(f0, fm), I_early=kl(f0, fe), G_noL2q=gWA("l2q"), G_noRo0=gWA("ro0"), G_full=gWA(None))
    r["R_route"] = float(np.log(r["I_early"] / r["I_mix"])); r["G_ratio"] = r["G_noL2q"] / r["G_noRo0"]
    open(out, "a").write(json.dumps(r) + "\n"); print(r, flush=True)
