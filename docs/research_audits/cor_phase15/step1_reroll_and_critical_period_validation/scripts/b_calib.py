"""XIII-E Workstream B development: immediate disruption (loss increase on a fixed development batch) caused by each intervention, measured on
the XIII-D base checkpoints (uP160 21/21/21 early base, uP160 23/23/23 mixture base) at steps 0, 25, 50, 100 and 546.
Interventions: ro0, l2q (route masks), sham (masked code path with zero bias), noise sigma (Gaussian noise added to the residual stream at
the layer-2 input), ro0p k (k of the 6 sources of every question masked for the layer-0 readout rows).  Development stream 95 800 000.
Output logs/b_calib.json and the frozen matching (sigma / k by log-linear interpolation on the grid):
  short DIST_E: sigma with the same mean delta over steps {25, 50, 100} as ro0 on the early base; DIST_M: same as l2q on the mixture base;
  late NOISE_LATE_E: sigma matching ro0 at 546 on the early base; NOISE_LATE_M: sigma on the mixture base matching that same delta (severity of
  the late early-route block); RO0P_LATE: k on the early base whose delta at 546 is closest to l2q's delta at 546 on the mixture base."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import native as NV, xc
NL, N = NV.NL, NV.N; NEG = -1e9
def mask(name, layer, k=0):
    m = np.zeros((43, 43), np.float32)
    if name == "ro0" and layer == 0: m[36:42, 0:36] = NEG
    if name == "l2q" and layer == 2:
        for qq in range(6): m[36 + qq, qq * 6:(qq + 1) * 6] = NEG
    if name == "ro0p" and layer == 0:
        for qq in range(6): m[36:42, qq * 6:qq * 6 + k] = NEG
    if name == "sham" and layer in (0, 2): return jnp.zeros((43, 43), jnp.float32)
    return jnp.asarray(m) if m.any() else None
def block_m(lay, x, bias):
    z = N.ln(x, lay["g1"], lay["b1"])
    q = jnp.einsum("ntd,hde->nhte", z, lay["WQ"]); k = jnp.einsum("ntd,hde->nhte", z, lay["WK"]); v = jnp.einsum("ntd,hde->nhte", z, lay["WV"])
    s = jnp.einsum("nhte,nhse->nhts", q, k) / jnp.sqrt(q.shape[-1]) + bias
    a = jax.nn.softmax(s, -1); x = x + jnp.einsum("nhte,hed->ntd", jnp.einsum("nhts,nhse->nhte", a, v), lay["WO"])
    h = jax.nn.gelu(N.ln(x, lay["g2"], lay["b2"]) @ lay["W1"] + lay["c1"], approximate=True); return x + h @ lay["W2"] + lay["c2"]
def loss(M, X, Y, name=None, k=0, sigma=0.0):
    x = N.embed(dict(M["emb"]), X)
    for l, lay in enumerate(M["layers"]):
        if l == 2 and name == "noise": x = x + sigma * jax.random.normal(jax.random.PRNGKey(7), x.shape)
        b = mask(name, l, k) if name not in (None, "noise") else None
        x = NL.block(lay, x, l=l if l < 2 else 10) if b is None else block_m(lay, x, b)
    lp = jax.nn.log_softmax(N.ln(x, M["gf"], M["bf"])[:, 36:42] @ M["WA"], -1); return float(-jnp.take_along_axis(lp, Y[..., None], -1).mean())
d = NL.Bm.sample_fast(NL.ENV, 2048, np.random.default_rng(95_800_000)); X, Y = jnp.asarray(d["X"]), jnp.asarray(d["Y"])
def init_model(parent, s):
    pf = NV.load_parent(parent); fr = lambda z: N.init("M", NL.ENV, z)
    return dict(emb={k: pf[k] for k in NL.FROZEN_KEYS}, layers=[pf["layers"][0], pf["layers"][1], fr(s + 7777)["layers"][2]], gf=pf["gf"], bf=pf["bf"], WA=fr(s + 7777)["WA"])
SIG = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]; R = {}
for base, spec, s in (("early", "uP160:fx:21:21:21:base", 21), ("mixture", "uP160:fx:23:23:23:base", 23)):
    R[base] = {}
    for st in (0, 25, 50, 100, 546):
        M = init_model("uP160", s) if st == 0 else xc.load(spec, st); b0 = loss(M, X, Y)
        r = dict(base_loss=b0, ro0=loss(M, X, Y, "ro0") - b0, l2q=loss(M, X, Y, "l2q") - b0, sham=loss(M, X, Y, "sham") - b0,
                 noise={g: loss(M, X, Y, "noise", sigma=g) - b0 for g in SIG}, ro0p={k: loss(M, X, Y, "ro0p", k=k) - b0 for k in range(1, 6)})
        R[base][st] = r; print(base, st, {k: (round(v, 4) if isinstance(v, float) else {a: round(b, 3) for a, b in v.items()}) for k, v in r.items()}, flush=True)
def match_sigma(base, steps, target):
    curve = np.array([np.mean([R[base][s]["noise"][g] for s in steps]) for g in SIG])
    if target <= curve[0]: return SIG[0]
    if target >= curve[-1]: return SIG[-1]
    i = int(np.searchsorted(curve, target)); f = (target - curve[i - 1]) / (curve[i] - curve[i - 1]); return float(np.exp(np.log(SIG[i - 1]) + f * (np.log(SIG[i]) - np.log(SIG[i - 1]))))
tE = np.mean([R["early"][s]["ro0"] for s in (25, 50, 100)]); tM = np.mean([R["mixture"][s]["l2q"] for s in (25, 50, 100)])
tLE = R["early"][546]["ro0"]; tLM = R["mixture"][546]["l2q"]
match = dict(DIST_E=dict(target=tE, sigma=match_sigma("early", (25, 50, 100), tE)), DIST_M=dict(target=tM, sigma=match_sigma("mixture", (25, 50, 100), tM)),
             NOISE_LATE_E=dict(target=tLE, sigma=match_sigma("early", (546,), tLE)), NOISE_LATE_M=dict(target=tLE, sigma=match_sigma("mixture", (546,), tLE)),
             RO0P_LATE=dict(target=tLM, k=int(min(range(1, 6), key=lambda k: abs(R["early"][546]["ro0p"][k] - tLM))), achieved={k: R["early"][546]["ro0p"][k] for k in range(1, 6)}))
json.dump(dict(R=R, match=match), open(os.path.join(HERE, "..", "logs", "b_calib.json"), "w"), indent=1, default=float); print(json.dumps(match, indent=1, default=float))
