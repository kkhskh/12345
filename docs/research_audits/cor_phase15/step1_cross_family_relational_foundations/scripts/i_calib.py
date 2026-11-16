"""XIV Stage I calibration (development): severity of the real block, the off-route shams and location-matched logit noise.
Severity measure (frozen in FREEZE_I): mean total-variation distance between the perturbed and unperturbed attention rows of the readout
queries in the intervened layer (all heads, readouts, episodes), on the base checkpoints at steps 25, 50 and 100 (the 100-step window).
Also reported: immediate loss change (mean NLL of the label under the perturbed forward minus unperturbed).
Noise: sigma * N(0,1) added to the attention logits on the real block's edge set (E: layer-0 readout->all sources; M: layer-2 readout->own
sources).  Writes logs/i_calib.json."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import xc, native as NV, maskfwd as MF
d = NV.Bm.sample_fast(NV.ENV, 512, np.random.default_rng(97_200_000)); X, Y = d["X"], d["Y"]
def run(M, layer, biases=None, noise=None):
    rec = {}; L = np.asarray(MF.forward(M, X, biases, noise, rec)); lp = L - np.log(np.exp(L - L.max(-1, keepdims=True)).sum(-1, keepdims=True)) - L.max(-1, keepdims=True)
    return np.asarray(rec[f"pattern{layer}"])[:, :, 36:42], float(-np.take_along_axis(lp, Y[..., None], -1).mean())
res = {}
for name, spec, real in (("E", "uP160:fx:21:21:21:base", "ro0"), ("M", "uP160:fx:23:23:23:base", "own2")):
    sham = "oq0" if name == "E" else "rd2"; lay, em = MF.edges(real); ls, sm = MF.edges(sham)
    for k in (25, 50, 100):
        M = xc.load(spec, k); A0, l0 = run(M, lay); out = {}
        A, l = run(M, lay, {lay: em}); out["real"] = dict(TV=float(0.5 * np.abs(A - A0).sum(-1).mean()), dloss=l - l0)
        A0s, _ = run(M, ls); A, l = run(M, ls, {ls: sm}); out["sham"] = dict(TV=float(0.5 * np.abs(A - A0s).sum(-1).mean()), dloss=l - l0)
        for sig in (0.5, 1, 2, 3, 4, 6, 8, 12, 16):
            tv, dl = [], []
            for r in range(3):
                A, l = run(M, lay, None, (lay, sig, em, jax.random.PRNGKey(1000 + r))); tv.append(0.5 * np.abs(A - A0).sum(-1).mean()); dl.append(l - l0)
            out[f"noise{sig}"] = dict(TV=float(np.mean(tv)), dloss=float(np.mean(dl)))
        res[f"{name}@{k}"] = out; print(name, k, {a: (round(b["TV"], 3), round(b["dloss"], 3)) for a, b in out.items()}, flush=True)
json.dump(res, open(os.path.join(HERE, "..", "logs", "i_calib.json"), "w"), indent=1)
