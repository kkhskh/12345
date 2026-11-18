"""XVIII development: test-time route necessity on step-1211 checkpoints (attention-statistic-free causal diagnostic).
KL(Bayes||model) on natural episodes with no block, with the E-route block (layer 0 readout -> all sources, 'ro0') and with the M-route block
(layer 2 readout q -> own-q sources, 'own2').  Usage: dev_necessity.py OUT STREAM SPEC..."""
import json, os, sys
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "..", "step1_cross_family_relational_foundations", "src"))
import numpy as np, jax.numpy as jnp
import xc, maskfwd as MFW, native as NV, genlib as GL, relpairs as RP
out, stream = sys.argv[1], int(sys.argv[2]); rng = np.random.default_rng(stream)
X = NV.Bm.sample_fast(NV.ENV, 1500, rng)["X"]; q = rng.integers(0, 6, 1500); PB = GL.posterior_closed(RP.ENV1, X, q); i = np.arange(1500)
def kl(L): P = RP.sm(np.asarray(L)[i, q]); return float((PB * (np.log(PB + 1e-12) - np.log(P + 1e-12))).sum(-1).mean())
for sp in sys.argv[3:]:
    M = xc.load(sp, 1211); r = dict(unit=sp)
    for nm, kind in (("none", None), ("ro0", "ro0"), ("own2", "own2"), ("own0", "own0")):
        b = None if kind is None else dict([MFW.edges(kind)]); r[nm] = kl(MFW.forward(M, X, biases=b))
    r["nec_E"] = r["ro0"] - r["none"]; r["nec_M"] = r["own2"] - r["none"]; open(out, "a").write(json.dumps(r) + "\n")
    print(sp, {k: round(v, 3) for k, v in r.items() if k != "unit"}, flush=True)
