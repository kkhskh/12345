"""XIV Stage I development: immediate effect of candidate masks on trained bases (does a candidate sham interrupt either route?).
For each base (E, M) and step (546, 1211): KL(Bayes||model), Bayes-argmax agreement, own-question readout attention mass in L0 (early
route) and max-head own mass in L2 (mixture route), under: none, real blocks (ro0, l2q=own2), candidate shams (oq0, oq2, rd0, rd2).
Natural episodes from metrics.sets(97_100_000) (development stream).  Writes logs/i_sham_dev.json."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import xc, metrics as MT, maskfwd as MF
S = MT.sets(97_100_000); X, qn, PB = S["Xn"][:1500], S["qn"][:1500], S["PBn"][:1500]
def evalm(M, kinds):
    biases = {}
    for k in kinds:
        l, m = MF.edges(k); biases[l] = biases.get(l, np.zeros((43, 43), bool)) | m
    rec = {}; L = np.asarray(MF.forward(M, X, biases, rec=rec))[np.arange(len(X)), qn]
    P = np.exp(L - L.max(-1, keepdims=True)); P /= P.sum(-1, keepdims=True)
    kl = float((PB * (np.log(PB + 1e-12) - np.log(P + 1e-12))).sum(-1).mean()); acc = float((P.argmax(-1) == PB.argmax(-1)).mean())
    own = np.zeros((6, 43), bool)
    for q in range(6): own[q, q * 6:(q + 1) * 6] = True
    A0 = np.asarray(rec["pattern0"])[:, :, 36:42]; A2 = np.asarray(rec["pattern2"])[:, :, 36:42]
    return dict(KL=kl, acc=acc, own_L0=float((A0 * own).sum(-1).mean()), own_L2_maxhead=float((A2 * own).sum(-1).mean((0, 2)).max()))
res = {}
for name, spec in (("E", "uP160:fx:21:21:21:base"), ("M", "uP160:fx:23:23:23:base")):
    for k in (546, 1211):
        M = xc.load(spec, k)
        for kinds in ((), ("ro0",), ("own2",), ("oq0",), ("oq2",), ("rd0",), ("rd2",)):
            r = evalm(M, kinds); res[f"{name}@{k}|{'+'.join(kinds) or 'none'}"] = r
            print(name, k, kinds or "none", {a: round(b, 3) for a, b in r.items()}, flush=True)
json.dump(res, open(os.path.join(HERE, "..", "logs", "i_sham_dev.json"), "w"), indent=1)
