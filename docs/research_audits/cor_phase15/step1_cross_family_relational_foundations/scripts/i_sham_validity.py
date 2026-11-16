"""XIV Stage I (frozen): validity of the off-route shams on the six baseline models at step 1211 (confirmation stream 95_300_000).
A sham is VALID on a base model if applying its mask (i) raises KL(Bayes||model) by <= 0.02 and (ii) does not reduce the own-question
readout attention mass of either route (layer 0, and max-head layer 2) by more than 0.01.  The real block's effect is reported alongside.
Writes logs/i_sham_validity.json."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, xc, metrics as MT, maskfwd as MF
S = MT.sets(95_300_000); X, qn, PB = S["Xn"], S["qn"], S["PBn"]
own = np.zeros((6, 43), bool)
for q in range(6): own[q, q * 6:(q + 1) * 6] = True
def ev(M, kinds):
    biases = {}
    for k in kinds: l, m = MF.edges(k); biases[l] = biases.get(l, np.zeros((43, 43), bool)) | m
    rec = {}; L = np.concatenate([np.asarray(MF.forward(M, X[i:i + 1000], biases, rec=rec))[np.arange(len(X[i:i + 1000])), qn[i:i + 1000]] for i in range(0, len(X), 1000)])
    P = np.exp(L - L.max(-1, keepdims=True)); P /= P.sum(-1, keepdims=True)
    A0 = np.asarray(rec["pattern0"])[:, :, 36:42]; A2 = np.asarray(rec["pattern2"])[:, :, 36:42]      # last batch (1000 episodes)
    return dict(KL=float((PB * (np.log(PB + 1e-12) - np.log(P + 1e-12))).sum(-1).mean()), own_L0=float((A0 * own).sum(-1).mean()), own_L2=float((A2 * own).sum(-1).mean((0, 2)).max()))
res = {}
for b, lw, sham, real, Ds in (("E", 21, "oq0", "ro0", (21, 801, 802)), ("M", 23, "rd2", "own2", (23, 801, 802))):
    for D in Ds:
        spec = f"uP160:fx:{lw}:{lw}:{D}:" + ("base" if D in (21, 23) else "stop1211_B_base"); M = xc.load(spec, 1211)
        r0, rs, rr = ev(M, ()), ev(M, (sham,)), ev(M, (real,))
        valid = (rs["KL"] - r0["KL"] <= 0.02) and (rs["own_L0"] >= r0["own_L0"] - 0.01) and (rs["own_L2"] >= r0["own_L2"] - 0.01)
        res[f"{b}:D{D}"] = dict(none=r0, sham=rs, real=rr, sham_valid=bool(valid)); print(b, D, valid, {k: {a: round(c, 3) for a, c in v.items()} for k, v in (("none", r0), ("sham", rs), ("real", rr))}, flush=True)
res["all_valid"] = all(v["sham_valid"] for k, v in res.items() if isinstance(v, dict))
json.dump(res, open(os.path.join(HERE, "..", "logs", "i_sham_validity.json"), "w"), indent=1)
