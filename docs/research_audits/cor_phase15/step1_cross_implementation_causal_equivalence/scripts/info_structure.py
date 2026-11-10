"""Natural conditional dependence of the site representations on generator count variables (mandate §5.3).
Usage: info_structure.py PHASE (cal | conf).  Fit on the first 3000 episodes of the fit stream, R² reported on 2000
held-out episodes (stream FIT + 1).  All feature sets of one comparison have equal dimensionality (capacity-matched).
Per source token t = (k, s) (both model kinds), target a[t] at the site (d = 64):
  own        onehot(n^{(-k)}_{g(s)})                                      6
  own+other  onehot(n^{(-k)}) (all three groups)                          18
  own+null   own + onehot of the two cross-pair counts (CROSS pairs)      18   (matched-capacity null)
  own+shuf   own + other groups' one-hot from a shuffled episode          18   (matched-capacity null)
Per readout token 36+q (both model kinds):
  additive   onehot(n^{(-q)})                                             18
  +pairwise  additive + onehot of the three (n_g, n_g') pairs             18+108
  +nullpair  additive + onehot of the three cross-pair-count pairs         18+108 (matched-capacity null)
plus decodability of n^{(-q)} (multinomial logistic, held-out) at the readout and at question q's source tokens for each
model kind, and the fraction of site variance explained by the additive count code.  Writes logs/PHASE/info_structure.json."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
import eqv as E

PH = sys.argv[1]
SETS = {"cal": (["P0", "P1", "T0", "T1"], 12_100_000), "conf": (["P2", "P3", "P4", "T2", "T10", "T11"], 12_300_000),
        "site": (["S5", "S6", "S7"], 12_300_000)}
names, FIT = SETS[PH]; names = [m for m in names if os.path.exists(E.ckpt_path(m))]
Xa = E.sample(3000, FIT)["X"]; Xb = E.sample(2000, FIT + 1)["X"]


def r2(Fa, Aa, Fb, Ab):
    W = np.linalg.lstsq(np.c_[np.ones(len(Fa)), Fa], Aa, rcond=None)[0]; P = np.c_[np.ones(len(Fb)), Fb] @ W
    return float(1 - ((Ab - P) ** 2).sum() / ((Ab - Ab.mean(0)) ** 2).sum())


def feats(X, rng_seed):
    n, nc = E.loo(X), E.loo(X, E.CROSS); sh = np.random.default_rng(rng_seed).permutation(len(X)); out = {}
    for t in range(36):
        k, g = t // 6, (t % 6) // 2; own = E.onehot(n[:, k, g][:, None]); oth = [x for x in range(3) if x != g]
        out[t] = dict(own=own, own_other=E.onehot(n[:, k]), own_null=np.c_[own, E.onehot(nc[:, k, :2])],
                      own_shuf=np.c_[own, E.onehot(n[sh][:, k][:, oth])])
    for q in range(E.Q):
        v, vc = n[:, q], nc[:, q]
        pw = np.concatenate([np.eye(36)[v[:, a] * 6 + v[:, b]] for a, b in ((0, 1), (0, 2), (1, 2))], 1)
        pwn = np.concatenate([np.eye(36)[vc[:, a] * 6 + vc[:, b]] for a, b in ((0, 1), (0, 2), (1, 2))], 1)
        out[36 + q] = dict(additive=E.onehot(v), pairwise=np.c_[E.onehot(v), pw], nullpair=np.c_[E.onehot(v), pwn])
    return out


Fa, Fb = feats(Xa, 1), feats(Xb, 2); R = dict(phase=PH, per_model={})
for m in names:
    M = E.Model(m); Aa, Ab = M.acts(Xa)[0], M.acts(Xb)[0]; res = {}
    for t in range(36):
        for nm in Fa[t]: res.setdefault(f"source|{nm}", []).append(r2(Fa[t][nm], Aa[:, t], Fb[t][nm], Ab[:, t]))
    for q in range(E.Q):
        t = 36 + q
        for nm in Fa[t]: res.setdefault(f"readout|{nm}", []).append(r2(Fa[t][nm], Aa[:, t], Fb[t][nm], Ab[:, t]))
    out = {k: float(np.mean(v)) for k, v in res.items()}
    nb = E.loo(Xb)
    for site in ("P", "T"):
        dec = E.fit_decoder(site, Xa, Aa)
        out[f"decode_acc_at_{'readout' if site == 'P' else 'q_source_tokens'}"] = float(np.mean([(E.decode(dec, Ab, q)[1] == nb[:, q]).mean() for q in range(E.Q)]))
    R["per_model"][m] = out; print(m, json.dumps({k: round(v, 3) for k, v in out.items()}), flush=True)
os.makedirs(os.path.join(HERE, "..", "logs", PH), exist_ok=True)
json.dump(R, open(os.path.join(HERE, "..", "logs", PH, "info_structure.json"), "w"), indent=1)
