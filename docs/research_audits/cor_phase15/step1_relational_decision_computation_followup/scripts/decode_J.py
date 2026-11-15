"""Workstream C2 (XIII-B): is the cross-group identity relation (the ingredient of J_H^beta) present in frozen layers 0-1?
Episodes: C = 2, beta = 0.6 (stream 77_800_000; 8000 train / 4000 test).  Targets for each group pair (g,h): n4_gh = number of questions at which
all four sources of g and h are equal (J_H^beta = prod_q (1 - beta + beta K^3 [all four equal at q]) is a monotone function of n4 given the pair),
and its centred version after regressing out the per-group statistics (own self-agreement counts of g and h).
Feature sets (ridge regression, alpha chosen on a validation split): resid1 and resid2 at all 36 source tokens (flattened), resid2 at the readout
tokens, raw one-hot input (linear reference: equality is NOT linear in one-hot inputs), per-group statistics only (self-agreement counts).
Reported: test R^2 of n4 and of the per-group-residualised n4.  Optional checkpoint of an adapted early01 model (adapt2.py) for comparison.
Usage: decode_J.py SPEC [ADAPT2_NPZ] -> logs/decode_J<TAG>.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
from sklearn.linear_model import Ridge
import ftlib as F, genlib as GL
np.seterr(all="ignore"); import warnings; warnings.filterwarnings("ignore")
E2 = GL.Env(3, 2, beta=0.6); rng = np.random.default_rng(77_800_000); d = GL.sample(E2, 12000, rng); X = d["X"]
A = X.reshape(len(X), 6, 3, 2); selfag = (A[..., 0] == A[..., 1]).sum(1)                         # (n,3) own agreement counts
pairs = [(0, 1), (0, 2), (1, 2)]
n4 = np.stack([((A[:, :, g, 0] == A[:, :, g, 1]) & (A[:, :, h, 0] == A[:, :, h, 1]) & (A[:, :, g, 0] == A[:, :, h, 0])).sum(1) for g, h in pairs], 1).astype(float)
tr, va, te = np.arange(7000), np.arange(7000, 8000), np.arange(8000, 12000)
def feats(spec_or_none, ck=None):
    out = {}
    if spec_or_none is not None:
        parent, spec, seed, suf = (spec_or_none.split(":") + ["", "", "0", ""])[:4]; M = F.load_model(parent, spec, seed=int(seed or 0), suffix=suf)
        if ck is not None:
            z = np.load(ck); M = dict(M); M["layers"] = [dict(M["layers"][0]), dict(M["layers"][1]), M["layers"][2]]
            for k, v in z.items():
                top, *rest = k.split("/")
                if top in ("l0", "l1") and rest: M["layers"][int(top[1])][rest[0]] = jnp.asarray(v)
        R1, R2 = [], []
        f = jax.jit(lambda X_: (lambda r: (F.forward(M, X_, rec=r), (r["resid1"], r["resid2"]))[1])({}))
        for i in range(0, len(X), 1000):
            a, b = f(jnp.asarray(X[i:i + 1000])); R1.append(np.asarray(a)); R2.append(np.asarray(b))
        R1, R2 = np.concatenate(R1), np.concatenate(R2)
        out["resid1_src"] = R1[:, :36].reshape(len(X), -1); out["resid2_src"] = R2[:, :36].reshape(len(X), -1); out["resid2_ro"] = R2[:, 36:].reshape(len(X), -1)
    oh = np.eye(3)[X].reshape(len(X), -1); out["raw_onehot"] = oh; out["pergroup_stats"] = np.eye(7)[selfag].reshape(len(X), -1)
    return out
def ridge_r2(Fm, y):
    best = None
    for al in (1e-1, 1, 10, 100, 1000):
        m = Ridge(alpha=al).fit(Fm[tr], y[tr]); v = 1 - ((m.predict(Fm[va]) - y[va]) ** 2).sum() / ((y[va] - y[va].mean()) ** 2).sum()
        if best is None or v > best[0]: best = (v, al)
    m = Ridge(alpha=best[1]).fit(Fm[np.r_[tr, va]], y[np.r_[tr, va]]); p = m.predict(Fm[te]); return float(1 - ((p - y[te]) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum())
# per-group-residualised targets (remove what per-group statistics explain, fitted on train)
PG = feats(None)["pergroup_stats"]; resid_t = np.zeros_like(n4)
for j in range(3):
    m = Ridge(alpha=1.0).fit(PG[tr], n4[tr, j]); resid_t[:, j] = n4[:, j] - m.predict(PG)
res = dict(n4_mean=n4.mean(0).tolist(), n4_var=n4.var(0).tolist())
spec = sys.argv[1] if len(sys.argv) > 1 else None; ck = sys.argv[2] if len(sys.argv) > 2 else None
Fs = feats(spec, ck)
for nm, Fm in Fs.items():
    res[nm] = dict(R2_n4=[ridge_r2(Fm, n4[:, j]) for j in range(3)], R2_n4_beyond_pergroup=[ridge_r2(Fm, resid_t[:, j]) for j in range(3)])
    print((spec or "-")[:6], ("ADAPTED " if ck else ""), nm, {k: [round(x, 3) for x in v] for k, v in res[nm].items()}, flush=True)
json.dump(res, open(os.path.join(HERE, "..", "logs", "decode_J" + os.environ.get("DJ_TAG", "") + ".json"), "w"), indent=1)
