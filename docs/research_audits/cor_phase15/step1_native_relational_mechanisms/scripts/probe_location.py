"""Workstream A2: information location in the original P-count Transformer (decodability only; NOT causal use).
Usage: probe_location.py PARENT [ORIG_TAG]   (ORIG_TAG: Investigation VIII trained original transfer, e.g. dev/xP80_orig -> layer-2 states)
Natural T4 decisions (stream 32_000_000), 9000 episodes x 1 random q; 6000 train / 3000 test.  Probes: multinomial logistic (linear) and a
small MLP (sklearn, 1 hidden layer of 128), both on held-out data; majority baseline.  Ridge R^2 for the Bayes centred logits.
Locations: readout token q and the two source tokens of each group at q, at resid_pre0 (embeddings), resid_pre1, resid_pre2 (= input of
the original downstream block) and, if ORIG_TAG given, resid_post2 (after the trained original downstream block).
Targets per decision: n_g (6 cl.), perf_g = [n_g = 5], agree_g(q), e_g = perf_g & agree_g(q), m_g (3 cl.), S = sum_g e_g, and the
reports of group g at q (pair one-hot, 9 cl.).  For source tokens: targets of the token's OWN group.   -> logs/probe_location_<PARENT>.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import native as NV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier
import warnings; warnings.filterwarnings("ignore")
parent = sys.argv[1]; orig = sys.argv[2] if len(sys.argv) > 2 else None
pf = NV.load_parent(parent); rng = np.random.default_rng(32_000_000)
n = 9000; d = NV.Bm.sample_fast(NV.ENV, n, rng); X = d["X"]; q = rng.integers(0, 6, n); idx = np.arange(n)
sys.path.insert(0, os.path.join(NV.ROOT, "step1_learned_relational_computation", "src")); import v8lib as V8
nn_ = V8.E.loo(X)[idx, q]                                   # n,3 counts (leave-question agreement)
a_ = X.reshape(n, 6, 3, 2)[idx, q]; agree = (a_[..., 0] == a_[..., 1]); perf = nn_ == 5; e = perf & agree
m = np.minimum(e.sum(1, keepdims=True) - e, 2); S = e.sum(1); rep = a_[..., 0] * 3 + a_[..., 1]
cB = NV.L.B.bayes_c(X, q)                                   # centred exact log posterior (n,3)
locs = {}
for u in (0, 1, 2, 3): locs[f"resid_pre{u}" if u < 3 else "pretrained_final_resid"] = NV.states_np(pf, X, u)
if orig:
    t = jax.tree_util.tree_map(jnp.asarray, V8.load_tree(os.path.join(NV.V8, "checkpoints", orig + ".npz")))
    z = jnp.asarray(locs["resid_pre2"]); locs["resid_post2"] = np.asarray(jax.jit(lambda z_: NV.NL.block(t["l2"], z_, l=2))(z))
tr, te = slice(0, 6000), slice(6000, n)
def clf(H, y):
    maj = np.bincount(y[te]).max() / len(y[te]) if len(np.unique(y[tr])) > 1 else 1.0
    if len(np.unique(y[tr])) < 2: return dict(lin=1.0, mlp=1.0, maj=1.0)
    lin = LogisticRegression(C=1.0, max_iter=3000).fit(H[tr], y[tr]).score(H[te], y[te])
    mlp = MLPClassifier((128,), max_iter=300, random_state=0).fit(H[tr], y[tr]).score(H[te], y[te])
    return dict(lin=round(lin, 3), mlp=round(mlp, 3), maj=round(float(maj), 3))
out = {}
for name, Z in locs.items():
    ro = Z[idx, 36 + q]; res = {}
    for g in range(3):
        for tname, tv in (("n", nn_[:, g]), ("perf", perf[:, g].astype(int)), ("agree_q", agree[:, g].astype(int)), ("e", e[:, g].astype(int)),
                          ("m", m[:, g]), ("rep", rep[:, g])):
            res[f"readout:{tname}_{g}"] = clf(ro, tv)
    res["readout:S"] = clf(ro, np.minimum(S, 2))
    rr = Ridge(1.0).fit(ro[tr], cB[tr]); res["readout:bayes_logits_R2"] = round(float(rr.score(ro[te], cB[te])), 3)
    for g in range(3):                                       # source tokens of group g at q (both sources concatenated)
        st = np.concatenate([Z[idx, q * 6 + 2 * g], Z[idx, q * 6 + 2 * g + 1]], -1)
        for tname, tv in (("n", nn_[:, g]), ("agree_q", agree[:, g].astype(int)), ("e", e[:, g].astype(int)), ("m", m[:, g])):
            res[f"src_g{g}:{tname}"] = clf(st, tv)
    st1 = Z[idx, q * 6 + 0]                                  # single source token (source 0 of group 0)
    res["src0_single:n_0"] = clf(st1, nn_[:, 0]); res["src0_single:agree_q_0"] = clf(st1, agree[:, 0].astype(int)); res["src0_single:e_0"] = clf(st1, e[:, 0].astype(int))
    out[name] = res; print(name, {k: v for k, v in res.items() if k.endswith("_0") or k.startswith("readout:S") or "R2" in k}, flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", f"probe_location_{parent}.json"), "w"), indent=1)
