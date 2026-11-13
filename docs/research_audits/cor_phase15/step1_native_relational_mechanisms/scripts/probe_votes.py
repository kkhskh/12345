"""A2 supplement: is candidate identity (each source's own report at q) retained in native states?  Linear/MLP held-out probes.
Usage: probe_votes.py PARENT -> logs/probe_votes_<PARENT>.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, native as NV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import warnings; warnings.filterwarnings("ignore")
parent = sys.argv[1]; pf = NV.load_parent(parent); rng = np.random.default_rng(32_000_001)
n = 9000; X = NV.Bm.sample_fast(NV.ENV, n, rng)["X"]; q = rng.integers(0, 6, n); idx = np.arange(n); out = {}
for u in (0, 1, 2, 3):
    Z = NV.states_np(pf, X, u); r = {}
    for s in (0, 2, 4):
        H = Z[idx, q * 6 + s]; y = X[idx, q, s]
        r[f"own_vote_src{s}"] = dict(lin=round(LogisticRegression(max_iter=3000).fit(H[:6000], y[:6000]).score(H[6000:], y[6000:]), 3),
                                     mlp=round(MLPClassifier((128,), max_iter=300, random_state=0).fit(H[:6000], y[:6000]).score(H[6000:], y[6000:]), 3))
    H = Z[idx, 36 + q]; ys = X[idx, q, 0]
    r["readout_vote_src0"] = dict(lin=round(LogisticRegression(max_iter=3000).fit(H[:6000], ys[:6000]).score(H[6000:], ys[6000:]), 3))
    out[f"u{u}"] = r; print(u, r, flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", f"probe_votes_{parent}.json"), "w"), indent=1)
