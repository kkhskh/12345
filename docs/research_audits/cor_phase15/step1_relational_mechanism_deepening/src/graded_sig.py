"""Workstream D signature computed on the TRAINING distribution only (natural T4 episodes, stream 9_800_000):
  GCD (graded-count decodability): ridge R^2 of the per-episode agreement counts of the two NON-colluder groups (0..6,
  graded) from the model's episode representation (continuation pp2_sig.load: TF readout-mean + episode token; DeepSets
  pool).  Held-out half of 3000 episodes.  Also SAT: logistic accuracy of the colluder identity (saturated-feature control).
No off-support input is used."""
import os, sys
HERE = os.path.dirname(__file__); sys.path.insert(0, HERE)
import tasks3  # noqa
import numpy as np, gen_family as G, pp2_sig as S
from sklearn.linear_model import Ridge, LogisticRegression
f4 = G.Fam(); _d = G.sample(f4, 3000, np.random.default_rng(9_800_000)); _cf = f4.configs
_cnt = np.stack([(_d["X"][:, :, a] == _d["X"][:, :, b]).sum(1) for a, b in G.GROUPS], -1)
_h = np.array([_cf[c][1] for c in _d["cfg"]])
_non = np.array([[g for g in range(3) if g != h] for h in _h]); _tgt = np.take_along_axis(_cnt, _non, 1).astype(float)   # N,2
def signature(apply):
    _, r = apply(_d["X"]); n = len(r); tr = np.arange(n) < n // 2; mu, sd = r[tr].mean(0), r[tr].std(0) + 1e-6; Z = (r - mu) / sd
    r2 = []
    for j in range(2):
        m = Ridge(alpha=1.0).fit(Z[tr], _tgt[tr, j]); pr = m.predict(Z[~tr]); r2.append(1 - ((pr - _tgt[~tr, j]) ** 2).mean() / _tgt[~tr, j].var())
    sat = LogisticRegression(max_iter=3000).fit(Z[tr], _h[tr]).score(Z[~tr], _h[~tr])
    return dict(GCD=float(np.mean(r2)), SAT=float(sat))
